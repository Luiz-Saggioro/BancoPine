"""Client models: portability/churn risk, next-best-offer, pricing elasticity, demand."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pine_portfolio import FEATURES_CHURN

PRODUCTS = {
    "has_card": "Payroll credit card",
    "has_insurance": "Credit life insurance",
    "has_investment": "CDB / investments",
    "has_home_equity": "Home equity",
}
_NBO_FEATURES = ["bureau_score", "log_income", "age", "dti", "months_on_book", "segment_code"]


@dataclass
class ChurnModel:
    """Portability (refinancing-away) model and holdout AUC."""

    model: HistGradientBoostingClassifier
    auc: float


def train_churn_model(book: pd.DataFrame, seed: int = 0) -> ChurnModel:
    """Gradient-boosted 12-month portability/prepayment classifier (retail book)."""
    retail = book[book["segment"] != "Corporate"]
    x, y = retail[FEATURES_CHURN], retail["churned_12m"]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    m = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=80, random_state=seed
    ).fit(xtr, ytr)
    return ChurnModel(m, float(roc_auc_score(yte, m.predict_proba(xte)[:, 1])))


def score_clients(book: pd.DataFrame, churn: ChurnModel, cdi_annual: float) -> pd.DataFrame:
    """Churn probability, annual margin at risk and the recommended retention action.

    Action tiers are relative: top 5% churn risk gets an offer (rate match when credit risk is
    low, otherwise a term-extension refinance), the next 15% a relationship call.
    """
    retail = book[book["segment"] != "Corporate"].copy()
    retail["p_churn"] = churn.model.predict_proba(retail[FEATURES_CHURN])[:, 1]
    funding_m = (1 + cdi_annual) ** (1 / 12) - 1
    spread = np.maximum(retail["rate_month"] - funding_m, 0) * 12
    retail["margin_at_risk"] = retail["p_churn"] * retail["balance"] * spread * (1 - retail["pd_hat"])
    hi, mid = np.quantile(retail["p_churn"], [0.95, 0.80]) if len(retail) else (1.0, 1.0)
    retail["action"] = np.select(
        [
            (retail["p_churn"] >= hi) & (retail["pd_hat"] < 0.03),
            (retail["p_churn"] >= hi),
            (retail["p_churn"] >= mid),
        ],
        ["Counter-offer: rate match", "Refinance with term extension", "Proactive relationship call"],
        default="Monitor",
    )
    return retail


def next_best_offer(book: pd.DataFrame) -> pd.DataFrame:
    """Per-product propensity models; returns the best product not yet held and its score."""
    retail = book[book["segment"] != "Corporate"]
    scores = {}
    for col in PRODUCTS:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        model.fit(retail[_NBO_FEATURES], retail[col])
        p = model.predict_proba(retail[_NBO_FEATURES])[:, 1]
        scores[col] = np.where(retail[col] == 1, 0.0, p)
    sc = pd.DataFrame(scores, index=retail.index)
    best = sc.idxmax(axis=1)
    return pd.DataFrame(
        {"nbo": best.map(PRODUCTS), "nbo_score": sc.max(axis=1)}, index=retail.index
    ).join(sc.rename(columns=PRODUCTS))


# ---------------------------------------------------------------------- pricing
def train_acceptance_model(offers: pd.DataFrame):
    """Logistic acceptance model P(accept | rate gap to competitor, score, tenure)."""
    cols = ["competitor_gap", "bureau_score", "employer_tenure_m"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    return model.fit(offers[cols], offers["accepted"]), cols


def pricing_curve(
    model, cols: list[str], rates: np.ndarray, competitor_rate: float, funding_month: float,
    pd_annual: float, lgd: float, leads: float, avg_ticket: float, term: int = 36,
    score: float = 680.0, tenure: float = 60.0,
) -> pd.DataFrame:
    """Expected monthly volume and lifetime contribution for each offered monthly rate.

    Contribution per R$ = (rate - funding) * avg life - PD * LGD, where avg life ~ term/2.
    """
    x = pd.DataFrame({
        "competitor_gap": rates - competitor_rate,
        "bureau_score": score,
        "employer_tenure_m": tenure,
    })[cols]
    p_accept = model.predict_proba(x)[:, 1]
    volume = leads * p_accept * avg_ticket
    unit = (rates - funding_month) * (term / 2) - pd_annual * (term / 24) * lgd
    return pd.DataFrame({
        "rate": rates, "p_accept": p_accept, "volume_mn": volume / 1e6,
        "contribution_mn": volume * unit / 1e6,
    })


# ----------------------------------------------------------------------- demand
def forecast_originations(weekly: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
    """ETS (additive damped trend) forecast of weekly originations with an 80% band."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = weekly.set_index("week")["originations_mn"].asfreq("W-MON")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False)
        fc = res.get_prediction(start=len(y), end=len(y) + weeks - 1).summary_frame(alpha=0.2)
    return pd.DataFrame(
        {"p50": fc["mean"], "p10": fc["pi_lower"], "p90": fc["pi_upper"]}, index=fc.index
    )
