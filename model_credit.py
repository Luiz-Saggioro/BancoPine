"""Credit-risk models: probability of default, rating migration, ECL and vintage curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from pine_config import RATINGS
from pine_portfolio import FEATURES_PD
from pine_reference import SEGMENT_PARAMS

FEATURE_LABELS = {
    "segment_code": "Segment", "region_code": "Region", "rating_num": "Internal rating",
    "rate_month": "Contract rate", "term_months": "Term", "months_on_book": "Months on book",
    "bureau_score": "Bureau score", "dti": "Debt-to-income", "ltv": "Loan-to-value",
    "employer_tenure_m": "Employer tenure", "log_income": "Income", "age": "Age",
    "dpd": "Days past due", "log_balance": "Balance", "products_held": "Products held",
}


@dataclass
class PdModel:
    """Fitted PD model with validation diagnostics."""

    model: HistGradientBoostingClassifier
    auc: float
    gini: float
    brier: float
    calibration: pd.DataFrame  # predicted vs observed by bin
    importance: pd.DataFrame  # feature, importance


def train_pd_model(book: pd.DataFrame, seed: int = 0) -> PdModel:
    """Gradient-boosted 12-month PD model with holdout AUC, Brier and calibration."""
    x, y = book[FEATURES_PD], book["default_12m"]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, min_samples_leaf=60,
        l2_regularization=1.0, random_state=seed,
    ).fit(xtr, ytr)
    p = model.predict_proba(xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    obs, pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
    sample = xte.sample(min(4000, len(xte)), random_state=seed)
    imp = permutation_importance(
        model, sample, yte.loc[sample.index], scoring="roc_auc", n_repeats=3, random_state=seed
    )
    importance = (
        pd.DataFrame({"feature": FEATURES_PD, "importance": imp.importances_mean})
        .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
        .sort_values("importance", ascending=False)
    )
    return PdModel(
        model=model, auc=auc, gini=2 * auc - 1, brier=float(brier_score_loss(yte, p)),
        calibration=pd.DataFrame({"predicted": pred, "observed": obs}), importance=importance,
    )


def score_pd(book: pd.DataFrame, pd_model: PdModel) -> pd.DataFrame:
    """Attach model PD (`pd_hat`) to every loan."""
    return book.assign(pd_hat=pd_model.model.predict_proba(book[FEATURES_PD])[:, 1])


# ------------------------------------------------------------------- migration
def migration_matrix(book: pd.DataFrame) -> pd.DataFrame:
    """Empirical 12-month rating transition matrix (rows: prior rating, cols: current)."""
    if book.empty:
        return pd.DataFrame(0.0, index=RATINGS, columns=RATINGS)
    m = pd.crosstab(book["prior_rating"], book["rating"], normalize="index")
    m = m.reindex(index=RATINGS, columns=RATINGS, fill_value=0.0)
    m.loc["Default"] = 0.0
    m.loc["Default", "Default"] = 1.0  # absorbing state
    return m


def project_rating_distribution(
    book: pd.DataFrame, matrix: pd.DataFrame, years: int = 2
) -> pd.DataFrame:
    """Balance-weighted rating distribution today and projected `years` ahead (Markov)."""
    cur = book.groupby("rating")["balance"].sum().reindex(RATINGS, fill_value=0.0)
    cur = cur / cur.sum() if cur.sum() else cur
    rows = {"Today": cur.to_numpy()}
    state = cur.to_numpy()
    p = matrix.to_numpy()
    for y in range(1, years + 1):
        state = state @ p
        rows[f"+{y}y projected"] = state
    return pd.DataFrame(rows, index=RATINGS)


# ------------------------------------------------------------------------- ECL
SCENARIOS = {"Base": 1.0, "Adverse": 1.35, "Severe": 1.8}


def expected_credit_loss(
    scored: pd.DataFrame, retail_mult: float, corporate_mult: float, scenario: str
) -> pd.DataFrame:
    """12-month ECL = PD x LGD x EAD by segment under a macro scenario.

    Args:
        scored: loan book with `pd_hat`.
        retail_mult: PD multiplier from the household NPL forecast.
        corporate_mult: PD multiplier from the corporate NPL forecast.
        scenario: key of SCENARIOS (scales the macro multiplier).
    """
    if scored.empty:
        return pd.DataFrame(columns=["segment", "ead_bn", "ecl_mn", "ecl_rate"])
    k = SCENARIOS[scenario]
    mult = np.where(scored["segment"] == "Corporate", corporate_mult, retail_mult) * k
    lgd = scored["segment"].map(lambda s: SEGMENT_PARAMS[s]["lgd"])
    lgd = lgd * np.where(scored["ltv"] > 0, 0.6 + scored["ltv"], 1.0)
    ecl = np.clip(scored["pd_hat"] * mult, 0, 1) * lgd * scored["balance"]
    out = (
        scored.assign(ecl=ecl)
        .groupby("segment", as_index=False)
        .agg(ead=("balance", "sum"), ecl=("ecl", "sum"))
    )
    return out.assign(ead_bn=out["ead"] / 1e9, ecl_mn=out["ecl"] / 1e6,
                      ecl_rate=out["ecl"] / out["ead"])[["segment", "ead_bn", "ecl_mn", "ecl_rate"]]


# -------------------------------------------------------------------- vintages
def _weibull_cdf(m: np.ndarray, lam: float, k: float) -> np.ndarray:
    return 1 - np.exp(-np.power(np.maximum(m, 0) / lam, k))


def simulate_vintages(selic_by_quarter: pd.Series | None = None, seed: int = 4) -> pd.DataFrame:
    """Observed cumulative over-90 curves for the last 12 origination cohorts.

    Cohort loss level rises with the Selic at origination (affordability) and with the
    share of private payroll in the mix. Only months already elapsed are observed.
    """
    rng = np.random.default_rng(seed)
    cohorts = pd.period_range(end=pd.Period("2026Q2", freq="Q"), periods=12, freq="Q")
    rows = []
    for i, c in enumerate(cohorts):
        age_max = (len(cohorts) - 1 - i) * 3 + 2
        sel = 12.0 if selic_by_quarter is None else float(selic_by_quarter.iloc[i % len(selic_by_quarter)])
        level = 0.030 + 0.0012 * (sel - 10) + 0.0015 * i / len(cohorts) + rng.normal(0, 0.002)
        for m in range(1, min(age_max, 30) + 1):
            val = level * _weibull_cdf(np.array([m]), 11.0, 1.6)[0] * (1 + rng.normal(0, 0.03))
            rows.append({"cohort": str(c), "mob": m, "over90": max(val, 0.0)})
    return pd.DataFrame(rows)


def fit_vintage_curves(vint: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Predict the full curve for each cohort: shared Weibull shape, cohort-specific level.

    Shape (lambda, k) is fitted on cohorts with >= 18 months; each cohort level is the
    least-squares scale on its observed points. Returns observed + predicted with a band.
    """
    mature = vint.groupby("cohort")["mob"].max()
    mat = vint[vint["cohort"].isin(mature[mature >= 18].index)]
    lvl0 = mat.groupby("cohort")["over90"].max()
    norm = mat.assign(y=mat["over90"] / mat["cohort"].map(lvl0) * 0.9)
    (lam, k), _ = curve_fit(_weibull_cdf, norm["mob"], norm["y"], p0=(10, 1.5), maxfev=5000)
    out = []
    for c, g in vint.groupby("cohort"):
        shape = _weibull_cdf(g["mob"].to_numpy(), lam, k)
        level = float(np.dot(shape, g["over90"]) / max(np.dot(shape, shape), 1e-12))
        resid = g["over90"].to_numpy() - level * shape
        sd = float(np.std(resid)) if len(g) > 2 else level * 0.1
        mobs = np.arange(1, horizon + 1)
        curve = level * _weibull_cdf(mobs, lam, k)
        unseen = mobs > g["mob"].max()
        widen = 1.2816 * (sd + 0.08 * level * np.sqrt(np.maximum(mobs - g["mob"].max(), 0) / 6))
        out.append(pd.DataFrame({
            "cohort": c, "mob": mobs, "predicted": curve,
            "lo": np.where(unseen, curve - widen, np.nan),
            "hi": np.where(unseen, curve + widen, np.nan),
            "is_forecast": unseen, "lifetime_level": level,
        }))
    pred = pd.concat(out, ignore_index=True)
    return pred.merge(vint, on=["cohort", "mob"], how="left")
