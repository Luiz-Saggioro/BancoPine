"""Simulated loan-level book calibrated to Banco Pine's published 2Q26 figures.

Loan-level data is confidential and not public. This module produces a statistically
realistic stand-in with the same segment balances, risk levels and business mix so the
predictive models, ledger and UI run end to end. To go to production, replace
`generate_loan_book` with a loader that returns the same columns from Pine's data warehouse
(see `LOAN_COLUMNS`); nothing downstream needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pine_config import RATINGS
from pine_reference import (
    CORPORATE_SECTORS,
    EMPLOYER_SECTORS,
    REGION_MIX,
    SEGMENT_BALANCE_2Q26_BN,
    SEGMENT_PARAMS,
    SEGMENT_YOY_GROWTH,
)

LOAN_COLUMNS: list[str] = [
    "loan_id", "client_id", "segment", "region", "sector", "rating", "prior_rating",
    "balance", "rate_month", "term_months", "months_on_book", "origination_q",
    "bureau_score", "dti", "ltv", "employer_tenure_m", "income", "age", "dpd",
    "default_12m", "churned_12m", "has_card", "has_insurance", "has_investment",
    "has_home_equity",
]
FEATURES_PD: list[str] = [
    "segment_code", "region_code", "rating_num", "rate_month", "term_months",
    "months_on_book", "bureau_score", "dti", "ltv", "employer_tenure_m", "log_income",
    "age", "dpd",
]
FEATURES_CHURN: list[str] = [
    "segment_code", "rate_month", "months_on_book", "bureau_score", "log_balance",
    "term_months", "age", "products_held",
]

_N_LOANS = {
    "Corporate": 900,
    "Consignado Privado": 14000,
    "Consignado INSS": 9000,
    "FGTS Antecipacao": 8000,
    "Home Equity": 1500,
    "Cartao Consignado": 6000,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(score: np.ndarray, target: float) -> float:
    """Find intercept b such that mean(sigmoid(score + b)) == target (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(score + mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _segment_frame(seg: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    p = SEGMENT_PARAMS[seg]
    corp = seg == "Corporate"
    sectors = CORPORATE_SECTORS if corp else EMPLOYER_SECTORS
    rating_num = np.clip(rng.normal(3.2 if corp else 3.6, 1.2, n).round(), 1, 7).astype(int)
    df = pd.DataFrame(
        {
            "segment": seg,
            "region": rng.choice(list(REGION_MIX), n, p=list(REGION_MIX.values())),
            "sector": rng.choice(list(sectors), n, p=list(sectors.values())),
            "rating_num": rating_num,
            "balance": rng.lognormal(0, 0.9 if corp else 0.7, n),
            "rate_month": np.clip(p["rate"] + rng.normal(0, p["rate"] * 0.12, n), 0.008, 0.05),
            "term_months": np.clip(rng.normal(p["term"], p["term"] * 0.25, n).round(), 6, 240),
            "bureau_score": np.clip(rng.normal(700 - 25 * (rating_num - 3), 70, n), 300, 1000),
            "income": rng.lognormal(np.log(250000 if corp else 4200), 0.6, n),
            "age": np.clip(rng.normal(45 if corp else 41, 11, n), 18, 85).round(),
        }
    )
    df["dti"] = np.clip(rng.normal(0.12 if corp else 0.28, 0.08, n), 0.01, 0.8)
    df["ltv"] = np.where(seg == "Home Equity", np.clip(rng.normal(0.45, 0.12, n), 0.1, 0.6), 0.0)
    tenure = rng.gamma(2.2, 30, n) if seg in ("Consignado Privado", "Cartao Consignado") else 0
    df["employer_tenure_m"] = np.round(tenure, 0) if np.ndim(tenure) else 0.0
    growth = SEGMENT_YOY_GROWTH[seg]
    young_share = min(0.85, growth / (1 + growth))  # fast-growing books are younger
    df["months_on_book"] = np.where(
        rng.random(n) < young_share, rng.integers(0, 12, n), rng.integers(12, 48, n)
    )
    df["dpd"] = rng.choice([0, 15, 45, 75], n, p=[0.93, 0.04, 0.02, 0.01])
    return df


def _assign_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    seg_churn = df["segment"].map(lambda s: SEGMENT_PARAMS[s]["churn"]).to_numpy()
    z = (
        0.55 * (df["rating_num"] - 3.5)
        - 0.006 * (df["bureau_score"] - 700)
        + 3.0 * (df["dti"] - 0.25)
        + 0.03 * df["dpd"]
        - 0.012 * np.minimum(df["employer_tenure_m"], 120)
        + 0.25 * (df["months_on_book"] < 6)
        + 1.8 * df["ltv"]
        + rng.normal(0, 0.6, len(df))
    ).to_numpy()
    pd_true = np.zeros(len(df))
    for seg in df["segment"].unique():
        m = (df["segment"] == seg).to_numpy()
        b = _calibrate_intercept(z[m], SEGMENT_PARAMS[seg]["pd"])
        pd_true[m] = _sigmoid(z[m] + b)
    rate_dev = df["rate_month"] / df.groupby("segment")["rate_month"].transform("mean") - 1
    zc = (
        0.004 * (df["bureau_score"] - 700)
        + 4.0 * rate_dev
        + 0.15 * np.log(df["balance"])
        - 0.02 * np.abs(df["months_on_book"] - 14)
        + rng.normal(0, 0.5, len(df))
    ).to_numpy()
    churn_p = _sigmoid(zc + np.log(seg_churn / (1 - seg_churn)))
    return df.assign(
        pd_true=pd_true,
        default_12m=(rng.random(len(df)) < pd_true).astype(int),
        churned_12m=(rng.random(len(df)) < churn_p).astype(int),
    )


def _assign_products(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    s = (df["bureau_score"] - 700) / 100
    inc = np.log(df["income"]) - np.log(4200)
    retail = df["segment"] != "Corporate"
    return df.assign(
        has_card=((rng.random(len(df)) < _sigmoid(-1.2 + 0.8 * s)) & retail).astype(int)
        | (df["segment"] == "Cartao Consignado").astype(int),
        has_insurance=(rng.random(len(df)) < _sigmoid(-1.5 + 0.02 * (df["age"] - 40))).astype(
            int
        ),
        has_investment=(rng.random(len(df)) < _sigmoid(-2.2 + 1.1 * inc + 0.4 * s)).astype(int),
        has_home_equity=(df["segment"] == "Home Equity").astype(int)
        | ((rng.random(len(df)) < _sigmoid(-3.5 + 0.9 * inc)) & retail).astype(int),
    )


def _scale_balances(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for seg, total_bn in SEGMENT_BALANCE_2Q26_BN.items():
        m = out["segment"] == seg
        out.loc[m, "balance"] = out.loc[m, "balance"] / out.loc[m, "balance"].sum() * total_bn * 1e9
    return out


def _prior_rating(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Rating 12 months earlier: mostly stable, defaults migrated from weak grades."""
    shift = rng.choice([-1, 0, 1], len(df), p=[0.12, 0.76, 0.12])
    prior = np.clip(df["rating_num"].to_numpy() - shift, 1, 7)
    prior = np.where(df["default_12m"] == 1, np.clip(prior + rng.integers(-1, 2, len(df)), 3, 7), prior)
    return pd.Series([RATINGS[i - 1] for i in prior], index=df.index)


def generate_loan_book(seed: int = 7) -> pd.DataFrame:
    """Build the calibrated simulated loan book.

    Args:
        seed: random seed (deterministic output).

    Returns:
        DataFrame with LOAN_COLUMNS plus model feature columns.
    """
    rng = np.random.default_rng(seed)
    parts = [_segment_frame(seg, n, rng) for seg, n in _N_LOANS.items()]
    df = pd.concat(parts, ignore_index=True)
    df = _assign_labels(df, rng)
    df = _assign_products(df, rng)
    df = _scale_balances(df)
    df["rating"] = np.where(
        df["default_12m"] == 1, "Default", [RATINGS[i - 1] for i in df["rating_num"]]
    )
    df["prior_rating"] = _prior_rating(df, rng)
    q_back = (df["months_on_book"] // 3).astype(int)
    df["origination_q"] = (pd.Period("2026Q2", freq="Q") - q_back.to_numpy()).astype(str)
    df["loan_id"] = [f"L{i:06d}" for i in range(len(df))]
    df["client_id"] = [f"C{i:06d}" for i in rng.permutation(len(df))]
    return add_model_features(df)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive encoded/log features used by the models (pure)."""
    seg_codes = {s: i for i, s in enumerate(SEGMENT_BALANCE_2Q26_BN)}
    reg_codes = {r: i for i, r in enumerate(REGION_MIX)}
    return df.assign(
        segment_code=df["segment"].map(seg_codes).astype(int),
        region_code=df["region"].map(reg_codes).astype(int),
        log_income=np.log(df["income"]),
        log_balance=np.log(df["balance"]),
        products_held=df[["has_card", "has_insurance", "has_investment", "has_home_equity"]].sum(
            axis=1
        ),
    )


def quarterly_segment_history(quarters: int = 9) -> pd.DataFrame:
    """Quarterly balance per segment (R$ bn), back-cast from 2Q26 using published YoY growth.

    The geometric path is anchored on the published 2Q26 level and 12-month growth;
    intermediate quarters are interpolated with mild seasonality, flagged as estimated.
    """
    periods = pd.period_range(end=pd.Period("2026Q2", freq="Q"), periods=quarters, freq="Q")
    rng = np.random.default_rng(3)
    rows = []
    for seg, bal in SEGMENT_BALANCE_2Q26_BN.items():
        q_growth = (1 + SEGMENT_YOY_GROWTH[seg]) ** 0.25
        for k, per in enumerate(periods):
            back = quarters - 1 - k
            noise = 1 + rng.normal(0, 0.012) if back not in (0, 4) else 1.0
            rows.append(
                {
                    "segment": seg,
                    "quarter": per.to_timestamp(how="end").normalize(),
                    "balance_bn": bal / q_growth**back * noise,
                }
            )
    return pd.DataFrame(rows)


def weekly_private_payroll_originations(weeks: int = 104, seed: int = 5) -> pd.DataFrame:
    """Weekly originations (R$ mn) for private payroll loans with a ramp and payday cycle."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp("2026-09-28"), periods=weeks, freq="W-MON")
    t = np.arange(weeks)
    ramp = 35 + 150 / (1 + np.exp(-(t - 60) / 9))  # product launch ramp (Mar 2025 law change)
    payday = 1 + 0.10 * np.cos(2 * np.pi * idx.day.to_numpy() / 30.5)
    vol = ramp * payday * (1 + rng.normal(0, 0.06, weeks))
    return pd.DataFrame({"week": idx, "originations_mn": vol})


def simulated_offers(n: int = 20000, seed: int = 9) -> pd.DataFrame:
    """Historic private-payroll offers with the offered monthly rate and acceptance outcome."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(0.012, 0.038, n)
    score = rng.normal(680, 80, n)
    comp = rate - rng.normal(0.025, 0.004, n)  # spread vs best competitor offer
    tenure = rng.gamma(2.2, 30, n)
    logit = 1.2 - 140 * comp + 0.004 * (score - 680) - 0.004 * tenure
    accepted = (rng.random(n) < _sigmoid(logit)).astype(int)
    return pd.DataFrame(
        {"rate_month": rate, "bureau_score": score, "competitor_gap": comp,
         "employer_tenure_m": tenure, "accepted": accepted}
    )
