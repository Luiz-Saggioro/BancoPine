"""Treasury and capital models: repricing gap, NII simulation, Basel Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pine_config import BASEL_MINIMUM
from pine_reference import BALANCE_SHEET_2Q26, SEGMENT_BALANCE_2Q26_BN, SEGMENT_PARAMS

BUCKETS = ["0-3m", "3-6m", "6-12m", "1-2y", "2y+"]
_BUCKET_EDGES = [0, 3, 6, 12, 24, 10_000]


@dataclass(frozen=True)
class Instrument:
    """One balance-sheet line."""

    name: str
    side: str  # "asset" | "liability"
    balance_bn: float
    floating: bool  # reprices with CDI every month
    spread: float  # annual spread over CDI (floating) or annual fixed rate (fixed)
    cdi_pct: float = 1.0  # % of CDI for floating instruments
    maturity_m: int = 1  # average life for fixed instruments (uniform runoff)


def default_balance_sheet() -> list[Instrument]:
    """Simplified 2Q26 balance sheet consistent with published totals (R$25.6bn funding)."""
    lines = [
        Instrument("Corporate loans", "asset", SEGMENT_BALANCE_2Q26_BN["Corporate"], True, 0.045),
        Instrument("Treasury (LFT)", "asset", BALANCE_SHEET_2Q26["treasury_securities_bn"], True, 0.0),
    ]
    for seg in SEGMENT_BALANCE_2Q26_BN:
        if seg == "Corporate":
            continue
        p = SEGMENT_PARAMS[seg]
        lines.append(Instrument(
            seg, "asset", SEGMENT_BALANCE_2Q26_BN[seg], False,
            (1 + p["rate"]) ** 12 - 1, maturity_m=int(p["term"] // 2),
        ))
    lines += [
        Instrument("CDB post-fixed", "liability", 14.0, True, 0.0, cdi_pct=1.03),
        Instrument("LCA / LCI", "liability", 4.0, True, 0.0, cdi_pct=0.92),
        Instrument("Securitizations", "liability", 2.0, True, 0.015),
        Instrument("Pre-fixed deposits", "liability", 3.0, False, 0.140, maturity_m=12),
        Instrument("Financial bills", "liability", 2.6, False, 0.135, maturity_m=24),
    ]
    return lines


def repricing_gap(lines: list[Instrument]) -> pd.DataFrame:
    """Assets minus liabilities repricing in each time bucket (R$ bn)."""
    rows = []
    for ins in lines:
        sign = 1 if ins.side == "asset" else -1
        if ins.floating:
            shares = [1.0, 0, 0, 0, 0]
        else:
            mat = max(ins.maturity_m, 1)
            shares = [
                max(0.0, min(hi, mat) - lo) / mat
                for lo, hi in zip(_BUCKET_EDGES[:-1], _BUCKET_EDGES[1:], strict=True)
            ]
        for b, s in zip(BUCKETS, shares, strict=True):
            rows.append({"bucket": b, "side": ins.side, "amount_bn": sign * ins.balance_bn * s})
    df = pd.DataFrame(rows).groupby(["bucket", "side"], as_index=False)["amount_bn"].sum()
    df["bucket"] = pd.Categorical(df["bucket"], BUCKETS, ordered=True)
    return df.sort_values("bucket")


def simulate_nii(
    lines: list[Instrument], selic_path: pd.Series, pass_through: float = 0.5
) -> pd.Series:
    """Monthly net interest income (R$ mn) along a Selic path.

    Floating lines reprice monthly. Fixed lines run off uniformly and are replaced at the
    initial rate plus `pass_through` x the change in Selic since today.
    """
    s0 = float(selic_path.iloc[0]) / 100
    out = []
    for d, selic in selic_path.items():
        cdi = float(selic) / 100 - 0.001
        nii = 0.0
        months = len(out) + 1
        for ins in lines:
            if ins.floating:
                annual = cdi * ins.cdi_pct + ins.spread
            else:
                run = min(months / max(ins.maturity_m, 1), 1.0)
                new_rate = ins.spread + pass_through * (float(selic) / 100 - s0)
                annual = (1 - run) * ins.spread + run * new_rate
            monthly = (1 + annual) ** (1 / 12) - 1
            nii += (1 if ins.side == "asset" else -1) * ins.balance_bn * 1e3 * monthly
        out.append((d, nii))
    return pd.Series(dict(out), name="nii_mn")


# ------------------------------------------------------------------------ capital
@dataclass
class CapitalProjection:
    """Basel ratio fan chart and breach probability."""

    fan: pd.DataFrame  # index=quarter, p10/p50/p90 Basel ratio
    p_breach: float  # probability of falling below the minimum within the horizon
    max_growth_90: float  # highest annual loan growth with P(breach) <= 10%


def _simulate_basel(
    growth: float, payout: float, quarters: int, ni_base: float, ni_vol: float,
    n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    pr = BALANCE_SHEET_2Q26["reference_equity_bn"] * 1e3
    rwa = BALANCE_SHEET_2Q26["rwa_bn"] * 1e3
    qg = (1 + growth) ** 0.25 - 1
    ratios = np.empty((n_paths, quarters))
    pr_p = np.full(n_paths, pr)
    rwa_p = np.full(n_paths, rwa)
    for q in range(quarters):
        ni = ni_base * (1 + 0.6 * qg) ** (q + 1) * np.exp(rng.normal(0, ni_vol, n_paths))
        pr_p = pr_p + ni * (1 - payout)
        rwa_p = rwa_p * (1 + qg + rng.normal(0, 0.015, n_paths))
        ratios[:, q] = pr_p / rwa_p
    return ratios


def project_basel(
    ni_history_mn: pd.Series, growth: float, payout: float, quarters: int = 8,
    n_paths: int = 3000, seed: int = 1,
) -> CapitalProjection:
    """Monte Carlo of the Basel ratio given loan growth, payout and earnings uncertainty.

    Earnings start at the mean of the last two quarters with volatility from the
    quarter-on-quarter log changes of history.
    """
    rng = np.random.default_rng(seed)
    ni = ni_history_mn.dropna()
    ni_base = float(ni.iloc[-2:].mean())
    ni_vol = float(np.clip(np.diff(np.log(ni.to_numpy())).std(ddof=1), 0.08, 0.35))
    ratios = _simulate_basel(growth, payout, quarters, ni_base, ni_vol, n_paths, rng)
    idx = pd.period_range(pd.Period("2026Q3", freq="Q"), periods=quarters, freq="Q")
    fan = pd.DataFrame(
        np.percentile(ratios, [10, 50, 90], axis=0).T, columns=["p10", "p50", "p90"],
        index=idx.to_timestamp(how="end").normalize(),
    )
    p_breach = float((ratios.min(axis=1) < BASEL_MINIMUM).mean())
    grid = np.arange(0.0, 0.81, 0.02)
    ok = [
        g for g in grid
        if (_simulate_basel(g, payout, quarters, ni_base, ni_vol, 800,
                            np.random.default_rng(seed)).min(axis=1) < BASEL_MINIMUM).mean() <= 0.10
    ]
    return CapitalProjection(fan, p_breach, float(max(ok)) if ok else 0.0)
