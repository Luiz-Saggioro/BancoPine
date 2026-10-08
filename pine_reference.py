"""Published Banco Pine reference figures used to calibrate the simulated loan book.

Sources (public press coverage of Banco Pine earnings releases):
- 2Q26: expanded portfolio R$21.8bn (+40% YoY); collateralized retail R$14.7bn (+50.6% YoY,
  +13.7% QoQ); wholesale R$7.1bn (+22% YoY); private payroll R$6.5bn (+183% YoY);
  net income R$165.7mn; ROAE 37.5%; NIM 12.5%; over-90 2.7%; coverage 105%;
  Basel 14.3%; reference equity R$2.55bn; RWA R$17.82bn; total funding R$25.6bn;
  shareholders' equity R$1.8bn; wholesale sector mix real estate 21%, grains 17%.
- 4Q25: expanded portfolio R$17.7bn; net income R$183.5mn; Basel 15.0%.
- 3Q25 net income R$103.6mn; 1H25 net income R$156.5mn (2Q25 R$83.0mn).

Replace or extend these with the Investor Relations spreadsheet (ri.pine.com) when available.
Values marked `estimated=True` are interpolations, not published numbers.
"""

from __future__ import annotations

import pandas as pd

# Balance by segment at 2Q26, R$ billions. Retail split inside R$14.7bn is an estimate
# consistent with "yield route" (private payroll + cards) = 51.8% of retail.
SEGMENT_BALANCE_2Q26_BN: dict[str, float] = {
    "Corporate": 7.1,
    "Consignado Privado": 6.5,
    "Consignado INSS": 3.6,
    "FGTS Antecipacao": 2.4,
    "Home Equity": 1.2,
    "Cartao Consignado": 1.0,
}

# Year-over-year growth used to back-cast quarterly history per segment.
SEGMENT_YOY_GROWTH: dict[str, float] = {
    "Corporate": 0.22,
    "Consignado Privado": 1.83,
    "Consignado INSS": 0.12,
    "FGTS Antecipacao": 0.05,
    "Home Equity": 0.20,
    "Cartao Consignado": 0.60,
}

# Per-segment risk and pricing parameters (monthly rates, 12m PD, LGD).
SEGMENT_PARAMS: dict[str, dict[str, float]] = {
    "Corporate": {"pd": 0.012, "lgd": 0.40, "rate": 0.0150, "term": 24, "churn": 0.06},
    "Consignado Privado": {"pd": 0.045, "lgd": 0.55, "rate": 0.0290, "term": 36, "churn": 0.14},
    "Consignado INSS": {"pd": 0.020, "lgd": 0.45, "rate": 0.0175, "term": 84, "churn": 0.12},
    "FGTS Antecipacao": {"pd": 0.008, "lgd": 0.30, "rate": 0.0170, "term": 60, "churn": 0.04},
    "Home Equity": {"pd": 0.018, "lgd": 0.25, "rate": 0.0140, "term": 180, "churn": 0.05},
    "Cartao Consignado": {"pd": 0.060, "lgd": 0.70, "rate": 0.0290, "term": 36, "churn": 0.08},
}

CORPORATE_SECTORS: dict[str, float] = {
    "Real Estate": 0.21,
    "Agribusiness": 0.17,
    "Energy": 0.14,
    "Infrastructure": 0.12,
    "Industry": 0.12,
    "Services": 0.10,
    "Financial": 0.08,
    "Retail Trade": 0.06,
}
EMPLOYER_SECTORS: dict[str, float] = {
    "Public Sector": 0.30,
    "Industry": 0.18,
    "Services": 0.22,
    "Retail Trade": 0.15,
    "Agribusiness": 0.08,
    "Construction": 0.07,
}
REGION_MIX: dict[str, float] = {
    "Southeast": 0.48,
    "South": 0.17,
    "Center-West": 0.14,
    "Northeast": 0.15,
    "North": 0.06,
}

BALANCE_SHEET_2Q26: dict[str, float] = {
    "reference_equity_bn": 2.55,
    "rwa_bn": 17.82,
    "basel_ratio": 0.143,
    "total_funding_bn": 25.6,
    "equity_bn": 1.8,
    "treasury_securities_bn": 6.0,  # estimate: liquidity book
    "npl90": 0.027,
    "coverage": 1.05,
}


def quarterly_results() -> pd.DataFrame:
    """Return quarterly net income (R$ mn), expanded portfolio (R$ bn) and Basel ratio."""
    rows = [
        ("4Q24", 67.1, 14.3, None, False),
        ("1Q25", 73.5, 14.9, None, True),
        ("2Q25", 83.0, 15.6, None, False),
        ("3Q25", 103.6, 16.6, None, True),
        ("4Q25", 183.5, 17.7, 0.150, False),
        ("1Q26", 149.9, 19.6, None, True),
        ("2Q26", 165.7, 21.8, 0.143, False),
    ]
    df = pd.DataFrame(
        rows, columns=["quarter", "net_income_mn", "portfolio_bn", "basel", "portfolio_estimated"]
    )
    df["period"] = pd.PeriodIndex(
        [f"20{q[2:]}Q{q[0]}" for q in df["quarter"]], freq="Q"
    ).to_timestamp(how="end").normalize()
    return df
