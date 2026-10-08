"""Cached access to live data and fitted models (the only place the UI gets data from)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

import model_clients as clients
import model_credit as credit
import model_forecasting as forecasting
import pine_live as live
from pine_config import BENCHMARK, PEERS, SGS_SERIES, TICKER, get_settings
from pine_db import get_engine
from pine_portfolio import (
    generate_loan_book,
    quarterly_segment_history,
    simulated_offers,
    weekly_private_payroll_originations,
)

log = logging.getLogger(__name__)
TTL = get_settings().live_ttl_seconds
MODEL_VERSION = "1.0.0"


@st.cache_resource(show_spinner=False)
def engine():
    """Shared SQLAlchemy engine."""
    return get_engine(get_settings().database_url)


# ------------------------------------------------------------------------ market
@st.cache_data(ttl=TTL, show_spinner="Loading B3 prices")
def prices() -> live.LiveResult:
    """Daily closes for Pine, peers and Ibovespa."""
    return live.fetch_prices([TICKER, *PEERS, BENCHMARK], years=5)


@st.cache_data(ttl=TTL, show_spinner="Training price forecast model")
def price_forecast(ticker: str, last_date: pd.Timestamp) -> forecasting.PriceForecast:
    """20-business-day quantile forecast for one ticker (cache keyed by last data date)."""
    px = prices().data
    return forecasting.quantile_price_forecast(px[ticker], px.get(BENCHMARK))


@st.cache_data(ttl=TTL, show_spinner="Scoring peer outlook")
def peer_outlook(last_date: pd.Timestamp) -> pd.DataFrame:
    """Model P10/P50/P90 20-day return for every listed bank."""
    px = prices().data
    rows = []
    for t in [TICKER, *PEERS]:
        if t in px and px[t].dropna().shape[0] > 400:
            lo, med, hi = forecasting.expected_return_distribution(px[t])
            rows.append({"ticker": t, "p10": np.expm1(lo), "p50": np.expm1(med), "p90": np.expm1(hi)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=TTL, show_spinner=False)
def regimes(ticker: str, last_date: pd.Timestamp) -> pd.DataFrame:
    """Market regime labels for a ticker."""
    return forecasting.detect_regimes(prices().data[ticker])


# ------------------------------------------------------------------------- macro
@dataclass
class Macro:
    """Monthly macro series, Focus expectations and model outputs."""

    selic: pd.Series
    ipca_m: pd.Series
    ipca12: pd.Series
    usd: pd.Series
    npl_pf: pd.Series
    npl_pj: pd.Series
    focus_selic: pd.DataFrame
    focus_ipca: pd.DataFrame
    sources: dict[str, bool]


@st.cache_data(ttl=TTL, show_spinner="Loading Banco Central data")
def macro() -> Macro:
    """Live macro data from SGS and Focus (with fallbacks)."""
    res = {k: live.fetch_sgs(code) for k, code in SGS_SERIES.items()}
    fs, fi = live.fetch_focus("Selic"), live.fetch_focus("IPCA")
    ipca_m = forecasting.monthly(res["ipca_month"].data)
    return Macro(
        selic=forecasting.monthly(res["selic_target"].data),
        ipca_m=ipca_m,
        ipca12=forecasting.ipca_12m(ipca_m).dropna(),
        usd=forecasting.monthly(res["usd_brl"].data, how="mean"),
        npl_pf=forecasting.monthly(res["npl_pf"].data),
        npl_pj=forecasting.monthly(res["npl_pj"].data),
        focus_selic=fs.data, focus_ipca=fi.data,
        sources={**{k: v.is_live for k, v in res.items()}, "focus": fs.is_live and fi.is_live},
    )


def selic_paths(m: Macro, shock_bps: int, months: int = 24) -> dict[str, pd.Series]:
    """Focus consensus path, shocked path and flat path."""
    base = forecasting.selic_path_from_focus(float(m.selic.iloc[-1]), m.focus_selic, months)
    return {
        "Focus consensus": base,
        f"Focus {shock_bps:+d} bp": base + shock_bps / 100,
        "Flat (unchanged)": pd.Series(float(m.selic.iloc[-1]), index=base.index),
    }


@st.cache_data(ttl=TTL, show_spinner="Forecasting inflation")
def ipca_forecast() -> pd.DataFrame:
    """SARIMA 12-month IPCA forecast."""
    return forecasting.forecast_ipca(macro().ipca_m)


@st.cache_resource(ttl=TTL, show_spinner="Fitting delinquency models")
def npl_models() -> dict[str, forecasting.NplModel]:
    """Household and corporate delinquency models."""
    m = macro()
    return {
        "PF": forecasting.fit_npl_model(m.npl_pf, m.selic, m.ipca12),
        "PJ": forecasting.fit_npl_model(m.npl_pj, m.selic, m.ipca12),
    }


@st.cache_data(ttl=TTL, show_spinner=False)
def npl_forecasts(shock_bps: int) -> dict[str, pd.DataFrame]:
    """NPL forecasts per path for households (PF) and companies (PJ)."""
    paths = selic_paths(macro(), shock_bps)
    return {
        k: pd.DataFrame({name: forecasting.forecast_npl(mdl, p) for name, p in paths.items()})
        for k, mdl in npl_models().items()
    }


def pd_multipliers(shock_bps: int) -> tuple[float, float]:
    """Retail and corporate PD multipliers implied by the shocked 12m NPL forecast."""
    fc = npl_forecasts(shock_bps)
    m = macro()
    col = f"Focus {shock_bps:+d} bp"
    retail = float(fc["PF"][col].iloc[-1] / m.npl_pf.iloc[-1])
    corp = float(fc["PJ"][col].iloc[-1] / m.npl_pj.iloc[-1])
    return float(np.clip(retail, 0.5, 3.0)), float(np.clip(corp, 0.5, 3.0))


# ------------------------------------------------------------------- credit book
@dataclass
class CreditBundle:
    """Scored loan book and fitted models."""

    book: pd.DataFrame
    pd_model: credit.PdModel
    churn: clients.ChurnModel


@st.cache_resource(show_spinner="Training credit and client models")
def credit_bundle() -> CreditBundle:
    """Generate the calibrated book and fit PD and churn models once per process."""
    seed = get_settings().random_seed
    book = generate_loan_book(seed)
    pd_model = credit.train_pd_model(book, seed)
    scored = credit.score_pd(book, pd_model)
    churn = clients.train_churn_model(scored, seed)
    return CreditBundle(scored, pd_model, churn)


@st.cache_data(show_spinner=False)
def segment_history() -> pd.DataFrame:
    """Quarterly balances per segment."""
    return quarterly_segment_history()


@st.cache_data(show_spinner=False)
def vintages() -> pd.DataFrame:
    """Observed and predicted vintage curves."""
    return credit.fit_vintage_curves(credit.simulate_vintages())


@st.cache_data(ttl=TTL, show_spinner=False)
def client_scores(cdi: float) -> pd.DataFrame:
    """Churn, margin at risk, retention action and next best offer for retail clients."""
    b = credit_bundle()
    scored = clients.score_clients(b.book, b.churn, cdi)
    return scored.join(clients.next_best_offer(b.book))


@st.cache_resource(show_spinner=False)
def acceptance_model():
    """Offer acceptance model for private payroll pricing."""
    return clients.train_acceptance_model(simulated_offers())


@st.cache_data(show_spinner=False)
def originations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly private payroll originations and their 12-week forecast."""
    w = weekly_private_payroll_originations()
    return w, clients.forecast_originations(w)
