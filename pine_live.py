"""Live data loaders: B3 prices (Yahoo Finance) and Banco Central do Brasil open data.

Every loader returns a `LiveResult` carrying the data and a flag telling the UI whether it
came from the live source or from the deterministic fallback (used when the network or the
upstream API is unavailable, so the dashboard never breaks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from pine_config import FOCUS_URL, HTTP_TIMEOUT_S, SGS_BASE_URL, TICKER_LABELS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveResult:
    """Data plus provenance."""

    data: pd.DataFrame
    is_live: bool
    source: str


# ------------------------------------------------------------------------ prices
def fetch_prices(tickers: list[str], years: int = 5) -> LiveResult:
    """Daily adjusted close prices, one column per ticker.

    Args:
        tickers: Yahoo Finance symbols (B3 tickers end in `.SA`).
        years: history length.

    Returns:
        LiveResult with a DatetimeIndex DataFrame of closes.
    """
    try:
        import yfinance as yf

        frames = {}
        for t in tickers:
            hist = yf.Ticker(t).history(period=f"{years}y", auto_adjust=True)
            if hist is not None and not hist.empty:
                frames[t] = hist["Close"]
        if not frames or tickers[0] not in frames:
            raise ValueError("primary ticker returned no data")
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.sort_index().ffill().dropna(how="all")
        return LiveResult(df, True, "Yahoo Finance (B3)")
    except Exception as exc:  # noqa: BLE001 - boundary: any upstream failure -> fallback
        log.warning("Price feed unavailable, using fallback: %s", type(exc).__name__)
        return LiveResult(_fallback_prices(tickers, years), False, "Fallback simulation")


# Approximate recent levels so fallback charts look plausible (not used when live).
_FALLBACK_LAST = {"PINE4.SA": 10.7, "^BVSP": 140000.0, "ITUB4.SA": 38.0, "BPAC11.SA": 45.0,
                  "ABCB4.SA": 22.0, "BPAN4.SA": 8.0, "BMGB4.SA": 4.0, "BRSR6.SA": 13.0}


def _fallback_prices(tickers: list[str], years: int) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * years)
    rng = np.random.default_rng(11)
    market = rng.normal(0.0003, 0.012, len(idx))
    out = {}
    for i, t in enumerate(tickers):
        beta = 1.0 if t == "^BVSP" else 0.7 + 0.1 * (i % 5)
        drift = 0.0011 if t == "PINE4.SA" else 0.0002
        idio = rng.normal(drift, 0.017 if t != "^BVSP" else 0.0, len(idx))
        rets = beta * market + idio
        path = np.exp(np.cumsum(rets))
        out[t] = path / path[-1] * _FALLBACK_LAST.get(t, 15.0 + 5 * i)
    return pd.DataFrame(out, index=idx)


# --------------------------------------------------------------------------- SGS
def fetch_sgs(code: int, years: int = 10) -> LiveResult:
    """Fetch one SGS series from the Banco Central API.

    The API limits daily series to 10-year windows, so the request is always bounded.
    """
    end = date.today()
    start = end - timedelta(days=365 * years - 5)
    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }
    try:
        resp = requests.get(
            SGS_BASE_URL.format(code=int(code)), params=params, timeout=HTTP_TIMEOUT_S
        )
        resp.raise_for_status()
        raw = pd.DataFrame(resp.json())
        if raw.empty or not {"data", "valor"} <= set(raw.columns):
            raise ValueError("unexpected SGS payload")
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["data"], format="%d/%m/%Y"),
                "value": pd.to_numeric(raw["valor"], errors="coerce"),
            }
        ).dropna()
        return LiveResult(df.set_index("date").sort_index(), True, "Banco Central (SGS)")
    except Exception as exc:  # noqa: BLE001
        log.warning("SGS %s unavailable, using fallback: %s", code, type(exc).__name__)
        return LiveResult(_fallback_sgs(code, years), False, "Fallback simulation")


def _fallback_sgs(code: int, years: int) -> pd.DataFrame:
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12 * years, freq="MS")
    t = np.linspace(0, 1, len(months))
    rng = np.random.default_rng(code)
    if code == 432:  # Selic: cycle shape, rounded to 25bp
        vals = 9.5 + 4.5 * np.sin(2 * np.pi * (t * 2.2 - 0.15)) + 1.5 * t
        vals = np.round(np.clip(vals, 2.0, 15.0) * 4) / 4
    elif code == 433:
        vals = 0.38 + 0.25 * np.sin(2 * np.pi * t * 9) + rng.normal(0, 0.12, len(t))
    elif code == 1:
        vals = 4.0 + 1.6 * t + 0.25 * np.sin(2 * np.pi * t * 4)
    else:  # NPL series, % of portfolio
        base = {21082: 3.2, 21083: 2.4, 21084: 3.9}.get(code, 3.0)
        vals = base + 0.6 * np.sin(2 * np.pi * (t * 2.2 - 0.4)) + rng.normal(0, 0.05, len(t))
    return pd.DataFrame({"value": vals}, index=pd.Index(months, name="date"))


# ------------------------------------------------------------------------- Focus
def fetch_focus(indicator: str) -> LiveResult:
    """Latest Focus survey median expectations (annual) for `Selic` or `IPCA`."""
    if indicator not in {"Selic", "IPCA"}:
        raise ValueError("indicator must be 'Selic' or 'IPCA'")
    params = {
        "$top": "200",
        "$filter": f"Indicador eq '{indicator}' and baseCalculo eq 0",
        "$orderby": "Data desc",
        "$format": "json",
        "$select": "Indicador,Data,DataReferencia,Mediana",
    }
    try:
        resp = requests.get(FOCUS_URL, params=params, timeout=HTTP_TIMEOUT_S)
        resp.raise_for_status()
        raw = pd.DataFrame(resp.json().get("value", []))
        if raw.empty:
            raise ValueError("empty Focus payload")
        latest = raw["Data"].max()
        df = raw[raw["Data"] == latest].assign(
            year=lambda d: pd.to_numeric(d["DataReferencia"], errors="coerce"),
            median=lambda d: pd.to_numeric(d["Mediana"], errors="coerce"),
            survey_date=pd.to_datetime(latest),
        )[["year", "median", "survey_date"]]
        df = df.dropna().sort_values("year").reset_index(drop=True)
        return LiveResult(df, True, "Banco Central (Focus)")
    except Exception as exc:  # noqa: BLE001
        log.warning("Focus %s unavailable, using fallback: %s", indicator, type(exc).__name__)
        y = date.today().year
        med = [12.25, 11.0, 10.25, 10.0] if indicator == "Selic" else [4.6, 4.1, 3.8, 3.5]
        df = pd.DataFrame(
            {"year": range(y, y + 4), "median": med, "survey_date": pd.Timestamp.today()}
        )
        return LiveResult(df, False, "Fallback simulation")


def label(ticker: str) -> str:
    """Human-readable name for a ticker."""
    return TICKER_LABELS.get(ticker, ticker)
