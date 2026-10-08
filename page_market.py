"""Page: Market & valuation (visuals 01-03)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import pine_datasets as ds
import pine_state as state
import ui_charts as charts
import ui_components as ui
from pine_config import BENCHMARK, TICKER, TICKER_LABELS
from pine_ledger import records_from_frame

WINDOWS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}


def _window(px: pd.DataFrame, flt: dict) -> pd.DataFrame:
    if flt.get("window_start"):
        return px.loc[flt["window_start"]:flt["window_end"]]
    n = WINDOWS[st.session_state.get("mkt_window") or "1Y"]
    return px.iloc[-n:]


def render() -> None:
    """Render the page."""
    flt = state.filters(st.session_state)
    st.title("Market and valuation")
    res = ds.prices()
    px = res.data
    ui.source_tags({res.source: res.is_live})
    last = px.index[-1]
    ticker = flt["ticker"] if flt["ticker"] in px else TICKER
    name = TICKER_LABELS[ticker]
    fc = ds.price_forecast(ticker, last)
    pine = px[TICKER].dropna()

    c = st.columns(4)
    c[0].metric("PINE4 last close", f"R$ {pine.iloc[-1]:.2f}",
                f"{pine.iloc[-1] / pine.iloc[-2] - 1:+.2%} on the day")
    c[1].metric("1-year return", f"{pine.iloc[-1] / pine.iloc[-min(252, len(pine))] - 1:+.1%}")
    c[2].metric("20-day median forecast", f"R$ {fc.path['p50'].iloc[-1]:.2f}",
                f"{fc.path['p50'].iloc[-1] / px[ticker].dropna().iloc[-1] - 1:+.1%}")
    c[3].metric("P10-P90 hit rate", f"{fc.coverage:.0%}",
                f"holdout MAE {fc.mae_model:.3f} vs naive {fc.mae_naive:.3f}", delta_color="off",
                help="Share of the last 120 holdout days whose 20-day outcome fell inside the band.")

    st.segmented_control("History window", list(WINDOWS), key="mkt_window", default="1Y", required=True)

    # 01 -----------------------------------------------------------------------
    ui.visual_header(1, f"{name}: price and 20-day probabilistic forecast",
                     "Quantile gradient boosting on momentum, volatility and market features. "
                     "Drag horizontally on the chart to set the analysis window for visuals 02 and 03.")
    hist = _window(px[[ticker]], flt)[ticker].dropna()
    pts = ui.selectable_chart(charts.price_forecast_fig(hist, fc.path, name), "v01", ("box",))
    if pts and "box" in pts[0]:
        xr = pts[0]["box"].get("x", [])
        if len(xr) == 2:
            a, b = sorted(pd.to_datetime(xr))
            state.set_filter(st.session_state, "window_start", a.normalize())
            state.set_filter(st.session_state, "window_end", b.normalize())
            st.rerun()
    recs = records_from_frame(fc.path.iloc[1:], model_name="price_quantile_gbm", model_version=ds.MODEL_VERSION,
                              target=ticker, as_of=last, unit="BRL")
    ui.autosave(recs, f"price_quantile_gbm_{ticker}",
                {"coverage": fc.coverage, "mae_model": fc.mae_model, "mae_naive": fc.mae_naive})

    # 02 -----------------------------------------------------------------------
    ui.visual_header(2, "Listed bank peers: realised return vs model outlook",
                     "Click a bank to switch visuals 01 and 03 to it. Bars use the active window; "
                     "dots show the model's median 20-day return with its P10-P90 range.")
    win = _window(px, flt)
    out = ds.peer_outlook(last)
    rets = (win.ffill().iloc[-1] / win.bfill().iloc[0] - 1).rename("window_ret")
    peer_df = out.merge(rets, left_on="ticker", right_index=True, how="left")
    peer_df["label"] = peer_df["ticker"].map(TICKER_LABELS)
    if peer_df.empty:
        ui.empty_state("Peer data unavailable.")
    else:
        pts = ui.selectable_chart(charts.peer_fig(peer_df, ticker), "v02")
        if pts:
            t = ui.point_value(pts[0])
            if t in TICKER_LABELS:
                state.set_filter(st.session_state, "ticker", t)
                st.rerun()
        ibov = win[BENCHMARK].dropna() if BENCHMARK in win else pd.Series(dtype=float)
        if len(ibov) > 2:
            ui.caption(f"Ibovespa over the same window: {ibov.iloc[-1] / ibov.iloc[0] - 1:+.1%}")

    # 03 -----------------------------------------------------------------------
    ui.visual_header(3, f"{name}: market regime detection",
                     "Unsupervised Gaussian mixture on rolling 20-day return and volatility. "
                     "Rising stressed-regime probability is an early warning for funding and "
                     "follow-on offering timing.")
    reg = ds.regimes(ticker, last)
    reg_w = reg.loc[hist.index[0]:hist.index[-1]] if len(hist) else reg
    if reg_w.empty:
        ui.empty_state("No regime data in the selected window.")
    else:
        st.plotly_chart(charts.regime_fig(px[ticker], reg_w, name), theme=None, config={"displaylogo": False})
        now = reg.iloc[-1]
        ui.caption(f"Current regime: {now['regime']} (P(stressed) {now['p_stressed']:.0%}, "
                   f"annualised 20-day volatility {now['vol20']:.0%}).")
