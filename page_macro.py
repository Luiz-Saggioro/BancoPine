"""Page: Macro drivers (visuals 04-05)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import pine_datasets as ds
import pine_state as state
import ui_charts as charts
import ui_components as ui
from pine_ledger import records_from_frame


def render() -> None:
    """Render the page."""
    flt = state.filters(st.session_state)
    shock = int(flt["selic_shock_bps"])
    st.title("Macro drivers")
    m = ds.macro()
    ui.source_tags({
        "Selic (SGS 432)": m.sources["selic_target"], "IPCA (SGS 433)": m.sources["ipca_month"],
        "NPL (SGS 21083/21084)": m.sources["npl_pf"], "Focus survey": m.sources["focus"],
    })
    paths = ds.selic_paths(m, shock)
    ipca_fc = ds.ipca_forecast()
    fc = ds.npl_forecasts(shock)
    models = ds.npl_models()
    r_mult, c_mult = ds.pd_multipliers(shock)

    c = st.columns(4)
    c[0].metric("Selic target", f"{m.selic.iloc[-1]:.2f}%",
                f"Focus year-end {m.focus_selic['median'].iloc[0]:.2f}%", delta_color="off")
    c[1].metric("IPCA 12m", f"{m.ipca12.iloc[-1]:.2f}%",
                f"model 12m ahead {ipca_fc['p50'].iloc[-1]:.2f}%", delta_color="off")
    c[2].metric("Household NPL in 12m (model)", f"{fc['PF'].iloc[-1, 1]:.2f}%",
                f"{fc['PF'].iloc[-1, 1] - m.npl_pf.iloc[-1]:+.2f} pp", delta_color="inverse")
    c[3].metric("Retail PD multiplier", f"{r_mult:.2f}x", f"corporate {c_mult:.2f}x", delta_color="off")

    # 04 -----------------------------------------------------------------------
    ui.visual_header(4, "Rates and inflation: actuals, market consensus and model forecast",
                     "Selic path interpolated from the latest Focus survey (and the sidebar shock); "
                     "IPCA forecast by seasonal ARIMA, compared with Focus year-end medians.")
    st.plotly_chart(charts.macro_fig(m.selic, paths, m.ipca12, ipca_fc, m.focus_ipca),
                    theme=None, config={"displaylogo": False})
    as_of = m.ipca12.index[-1]
    ui.autosave(records_from_frame(ipca_fc, model_name="ipca_sarima", model_version=ds.MODEL_VERSION,
                                   target="IPCA_12M", as_of=as_of, unit="%"), "ipca_sarima")

    # 05 -----------------------------------------------------------------------
    ui.visual_header(5, "System delinquency outlook under Selic scenarios",
                     "Ridge model of NPL 90+ on its own lag, Selic at 6/9/12-month lags and inflation. "
                     "The shocked path drives the PD multipliers used in the ECL visual (10).")
    st.plotly_chart(charts.npl_fig(m.npl_pf, m.npl_pj, fc["PF"], fc["PJ"]), theme=None, config={"displaylogo": False})
    ui.caption(f"In-sample R-squared: households {models['PF'].r2:.2f}, companies {models['PJ'].r2:.2f}.")
    for seg, ser in (("PF", m.npl_pf), ("PJ", m.npl_pj)):
        col = f"Focus {shock:+d} bp"
        frame = pd.DataFrame({"p50": fc[seg][col]})
        recs = records_from_frame(frame, model_name="npl_ridge", model_version=ds.MODEL_VERSION,
                                  target=f"NPL_{seg}", as_of=ser.index[-1], unit="%",
                                  scenario=f"selic{shock:+d}bp")
        ui.autosave(recs, f"npl_ridge_{seg}", {"r2": models[seg].r2})
