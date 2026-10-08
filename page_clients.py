"""Page: Clients and origination (visuals 12-13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import model_clients as cm
import pine_datasets as ds
import pine_state as state
import ui_charts as charts
import ui_components as ui
from pine_config import PRIVATE_PAYROLL_RATE_CAP
from pine_ledger import records_from_frame
from pine_reference import SEGMENT_PARAMS

_TABLE_COLS = {
    "client_id": "Client", "segment": "Segment", "region": "Region", "balance": "Balance (R$)",
    "p_churn": "P(portability)", "pd_hat": "Model PD", "margin_at_risk": "Margin at risk (R$/yr)",
    "action": "Recommended action", "nbo": "Next best offer", "nbo_score": "Offer propensity",
}


def _cdi() -> float:
    m = ds.macro()
    return float(m.selic.iloc[-1]) / 100 - 0.001


def render() -> None:
    """Render the page."""
    flt = state.filters(st.session_state)
    st.title("Clients and origination")
    cdi = _cdi()
    scores = ds.client_scores(round(cdi, 4))
    view = state.apply_book_filters(scores, flt)

    c = st.columns(4)
    c[0].metric("Retail clients in view", f"{len(view):,}")
    c[1].metric("Expected portability (12m)", f"{view['p_churn'].mean():.1%}" if len(view) else "n/a")
    c[2].metric("Annual margin at risk", f"R$ {view['margin_at_risk'].sum() / 1e6:,.1f} mn")
    c[3].metric("Churn model AUC (holdout)", f"{ds.credit_bundle().churn.auc:.3f}")

    # 12 -----------------------------------------------------------------------
    ui.visual_header(12, "Portability risk and retention playbook",
                     "Gradient-boosted portability model plus next-best-offer propensities. Lasso or box "
                     "select clients to build a retention list; the table and export follow the selection.")
    if view.empty:
        ui.empty_state("No retail clients for the current filters (corporate is excluded here).")
    else:
        sample = view.sample(min(5000, len(view)), random_state=0)
        key = state.chart_key(st.session_state, "v12")
        event = st.plotly_chart(charts.churn_fig(sample), key=key, on_select="rerun",
                                selection_mode=("lasso", "box"), theme=None, config={"displaylogo": False})
        pts = (event or {}).get("selection", {}).get("points", []) if event else []
        ids = {p["customdata"][0] for p in pts if p.get("customdata")}
        table = (view[view["client_id"].isin(ids)] if ids else
                 view[view["action"] != "Monitor"].nlargest(200, "margin_at_risk"))
        ui.caption(f"{'Selected' if ids else 'Top 200 by margin at risk'}: {len(table):,} clients, "
                   f"R$ {table['margin_at_risk'].sum() / 1e6:,.2f} mn annual margin at risk.")
        out = table[list(_TABLE_COLS)].rename(columns=_TABLE_COLS).sort_values(
            "Margin at risk (R$/yr)", ascending=False)
        st.dataframe(out, hide_index=True, height=280, column_config={
            "Balance (R$)": st.column_config.NumberColumn(format="%.0f"),
            "P(portability)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Model PD": st.column_config.NumberColumn(format="%.3f"),
            "Margin at risk (R$/yr)": st.column_config.NumberColumn(format="%.0f"),
            "Offer propensity": st.column_config.NumberColumn(format="%.2f"),
        })
        st.download_button("Export retention list (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="pine_retention_list.csv", mime="text/csv")
        mix = view.groupby("action")["margin_at_risk"].sum().sort_values(ascending=False)
        today = ui.as_of_today()
        recs = []
        for action, val in mix.items():
            recs += records_from_frame(pd.DataFrame({"p50": [val]}, index=[today + pd.DateOffset(years=1)]),
                                       model_name="churn_gbm", model_version=ds.MODEL_VERSION,
                                       target=f"MARGIN_AT_RISK_{action}".upper().replace(" ", "_")[:64],
                                       as_of=today, unit="BRL")
        ui.autosave(recs, "churn_gbm", {"auc": ds.credit_bundle().churn.auc})

    # 13 -----------------------------------------------------------------------
    ui.visual_header(13, "Private payroll (consignado privado): demand forecast and pricing under the cap",
                     "ETS forecast of weekly originations; logistic acceptance model turns each offered "
                     "rate into expected volume and lifetime contribution net of funding and expected loss.")
    weekly, fc = ds.originations()
    p = SEGMENT_PARAMS["Consignado Privado"]
    r_mult, _ = ds.pd_multipliers(int(flt["selic_shock_bps"]))
    k = st.columns(3)
    competitor = k[0].slider("Best competitor rate (% per month)", 1.2, 3.5, 1.99, 0.01) / 100
    chosen = k[1].slider("Pine offered rate (% per month)", 1.0, 3.5, 1.89, 0.01) / 100
    leads = k[2].number_input("Qualified leads per month", 5000, 500000, 60000, step=5000)
    model, cols = ds.acceptance_model()
    rates = np.unique(np.append(np.round(np.arange(0.010, 0.0351, 0.0005), 4),
                                PRIVATE_PAYROLL_RATE_CAP))
    funding_m = (1 + cdi * 1.03) ** (1 / 12) - 1
    curve = cm.pricing_curve(model, cols, rates, competitor, funding_m, p["pd"] * r_mult, p["lgd"],
                             leads, avg_ticket=8500.0)
    st.plotly_chart(charts.pricing_fig(weekly, fc, curve, chosen, PRIVATE_PAYROLL_RATE_CAP),
                    theme=None, config={"displaylogo": False})
    ok = curve[curve["rate"] <= PRIVATE_PAYROLL_RATE_CAP + 1e-9]
    best = ok.loc[ok["contribution_mn"].idxmax()]
    here = curve.iloc[(curve["rate"] - chosen).abs().argmin()]
    ui.caption(
        f"Optimal rate under the {PRIVATE_PAYROLL_RATE_CAP:.2%} cap: {best['rate']:.2%}/month "
        f"(acceptance {best['p_accept']:.0%}, R$ {best['contribution_mn']:,.1f} mn contribution). "
        f"At {chosen:.2%}: acceptance {here['p_accept']:.0%}, R$ {here['contribution_mn']:,.1f} mn. "
        f"Funding cost {funding_m:.2%}/month (103% of CDI)."
    )
    ui.autosave(records_from_frame(fc, model_name="originations_ets", model_version=ds.MODEL_VERSION,
                                   target="CONSIG_PRIV_WEEKLY_ORIG", as_of=weekly["week"].iloc[-1],
                                   unit="BRL mn"), "originations_ets")
