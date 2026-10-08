"""Page: Treasury and capital (visuals 14-15)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import model_treasury as treasury
import pine_datasets as ds
import pine_state as state
import ui_charts as charts
import ui_components as ui
from pine_config import BASEL_MINIMUM
from pine_ledger import records_from_frame
from pine_reference import quarterly_results


@st.cache_data(show_spinner="Running capital Monte Carlo")
def _basel(growth: float, payout: float) -> treasury.CapitalProjection:
    q = quarterly_results()
    return treasury.project_basel(q.set_index("period")["net_income_mn"], growth, payout)


def render() -> None:
    """Render the page."""
    flt = state.filters(st.session_state)
    shock = int(flt["selic_shock_bps"])
    st.title("Treasury and capital")
    m = ds.macro()
    ui.source_tags({"Selic (SGS 432)": m.sources["selic_target"], "Focus survey": m.sources["focus"]})
    lines = treasury.default_balance_sheet()
    paths = ds.selic_paths(m, shock, months=12)
    nii = {k: treasury.simulate_nii(lines, v) for k, v in paths.items()}
    base_key, shock_key = list(paths)[0], list(paths)[1]
    nii_base, nii_shock = nii[base_key].sum(), nii[shock_key].sum()

    q = quarterly_results()
    c = st.columns(4)
    c[0].metric("12m NII, Focus path", f"R$ {nii_base / 1e3:,.2f} bn")
    c[1].metric(f"NII impact of {shock:+d} bp", f"R$ {(nii_shock - nii_base):+,.0f} mn",
                f"{nii_shock / nii_base - 1:+.1%}")
    c[2].metric("Basel ratio (2Q26)", f"{q['basel'].dropna().iloc[-1]:.1%}")
    c[3].metric("Net income 2Q26", f"R$ {q['net_income_mn'].iloc[-1]:,.1f} mn")

    # 14 -----------------------------------------------------------------------
    ui.visual_header(14, "Interest-rate risk: repricing gap and NII simulation",
                     "Floating funding (CDB, LCA/LCI at % of CDI) reprices monthly while payroll loans are "
                     "fixed-rate: the bank is liability-sensitive, so rate cuts widen NII. Use the sidebar "
                     "shock to stress the path.")
    st.plotly_chart(charts.alm_fig(treasury.repricing_gap(lines), nii), theme=None, config={"displaylogo": False})
    for name, s in nii.items():
        recs = records_from_frame(pd.DataFrame({"p50": s}), model_name="nii_simulation",
                                  model_version=ds.MODEL_VERSION, target="NII_MONTHLY",
                                  as_of=ui.as_of_today(), unit="BRL mn",
                                  scenario=name.lower().replace(" ", "_")[:64])
        ui.autosave(recs, "nii_simulation")

    # 15 -----------------------------------------------------------------------
    ui.visual_header(15, "Capital adequacy projection and growth capacity",
                     "Monte Carlo (3,000 paths) of reference equity and RWA over 8 quarters with earnings "
                     "uncertainty from reported quarterly results.")
    k = st.columns(2)
    growth = k[0].slider("Annual loan growth", 0.0, 0.8, 0.30, 0.02, format="%.2f")
    payout = k[1].slider("Dividend / JCP payout", 0.0, 0.8, 0.40, 0.05, format="%.2f")
    proj = _basel(growth, payout)
    st.plotly_chart(charts.basel_fig(q, proj.fan), theme=None, config={"displaylogo": False})
    kc = st.columns(3)
    kc[0].metric("Median ratio in 8 quarters", f"{proj.fan['p50'].iloc[-1]:.1%}")
    kc[1].metric("Probability of breaching 10.5%", f"{proj.p_breach:.0%}",
                 delta="above 10% tolerance" if proj.p_breach > 0.10 else "within tolerance",
                 delta_color="inverse" if proj.p_breach > 0.10 else "off")
    kc[2].metric("Max growth at 90% confidence", f"{proj.max_growth_90:.0%} per year",
                 help=f"Highest annual loan growth with at most 10% chance of falling below "
                      f"{BASEL_MINIMUM:.1%} at the selected payout.")
    recs = records_from_frame(proj.fan, model_name="basel_monte_carlo", model_version=ds.MODEL_VERSION,
                              target="BASEL_RATIO", as_of=ui.as_of_today(), unit="ratio",
                              scenario=f"g{growth:.2f}_p{payout:.2f}")
    if (growth, payout) == (0.30, 0.40):  # daily baseline; other plans are saved explicitly
        ui.autosave(recs, "basel_monte_carlo",
                    {"p_breach": proj.p_breach, "max_growth_90": proj.max_growth_90})
    ui.save_button(recs, "Save this capital plan to the ledger", "save_basel")
