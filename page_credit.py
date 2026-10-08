"""Page: Credit portfolio and risk (visuals 06-11)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import model_credit as credit
import model_forecasting as forecasting
import pine_datasets as ds
import pine_state as state
import ui_charts as charts
import ui_components as ui
from pine_config import RATINGS
from pine_ledger import records_from_frame


def _kpis(book: pd.DataFrame, full: pd.DataFrame) -> None:
    c = st.columns(4)
    bal = book["balance"].sum()
    c[0].metric("Exposure in view", f"R$ {bal / 1e9:,.2f} bn", f"{bal / full['balance'].sum():.0%} of book",
                delta_color="off")
    w = book["balance"] / bal if bal else 0
    c[1].metric("Balance-weighted model PD", f"{(book['pd_hat'] * w).sum():.2%}")
    c[2].metric("Observed 12m default (sim.)", f"{(book['default_12m'] * w).sum():.2%}")
    c[3].metric("Loans in view", f"{len(book):,}")


def render() -> None:
    """Render the page."""
    flt = state.filters(st.session_state)
    st.title("Credit portfolio and risk")
    ui.caption("Loan-level book is a calibrated simulation of the published 2Q26 mix (R$21.8 bn, "
               "over-90 2.7%). Replace `generate_loan_book` with the warehouse loader in production.")
    b = ds.credit_bundle()
    book = b.book
    view = state.apply_book_filters(book, flt)
    _kpis(view, book)

    # 06 -----------------------------------------------------------------------
    ui.visual_header(6, "Exposure map: segment by region, colored by model PD",
                     "Bubble size is exposure. Click a bubble to filter every visual on this and the "
                     "client page to that segment and region.")
    agg = (state.apply_book_filters(book, {"rating": flt["rating"]})
           .assign(w=lambda d: d["pd_hat"] * d["balance"])
           .groupby(["segment", "region"], as_index=False).agg(balance=("balance", "sum"), w=("w", "sum")))
    agg["pd"] = agg["w"] / agg["balance"]
    pts = ui.selectable_chart(charts.exposure_matrix_fig(agg, flt["segment"], flt["region"]), "v06")
    if pts:
        seg, reg = ui.point_value(pts[0], index=0), ui.point_value(pts[0], index=1)
        state.set_filter(st.session_state, "segment", seg)
        state.set_filter(st.session_state, "region", reg)
        st.rerun()

    left, right = st.columns(2)
    # 07 -----------------------------------------------------------------------
    with left:
        ui.visual_header(7, "Portfolio growth and 4-quarter forecast",
                         "Damped-trend exponential smoothing on log balances. Click a bar to filter by segment.")
        hist = ds.segment_history()
        segs = [flt["segment"]] if flt["segment"] else hist["segment"].unique().tolist()
        total = hist[hist["segment"].isin(segs)].groupby("quarter")["balance_bn"].sum()
        fc = forecasting.forecast_balance(total)
        pts = ui.selectable_chart(charts.growth_fig(hist, fc, flt["segment"]), "v07")
        if pts and ui.point_value(pts[0]):
            state.set_filter(st.session_state, "segment", ui.point_value(pts[0]))
            st.rerun()
        target = f"BALANCE_{flt['segment'] or 'TOTAL'}".upper().replace(" ", "_")
        ui.autosave(records_from_frame(fc, model_name="balance_holt", model_version=ds.MODEL_VERSION,
                                       target=target, as_of=total.index[-1], unit="BRL bn"),
                    f"balance_holt_{target}")
    # 08 -----------------------------------------------------------------------
    with right:
        ui.visual_header(8, "Probability-of-default model",
                         f"Gradient boosting, holdout AUC {b.pd_model.auc:.3f} (Gini {b.pd_model.gini:.2f}), "
                         f"Brier {b.pd_model.brier:.4f}. Calibration reflects the current filters.")
        if len(view) < 200 or view["default_12m"].sum() < 10:
            ui.empty_state("Too few loans or defaults in this filter for a calibration curve.")
            calib = b.pd_model.calibration
        else:
            calib = (view.assign(bin=pd.qcut(view["pd_hat"], 10, duplicates="drop"))
                     .groupby("bin", observed=True)
                     .agg(predicted=("pd_hat", "mean"), observed=("default_12m", "mean")))
        st.plotly_chart(charts.pd_model_fig(calib, b.pd_model.importance), theme=None, config={"displaylogo": False})

    # 09 -----------------------------------------------------------------------
    ui.visual_header(9, "Rating migration and projected rating mix",
                     "Empirical 12-month transition matrix for the filtered book, projected forward as a "
                     "Markov chain. Click a rating bar to filter the book to that rating.")
    mig_base = state.apply_book_filters(book, {**flt, "rating": None})
    matrix = credit.migration_matrix(mig_base)
    dist = credit.project_rating_distribution(mig_base, matrix, years=2)
    pts = ui.selectable_chart(charts.migration_fig(matrix, dist, flt["rating"]), "v09")
    if pts:
        r = ui.point_value(pts[0])
        if r in RATINGS:
            state.set_filter(st.session_state, "rating", r)
            st.rerun()
    ui.caption(f"Projected share of balance in default in 2 years: {dist.iloc[-1, -1]:.1%} "
               f"(today {dist.iloc[-1, 0]:.1%}).")

    # 10 -----------------------------------------------------------------------
    shock = int(flt["selic_shock_bps"])
    r_mult, c_mult = ds.pd_multipliers(shock)
    ui.visual_header(10, "Expected credit loss by segment under macro scenarios",
                     f"ECL = model PD x LGD x EAD. PD multipliers come from the delinquency model "
                     f"(retail {r_mult:.2f}x, corporate {c_mult:.2f}x at {shock:+d} bp). "
                     "Click a segment to filter.")
    ecl_base = state.apply_book_filters(book, {**flt, "segment": None})
    ecl = pd.concat([credit.expected_credit_loss(ecl_base, r_mult, c_mult, s).assign(scenario=s)
                     for s in credit.SCENARIOS], ignore_index=True)
    if ecl.empty:
        ui.empty_state("No exposure for the current filters.")
    else:
        pts = ui.selectable_chart(charts.ecl_fig(ecl, flt["segment"]), "v10")
        if pts and ui.point_value(pts[0], index=0):
            state.set_filter(st.session_state, "segment", ui.point_value(pts[0], index=0))
            st.rerun()
        tot = ecl.groupby("scenario")["ecl_mn"].sum()
        chosen = flt["scenario"]
        ui.caption(f"Total 12-month ECL, {chosen} scenario: R$ {tot[chosen]:,.0f} mn "
                   f"(Base R$ {tot['Base']:,.0f} mn, Severe R$ {tot['Severe']:,.0f} mn).")
        today = ui.as_of_today()
        recs = []
        for scen, val in tot.items():
            recs += records_from_frame(pd.DataFrame({"p50": [val]}, index=[today + pd.DateOffset(years=1)]),
                                       model_name="ecl_pd_lgd_ead", model_version=ds.MODEL_VERSION,
                                       target="ECL_12M_TOTAL", as_of=today, unit="BRL mn",
                                       scenario=f"{scen.lower()}_selic{shock:+d}bp")
        ui.autosave(recs, "ecl_pd_lgd_ead", {"pd_auc": b.pd_model.auc, "pd_brier": b.pd_model.brier})

    # 11 -----------------------------------------------------------------------
    ui.visual_header(11, "Vintage curves: cumulative over-90 by origination cohort",
                     "Shared Weibull shape fitted on mature cohorts, cohort-specific level on observed "
                     "months; dotted lines are model projections. Click a cohort to see its band.")
    v = ds.vintages()
    pts = ui.selectable_chart(charts.vintage_fig(v, flt["cohort"]), "v11")
    if pts and ui.point_value(pts[0]):
        flt["cohort"] = str(ui.point_value(pts[0]))
        st.rerun()
    levels = v.groupby("cohort")["lifetime_level"].first()
    ui.caption(f"Projected lifetime over-90: newest cohort {levels.iloc[-1]:.2%} vs "
               f"oldest {levels.iloc[0]:.2%}.")
    today = ui.as_of_today()
    recs = []
    for c, val in levels.items():
        recs += records_from_frame(pd.DataFrame({"p50": [val]}, index=[today + pd.DateOffset(months=30)]),
                                   model_name="vintage_weibull", model_version=ds.MODEL_VERSION,
                                   target=f"LIFETIME_OVER90_{c}", as_of=today, unit="ratio")
    ui.autosave(recs, "vintage_weibull")
