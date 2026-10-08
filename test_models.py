from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model_credit as credit
import model_forecasting as forecasting
import model_treasury as treasury
from pine_config import RATINGS
from pine_reference import quarterly_results


def test_price_forecast_is_ordered(close):
    fc = forecasting.quantile_price_forecast(close, max_iter=40)
    assert len(fc.path) == 21
    assert (fc.path["p10"] <= fc.path["p50"] + 1e-9).all()
    assert (fc.path["p50"] <= fc.path["p90"] + 1e-9).all()
    assert 0 <= fc.coverage <= 1


def test_price_forecast_needs_history():
    short = pd.Series(np.linspace(1, 2, 100), index=pd.bdate_range("2024-01-01", periods=100))
    with pytest.raises(ValueError):
        forecasting.quantile_price_forecast(short, max_iter=10)


def test_regimes_labels(close):
    reg = forecasting.detect_regimes(close)
    assert set(reg["regime"]) <= {"Calm", "Normal", "Stressed"}
    assert reg["p_stressed"].between(0, 1).all()


def test_ipca_12m_compounding():
    s = pd.Series([1.0] * 12, index=pd.date_range("2025-01-01", periods=12, freq="MS"))
    assert np.isclose(forecasting.ipca_12m(s).iloc[-1], (1.01**12 - 1) * 100)


def test_selic_path_hits_focus_anchor():
    focus = pd.DataFrame({"year": [pd.Timestamp.today().year + 1], "median": [10.0]})
    path = forecasting.selic_path_from_focus(14.0, focus, months=30)
    assert path.iloc[0] == 14.0
    assert np.isclose(path.loc[pd.Timestamp(pd.Timestamp.today().year + 1, 12, 1)], 10.0)


def test_npl_model_responds_to_selic():
    idx = pd.date_range("2016-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(0)
    selic = pd.Series(10 + 4 * np.sin(np.arange(120) / 15), index=idx)
    npl = 2 + 0.2 * selic.shift(9).bfill() + rng.normal(0, 0.03, 120)
    ipca = pd.Series(4.0, index=idx)
    m = forecasting.fit_npl_model(npl, selic, ipca)
    future = pd.Series(10.0, index=pd.date_range(idx[-1], periods=13, freq="MS"))
    low = forecasting.forecast_npl(m, future)
    high = forecasting.forecast_npl(m, future + 3)
    assert high.iloc[-1] > low.iloc[-1]


def test_pd_model_quality(book):
    m = credit.train_pd_model(book)
    assert m.auc > 0.7
    scored = credit.score_pd(book, m)
    assert scored["pd_hat"].between(0, 1).all()


def test_migration_rows_sum_to_one(book):
    mat = credit.migration_matrix(book)
    assert list(mat.index) == RATINGS
    assert np.allclose(mat.sum(axis=1), 1.0)


def test_migration_empty_book():
    empty = pd.DataFrame(columns=["prior_rating", "rating", "balance"])
    assert credit.migration_matrix(empty).shape == (8, 8)


def test_ecl_scales_with_scenario(book):
    scored = book.assign(pd_hat=0.02)
    base = credit.expected_credit_loss(scored, 1.0, 1.0, "Base")["ecl_mn"].sum()
    severe = credit.expected_credit_loss(scored, 1.0, 1.0, "Severe")["ecl_mn"].sum()
    assert severe == pytest.approx(base * credit.SCENARIOS["Severe"])


def test_vintage_forecast_extends_curves():
    v = credit.fit_vintage_curves(credit.simulate_vintages())
    assert v["is_forecast"].any()
    assert (v.groupby("cohort")["mob"].max() == 30).all()


def test_nii_liability_sensitive():
    lines = treasury.default_balance_sheet()
    idx = pd.date_range("2026-10-01", periods=12, freq="MS")
    base = treasury.simulate_nii(lines, pd.Series(14.0, index=idx)).sum()
    cut = treasury.simulate_nii(lines, pd.Series(12.0, index=idx)).sum()
    assert cut > base


def test_basel_projection_monotone_in_growth():
    ni = quarterly_results().set_index("period")["net_income_mn"]
    slow = treasury.project_basel(ni, 0.05, 0.4, n_paths=500)
    fast = treasury.project_basel(ni, 0.6, 0.4, n_paths=500)
    assert fast.fan["p50"].iloc[-1] < slow.fan["p50"].iloc[-1]
    assert 0 <= fast.p_breach <= 1
