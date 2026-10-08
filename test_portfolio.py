from __future__ import annotations

import numpy as np

from pine_portfolio import LOAN_COLUMNS, quarterly_segment_history
from pine_reference import SEGMENT_BALANCE_2Q26_BN, quarterly_results


def test_book_matches_published_balances(book):
    totals = book.groupby("segment")["balance"].sum() / 1e9
    for seg, bn in SEGMENT_BALANCE_2Q26_BN.items():
        assert np.isclose(totals[seg], bn)
    assert np.isclose(totals.sum(), 21.8)


def test_book_has_contract_columns(book):
    assert set(LOAN_COLUMNS) <= set(book.columns)
    assert book["loan_id"].is_unique


def test_default_rate_near_published_npl(book):
    w = (book["default_12m"] * book["balance"]).sum() / book["balance"].sum()
    assert 0.015 < w < 0.045


def test_history_anchored_on_2q26():
    h = quarterly_segment_history()
    last = h[h["quarter"] == h["quarter"].max()]["balance_bn"].sum()
    assert np.isclose(last, 21.8)


def test_quarterly_results_periods():
    q = quarterly_results()
    assert q["period"].is_monotonic_increasing
    assert q["net_income_mn"].iloc[-1] == 165.7
