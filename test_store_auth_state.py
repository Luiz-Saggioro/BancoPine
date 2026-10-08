from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import pine_auth as auth
import pine_state as state
from pine_ledger import (
    PredictionRecord,
    evaluate_predictions,
    load_predictions,
    records_from_frame,
    save_predictions,
)


# ---------------------------------------------------------------- prediction ledger
def _recs(value: float = 10.0) -> list[PredictionRecord]:
    fc = pd.DataFrame({"p10": [9.0], "p50": [value], "p90": [11.0]},
                      index=[pd.Timestamp("2026-01-10")])
    return records_from_frame(fc, model_name="m", model_version="1", target="PINE4.SA",
                              as_of=datetime(2026, 1, 2), unit="BRL")


def test_save_is_idempotent(engine):
    assert save_predictions(engine, _recs(), "u") == 1
    assert save_predictions(engine, _recs(), "u") == 0
    assert len(load_predictions(engine)) == 1


def test_overwrite_updates(engine):
    save_predictions(engine, _recs(10.0), "u")
    save_predictions(engine, _recs(12.0), "u", overwrite=True)
    assert load_predictions(engine)["value"].iloc[0] == 12.0


def test_evaluate_matured(engine):
    save_predictions(engine, _recs(10.0), "u")
    preds = load_predictions(engine)
    actual = pd.Series([10.5], index=[pd.Timestamp("2026-01-09")])
    ev = evaluate_predictions(preds, {"PINE4.SA": actual})
    assert ev["in_band"].iloc[0]
    assert ev["abs_pct_error"].iloc[0] == pytest.approx(0.5 / 10.5)


def test_evaluate_empty():
    assert evaluate_predictions(pd.DataFrame(columns=["target_date"]), {}).empty


# ------------------------------------------------------------------------- auth
def test_password_hash_roundtrip():
    h = auth.hash_password("CorrectHorse123")
    assert auth.verify_password("CorrectHorse123", h)
    assert not auth.verify_password("wrong", h)


@pytest.mark.parametrize("bad", ["ab", "drop table;", "x" * 65, ""])
def test_username_whitelist(bad):
    with pytest.raises(auth.AuthError):
        auth.validate_username(bad)


def test_weak_password_rejected(engine):
    with pytest.raises(auth.AuthError):
        auth.create_user(engine, "ana", "Ana", "viewer", "short")


def test_login_and_lockout(engine):
    auth.create_user(engine, "ana", "Ana", "analyst", "LongPassword1")
    user = auth.authenticate(engine, "ANA", "LongPassword1", 3, 15)
    assert user.role == "analyst" and user.can_write
    for _ in range(3):
        with pytest.raises(auth.AuthError):
            auth.authenticate(engine, "ana", "nope", 3, 15)
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.authenticate(engine, "ana", "LongPassword1", 3, 15)


def test_bootstrap_admin_only_once(engine):
    assert auth.ensure_bootstrap_admin(engine, "root", "AdminPassword9", "Root")
    assert not auth.ensure_bootstrap_admin(engine, "root2", "AdminPassword9", "Root")


def test_disabled_user_cannot_login(engine):
    auth.create_user(engine, "bob", "Bob", "viewer", "LongPassword1")
    auth.set_active(engine, "bob", False)
    with pytest.raises(auth.AuthError):
        auth.authenticate(engine, "bob", "LongPassword1", 5, 15)


# ------------------------------------------------------------------ cross-filters
def test_filters_whitelist_and_clear():
    s: dict = {}
    state.set_filter(s, "segment", "Corporate")
    state.set_filter(s, "region", "Mars")  # ignored
    assert state.filters(s)["segment"] == "Corporate"
    assert state.filters(s)["region"] is None
    state.clear_filters(s)
    assert state.filters(s)["segment"] is None
    assert state.chart_key(s, "v06") == "v06_e1"


def test_apply_book_filters(book):
    out = state.apply_book_filters(book, {"segment": "Home Equity", "region": None, "rating": None})
    assert set(out["segment"]) == {"Home Equity"}


def test_selection_changed_only_once():
    s: dict = {}
    pts = [{"x": 1}]
    assert state.selection_changed(s, "c", pts)
    assert not state.selection_changed(s, "c", pts)
    assert not state.selection_changed(s, "c", [])
