"""Shared fixtures: deterministic book, in-memory DB, synthetic series. No network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pine_db import get_engine
from pine_portfolio import generate_loan_book


@pytest.fixture(scope="session")
def book() -> pd.DataFrame:
    return generate_loan_book(seed=7)


@pytest.fixture()
def engine(tmp_path):
    return get_engine(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture(scope="session")
def close() -> pd.Series:
    idx = pd.bdate_range("2021-01-01", periods=1000)
    rng = np.random.default_rng(1)
    return pd.Series(10 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(idx)))), index=idx)
