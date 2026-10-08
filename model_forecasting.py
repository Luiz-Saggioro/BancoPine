"""Time-series and market models: price quantile forecasts, regimes, macro and growth."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)
HORIZON_GRID: tuple[int, ...] = (1, 5, 10, 15, 20)


# ------------------------------------------------------------------ price features
def price_features(close: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """Technical features from a close series (pure, no look-ahead)."""
    lr = np.log(close).diff()
    feats = pd.DataFrame(
        {
            "r1": lr,
            "r5": lr.rolling(5).sum(),
            "r20": lr.rolling(20).sum(),
            "r60": lr.rolling(60).sum(),
            "vol20": lr.rolling(20).std(),
            "vol60": lr.rolling(60).std(),
            "ma50_gap": close / close.rolling(50).mean() - 1,
            "dd252": close / close.rolling(252, min_periods=60).max() - 1,
        },
        index=close.index,
    )
    if bench is not None:
        blr = np.log(bench.reindex(close.index).ffill()).diff()
        feats["b5"] = blr.rolling(5).sum()
        feats["b20"] = blr.rolling(20).sum()
    return feats


@dataclass
class PriceForecast:
    """Forecast path and holdout diagnostics."""

    path: pd.DataFrame  # index=date, columns p10, p50, p90
    coverage: float  # share of holdout outcomes inside P10-P90 (h=20)
    mae_model: float  # holdout MAE of median log-return (h=20)
    mae_naive: float  # holdout MAE of zero-return random walk (h=20)


def _fit_quantiles(x: pd.DataFrame, y: pd.Series, max_iter: int) -> dict[float, object]:
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=max_iter, max_depth=3,
            learning_rate=0.05, min_samples_leaf=40, random_state=0,
        )
        models[q] = m.fit(x, y)
    return models


def quantile_price_forecast(
    close: pd.Series, bench: pd.Series | None = None, horizon: int = 20, max_iter: int = 150
) -> PriceForecast:
    """Direct multi-horizon quantile gradient boosting forecast of the close price.

    For each horizon in HORIZON_GRID a set of quantile models predicts the forward
    log return; intermediate days are interpolated. Holdout diagnostics use the last
    120 observations for the 20-day model.
    """
    close = close.dropna()
    feats = price_features(close, bench)
    grid = [h for h in HORIZON_GRID if h <= horizon] or [horizon]
    last_x = feats.iloc[[-1]]
    preds = {0: (0.0, 0.0, 0.0)}
    diag = (np.nan, np.nan, np.nan)
    for h in grid:
        y = np.log(close).shift(-h) - np.log(close)
        data = feats.assign(y=y).dropna()
        if len(data) < 300:
            raise ValueError("not enough history for the price model")
        if h == grid[-1]:
            diag = _holdout(data, max_iter)
        models = _fit_quantiles(data.drop(columns="y"), data["y"], max_iter)
        qs = sorted(float(models[q].predict(last_x)[0]) for q in QUANTILES)
        preds[h] = tuple(qs)
    steps = np.arange(0, horizon + 1)
    known = sorted(preds)
    path = {
        name: np.interp(steps, known, [preds[k][i] for k in known])
        for i, name in enumerate(("p10", "p50", "p90"))
    }
    dates = pd.bdate_range(close.index[-1], periods=horizon + 1)
    out = pd.DataFrame({k: close.iloc[-1] * np.exp(v) for k, v in path.items()}, index=dates)
    return PriceForecast(out, *diag)


def _holdout(data: pd.DataFrame, max_iter: int) -> tuple[float, float, float]:
    train, test = data.iloc[:-140], data.iloc[-120:]  # 20-day embargo between sets
    models = _fit_quantiles(train.drop(columns="y"), train["y"], max_iter)
    xt = test.drop(columns="y")
    lo, med, hi = (models[q].predict(xt) for q in QUANTILES)
    y = test["y"].to_numpy()
    coverage = float(np.mean((y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))))
    return coverage, float(np.mean(np.abs(y - med))), float(np.mean(np.abs(y)))


def expected_return_distribution(close: pd.Series, horizon: int = 20) -> tuple[float, ...]:
    """P10/P50/P90 forward log return for one asset (lighter model, for peer screens)."""
    feats = price_features(close.dropna())
    y = np.log(close).shift(-horizon) - np.log(close)
    data = feats.assign(y=y).dropna()
    models = _fit_quantiles(data.drop(columns="y"), data["y"], max_iter=80)
    x = feats.iloc[[-1]]
    return tuple(sorted(float(models[q].predict(x)[0]) for q in QUANTILES))


# ------------------------------------------------------------------------ regimes
REGIME_NAMES = ("Calm", "Normal", "Stressed")


def detect_regimes(close: pd.Series, n_regimes: int = 3, seed: int = 0) -> pd.DataFrame:
    """Unsupervised market-regime labels from rolling return/volatility (Gaussian mixture).

    Returns a frame with r20, vol20 (annualised), regime label and the probability of
    being in the stressed regime.
    """
    lr = np.log(close.dropna()).diff()
    x = pd.DataFrame(
        {"r20": lr.rolling(20).sum(), "vol20": lr.rolling(20).std() * np.sqrt(252)}
    ).dropna()
    gm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=seed)
    z = (x - x.mean()) / x.std()
    gm.fit(z)
    order = np.argsort(gm.means_[:, 1])  # sort components by volatility
    rank = {comp: i for i, comp in enumerate(order)}
    labels = np.vectorize(rank.get)(gm.predict(z))
    proba = gm.predict_proba(z)[:, order[-1]]
    names = REGIME_NAMES if n_regimes == 3 else tuple(f"Regime {i + 1}" for i in range(n_regimes))
    return x.assign(regime=[names[i] for i in labels], p_stressed=proba)


# ------------------------------------------------------------------------- macro
def monthly(series: pd.DataFrame, how: str = "last") -> pd.Series:
    """Resample a daily/monthly SGS frame (column `value`) to month-start frequency."""
    s = series["value"].resample("MS")
    return (s.last() if how == "last" else s.mean()).dropna()


def ipca_12m(ipca_month: pd.Series) -> pd.Series:
    """Accumulated 12-month IPCA (%), from monthly % changes."""
    return ((1 + ipca_month / 100).rolling(12).apply(np.prod, raw=True) - 1) * 100


def forecast_ipca(ipca_month: pd.Series, months: int = 12) -> pd.DataFrame:
    """Seasonal ARIMA forecast of monthly IPCA, returned as 12m-accumulated with 80% band."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = ipca_month.dropna().iloc[-120:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), trend="c").fit(disp=False)
    fc = res.get_forecast(months)
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.2)
    hist = y.iloc[-11:]
    out = []
    for col, path in (("p50", mean), ("p10", ci.iloc[:, 0]), ("p90", ci.iloc[:, 1])):
        acc = ipca_12m(pd.concat([hist, path])).iloc[-months:]
        out.append(acc.rename(col))
    return pd.concat(out, axis=1)


def selic_path_from_focus(current: float, focus: pd.DataFrame, months: int = 24) -> pd.Series:
    """Monthly Selic path linearly interpolated toward Focus year-end medians."""
    start = pd.Timestamp.today().normalize().replace(day=1)
    idx = pd.date_range(start, periods=months + 1, freq="MS")
    anchors = {start: current}
    for _, row in focus.iterrows():
        anchors[pd.Timestamp(int(row["year"]), 12, 1)] = float(row["median"])
    anc = pd.Series(anchors).sort_index()
    x = (anc.index - start).days.to_numpy()
    xi = (idx - start).days.to_numpy()
    return pd.Series(np.interp(xi, x, anc.to_numpy()), index=idx, name="selic")


# --------------------------------------------------------------- NPL (delinquency)
@dataclass
class NplModel:
    """Ridge model of system delinquency driven by lagged Selic and inflation."""

    model: Ridge
    lags: tuple[int, ...]
    history: pd.DataFrame
    r2: float


def _npl_design(npl: pd.Series, selic: pd.Series, ipca12: pd.Series, lags: tuple[int, ...]):
    df = pd.DataFrame({"npl": npl, "selic": selic, "ipca12": ipca12}).dropna()
    x = pd.DataFrame({"npl_l1": df["npl"].shift(1)}, index=df.index)
    for lag in lags:
        x[f"selic_l{lag}"] = df["selic"].shift(lag)
    x["ipca12_l3"] = df["ipca12"].shift(3)
    return df, x


def fit_npl_model(
    npl: pd.Series, selic: pd.Series, ipca12: pd.Series, lags: tuple[int, ...] = (6, 9, 12)
) -> NplModel:
    """Fit NPL_t = f(NPL_t-1, Selic_t-lag, IPCA12_t-3) by ridge regression."""
    df, x = _npl_design(npl, selic, ipca12, lags)
    data = x.assign(y=df["npl"]).dropna()
    if len(data) < 24:
        raise ValueError("not enough overlapping NPL / Selic history")
    m = Ridge(alpha=1.0).fit(data.drop(columns="y"), data["y"])
    r2 = float(m.score(data.drop(columns="y"), data["y"]))
    return NplModel(m, lags, df, r2)


def forecast_npl(model: NplModel, selic_path: pd.Series, months: int = 12) -> pd.Series:
    """Recursive NPL forecast given a future monthly Selic path."""
    hist = model.history.copy()
    sel = pd.concat([hist["selic"], selic_path[selic_path.index > hist.index[-1]]])
    sel = sel[~sel.index.duplicated()]
    npl = hist["npl"].copy()
    ipca_last = hist["ipca12"].iloc[-1]
    idx = pd.date_range(hist.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    for d in idx:
        row = {"npl_l1": npl.iloc[-1]}
        for lag in model.lags:
            ref = d - pd.DateOffset(months=lag)
            row[f"selic_l{lag}"] = float(sel.asof(ref)) if ref >= sel.index[0] else sel.iloc[0]
        row["ipca12_l3"] = ipca_last
        val = float(model.model.predict(pd.DataFrame([row]))[0])
        npl = pd.concat([npl, pd.Series([val], index=[d])])
    return npl.loc[idx]


# ------------------------------------------------------------- portfolio growth
def forecast_balance(series: pd.Series, steps: int = 4) -> pd.DataFrame:
    """Damped-trend Holt forecast of a quarterly balance with an 80% band."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.log(series.dropna())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
    mean = res.forecast(steps)
    sigma = float(np.std(res.resid, ddof=1)) if len(y) > 3 else 0.03
    h = np.arange(1, steps + 1)
    band = 1.2816 * sigma * np.sqrt(h)
    idx = pd.date_range(series.index[-1], periods=steps + 1, freq="QE")[1:]
    return pd.DataFrame(
        {"p50": np.exp(mean.to_numpy()), "p10": np.exp(mean.to_numpy() - band),
         "p90": np.exp(mean.to_numpy() + band)},
        index=idx,
    )
