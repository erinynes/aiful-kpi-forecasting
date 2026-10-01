"""Forecast AIFUL monthly KPIs and test whether Japanese macro drivers add accuracy.

Design
------
Targets (monthly, AIFUL standalone):
  applications        - unsecured personal-loan applications
  new_accounts        - new unsecured accounts acquired
  unsecured_loans_mil - unsecured personal-loan balance (¥mil)

Models compared on an expanding-window, rolling-origin backtest (h = 1..6 months):
  snaive    - seasonal naive (same month last year)
  ets       - Holt-Winters, damped additive trend + additive seasonality, on log scale
  sarima    - SARIMA(1,1,1)(0,1,1)12 on log scale + intervention dummies
  sarimax   - identical SARIMA + macro drivers (lagged 6 months)
  lgbm      - LightGBM direct multi-horizon model on log-growth, with macro features

Fairness rule: sarima and sarimax share the same intervention dummies (COVID, the
June-2023 change in how applications are counted), so the only difference is macro.
Macro drivers enter with a 6-month lag, so every value needed for h <= 6 is already
published at the forecast origin (no look-ahead).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TARGETS = ["applications", "new_accounts", "unsecured_loans_mil"]
MACRO_LAG = 6
H = 6
MACRO_FEATURES = ["d_unemp", "consumer_confidence", "core_cpi_yoy"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    k = pd.read_csv(PROC / "aiful_monthly_kpis.csv", index_col=0)
    k.index = pd.PeriodIndex(k.index, freq="M")
    m = pd.read_csv(PROC / "macro_monthly.csv", index_col=0)
    m.index = pd.PeriodIndex(m.index, freq="M")
    m["d_unemp"] = m["unemployment_rate"].diff(12)
    return k, m


def interventions(index: pd.PeriodIndex, target: str) -> pd.DataFrame:
    d = pd.DataFrame(index=index)
    # first COVID state of emergency: Apr-May 2020 lockdown, demand collapse through Jun
    d["covid"] = ((index >= pd.Period("2020-04", "M")) & (index <= pd.Period("2020-06", "M"))).astype(float)
    if target in ("applications",):
        # AIFUL disclosed that applications were over-counted (duplicates) until May 2023
        d["count_method_pre_2023_06"] = (index < pd.Period("2023-06", "M")).astype(float)
    return d


def macro_exog(m: pd.DataFrame, index: pd.PeriodIndex, scenario: dict | None = None) -> pd.DataFrame:
    """Macro drivers lagged MACRO_LAG months, aligned to `index`.
    Months beyond the last published value are held flat (or shifted by a scenario)."""
    src = m[MACRO_FEATURES].copy()
    full = pd.period_range(src.index.min(), index.max(), freq="M")
    src = src.reindex(full).ffill()
    if scenario:
        last = m[MACRO_FEATURES].dropna().index.max()
        for col, shock in scenario.items():
            src.loc[src.index > last, col] += shock
    return src.shift(MACRO_LAG).reindex(index)


# ---------------------------------------------------------------- models
def f_snaive(y: pd.Series, h: int, **_) -> np.ndarray:
    return np.array([y.iloc[-12 + (i % 12)] for i in range(h)])


def f_ets(y: pd.Series, h: int, **_) -> np.ndarray:
    fit = ExponentialSmoothing(np.log(y.values), trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=12).fit(optimized=True)
    return np.exp(fit.forecast(h))


def _sarimax(y, h, exog_tr, exog_fc, return_ci=False):
    mod = SARIMAX(np.log(y.values), exog=None if exog_tr is None else exog_tr.values,
                  order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False, maxiter=200)
    fc = res.get_forecast(h, exog=None if exog_fc is None else exog_fc.values)
    if return_ci:
        return res, fc
    return np.exp(fc.predicted_mean)


def f_sarima(y, h, target, fc_index, **_):
    ex_tr = interventions(y.index, target)
    ex_fc = interventions(fc_index, target)
    return _sarimax(y, h, ex_tr, ex_fc)


def f_sarimax(y, h, target, fc_index, macro, **_):
    ex_tr = interventions(y.index, target).join(macro_exog(macro, y.index))
    ex_fc = interventions(fc_index, target).join(macro_exog(macro, fc_index))
    return _sarimax(y, h, ex_tr, ex_fc)


def _lgbm_frame(y: pd.Series, macro: pd.DataFrame, target: str) -> pd.DataFrame:
    ly = np.log(y)
    X = pd.DataFrame(index=y.index)
    for L in (0, 1, 2, 11):
        X[f"lag{L}"] = ly.shift(L) - ly.shift(12)  # level relative to a year ago
    X["g1"] = ly - ly.shift(1)
    X["g12"] = ly - ly.shift(12)
    X["month"] = y.index.month
    X = X.join(macro_exog(macro, y.index).add_prefix("mx_"))
    return X


def f_lgbm(y, h, target, fc_index, macro, **_):
    X = _lgbm_frame(y, macro, target)
    ly = np.log(y)
    preds = []
    for step in range(1, h + 1):
        tgt = ly.shift(-step) - ly  # log growth over `step` months
        tr = X.join(tgt.rename("t")).join(pd.Series((y.index + step).month, index=y.index, name="tgt_month"))
        tr["seasonal_ref"] = ly.shift(12 - step) - ly if step <= 12 else np.nan
        train = tr.dropna()
        feats = [c for c in train.columns if c != "t"]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                  min_child_samples=8, subsample=0.9, subsample_freq=1,
                                  colsample_bytree=0.9, verbose=-1, random_state=0)
        model.fit(train[feats], train["t"])
        last = tr.iloc[[-1]][feats]
        preds.append(float(np.exp(ly.iloc[-1] + model.predict(last)[0])))
    return np.array(preds)


MODELS = {"snaive": f_snaive, "ets": f_ets, "sarima": f_sarima, "sarimax": f_sarimax, "lgbm": f_lgbm}


# ---------------------------------------------------------------- backtest
def backtest(k: pd.DataFrame, m: pd.DataFrame, start="2021-09", end=None, models=None) -> pd.DataFrame:
    models = models or list(MODELS)
    rows = []
    for target in TARGETS:
        y_all = k[target].dropna()
        last_origin = pd.Period(end, "M") if end else y_all.index.max() - H
        origins = pd.period_range(start, last_origin, freq="M")
        for origin in origins:
            y = y_all.loc[:origin]
            fc_index = pd.period_range(origin + 1, origin + H, freq="M")
            actual = y_all.reindex(fc_index).values
            for name in models:
                try:
                    p = MODELS[name](y, H, target=target, fc_index=fc_index, macro=m)
                except Exception as e:  # keep the loop alive, record failure
                    p = np.full(H, np.nan)
                    print(target, origin, name, "failed:", e)
                for i in range(H):
                    rows.append((target, str(origin), name, i + 1, str(fc_index[i]), actual[i], p[i]))
        print(f"backtest done: {target} ({len(origins)} origins)")
    bt = pd.DataFrame(rows, columns=["target", "origin", "model", "h", "month", "actual", "forecast"])
    bt["ape"] = (bt["forecast"] - bt["actual"]).abs() / bt["actual"] * 100
    return bt


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """DM test on squared log errors with Newey-West variance (Harvey et al. small-sample correction)."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return dm, p


# ---------------------------------------------------------------- final forecast
@dataclass
class Forecast:
    target: str
    table: pd.DataFrame  # mean, lo80, hi80, lo95, hi95


def final_forecast(k, m, target, model="sarimax", horizon=12, scenario=None) -> Forecast:
    y = k[target].dropna()
    fc_index = pd.period_range(y.index.max() + 1, periods=horizon, freq="M")
    ex_tr = interventions(y.index, target)
    ex_fc = interventions(fc_index, target)
    if model == "sarimax":
        ex_tr = ex_tr.join(macro_exog(m, y.index))
        ex_fc = ex_fc.join(macro_exog(m, fc_index, scenario))
    _, fc = _sarimax(y, horizon, ex_tr, ex_fc, return_ci=True)
    ci80 = np.exp(fc.conf_int(alpha=0.2))
    ci95 = np.exp(fc.conf_int(alpha=0.05))
    t = pd.DataFrame({"mean": np.exp(fc.predicted_mean), "lo80": ci80[:, 0], "hi80": ci80[:, 1],
                      "lo95": ci95[:, 0], "hi95": ci95[:, 1]}, index=fc_index)
    return Forecast(target, t)
