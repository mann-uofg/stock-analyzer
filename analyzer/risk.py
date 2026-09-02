"""Risk metrics: beta, Jensen's alpha, realised volatility, drawdown.

Betas are computed on **daily** returns against each benchmark over the
configured windows. Alpha follows CAPM:

    alpha = R_asset - [ R_f + beta * (R_market - R_f) ]

expressed as an annualised percentage.

Note on comparability: Yahoo's own ``info['beta']`` is estimated from five
years of *monthly* returns, so it will not match the 1y/3y daily figures here.
Neither is wrong - they are different estimators, and the daily windows react
far faster to a change in regime. Where the regression explains little of the
variance, ``low_explanatory_power`` is set so the number is not mistaken for a
stable risk measure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import BETA_WINDOWS, HV_WINDOW, TRADING_DAYS


def _returns(df: pd.DataFrame) -> pd.Series:
    """Daily simple returns from adjusted closes (falls back to Close)."""
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].astype(float).pct_change().dropna()


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two return series on a shared, tz-naive daily index."""
    a = a.copy()
    b = b.copy()
    for s in (a, b):
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    a.index = pd.to_datetime(a.index).normalize()
    b.index = pd.to_datetime(b.index).normalize()
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return joined.iloc[:, 0], joined.iloc[:, 1]


def beta_alpha(
    asset: pd.Series, market: pd.Series, risk_free: float, window: int
) -> dict[str, float | None]:
    """Beta and annualised Jensen's alpha over the trailing ``window`` days."""
    empty = {
        "beta": None, "alpha_annual_pct": None, "r_squared": None,
        "low_explanatory_power": None,
    }

    a, m = _align(asset, market)
    if len(a) < max(30, window // 4):
        return {**empty, "observations": len(a)}

    a, m = a.iloc[-window:], m.iloc[-window:]
    var_m = float(np.var(m, ddof=1))
    if var_m == 0 or not np.isfinite(var_m):
        return {**empty, "observations": len(a)}

    cov = float(np.cov(a, m, ddof=1)[0, 1])
    beta = cov / var_m

    periods = len(a)
    # Annualise realised returns geometrically over the observed window.
    asset_ann = float((1 + a).prod() ** (TRADING_DAYS / periods) - 1)
    market_ann = float((1 + m).prod() ** (TRADING_DAYS / periods) - 1)
    alpha = asset_ann - (risk_free + beta * (market_ann - risk_free))

    corr = float(np.corrcoef(a, m)[0, 1])
    r_squared = corr**2

    return {
        "beta": beta,
        "alpha_annual_pct": alpha * 100,
        "r_squared": r_squared,
        "observations": periods,
        # A beta estimated off near-zero correlation is not meaningful, however
        # precise it looks. Defensive names routinely decouple over a single
        # year of daily returns and print betas near zero or negative; the
        # figure is real but carries little explanatory power, so flag it
        # instead of letting it read as a stable risk measure.
        "low_explanatory_power": bool(r_squared < 0.10),
    }


def _monthly_returns(df: pd.DataFrame) -> pd.Series:
    """Month-end returns, used for the industry-convention beta."""
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    series = df[col].astype(float).copy()
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series.resample("ME").last().pct_change().dropna()


def monthly_beta(
    asset_df: pd.DataFrame, market_df: pd.DataFrame, months: int = 60
) -> dict[str, Any]:
    """Beta from ~5 years of monthly returns.

    This is the estimator Yahoo, Google Finance and most data vendors publish,
    so it is the figure to compare against when sanity-checking this tool
    against a public source. It is deliberately slower-moving than the daily
    betas above: monthly sampling filters out day-to-day noise, and a 5-year
    window spans multiple regimes.
    """
    a, m = _align(_monthly_returns(asset_df), _monthly_returns(market_df))
    if len(a) < 24:  # under two years of months is not worth reporting
        return {"beta": None, "r_squared": None, "months": len(a)}

    a, m = a.iloc[-months:], m.iloc[-months:]
    var_m = float(np.var(m, ddof=1))
    if var_m == 0 or not np.isfinite(var_m):
        return {"beta": None, "r_squared": None, "months": len(a)}

    beta = float(np.cov(a, m, ddof=1)[0, 1] / var_m)
    corr = float(np.corrcoef(a, m)[0, 1])
    return {"beta": beta, "r_squared": corr**2, "months": int(len(a))}


def historical_volatility(df: pd.DataFrame, window: int = HV_WINDOW) -> pd.Series:
    """Annualised rolling realised volatility, in percent."""
    rets = np.log(df["Close"].astype(float)).diff()
    return (rets.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100).dropna()


def max_drawdown(df: pd.DataFrame, lookback: int = TRADING_DAYS) -> float | None:
    close = df["Close"].astype(float).iloc[-lookback:]
    if close.empty:
        return None
    peak = close.cummax()
    dd = (close / peak - 1).min()
    return float(dd) * 100


def compute(
    df: pd.DataFrame, benchmarks: dict[str, pd.DataFrame], risk_free: float
) -> dict[str, Any]:
    """Full risk panel for one asset against every supplied benchmark."""
    asset_rets = _returns(df)
    out: dict[str, Any] = {"risk_free_rate_pct": risk_free * 100}

    bench_block: dict[str, Any] = {}
    for symbol, bdf in benchmarks.items():
        if bdf is None or bdf.empty:
            continue
        m_rets = _returns(bdf)
        bench_block[symbol] = {
            label: beta_alpha(asset_rets, m_rets, risk_free, window)
            for label, window in BETA_WINDOWS.items()
        }
        # The public-convention estimator, for direct comparison against any
        # finance website.
        bench_block[symbol]["5y_monthly"] = monthly_beta(df, bdf)
    out["benchmarks"] = bench_block

    hv = historical_volatility(df)
    hv_current = float(hv.iloc[-1]) if not hv.empty else None
    out["volatility"] = {
        "hv_30d_annual_pct": hv_current,
        "hv_percentile_1y": (
            float((hv.iloc[-TRADING_DAYS:] < hv_current).mean() * 100)
            if hv_current is not None and len(hv) >= 60
            else None
        ),
        "hv_1y_min": float(hv.iloc[-TRADING_DAYS:].min()) if len(hv) >= 60 else None,
        "hv_1y_max": float(hv.iloc[-TRADING_DAYS:].max()) if len(hv) >= 60 else None,
    }

    rets_1y = asset_rets.iloc[-TRADING_DAYS:]
    if len(rets_1y) >= 60:
        ann_ret = float((1 + rets_1y).prod() ** (TRADING_DAYS / len(rets_1y)) - 1)
        ann_vol = float(rets_1y.std() * np.sqrt(TRADING_DAYS))
        downside = rets_1y[rets_1y < 0]
        down_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else None
        out["ratios"] = {
            "annual_return_pct": ann_ret * 100,
            "annual_volatility_pct": ann_vol * 100,
            "sharpe": (ann_ret - risk_free) / ann_vol if ann_vol else None,
            "sortino": (ann_ret - risk_free) / down_vol if down_vol else None,
        }
    else:
        out["ratios"] = {}

    out["max_drawdown_1y_pct"] = max_drawdown(df)
    return out
