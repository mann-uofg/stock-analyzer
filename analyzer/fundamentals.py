"""Fundamentals, earnings history, forward consensus, and the catalyst clock.

Free Yahoo data covers EPS surprise history and forward EPS/revenue consensus,
but exposes **no historical revenue estimates** - so revenue "surprise" cannot
be computed. We report revenue actuals with YoY growth instead and say so,
rather than inventing a comparison.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd


def _row(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """Find a statement row by any of several label spellings."""
    if df is None or df.empty:
        return None
    lowered = {str(idx).strip().lower(): idx for idx in df.index}
    for name in candidates:
        key = name.strip().lower()
        if key in lowered:
            return df.loc[lowered[key]]
    return None


def _ttm(series: pd.Series | None, periods: int = 4) -> float | None:
    """Trailing-twelve-month sum of the most recent quarterly values."""
    if series is None:
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < periods:
        return None
    return float(clean.iloc[:periods].sum())


def _f(value: Any) -> float | None:
    """Coerce to a finite float, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _naive(ts: Any) -> dt.datetime | None:
    try:
        stamp = pd.Timestamp(ts)
    except Exception:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.to_pydatetime()


def earnings_panel(
    earnings_df: pd.DataFrame, price_df: pd.DataFrame, quarters: int = 4
) -> dict[str, Any]:
    """Past surprise history, the next catalyst, and post-earnings move stats."""
    panel: dict[str, Any] = {
        "history": [],
        "next_earnings_date": None,
        "days_to_earnings": None,
        "avg_abs_post_earnings_move_pct": None,
        "post_earnings_moves": [],
        "avg_eps_surprise_pct": None,
        "beat_rate_pct": None,
    }

    if earnings_df is None or earnings_df.empty:
        return panel

    df = earnings_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
    df = df[df.index.notna()].sort_index(ascending=False)

    now = pd.Timestamp.now(tz="UTC")
    est_col = next((c for c in df.columns if "estimate" in c.lower()), None)
    act_col = next((c for c in df.columns if "reported" in c.lower()), None)
    sur_col = next((c for c in df.columns if "surprise" in c.lower()), None)

    # --- Upcoming catalyst ---
    future = df[df.index > now]
    if not future.empty:
        nxt = future.index.min()
        panel["next_earnings_date"] = nxt.date().isoformat()
        panel["days_to_earnings"] = int((nxt - now).days)

    # --- Reported history ---
    reported = df[df.index <= now]
    if act_col:
        reported = reported[pd.to_numeric(reported[act_col], errors="coerce").notna()]

    surprises: list[float] = []
    beats = 0
    for stamp, row in reported.head(quarters).iterrows():
        est = _f(row.get(est_col)) if est_col else None
        act = _f(row.get(act_col)) if act_col else None
        sur = _f(row.get(sur_col)) if sur_col else None
        if sur is None and est not in (None, 0) and act is not None:
            sur = (act - est) / abs(est) * 100
        if sur is not None:
            surprises.append(sur)
            beats += int(sur > 0)
        panel["history"].append(
            {
                "date": stamp.date().isoformat(),
                "eps_estimate": est,
                "eps_actual": act,
                "eps_surprise_pct": sur,
            }
        )

    if surprises:
        panel["avg_eps_surprise_pct"] = float(np.mean(surprises))
        panel["beat_rate_pct"] = beats / len(surprises) * 100

    # --- Historical post-earnings reaction ---
    #
    # Reports land after the close, so the tradeable reaction is the following
    # session's close-to-close move.
    if price_df is not None and not price_df.empty:
        close = price_df["Close"].astype(float).copy()
        idx = pd.to_datetime(close.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        close.index = idx.normalize()

        moves = []
        for stamp in reported.head(8).index:
            day = _naive(stamp)
            if day is None:
                continue
            after = close[close.index > pd.Timestamp(day).normalize()]
            before = close[close.index <= pd.Timestamp(day).normalize()]
            if after.empty or before.empty:
                continue
            move = (after.iloc[0] / before.iloc[-1] - 1) * 100
            if np.isfinite(move):
                moves.append({"date": stamp.date().isoformat(), "move_pct": float(move)})

        if moves:
            panel["post_earnings_moves"] = moves
            panel["avg_abs_post_earnings_move_pct"] = float(
                np.mean([abs(m["move_pct"]) for m in moves])
            )

    return panel


def consensus_panel(estimates: dict[str, Any]) -> dict[str, Any]:
    """Forward EPS and revenue consensus, plus analyst price targets."""
    label_map = {
        "0q": "current_quarter",
        "+1q": "next_quarter",
        "0y": "current_year",
        "+1y": "next_year",
    }
    out: dict[str, Any] = {"eps": {}, "revenue": {}, "price_targets": None}

    for source, key, value_name in (
        ("eps_estimate", "eps", "yearAgoEps"),
        ("revenue_estimate", "revenue", "yearAgoRevenue"),
    ):
        block = estimates.get(source) or {}
        for period, row in block.items():
            label = label_map.get(str(period))
            if not label or not isinstance(row, dict):
                continue
            out[key][label] = {
                "consensus": _f(row.get("avg")),
                "low": _f(row.get("low")),
                "high": _f(row.get("high")),
                "analysts": _f(row.get("numberOfAnalysts")),
                "yoy_growth_pct": (
                    _f(row.get("growth")) * 100 if _f(row.get("growth")) is not None else None
                ),
                "year_ago": _f(row.get(value_name)),
            }

    targets = estimates.get("price_targets")
    if isinstance(targets, dict) and targets:
        out["price_targets"] = {k: _f(v) for k, v in targets.items()}

    return out


def valuation_panel(
    info: dict[str, Any], fin: dict[str, pd.DataFrame], market_cap: float | None
) -> dict[str, Any]:
    """Valuation multiples, computed from statements where Yahoo omits them."""
    out: dict[str, Any] = {
        "trailing_pe": _f(info.get("trailingPE")),
        "forward_pe": _f(info.get("forwardPE")),
        "peg_ratio": _f(info.get("trailingPegRatio")) or _f(info.get("pegRatio")),
        "price_to_sales": _f(info.get("priceToSalesTrailing12Months")),
        "price_to_book": _f(info.get("priceToBook")),
        "ev_to_ebitda": _f(info.get("enterpriseToEbitda")),
        "ev_to_revenue": _f(info.get("enterpriseToRevenue")),
        "profit_margin_pct": (
            _f(info.get("profitMargins")) * 100 if _f(info.get("profitMargins")) is not None else None
        ),
        "return_on_equity_pct": (
            _f(info.get("returnOnEquity")) * 100
            if _f(info.get("returnOnEquity")) is not None
            else None
        ),
        "debt_to_equity": _f(info.get("debtToEquity")),
        "market_cap": market_cap or _f(info.get("marketCap")),
        "enterprise_value": _f(info.get("enterpriseValue")),
    }

    cash_q = fin.get("cashflow_q", pd.DataFrame())
    income_q = fin.get("income_q", pd.DataFrame())

    # --- Free cash flow yield ---
    fcf = _ttm(_row(cash_q, "Free Cash Flow"))
    if fcf is None:
        ocf = _ttm(_row(cash_q, "Operating Cash Flow", "Total Cash From Operating Activities"))
        capex = _ttm(_row(cash_q, "Capital Expenditure", "Capital Expenditures"))
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is reported negative
    out["free_cash_flow_ttm"] = fcf

    cap = out["market_cap"]
    out["fcf_yield_pct"] = (fcf / cap * 100) if fcf and cap else None

    # --- Revenue trend (stands in for unavailable revenue-surprise history) ---
    revenue = _row(income_q, "Total Revenue", "Operating Revenue")
    rev_ttm = _ttm(revenue)
    out["revenue_ttm"] = rev_ttm
    if revenue is not None:
        clean = pd.to_numeric(revenue, errors="coerce").dropna()
        if len(clean) >= 5:
            latest, year_ago = float(clean.iloc[0]), float(clean.iloc[4])
            out["revenue_yoy_growth_pct"] = (
                (latest / year_ago - 1) * 100 if year_ago else None
            )
        out["latest_quarter_revenue"] = float(clean.iloc[0]) if len(clean) else None

    # Fall back to computing EV/EBITDA when Yahoo omits it.
    if out["ev_to_ebitda"] is None:
        ebitda = _ttm(_row(income_q, "EBITDA", "Normalized EBITDA"))
        ev = out["enterprise_value"]
        if ebitda and ev:
            out["ev_to_ebitda"] = ev / ebitda

    out["note"] = (
        "Historical revenue estimates are not available from free Yahoo data, "
        "so revenue surprise history cannot be computed; revenue actuals and "
        "YoY growth are shown instead."
    )
    return out


def compute(
    info: dict[str, Any],
    earnings_df: pd.DataFrame,
    estimates: dict[str, Any],
    fin: dict[str, pd.DataFrame],
    price_df: pd.DataFrame,
    market_cap: float | None = None,
) -> dict[str, Any]:
    """Assemble the full fundamental panel."""
    return {
        "profile": {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "employees": info.get("fullTimeEmployees"),
            "currency": info.get("currency"),
        },
        "valuation": valuation_panel(info, fin, market_cap),
        "earnings": earnings_panel(earnings_df, price_df),
        "consensus": consensus_panel(estimates),
    }
