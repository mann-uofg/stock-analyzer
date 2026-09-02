"""Portfolio value through time.

WHAT THIS IS, PRECISELY
-----------------------
A holdings export carries no purchase dates, so the account's actual historical
value is not recoverable. This computes something different and well defined:
what **the basket you hold today** would have been worth across each window, at
today's share counts.

That distinction matters. It measures the securities you chose, not your timing
- so a position bought last week is charted as though you had held it all year.
Read it as "how has this basket behaved", never as a performance record.

Two details are handled properly rather than approximated:

* **Intraday** windows use 5-minute and 30-minute bars, because a daily close
  series cannot show a single day.
* **Currency** is converted at each date's own rate, not today's. Using a
  single spot rate would fold every past FX move into the equity line and
  attribute it to stock selection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# label -> (yfinance period, interval). Ordered as the selector displays them.
PERIODS: dict[str, tuple[str, str]] = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "30m"),
    "1M": ("1mo", "1d"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "All": ("max", "1wk"),
}

INTRADAY = {"1D", "5D"}


def _naive_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop timezone so series from different venues align."""
    out = frame.copy()
    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = index
    return out


def close_series(history: pd.DataFrame | None) -> pd.Series | None:
    if history is None or history.empty or "Close" not in history:
        return None
    series = _naive_index(history)["Close"].astype(float).dropna()
    return series if not series.empty else None


def build(
    holdings: list[dict[str, Any]],
    histories: dict[str, pd.DataFrame],
    fx_histories: dict[str, pd.DataFrame] | None = None,
    base_currency: str | None = None,
) -> dict[str, Any]:
    """Value of the current basket over the window covered by ``histories``.

    Returns the series plus the diagnostics a caller needs to caption it
    honestly: where it starts, and whether any holding was excluded.
    """
    fx_histories = fx_histories or {}
    closes: dict[str, pd.Series] = {}
    quantities: dict[str, float] = {}
    currencies: dict[str, str] = {}
    missing: list[str] = []

    for holding in holdings:
        symbol = holding["symbol"]
        series = close_series(histories.get(symbol))
        if series is None or not holding.get("quantity"):
            missing.append(symbol)
            continue
        closes[symbol] = series
        quantities[symbol] = float(holding["quantity"])
        currencies[symbol] = (holding.get("currency") or base_currency or "").upper()

    if not closes:
        return {"series": pd.Series(dtype=float), "missing": missing,
                "limited_by": None, "start": None, "end": None}

    # Outer join then forward fill: holdings trade on different calendars -
    # crypto every day, equities only on sessions - and an inner join would
    # discard every weekend the book still had a value on.
    prices = pd.DataFrame(closes).sort_index().ffill()

    # Only from the point every holding has a real price; before that the
    # forward fill would be inventing history. The newest listing therefore
    # sets the start for the whole basket, which is worth naming - otherwise a
    # one-year chart silently showing four months looks like a bug.
    starts = {
        col: prices[col].first_valid_index() for col in prices.columns
        if prices[col].first_valid_index() is not None
    }
    limited_by = None
    first_valid = None
    if starts:
        limited_by, first_valid = max(starts.items(), key=lambda kv: kv[1])
        # Only worth flagging if this holding actually truncated the others.
        if sum(1 for v in starts.values() if v < first_valid) == 0:
            limited_by = None
        prices = prices.loc[first_valid:]
    prices = prices.dropna(how="any")
    if prices.empty:
        return {"series": pd.Series(dtype=float), "missing": missing,
                "limited_by": limited_by, "start": None, "end": None}

    # Convert each holding at the rate that applied on each date.
    rates: dict[str, pd.Series] = {}
    for currency, history in fx_histories.items():
        series = close_series(history)
        if series is not None:
            rates[currency.upper()] = series.reindex(
                prices.index, method="ffill"
            ).bfill()

    value = pd.Series(0.0, index=prices.index)
    for symbol, quantity in quantities.items():
        line = prices[symbol] * quantity
        currency = currencies.get(symbol)
        if currency and base_currency and currency != base_currency:
            rate = rates.get(currency)
            line = line * (rate if rate is not None else 1.0)
        value = value.add(line, fill_value=0.0)

    value = value.dropna()
    return {
        "series": value,
        "missing": missing,
        "limited_by": limited_by,
        "start": value.index[0] if not value.empty else None,
        "end": value.index[-1] if not value.empty else None,
    }


def summarise(series: pd.Series) -> dict[str, Any]:
    """Change over the window, plus its extremes."""
    if series is None or series.empty or len(series) < 2:
        return {"start_value": None, "end_value": None, "change": None,
                "change_pct": None, "high": None, "low": None}

    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    return {
        "start_value": start,
        "end_value": end,
        "change": end - start,
        "change_pct": ((end / start - 1) * 100) if start else None,
        "high": float(series.max()),
        "low": float(series.min()),
    }


def normalise(series: pd.Series) -> pd.Series:
    """Rebase to percentage change from the first point, for comparison."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    first = float(series.iloc[0])
    if not first:
        return pd.Series(dtype=float)
    return (series / first - 1) * 100
