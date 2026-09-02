"""Market data acquisition via yfinance.

Every accessor here is defensive: Yahoo's endpoints routinely return partial
payloads, rename fields between releases, or rate-limit outright. Nothing in
this module raises on missing data - callers get ``None`` and decide.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Callable

import pandas as pd
import yfinance as yf

from .cache import get_or_fetch
from .config import BENCHMARKS, RISK_FREE_TICKER, SETTINGS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# yfinance prints raw HTTP bodies for every 404 - routine for ETFs and crypto,
# which simply have no fundamentals. We surface those gaps deliberately through
# the report's "n/a" markers, so the raw noise is suppressed here.
for _name in ("yfinance", "peewee", "urllib3"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


class DataError(RuntimeError):
    """Raised only when a ticker is unusable (no price history at all)."""


def _retry(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 1.0) -> Any:
    """Retry with exponential backoff. Returns None if every attempt fails."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                return None
            time.sleep(base_delay * (2**i))
    return None


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Call ``fn``, returning ``default`` on any failure."""
    try:
        result = fn()
    except Exception:
        return default
    return default if result is None else result


def _frame_ok(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame) and not obj.empty


# --- Price history --------------------------------------------------------


def price_history(ticker: str, period: str = "3y", interval: str = "1d",
                  use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV history. Raises DataError if the ticker yields nothing."""

    def _fetch() -> pd.DataFrame:
        df = _retry(
            lambda: yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=False, timeout=SETTINGS.request_timeout
            )
        )
        return df if _frame_ok(df) else pd.DataFrame()

    df = get_or_fetch("hist", f"{ticker}:{period}:{interval}", _fetch, use_cache=use_cache)

    if not _frame_ok(df):
        raise DataError(
            f"No price history returned for '{ticker}'. "
            "Check the symbol is valid and that Yahoo Finance is reachable."
        )

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Drop rows with no close; forward-fill incidental gaps in OHLC.
    df = df.dropna(subset=["Close"])
    return df


def benchmark_history(period: str = "3y", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Price history for each configured benchmark, keyed by symbol."""
    out: dict[str, pd.DataFrame] = {}
    for sym in BENCHMARKS:
        try:
            out[sym] = price_history(sym, period=period, use_cache=use_cache)
        except DataError:
            continue
    return out


def risk_free_rate(use_cache: bool = True) -> float:
    """Annualised risk-free rate from the 13-week T-bill, as a decimal."""

    def _fetch() -> float | None:
        df = _retry(
            lambda: yf.Ticker(RISK_FREE_TICKER).history(period="5d", timeout=SETTINGS.request_timeout)
        )
        if _frame_ok(df) and "Close" in df:
            series = df["Close"].dropna()
            if not series.empty:
                return float(series.iloc[-1]) / 100.0  # ^IRX quotes percent
        return None

    rate = get_or_fetch("riskfree", "irx", _fetch, ttl=3600, use_cache=use_cache)
    if rate is None or not (0 <= rate < 0.25):
        return SETTINGS.risk_free_fallback
    return rate


# --- Profile & fundamentals ----------------------------------------------


def ticker_info(ticker: str, use_cache: bool = True) -> dict[str, Any]:
    """The ``.info`` blob. Frequently partial; never trust a key exists."""

    def _fetch() -> dict[str, Any]:
        info = _retry(lambda: yf.Ticker(ticker).info)
        return info if isinstance(info, dict) else {}

    return get_or_fetch("info", ticker, _fetch, use_cache=use_cache) or {}


def fast_quote(ticker: str) -> dict[str, Any]:
    """Live-ish quote via fast_info. Not cached - this is the freshest read."""
    def _fetch() -> dict[str, Any]:
        fi = yf.Ticker(ticker).fast_info
        return {
            "last_price": _safe(lambda: float(fi["lastPrice"])),
            "previous_close": _safe(lambda: float(fi["previousClose"])),
            "open": _safe(lambda: float(fi["open"])),
            "day_high": _safe(lambda: float(fi["dayHigh"])),
            "day_low": _safe(lambda: float(fi["dayLow"])),
            "market_cap": _safe(lambda: float(fi["marketCap"])),
            "shares": _safe(lambda: float(fi["shares"])),
        }

    return _retry(_fetch) or {}


def earnings_history(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """Past and upcoming earnings dates with EPS estimate/actual/surprise."""

    def _fetch() -> pd.DataFrame:
        df = _retry(lambda: yf.Ticker(ticker).get_earnings_dates(limit=16))
        return df if _frame_ok(df) else pd.DataFrame()

    return get_or_fetch("earnings", ticker, _fetch, ttl=6 * 3600, use_cache=use_cache)


def analyst_estimates(ticker: str, use_cache: bool = True) -> dict[str, Any]:
    """Forward EPS/revenue consensus and price targets, where exposed."""

    def _fetch() -> dict[str, Any]:
        t = yf.Ticker(ticker)
        payload: dict[str, Any] = {}

        eps = _safe(lambda: t.earnings_estimate)
        if _frame_ok(eps):
            payload["eps_estimate"] = eps.to_dict(orient="index")

        rev = _safe(lambda: t.revenue_estimate)
        if _frame_ok(rev):
            payload["revenue_estimate"] = rev.to_dict(orient="index")

        targets = _safe(lambda: t.analyst_price_targets)
        if isinstance(targets, dict) and targets:
            payload["price_targets"] = targets

        rec = _safe(lambda: t.recommendations)
        if _frame_ok(rec):
            payload["recommendations"] = rec.head(4).to_dict(orient="records")

        return payload

    return get_or_fetch("estimates", ticker, _fetch, ttl=6 * 3600, use_cache=use_cache) or {}


def financials(ticker: str, use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Quarterly and annual statements needed for FCF / EV-EBITDA work."""

    def _fetch() -> dict[str, pd.DataFrame]:
        t = yf.Ticker(ticker)
        frames = {
            "income_q": _safe(lambda: t.quarterly_income_stmt, pd.DataFrame()),
            "income_a": _safe(lambda: t.income_stmt, pd.DataFrame()),
            "cashflow_q": _safe(lambda: t.quarterly_cashflow, pd.DataFrame()),
            "cashflow_a": _safe(lambda: t.cashflow, pd.DataFrame()),
            "balance_q": _safe(lambda: t.quarterly_balance_sheet, pd.DataFrame()),
        }
        return {k: (v if _frame_ok(v) else pd.DataFrame()) for k, v in frames.items()}

    return get_or_fetch("financials", ticker, _fetch, ttl=12 * 3600, use_cache=use_cache) or {}


# --- Options --------------------------------------------------------------


def option_expirations(ticker: str, use_cache: bool = True) -> list[str]:
    """Available expiry dates, oldest first. Empty list if none are listed."""

    def _fetch() -> list[str]:
        exps = _retry(lambda: yf.Ticker(ticker).options)
        return list(exps) if exps else []

    return get_or_fetch("expiries", ticker, _fetch, ttl=3600, use_cache=use_cache) or []


def option_chain(ticker: str, expiry: str, use_cache: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calls, puts) for one expiry. Returns empty frames when unavailable."""

    def _fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
        chain = _retry(lambda: yf.Ticker(ticker).option_chain(expiry))
        if chain is None:
            return pd.DataFrame(), pd.DataFrame()
        calls = chain.calls if _frame_ok(chain.calls) else pd.DataFrame()
        puts = chain.puts if _frame_ok(chain.puts) else pd.DataFrame()
        return calls, puts

    result = get_or_fetch("chain", f"{ticker}:{expiry}", _fetch, ttl=600, use_cache=use_cache)
    return result if result else (pd.DataFrame(), pd.DataFrame())


def fx_rate(base: str, quote: str, use_cache: bool = True) -> float | None:
    """Spot rate to convert ``base`` into ``quote`` (e.g. USD -> CAD).

    Returns None when the pair is unavailable, so callers can fall back to
    reporting per-currency rather than inventing a conversion.
    """
    base, quote = base.strip().upper(), quote.strip().upper()
    if not base or not quote:
        return None
    if base == quote:
        return 1.0

    def _fetch() -> float | None:
        for pair, invert in ((f"{base}{quote}=X", False), (f"{quote}{base}=X", True)):
            df = _retry(
                lambda p=pair: yf.Ticker(p).history(
                    period="5d", timeout=SETTINGS.request_timeout
                ),
                attempts=2,
            )
            if _frame_ok(df) and "Close" in df:
                series = df["Close"].dropna()
                if not series.empty:
                    rate = float(series.iloc[-1])
                    if rate > 0:
                        return 1 / rate if invert else rate
        return None

    return get_or_fetch("fx", f"{base}{quote}", _fetch, ttl=3600, use_cache=use_cache)


def news(ticker: str, limit: int = 30, use_cache: bool = True) -> list[dict[str, Any]]:
    """Recent headlines. Purely supplementary - absence is not an error.

    The summary is kept alongside the title because the speaker is frequently
    named in the body rather than the headline - "chip maker rallies" in the
    title, "after Jensen Huang said" in the first line.
    """

    def _fetch() -> list[dict[str, Any]]:
        items = _retry(lambda: yf.Ticker(ticker).news) or []
        out = []
        for item in items[:limit]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            title = content.get("title") or item.get("title")
            if not title:
                continue
            provider = content.get("provider") or {}
            link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            out.append(
                {
                    "title": title,
                    "summary": (content.get("summary")
                                or content.get("description") or "").strip(),
                    "publisher": (
                        provider.get("displayName")
                        if isinstance(provider, dict)
                        else item.get("publisher")
                    ),
                    "link": link.get("url") if isinstance(link, dict) else None,
                    "published": content.get("pubDate") or content.get("displayTime"),
                }
            )
        return out

    return get_or_fetch("news", ticker, _fetch, ttl=1800, use_cache=use_cache) or []
