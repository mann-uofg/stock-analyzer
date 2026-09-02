"""Pipeline orchestration: ticker in, complete analysis payload out."""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

from . import datafeed, fundamentals, indicators, options, risk, scoring
from .datafeed import DataError

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def analyse(
    ticker: str,
    period: str = "5y",
    use_cache: bool = True,
    skip_options: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the full analysis for one ticker.

    Raises ``DataError`` only when the symbol has no price history at all;
    every other missing input degrades into a ``None``/unavailable marker.
    """
    say = progress or _noop
    ticker = ticker.strip().upper()

    say("Fetching price history")
    price_df = datafeed.price_history(ticker, period=period, use_cache=use_cache)

    say("Fetching benchmarks")
    benchmarks = datafeed.benchmark_history(period=period, use_cache=use_cache)
    risk_free = datafeed.risk_free_rate(use_cache=use_cache)

    say("Fetching quote and profile")
    quote = datafeed.fast_quote(ticker)
    info = datafeed.ticker_info(ticker, use_cache=use_cache)

    spot = quote.get("last_price") or float(price_df["Close"].iloc[-1])

    say("Computing technical indicators")
    tech = indicators.compute(price_df)

    say("Computing risk metrics")
    risk_panel = risk.compute(price_df, benchmarks, risk_free)
    hv_series = risk.historical_volatility(price_df)
    hv_current = risk_panel.get("volatility", {}).get("hv_30d_annual_pct")

    if skip_options:
        opts: dict[str, Any] = {"available": False, "reason": "skipped by request"}
    else:
        say("Analysing option chain")
        expiries = datafeed.option_expirations(ticker, use_cache=use_cache)
        opts = options.analyse(
            ticker, spot, expiries,
            lambda t, e: datafeed.option_chain(t, e, use_cache=use_cache),
            risk_free, hv_series, hv_current,
        )

    say("Fetching fundamentals and earnings")
    earnings_df = datafeed.earnings_history(ticker, use_cache=use_cache)
    estimates = datafeed.analyst_estimates(ticker, use_cache=use_cache)
    financial_statements = datafeed.financials(ticker, use_cache=use_cache)
    fund = fundamentals.compute(
        info, earnings_df, estimates, financial_statements, price_df,
        quote.get("market_cap"),
    )

    say("Scoring")
    verdict = scoring.compute(tech, risk_panel, opts, fund, spot)

    prev = quote.get("previous_close")
    change_pct = ((spot / prev - 1) * 100) if prev else None

    return {
        "meta": {
            "ticker": ticker,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "period": period,
            "data_source": "Yahoo Finance via yfinance",
            "bars_analysed": tech.get("bars"),
        },
        "quote": {
            **quote,
            "spot": spot,
            "change_pct": change_pct,
            "last_bar_date": str(price_df.index[-1].date()),
        },
        "technical": tech,
        "risk": risk_panel,
        "options": opts,
        "fundamental": fund,
        "verdict": verdict,
        "news": datafeed.news(ticker, use_cache=use_cache),
    }


__all__ = ["analyse", "DataError"]
