"""Comparing a book against the index you could have bought instead.

An honest limitation, stated up front: a holdings export carries no purchase
dates, so a true time-weighted return cannot be computed. What *can* be
computed exactly is how the securities you currently hold, at their current
weights, performed over a fixed window versus a benchmark over the same window.

That answers the question that matters - "is this selection beating the index?"
- without pretending to be a performance record.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}

# Benchmarks a Canadian retail investor would realistically buy instead.
DEFAULT_BENCHMARKS = ("SPY", "QQQ", "XEQT.TO")


def period_return(history: pd.DataFrame, days: int) -> float | None:
    """Percentage price return over the last ``days`` sessions."""
    if history is None or history.empty or "Close" not in history:
        return None
    close = history["Close"].astype(float).dropna()
    if len(close) <= days:
        return None
    prior = float(close.iloc[-days - 1])
    if prior == 0:
        return None
    return (float(close.iloc[-1]) / prior - 1) * 100


def weighted_return(
    rows: list[dict[str, Any]], histories: dict[str, pd.DataFrame], days: int
) -> tuple[float | None, float]:
    """Value-weighted return of the current book, and the share of it covered.

    Coverage matters: if half the book has too little history, a return computed
    on the other half is not the portfolio's return, and the caller should say
    so rather than print a confident number.
    """
    total_weight = 0.0
    covered_weight = 0.0
    accumulated = 0.0

    for row in rows:
        value = row.get("market_value_base")
        if value is None:
            continue
        total_weight += value
        result = period_return(histories.get(row["symbol"]), days)
        if result is None:
            continue
        covered_weight += value
        accumulated += value * result

    if covered_weight == 0:
        return None, 0.0
    return accumulated / covered_weight, (
        covered_weight / total_weight if total_weight else 0.0
    )


def compare(
    rows: list[dict[str, Any]],
    histories: dict[str, pd.DataFrame],
    benchmark_histories: dict[str, pd.DataFrame],
    actual_pnl_pct: float | None = None,
) -> dict[str, Any]:
    """Book versus each benchmark across every period."""
    table: list[dict[str, Any]] = []
    verdicts: list[str] = []

    for label, days in PERIODS.items():
        book, coverage = weighted_return(rows, histories, days)
        entry: dict[str, Any] = {
            "period": label,
            "portfolio_pct": book,
            "coverage_pct": coverage * 100,
        }
        for symbol, history in benchmark_histories.items():
            entry[symbol] = period_return(history, days)
        table.append(entry)

    # The one-year comparison is the headline; shorter windows are noise.
    year = next((r for r in table if r["period"] == "1Y"), None)
    best_benchmark = None
    if year and year.get("portfolio_pct") is not None:
        gaps = {
            symbol: year["portfolio_pct"] - value
            for symbol, value in year.items()
            if symbol in benchmark_histories and value is not None
        }
        if gaps:
            best_benchmark = min(gaps.items(), key=lambda kv: kv[1])
            symbol, gap = best_benchmark
            if gap < -5:
                verdicts.append(
                    f"Over the past year your holdings returned "
                    f"{year['portfolio_pct']:+.1f}% against {symbol}'s "
                    f"{year[symbol]:+.1f}% — behind by {abs(gap):.1f} points. "
                    "Simply owning the index would have done better."
                )
            elif gap > 5:
                verdicts.append(
                    f"Over the past year your holdings returned "
                    f"{year['portfolio_pct']:+.1f}% against {symbol}'s "
                    f"{year[symbol]:+.1f}% — ahead by {gap:.1f} points."
                )
            else:
                verdicts.append(
                    f"Over the past year your holdings tracked {symbol} closely "
                    f"({year['portfolio_pct']:+.1f}% versus {year[symbol]:+.1f}%). "
                    "You are taking single-stock risk for index-like returns."
                )

    # The most confusing number on this page is the gap between what the
    # securities did and what the holder made. Naming it prevents reading a
    # security's run-up as personal performance.
    if year and year.get("portfolio_pct") is not None and actual_pnl_pct is not None:
        divergence = year["portfolio_pct"] - actual_pnl_pct
        if abs(divergence) > 20:
            direction = "far better" if divergence > 0 else "far worse"
            verdicts.append(
                f"Note the gap: these securities returned "
                f"{year['portfolio_pct']:+.1f}% over the past year, but your "
                f"position is {actual_pnl_pct:+.1f}%. The stocks did "
                f"{direction} than you did — entry price and timing account "
                "for the difference, not stock selection."
            )

    low_coverage = [r["period"] for r in table
                    if r["coverage_pct"] < 80 and r["portfolio_pct"] is not None]
    if low_coverage:
        verdicts.append(
            "Periods " + ", ".join(low_coverage) + " cover under 80% of the book "
            "— some holdings lack that much history, so treat those rows as "
            "indicative."
        )

    return {
        "table": table,
        "commentary": verdicts,
        "note": (
            "This compares the securities you hold now, at today's weights, "
            "against the same window for each benchmark. It is not a record of "
            "your actual returns — a holdings export carries no purchase dates."
        ),
    }
