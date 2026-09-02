"""Correlation and stress testing for a book.

Position weights understate concentration. Three semiconductor names at 10%
each look like 30% spread over three ideas; if they move together at 0.85
correlation they behave like one 30% position that happens to have three
tickers. This module measures that directly, then asks what a bad day does to
the whole book.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Above this, two holdings are not meaningfully separate bets.
CLUSTER_THRESHOLD = 0.75

# Scenarios chosen to be recognisable rather than exhaustive.
SCENARIOS = (
    {"name": "Market falls 10%", "market_move_pct": -10.0,
     "detail": "A routine correction."},
    {"name": "Market falls 20%", "market_move_pct": -20.0,
     "detail": "A bear market, roughly 2022."},
    {"name": "Market rises 10%", "market_move_pct": 10.0,
     "detail": "For symmetry — high beta cuts both ways."},
)


def returns_frame(histories: dict[str, pd.DataFrame], days: int = 252) -> pd.DataFrame:
    """Aligned daily returns for every holding with usable history."""
    series: dict[str, pd.Series] = {}
    for symbol, history in histories.items():
        if history is None or history.empty or "Close" not in history:
            continue
        close = history["Close"].astype(float).dropna()
        if len(close) < 40:
            continue
        index = pd.to_datetime(close.index)
        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)
        close.index = index.normalize()
        series[symbol] = close.pct_change().dropna()

    if not series:
        return pd.DataFrame()
    frame = pd.DataFrame(series).dropna(how="all")
    return frame.iloc[-days:] if len(frame) > days else frame


def correlation_matrix(histories: dict[str, pd.DataFrame],
                       days: int = 252) -> pd.DataFrame:
    frame = returns_frame(histories, days)
    if frame.empty or frame.shape[1] < 2:
        return pd.DataFrame()
    return frame.corr(min_periods=30)


def clusters(corr: pd.DataFrame, threshold: float = CLUSTER_THRESHOLD
             ) -> list[list[str]]:
    """Group holdings that move together, by transitive closure above threshold."""
    if corr.empty:
        return []

    symbols = list(corr.columns)
    parent = {s: s for s in symbols}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(symbols):
        for b in symbols[i + 1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= threshold:
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        groups.setdefault(find(symbol), []).append(symbol)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def stress(
    rows: list[dict[str, Any]],
    betas: dict[str, float | None],
    scenarios: tuple[dict[str, Any], ...] = SCENARIOS,
) -> dict[str, Any]:
    """Estimated book impact per scenario, via each holding's beta.

    Beta is a linear approximation fitted in ordinary conditions, and
    correlations converge toward 1 in a real crash - so these figures are a
    floor on the pain, not a forecast.
    """
    priced = [r for r in rows if r.get("market_value_base") is not None]
    total = sum(r["market_value_base"] for r in priced)
    if not total:
        return {"scenarios": [], "portfolio_beta": None, "covered_pct": 0.0}

    covered = sum(
        r["market_value_base"] for r in priced if betas.get(r["symbol"]) is not None
    )
    portfolio_beta = (
        sum(
            (betas.get(r["symbol"]) or 0) * r["market_value_base"]
            for r in priced if betas.get(r["symbol"]) is not None
        ) / covered
        if covered else None
    )

    out = []
    for scenario in scenarios:
        move = scenario["market_move_pct"]
        if portfolio_beta is None:
            continue
        # Holdings without a beta are assumed to move with the market, which is
        # conservative for the loss cases.
        impact_pct = move * portfolio_beta
        out.append({
            "name": scenario["name"],
            "detail": scenario["detail"],
            "market_move_pct": move,
            "portfolio_move_pct": impact_pct,
            "value_change": total * impact_pct / 100,
            "resulting_value": total * (1 + impact_pct / 100),
        })

    return {
        "scenarios": out,
        "portfolio_beta": portfolio_beta,
        "covered_pct": covered / total * 100 if total else 0.0,
        "total_value": total,
        "note": (
            "Estimated from each holding's beta to the benchmark. Betas are "
            "fitted in normal conditions and correlations rise toward 1 in a "
            "real sell-off, so treat these as a floor rather than a forecast."
        ),
    }


def diversification_findings(
    rows: list[dict[str, Any]], corr: pd.DataFrame
) -> list[dict[str, str]]:
    """Where the book is less diversified than the position count suggests."""
    findings: list[dict[str, str]] = []
    if corr.empty:
        return findings

    weights = {
        r["symbol"]: (r.get("weight_pct") or 0.0) for r in rows
    }

    for group in clusters(corr):
        combined = sum(weights.get(s, 0.0) for s in group)
        if combined < 15:
            continue
        pairs = [
            corr.loc[a, b] for i, a in enumerate(group) for b in group[i + 1:]
            if pd.notna(corr.loc[a, b])
        ]
        average = float(np.mean(pairs)) if pairs else float("nan")
        findings.append({
            "level": "warning" if combined >= 30 else "note",
            "headline": f"{', '.join(group)} move together ({combined:.0f}% of the book)",
            "detail": (
                f"Average correlation {average:.2f}. These behave as one "
                f"position of {combined:.0f}%, not {len(group)} separate ideas — "
                "a shock to one is a shock to all of them."
            ),
        })

    # The opposite finding is worth stating too.
    if not findings and corr.shape[1] >= 3:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        average = float(np.nanmean(upper.to_numpy()))
        if average < 0.4:
            findings.append({
                "level": "good",
                "headline": f"Holdings are genuinely diversified (average correlation {average:.2f})",
                "detail": "No cluster large enough to dominate the book.",
            })
    return findings
