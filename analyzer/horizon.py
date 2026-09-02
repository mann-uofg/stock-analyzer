"""Near-term versus long-term scoring.

The composite verdict in ``scoring.py`` answers "is this attractive?". It does
not answer "attractive *when?*" - and those are different questions. A stock
with deteriorating fundamentals can be an excellent two-week trade; a stock
with poor momentum can be an excellent three-year hold.

This module reweights the same bucket scores against two horizons:

* **Near term** (days to weeks) leans on momentum, trend and positioning.
  Valuation barely moves a stock over a fortnight, so fundamentals carry no
  weight here.
* **Long term** (quarters to years) leans on fundamentals and the primary
  trend. Today's RSI is noise at that horizon.

Both return 0-100 on the same scale as the headline score, so they are directly
comparable across a watchlist.
"""

from __future__ import annotations

from typing import Any

NEAR_TERM_WEIGHTS = {
    "momentum": 0.32,
    "trend": 0.26,
    "volume": 0.18,
    "options": 0.14,
    "volatility": 0.10,
    "fundamental": 0.00,
}

LONG_TERM_WEIGHTS = {
    "fundamental": 0.46,
    "trend": 0.30,
    "volatility": 0.10,
    "momentum": 0.09,
    "volume": 0.05,
    "options": 0.00,
}


def _weighted(buckets: dict[str, Any], weights: dict[str, float]) -> float | None:
    """Weighted mean of bucket scores in [-1, 1], renormalised over what exists.

    Renormalisation matters: when a ticker has no option chain, the options
    bucket is absent, and without it the remaining weights would sum to less
    than one and silently drag every score toward neutral.
    """
    total_weight = 0.0
    total = 0.0
    for name, weight in weights.items():
        bucket = buckets.get(name)
        if not bucket or weight == 0:
            continue
        # A bucket the scorer marked unavailable holds 0.0 as a placeholder,
        # not as a neutral reading; including it would pull the horizon score
        # toward the midpoint for every ETF and coin.
        if bucket.get("available") is False:
            continue
        score = bucket.get("score")
        if score is None:
            continue
        total += float(score) * weight
        total_weight += weight

    if total_weight == 0:
        return None
    return total / total_weight


def _to_100(score: float | None) -> float | None:
    return None if score is None else round((score + 1) * 50, 1)


def compute(payload: dict[str, Any]) -> dict[str, Any]:
    """Horizon scores and a plain-language read for one analysed ticker."""
    verdict = payload.get("verdict") or {}
    buckets = verdict.get("buckets") or {}

    near = _to_100(_weighted(buckets, NEAR_TERM_WEIGHTS))
    long = _to_100(_weighted(buckets, LONG_TERM_WEIGHTS))

    out: dict[str, Any] = {
        "near_term_score": near,
        "long_term_score": long,
        "catalyst_days": None,
        "bias": None,
        "summary": None,
    }

    earnings = (payload.get("fundamental") or {}).get("earnings") or {}
    days = earnings.get("days_to_earnings")
    out["catalyst_days"] = days

    if near is None or long is None:
        return out

    gap = near - long
    if abs(gap) < 8:
        bias = "balanced"
    elif gap > 0:
        bias = "near term"
    else:
        bias = "long term"
    out["bias"] = bias

    # A near-dated earnings report is the dominant near-term consideration, so
    # it is called out regardless of which side the scores favour.
    catalyst = ""
    if isinstance(days, (int, float)) and 0 <= days <= 21:
        catalyst = f" Earnings in {int(days)} days dominate the near-term path."

    if bias == "near term":
        summary = (
            f"Setup favours a shorter hold — momentum and positioning "
            f"({near:.0f}) outrun the fundamental case ({long:.0f})."
        )
    elif bias == "long term":
        summary = (
            f"Better as a hold than a trade — fundamentals ({long:.0f}) are "
            f"ahead of current momentum ({near:.0f})."
        )
    else:
        summary = (
            f"Reads the same on both horizons ({near:.0f} near, {long:.0f} long)."
        )
    out["summary"] = summary + catalyst
    return out


def rank(rows: list[dict[str, Any]], horizon: str = "near_term") -> list[dict[str, Any]]:
    """Sort analysed rows by the chosen horizon score, best first.

    Rows missing that score sort last rather than being dropped, so a ticker
    that failed to fetch stays visible instead of vanishing from the list.
    """
    key = "near_term_score" if horizon == "near_term" else "long_term_score"
    return sorted(
        rows,
        key=lambda r: (r.get(key) is not None, r.get(key) or 0),
        reverse=True,
    )
