"""Position sizing from a fixed-fractional risk budget.

A trade plan that gives an entry and a stop is only half an answer. The other
half is *how much*, and it is the half that decides whether a run of bad calls
is an inconvenience or the end of the account.

The rule here is fixed-fractional: risk a constant small percentage of the
account on each idea, and let the distance to the stop determine the share
count. A wide stop therefore buys fewer shares, not more risk - which is the
opposite of what sizing by dollar amount does.
"""

from __future__ import annotations

import math
from typing import Any

# Above this, one position dominates the portfolio regardless of the stop.
DEFAULT_MAX_POSITION_PCT = 20.0

# Risking more than this per idea is how small accounts die.
HIGH_RISK_PCT = 3.0


def size_position(
    account_value: float,
    risk_pct: float,
    entry: float,
    stop: float,
    *,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    allow_fractional: bool = True,
) -> dict[str, Any]:
    """Shares to buy so that being stopped out costs ``risk_pct`` of the account.

    Returns the sizing plus whichever constraint bound it, so the number is
    explainable rather than arbitrary.
    """
    out: dict[str, Any] = {
        "valid": False, "shares": None, "position_value": None,
        "position_pct": None, "dollars_at_risk": None, "risk_per_share": None,
        "bound_by": None, "warnings": [],
    }

    if account_value is None or account_value <= 0:
        out["warnings"].append("Set your account size to size positions.")
        return out
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (entry, stop, risk_pct)):
        out["warnings"].append("Entry, stop and risk must be numbers.")
        return out
    if entry <= 0 or risk_pct <= 0:
        out["warnings"].append("Entry price and risk percentage must be positive.")
        return out

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        out["warnings"].append("Stop equals entry - there is no defined risk.")
        return out

    risk_budget = account_value * risk_pct / 100.0
    raw_shares = risk_budget / risk_per_share

    # The risk budget is the primary constraint; the position cap is a second,
    # independent one that stops a very tight stop from justifying a position
    # that dominates the book.
    bound_by = "risk budget"
    max_value = account_value * max_position_pct / 100.0
    if raw_shares * entry > max_value:
        raw_shares = max_value / entry
        bound_by = f"{max_position_pct:.0f}% position cap"

    shares = raw_shares if allow_fractional else math.floor(raw_shares)
    if shares <= 0:
        out["warnings"].append(
            f"One share costs {entry:,.2f} but the risk budget only allows "
            f"{risk_budget:,.2f}. Either this trade is too large for the "
            "account, or fractional shares are needed."
        )
        return out

    position_value = shares * entry
    dollars_at_risk = shares * risk_per_share

    if risk_pct > HIGH_RISK_PCT:
        out["warnings"].append(
            f"Risking {risk_pct:.1f}% per trade means roughly "
            f"{int(100 / risk_pct)} consecutive losses would halve the account. "
            "1-2% is the usual ceiling."
        )

    out.update({
        "valid": True,
        "shares": round(shares, 4) if allow_fractional else int(shares),
        "position_value": round(position_value, 2),
        "position_pct": round(position_value / account_value * 100, 2),
        "dollars_at_risk": round(dollars_at_risk, 2),
        "risk_per_share": round(risk_per_share, 4),
        "risk_budget": round(risk_budget, 2),
        "bound_by": bound_by,
    })
    return out


def sizing_for_setup(
    setup: dict[str, Any], account_value: float, risk_pct: float,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    allow_fractional: bool = True,
) -> dict[str, Any]:
    """Size a trade plan produced by ``scoring.build_trade_setup``.

    Entry uses the far edge of the range - the least favourable fill - so the
    share count is not flattered by assuming a perfect entry.
    """
    if not setup or not setup.get("valid"):
        return {"valid": False, "warnings": ["No trade setup to size."]}

    entry = (
        setup["entry_high"] if setup.get("direction") == "long" else setup["entry_low"]
    )
    result = size_position(
        account_value, risk_pct, entry, setup["stop_loss"],
        max_position_pct=max_position_pct, allow_fractional=allow_fractional,
    )
    result["entry_used"] = entry
    result["direction"] = setup.get("direction")

    if result.get("valid"):
        # What the plan is worth if it works, against what it costs if it does not.
        for label, key in (("target_1", "target_1"), ("target_2", "target_2")):
            target = setup.get(key)
            if target:
                gain = abs(target - entry) * result["shares"]
                result[f"profit_at_{label}"] = round(gain, 2)
    return result
