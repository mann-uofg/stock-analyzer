"""Book-level review: what the portfolio as a whole is saying.

The per-position table answers "how is each holding doing". This answers the
question that actually drives decisions: *given everything I own, what should I
be paying attention to?*

Deterministic and instant. The local model can write prose on top of these
findings, but the findings themselves are computed here so the page is useful
with no model running.
"""

from __future__ import annotations

from typing import Any

# A finding worth surfacing, ordered by how much it should change behaviour.
SEVERITY_ORDER = {"critical": 0, "warning": 1, "note": 2, "good": 3}


def _weighted_score(rows: list[dict[str, Any]], analysed: dict[str, dict],
                    key: str) -> float | None:
    """Value-weighted average of a horizon score across priced holdings."""
    total = 0.0
    weight = 0.0
    for row in rows:
        value = row.get("market_value_base")
        score = (analysed.get(row["symbol"]) or {}).get(key)
        if value is None or score is None:
            continue
        total += value * score
        weight += value
    return total / weight if weight else None


def compute(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    analysed: dict[str, dict[str, Any]],
    base_currency: str | None = None,
) -> dict[str, Any]:
    """Findings about the book as a whole."""
    findings: list[dict[str, str]] = []

    def add(level: str, headline: str, detail: str) -> None:
        findings.append({"level": level, "headline": headline, "detail": detail})

    near = _weighted_score(rows, analysed, "near_term_score")
    long = _weighted_score(rows, analysed, "long_term_score")

    # --- Concentration -----------------------------------------------------
    top_weight = summary.get("top_weight_pct")
    effective = summary.get("effective_positions")
    count = summary.get("positions") or 0

    if top_weight and top_weight >= 30:
        add(
            "critical" if top_weight >= 40 else "warning",
            f"One position is {top_weight:.0f}% of the book",
            "A single holding this size drives your return more than every "
            "other decision combined. That is a bet on one company, not a "
            "portfolio.",
        )
    if effective and count and effective < count * 0.6:
        add(
            "warning",
            f"Effectively {effective:.1f} positions, not {count}",
            "Weights are lopsided enough that the smaller holdings barely "
            "affect the outcome. Adding more names does not diversify if they "
            "stay tiny.",
        )

    allocation = summary.get("sector_allocation_pct") or {}
    if allocation:
        sector, weight = max(allocation.items(), key=lambda kv: kv[1])
        if weight >= 50 and sector != "Unknown":
            add(
                "warning",
                f"{sector} is {weight:.0f}% of the book",
                "Positions in one sector move together, especially when it "
                "sells off. Concentration here is the risk that shows up all "
                "at once rather than gradually.",
            )

    # --- Currency ----------------------------------------------------------
    currencies: dict[str, float] = {}
    for row in rows:
        value = row.get("market_value_base")
        if value is None:
            continue
        ccy = (row.get("currency") or "?").upper()
        currencies[ccy] = currencies.get(ccy, 0.0) + value
    total_value = sum(currencies.values())
    if total_value and len(currencies) > 1:
        foreign = {
            c: v / total_value * 100 for c, v in currencies.items() if c != base_currency
        }
        if foreign:
            biggest, share = max(foreign.items(), key=lambda kv: kv[1])
            if share >= 25:
                add(
                    "note",
                    f"{share:.0f}% of the book is in {biggest}",
                    f"Your return depends on the {biggest}/{base_currency} rate "
                    "as well as on the holdings. A favourable move in one can "
                    "mask a loss in the other.",
                )

    # --- Momentum of the book ---------------------------------------------
    if near is not None and long is not None:
        if near < 40 and long >= 55:
            add(
                "note",
                "Weak near term, sound long term",
                f"Value-weighted near-term score is {near:.0f} against "
                f"{long:.0f} long term. The holdings look better as positions "
                "to sit on than to add to right now.",
            )
        elif near >= 60 and long < 45:
            add(
                "warning",
                "Running on momentum, not fundamentals",
                f"Near-term {near:.0f} versus long-term {long:.0f}. This book "
                "depends on the trend continuing rather than on the businesses "
                "compounding.",
            )
        elif near >= 60 and long >= 60:
            add("good", "Both horizons agree",
                f"Near-term {near:.0f} and long-term {long:.0f}.")

    # --- Individual names --------------------------------------------------
    scored = [
        (row, analysed.get(row["symbol"]) or {})
        for row in rows if analysed.get(row["symbol"])
    ]
    weak = [
        (r, a) for r, a in scored
        if (a.get("near_term_score") or 100) < 35 and (r.get("weight_pct") or 0) >= 8
    ]
    if weak:
        names = ", ".join(f"{r['symbol']} ({a['near_term_score']:.0f})" for r, a in weak)
        add(
            "warning",
            "Meaningful positions scoring poorly",
            f"{names}. These are large enough to matter and are not working; "
            "worth a decision rather than drift.",
        )

    losers = [r for r in rows if (r.get("unrealised_pnl_pct") or 0) < -20]
    if losers and len(losers) >= max(2, len(rows) // 3):
        add(
            "note",
            f"{len(losers)} positions down more than 20%",
            "Check whether the thesis changed or only the price. Those are "
            "different situations and call for different actions.",
        )

    soon = [
        (r, a) for r, a in scored
        if isinstance(a.get("days_to_earnings"), (int, float))
        and 0 <= a["days_to_earnings"] <= 14
    ]
    if soon:
        names = ", ".join(
            f"{r['symbol']} in {int(a['days_to_earnings'])}d" for r, a in soon
        )
        add("note", "Earnings inside two weeks", f"{names}. Binary events on "
            "positions you already hold — size accordingly.")

    unpriced = [r["symbol"] for r in rows if r.get("price") is None]
    if unpriced:
        add(
            "critical",
            "Some holdings could not be priced",
            ", ".join(unpriced) + ". Totals below exclude them, so the book is "
            "worth more than shown.",
        )

    if not findings:
        add("good", "Nothing demanding attention",
            "No concentration, currency or momentum flags on this book.")

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f["level"], 9))

    return {
        "near_term_score": near,
        "long_term_score": long,
        "findings": findings,
        "headline": findings[0]["headline"] if findings else None,
    }
