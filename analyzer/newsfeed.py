"""News, sorted by who spoke and whether the price actually moved.

The premise: most headlines are noise, and the ones that are not usually share
a shape - **a specific person with the power to move money said something.**
A president naming a company, a chief executive naming someone else's company,
a central banker describing the path of rates. Those move prices in a way that
a routine product announcement does not.

So this module does two things a plain feed does not:

1. **Attributes the headline.** It looks for named decision makers and sorts
   them by the reach they actually have - policy makers who move whole markets,
   then executives who move sectors, then analysts who move a single name.
2. **Attaches the realised move.** For every headline it measures what the
   stock did from the session before publication to now. This is the only
   honest way to separate talk that mattered from talk that did not, and it is
   computed from prices rather than guessed from wording.

Everything is local: headlines come from the same Yahoo feed as the price data,
and the classification is plain pattern matching on this machine. No sentiment
API, no third party sees which companies you follow.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Decision makers, grouped by how far their words carry. The tiers drive both
# ordering and the "reach" label shown against each headline.
DECISION_MAKERS: dict[str, dict[str, Any]] = {
    "policy": {
        "label": "Policy makers",
        "reach": "Moves the whole market",
        "blurb": (
            "Rates, tariffs and regulation reprice everything at once. When "
            "these names speak, single-stock analysis is temporarily beside "
            "the point."
        ),
        "names": (
            "trump", "powell", "jerome powell", "federal reserve", "the fed",
            "fomc", "treasury", "yellen", "bessent", "white house", "congress",
            "senate", "sec chair", "european central bank", "ecb", "lagarde",
            "bank of canada", "tariff", "executive order",
        ),
    },
    "executive": {
        "label": "Executives",
        "reach": "Moves a company or a sector",
        "blurb": (
            "A chief executive naming a supplier, a partner or a rival is the "
            "single most reliable source of a violent one-day move in a name "
            "that had no news of its own."
        ),
        "names": (
            "jensen huang", "elon musk", "tim cook", "satya nadella",
            "sundar pichai", "mark zuckerberg", "andy jassy", "lisa su",
            "sam altman", "warren buffett", "larry ellison", "michael dell",
            "pat gelsinger", "c.c. wei", "hock tan", "ceo", "chief executive",
            "founder", "chairman",
        ),
    },
    "analyst": {
        "label": "Analysts and institutions",
        "reach": "Moves the name for a session or two",
        "blurb": (
            "Upgrades, downgrades and price targets. Real but short-lived, "
            "and largely priced within a day or two of the note."
        ),
        "names": (
            "upgrade", "downgrade", "price target", "initiated coverage",
            "overweight", "underweight", "outperform", "underperform",
            "goldman", "morgan stanley", "jpmorgan", "j.p. morgan",
            "bank of america", "wedbush", "piper", "barclays", "citi",
            "raymond james", "bernstein", "melius", "loop capital",
            "reiterates", "buy rating", "sell rating",
        ),
    },
}

# Event types, for headlines with no named speaker.
EVENT_TYPES: dict[str, dict[str, Any]] = {
    "earnings": {
        "label": "Earnings and guidance",
        "blurb": "The scheduled repricing. Guidance usually matters more than the quarter.",
        "terms": ("earnings", "quarterly results", "q1", "q2", "q3", "q4",
                  "guidance", "outlook", "beats", "misses", "revenue", "eps",
                  "forecast", "profit"),
    },
    "corporate": {
        "label": "Deals and corporate action",
        "blurb": "Acquisitions, splits, buybacks and partnerships - structural, not sentiment.",
        "terms": ("acquisition", "acquires", "merger", "buyback", "stock split",
                  "dividend", "partnership", "deal", "contract", "stake",
                  "spin-off", "ipo", "investment in"),
    },
    "product": {
        "label": "Products and operations",
        "blurb": "Launches, capacity, supply. Slow-burning unless it changes the numbers.",
        "terms": ("launch", "unveils", "chip", "product", "factory", "plant",
                  "production", "supply", "capacity", "recall", "data center",
                  "model", "release"),
    },
    "legal": {
        "label": "Legal and regulatory",
        "blurb": "Investigations, lawsuits, approvals. Usually an overhang rather than a shock.",
        "terms": ("lawsuit", "investigation", "probe", "antitrust", "fine",
                  "settlement", "court", "regulator", "approval", "ban",
                  "export control", "sanction"),
    },
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.tz_convert("UTC").to_pydatetime()
    except Exception:
        return None


def _find(text: str, terms) -> str | None:
    """First term present as a whole word or phrase."""
    lowered = text.lower()
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
            return term
    return None


def classify(headline: str) -> dict[str, Any]:
    """Attribute a headline to a speaker tier, or failing that an event type."""
    text = headline or ""
    for tier, spec in DECISION_MAKERS.items():
        hit = _find(text, spec["names"])
        if hit:
            return {
                "bucket": tier,
                "label": spec["label"],
                "reach": spec["reach"],
                "matched": hit,
                "is_decision_maker": True,
            }
    for kind, spec in EVENT_TYPES.items():
        hit = _find(text, spec["terms"])
        if hit:
            return {
                "bucket": kind,
                "label": spec["label"],
                "reach": "Company-specific",
                "matched": hit,
                "is_decision_maker": False,
            }
    return {
        "bucket": "other", "label": "Everything else",
        "reach": "Background", "matched": None, "is_decision_maker": False,
    }


def price_reaction(history: pd.DataFrame | None, published: datetime | None
                   ) -> dict[str, Any]:
    """What the stock has done since the session before a headline landed.

    Compared at date granularity, from the close *preceding* the publication
    date to the latest price available. Comparing against a timestamp instead
    silently blanks every fresh headline: a daily bar is stamped at midnight,
    so nothing sorts "after" a story published at 14:23 today - and the news
    worth reading is exactly the news from the last few hours.

    Attribution is loose by nature: other things happen the same day. But a
    headline followed by a 9% move is categorically different from one
    followed by 0.2%, and that is what makes the feed worth scanning.
    """
    blank = {"move_pct": None, "since": None, "sessions": None,
             "same_session": None}
    if history is None or history.empty or published is None or "Close" not in history:
        return blank

    close = history["Close"].astype(float).dropna()
    if close.empty:
        return blank

    index = pd.to_datetime(close.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    close.index = index.normalize()

    published_date = pd.Timestamp(published).tz_localize(None).normalize()
    before = close[close.index < published_date]
    if before.empty:
        return blank

    start, end = float(before.iloc[-1]), float(close.iloc[-1])
    if not start:
        return blank

    after = close[close.index >= published_date]
    return {
        "move_pct": (end / start - 1) * 100,
        "since": before.index[-1].date().isoformat(),
        "sessions": int(len(after)),
        # One session means the move is today's, still in progress.
        "same_session": len(after) <= 1,
    }


def build(items: list[dict[str, Any]], history: pd.DataFrame | None = None,
          symbol: str | None = None) -> dict[str, Any]:
    """Classify, time, and measure a list of raw headlines."""
    enriched: list[dict[str, Any]] = []
    for item in items or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        published = _parse_time(item.get("published"))
        summary = (item.get("summary") or "").strip()
        # Match against headline and summary together: the speaker is often
        # named only in the body.
        classification = classify(f"{title}. {summary}")
        reaction = price_reaction(history, published)

        age_hours = None
        if published:
            age_hours = (_now() - published).total_seconds() / 3600

        enriched.append({
            "title": title,
            "summary": summary,
            "publisher": item.get("publisher"),
            "link": item.get("link"),
            "symbol": symbol,
            "published": published.isoformat() if published else None,
            "age_hours": age_hours,
            **classification,
            **reaction,
        })

    # Newest first; undated last rather than dropped.
    enriched.sort(key=lambda i: (i["age_hours"] is None, i["age_hours"] or 0))

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in enriched:
        groups.setdefault(item["bucket"], []).append(item)

    movers = [
        i for i in enriched
        if i.get("move_pct") is not None and abs(i["move_pct"]) >= 3
    ]
    movers.sort(key=lambda i: -abs(i["move_pct"]))

    return {
        "items": enriched,
        "groups": groups,
        "movers": movers,
        "decision_maker_count": sum(1 for i in enriched if i["is_decision_maker"]),
        "total": len(enriched),
    }


def _normalise_title(title: str) -> str:
    """Key for detecting the same story republished against several tickers."""
    return re.sub(r"[^a-z0-9 ]+", "", (title or "").lower()).strip()[:90]


# How far a speaker's words carry, as a multiplier on a story's importance.
TIER_WEIGHT = {
    "policy": 3.0, "executive": 2.4, "analyst": 1.5,
    "earnings": 1.6, "corporate": 1.3, "legal": 1.3,
    "product": 1.0, "other": 0.7,
}


# Beyond this many sessions, a price move cannot honestly be attributed to a
# headline - it is just drift. A story from seven weeks ago sitting above a
# "+54% since" claim is not analysis, it is a coincidence with a number on it.
ATTRIBUTION_SESSIONS = 5


def is_attributable(item: dict[str, Any]) -> bool:
    """Whether the measured move is close enough in time to mean anything."""
    sessions = item.get("sessions")
    return (
        item.get("move_pct") is not None
        and sessions is not None
        and sessions <= ATTRIBUTION_SESSIONS
    )


def importance(item: dict[str, Any], owned: bool, weight_pct: float = 0.0) -> float:
    """Rank a story by how much it should change what you do today.

    Four things decide that, and recency is deliberately the weakest of them.
    A three-day-old headline that moved a position you hold 9% matters more
    than a fresh note on a stock you merely watch.
    """
    # Only a move that happened near the headline counts toward its importance.
    # Without this gate an old story about a stock that has since run 50%
    # outranks today's genuine news.
    move = abs(item.get("move_pct") or 0.0) if is_attributable(item) else 0.0
    # Moves saturate: past roughly 12% the story is already "very big", and
    # letting a 40% mover swamp everything hides the rest of the page.
    magnitude = min(move, 12.0) / 12.0

    tier = TIER_WEIGHT.get(item.get("bucket", "other"), 0.7)

    # Something you own outranks something you watch, and a large position
    # outranks a token one.
    ownership = 1.0
    if owned:
        ownership = 1.6 + min(weight_pct, 30.0) / 30.0

    age = item.get("age_hours")
    recency = 1.0 if age is None else max(0.35, 1.0 - (age / 96.0))

    return (0.35 + magnitude * 1.9) * tier * ownership * recency


def aggregate(
    per_symbol: dict[str, list[dict[str, Any]]],
    owned: set[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Merge every symbol's headlines into one ranked stream.

    The same wire story is frequently published against several tickers, so
    identical titles are collapsed into a single entry listing every symbol it
    touches - otherwise a market-wide piece would fill the page once per
    holding.
    """
    owned = owned or set()
    weights = weights or {}
    merged: dict[str, dict[str, Any]] = {}

    for symbol, items in per_symbol.items():
        for item in items:
            key = _normalise_title(item.get("title", ""))
            if not key:
                continue
            existing = merged.get(key)
            if existing:
                if symbol not in existing["symbols"]:
                    existing["symbols"].append(symbol)
                # Keep the largest move across the tickers it was filed under,
                # and prefer a held position's reaction over a watched one's.
                if abs(item.get("move_pct") or 0) > abs(existing.get("move_pct") or 0):
                    existing["move_pct"] = item.get("move_pct")
                    existing["sessions"] = item.get("sessions")
                    existing["same_session"] = item.get("same_session")
                    existing["symbol"] = symbol
                existing["owned"] = existing["owned"] or symbol in owned
                continue

            entry = dict(item)
            entry["symbols"] = [symbol]
            entry["owned"] = symbol in owned
            merged[key] = entry

    stream = list(merged.values())
    for entry in stream:
        best_weight = max((weights.get(s, 0.0) for s in entry["symbols"]), default=0.0)
        entry["weight_pct"] = best_weight
        entry["importance"] = importance(entry, entry["owned"], best_weight)

    stream.sort(key=lambda i: -i["importance"])

    held = [i for i in stream if i["owned"]]
    watched = [i for i in stream if not i["owned"]]
    macro = [i for i in stream if i["bucket"] == "policy"]
    # A "mover" must be both large and recent enough for the headline to be a
    # plausible cause.
    movers = [i for i in stream
              if is_attributable(i) and abs(i["move_pct"]) >= 4]
    movers.sort(key=lambda i: -abs(i["move_pct"]))

    return {
        "stream": stream,
        "held": held,
        "watched": watched,
        "macro": macro,
        "movers": movers,
        "total": len(stream),
        "symbols_covered": len(per_symbol),
        "decision_maker_count": sum(1 for i in stream if i["is_decision_maker"]),
    }


# Section order and copy for the view, so the page reads top-down by reach.
SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("policy", DECISION_MAKERS["policy"]["label"], DECISION_MAKERS["policy"]["blurb"]),
    ("executive", DECISION_MAKERS["executive"]["label"], DECISION_MAKERS["executive"]["blurb"]),
    ("analyst", DECISION_MAKERS["analyst"]["label"], DECISION_MAKERS["analyst"]["blurb"]),
    ("earnings", EVENT_TYPES["earnings"]["label"], EVENT_TYPES["earnings"]["blurb"]),
    ("corporate", EVENT_TYPES["corporate"]["label"], EVENT_TYPES["corporate"]["blurb"]),
    ("legal", EVENT_TYPES["legal"]["label"], EVENT_TYPES["legal"]["blurb"]),
    ("product", EVENT_TYPES["product"]["label"], EVENT_TYPES["product"]["blurb"]),
    ("other", "Everything else", "No named speaker and no obvious event type."),
)
