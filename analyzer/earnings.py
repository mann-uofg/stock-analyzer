"""The upcoming earnings calendar, and what history says about each name.

Earnings are the one scheduled event that reliably overrides everything else on
a chart. A stock can be in a clean uptrend with every indicator agreeing and
still fall 12% the morning after a report, so knowing what is coming - and how
violently this particular company tends to react - matters more than most of
what the other pages measure.

Three things are assembled here:

* **When.** The next scheduled report, and whether it lands before the open or
  after the close. That decides which session absorbs the move.
* **What is expected.** The consensus EPS, and the spread between the highest
  and lowest analyst, which says how much disagreement the single number hides.
* **How far it might move.** The options market's implied move, set against
  what this company has actually done on its last several reports. That
  comparison is the useful part: an implied move well above its own history
  says the market is bracing for something.

Yahoo's schedule data is imperfect and this module is deliberately explicit
about that rather than presenting a placeholder as a fact - see
``classify_session``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from . import datafeed

# When the report lands, relative to the trading session.
BEFORE, AFTER, UNKNOWN = "before", "after", "unknown"

SESSION_LABEL = {
    BEFORE: "Before the open",
    AFTER: "After the close",
    UNKNOWN: "Time unconfirmed",
}

SESSION_NOTE = {
    BEFORE: "The reaction lands in that day's regular session.",
    AFTER: "The reaction lands the following morning, usually with a gap.",
    UNKNOWN: "Yahoo has not published a confirmed time for this one.",
}

# Market hours in the exchange's own timezone.
_OPEN = dt.time(9, 30)
_CLOSE = dt.time(16, 0)


def _f(value: Any) -> float | None:
    """A finite float, or None. Yahoo returns NaN for anything it lacks."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def classify_session(when: Any) -> str:
    """Whether a report lands before the open or after the close.

    Yahoo timestamps future reports inconsistently: NVIDIA, which has always
    reported after the close, currently carries a 15:00 timestamp for its next
    one. A time inside market hours is therefore treated as a placeholder and
    reported as unconfirmed, rather than announcing a mid-session report that
    almost certainly is not happening.
    """
    if when is None:
        return UNKNOWN
    try:
        clock = when.time()
    except AttributeError:
        return UNKNOWN
    if clock < _OPEN:
        return BEFORE
    if clock >= _CLOSE:
        return AFTER
    return UNKNOWN


def _naive_date(stamp: Any) -> dt.date | None:
    try:
        return pd.Timestamp(stamp).date()
    except Exception:
        return None


def upcoming(symbol: str, earnings_df: pd.DataFrame | None = None,
             calendar: dict[str, Any] | None = None,
             today: dt.date | None = None) -> dict[str, Any] | None:
    """The next scheduled report for one symbol, or None if nothing is listed.

    ``today`` is injectable so the behaviour around the event date is testable
    without waiting for the calendar to roll over.
    """
    today = today or dt.date.today()

    if earnings_df is None:
        earnings_df = datafeed.earnings_history(symbol)
    if calendar is None:
        calendar = datafeed.earnings_calendar(symbol)

    when = None
    estimate = None

    if isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty:
        reported_col = next(
            (c for c in earnings_df.columns if "reported" in c.lower()), None
        )
        estimate_col = next(
            (c for c in earnings_df.columns if "estimate" in c.lower()), None
        )
        # Future rows are the ones with no reported figure yet. Sorting
        # ascending puts the nearest one first; the frame arrives newest-first.
        for stamp in sorted(earnings_df.index):
            day = _naive_date(stamp)
            if day is None or day < today:
                continue
            row = earnings_df.loc[stamp]
            # A duplicated index yields a frame rather than a row.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if reported_col and _f(row.get(reported_col)) is not None:
                continue  # already reported
            when = stamp
            if estimate_col:
                estimate = _f(row.get(estimate_col))
            break

    cal_date = None
    if isinstance(calendar, dict):
        dates = calendar.get("Earnings Date") or []
        if isinstance(dates, (list, tuple)) and dates:
            cal_date = dates[0] if isinstance(dates[0], dt.date) else None
        elif isinstance(dates, dt.date):
            cal_date = dates

    if when is None and cal_date is None:
        return None

    day = _naive_date(when) if when is not None else cal_date
    if day is None or day < today:
        return None

    high = _f((calendar or {}).get("Earnings High"))
    low = _f((calendar or {}).get("Earnings Low"))
    average = _f((calendar or {}).get("Earnings Average"))
    consensus = estimate if estimate is not None else average

    # How far apart the most and least optimistic analysts are, as a share of
    # the consensus. A wide spread means the single headline number is holding
    # together very different views.
    spread_pct = None
    if high is not None and low is not None and consensus:
        spread_pct = abs(high - low) / abs(consensus) * 100

    session = classify_session(when)
    return {
        "symbol": symbol,
        "date": day.isoformat(),
        "days_away": (day - today).days,
        "session": session,
        "session_label": SESSION_LABEL[session],
        "eps_estimate": consensus,
        "eps_high": high,
        "eps_low": low,
        "eps_spread_pct": spread_pct,
        "revenue_estimate": _f((calendar or {}).get("Revenue Average")),
        # Yahoo's two sources disagree by a day often enough to be worth
        # surfacing rather than silently preferring one.
        "date_disputed": bool(
            when is not None and cal_date is not None
            and _naive_date(when) != cal_date
        ),
    }


def past_events(earnings_df: pd.DataFrame | None,
                price_df: pd.DataFrame | None,
                limit: int = 8) -> list[dict[str, Any]]:
    """Reported quarters, each with its surprise and the move that followed.

    The move is the first session's close-to-close change after the report,
    which is the reaction a holder actually experiences whether the news landed
    before the open or after the close.
    """
    if not isinstance(earnings_df, pd.DataFrame) or earnings_df.empty:
        return []

    reported_col = next(
        (c for c in earnings_df.columns if "reported" in c.lower()), None
    )
    estimate_col = next(
        (c for c in earnings_df.columns if "estimate" in c.lower()), None
    )
    surprise_col = next(
        (c for c in earnings_df.columns if "surprise" in c.lower()), None
    )
    if reported_col is None:
        return []

    close = None
    if isinstance(price_df, pd.DataFrame) and not price_df.empty:
        series = price_df["Close"].astype(float).copy()
        index = pd.to_datetime(series.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        series.index = index.normalize()
        close = series

    events: list[dict[str, Any]] = []
    for stamp in sorted(earnings_df.index, reverse=True):
        row = earnings_df.loc[stamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        actual = _f(row.get(reported_col))
        if actual is None:
            continue  # not yet reported

        day = _naive_date(stamp)
        estimate = _f(row.get(estimate_col)) if estimate_col else None
        surprise = _f(row.get(surprise_col)) if surprise_col else None
        if surprise is None and estimate not in (None, 0) and estimate:
            surprise = (actual - estimate) / abs(estimate) * 100

        move = None
        if close is not None and day is not None:
            marker = pd.Timestamp(day).normalize()
            after = close[close.index > marker]
            before = close[close.index <= marker]
            if not after.empty and not before.empty:
                candidate = (after.iloc[0] / before.iloc[-1] - 1) * 100
                if np.isfinite(candidate):
                    move = float(candidate)

        events.append({
            "date": day.isoformat() if day else None,
            "eps_estimate": estimate,
            "reported_eps": actual,
            "surprise_pct": surprise,
            "move_pct": move,
            "beat": None if surprise is None else surprise > 0,
        })
        if len(events) >= limit:
            break
    return events


def history_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """What the past reports say in aggregate."""
    surprises = [e["surprise_pct"] for e in events if e.get("surprise_pct") is not None]
    moves = [e["move_pct"] for e in events if e.get("move_pct") is not None]

    stats: dict[str, Any] = {
        "quarters": len(events),
        "beat_rate_pct": None,
        "avg_surprise_pct": None,
        "typical_move_pct": None,
        "largest_move_pct": None,
        "up_rate_pct": None,
    }
    if surprises:
        stats["beat_rate_pct"] = sum(1 for s in surprises if s > 0) / len(surprises) * 100
        stats["avg_surprise_pct"] = float(np.mean(surprises))
    if moves:
        # Median, not mean: one 25% quarter should not set the expectation for
        # a stock that usually moves 4%.
        stats["typical_move_pct"] = float(np.median([abs(m) for m in moves]))
        stats["largest_move_pct"] = float(max(moves, key=abs))
        stats["up_rate_pct"] = sum(1 for m in moves if m > 0) / len(moves) * 100
    return stats


# A straddle prices every day between now and its expiry, so it only isolates
# an earnings move when it barely outlives the event. These two bounds decide
# whether the number may be called an earnings expectation at all.
MAX_EXPIRY_GAP_DAYS = 7    # expiry must land just after the report
MAX_EVENT_HORIZON_DAYS = 21  # and the report must be close enough to dominate


def implied_move(symbol: str, event_date: str, spot: float | None = None,
                 today: dt.date | None = None) -> dict[str, Any] | None:
    """What the options market is pricing, and whether it is about this event.

    Uses the at-the-money straddle on the first expiry after the report: buying
    both the call and the put costs roughly what the market thinks the move is
    worth, so that premium as a share of the price is the expected move. It is
    the number options desks quote, and it needs no volatility model to read.

    The catch, and the reason for ``basis``: a straddle prices volatility from
    today until expiry, not the report alone. With earnings ten weeks out and
    only a monthly expiry available, the straddle is ten weeks of ordinary
    movement with one event buried inside it - NVIDIA's reads 14% against a
    company that typically moves 3% on results. Quoting that as the earnings
    expectation would be wrong by a factor of four.

    So the reading is labelled ``event`` only when the expiry lands within a
    week of the report and the report is near enough to dominate the premium.
    Otherwise it is labelled ``period`` and describes exactly what it is: the
    range priced between now and that expiry.
    """
    today = today or dt.date.today()
    try:
        event = dt.date.fromisoformat(event_date)
    except (TypeError, ValueError):
        return None

    try:
        expiries = datafeed.option_expirations(symbol)
    except Exception:
        return None
    if not expiries:
        return None

    # The first expiry that actually covers the event. An expiry before it
    # prices a different week entirely.
    chosen = None
    for expiry in sorted(expiries):
        try:
            when = dt.date.fromisoformat(expiry)
        except ValueError:
            continue
        if when >= event:
            chosen = expiry
            break
    if chosen is None:
        return None

    if spot is None:
        quote = datafeed.fast_quote(symbol) or {}
        spot = _f(quote.get("last_price")) or _f(quote.get("regularMarketPrice"))
    if not spot:
        return None

    try:
        calls, puts = datafeed.option_chain(symbol, chosen)
    except Exception:
        return None
    if calls is None or puts is None or calls.empty or puts.empty:
        return None

    def _mid(frame: pd.DataFrame) -> tuple[float | None, float | None]:
        """The at-the-money contract's mid price, and its strike."""
        frame = frame.dropna(subset=["strike"]).copy()
        if frame.empty:
            return None, None
        frame["_gap"] = (frame["strike"].astype(float) - spot).abs()
        row = frame.sort_values("_gap").iloc[0]
        bid, ask = _f(row.get("bid")), _f(row.get("ask"))
        if bid and ask and ask >= bid:
            return (bid + ask) / 2, _f(row.get("strike"))
        # Fall back to the last trade when the book is empty, which is common
        # on thin names outside market hours.
        return _f(row.get("lastPrice")), _f(row.get("strike"))

    call_mid, strike = _mid(calls)
    put_mid, _ = _mid(puts)
    if call_mid is None or put_mid is None:
        return None

    straddle = call_mid + put_mid
    if straddle <= 0:
        return None

    expiry_date = dt.date.fromisoformat(chosen)
    gap_days = (expiry_date - event).days
    horizon_days = (event - today).days
    event_dominated = (
        gap_days <= MAX_EXPIRY_GAP_DAYS
        and horizon_days <= MAX_EVENT_HORIZON_DAYS
    )

    return {
        "expiry": chosen,
        "strike": strike,
        "straddle": straddle,
        "move_pct": straddle / spot * 100,
        "spot": spot,
        "days_to_expiry": (expiry_date - today).days,
        "gap_days": gap_days,
        "basis": "event" if event_dominated else "period",
    }


def brief(symbol: str, price_df: pd.DataFrame | None = None,
          with_options: bool = True) -> dict[str, Any]:
    """Everything the detail panel needs for one company.

    ``with_options`` is separable because the chain is by far the slowest call
    here; the calendar listing never needs it, and only the opened company does.
    """
    earnings_df = datafeed.earnings_history(symbol)
    calendar = datafeed.earnings_calendar(symbol)
    if price_df is None:
        try:
            price_df = datafeed.price_history(symbol, period="3y")
        except Exception:
            price_df = None

    event = upcoming(symbol, earnings_df, calendar)
    events = past_events(earnings_df, price_df)
    stats = history_stats(events)

    expected = None
    if with_options and event:
        try:
            expected = implied_move(symbol, event["date"])
        except Exception:
            expected = None

    # The comparison that makes both numbers mean something: options pricing a
    # move far above what this company usually delivers is the market bracing
    # for something specific.
    #
    # Only when the straddle is actually about this report. Setting a ten-week
    # range against a one-day earnings move would manufacture alarm out of
    # nothing but time value.
    ratio = None
    if (expected and expected.get("basis") == "event"
            and stats.get("typical_move_pct")):
        typical = stats["typical_move_pct"]
        if typical > 0:
            ratio = expected["move_pct"] / typical

    return {
        "symbol": symbol,
        "event": event,
        "history": events,
        "stats": stats,
        "implied": expected,
        "implied_vs_typical": ratio,
    }
