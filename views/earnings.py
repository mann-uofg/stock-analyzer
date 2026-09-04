"""Earnings view: who reports when, across the whole market.

Earnings are the one scheduled event that overrides everything else on a chart.
A stock can look clean on every indicator and still fall 12% the morning after
a report, so this page answers the questions that precede a decision: what is
coming, whether it lands before the open or after the close, what is expected
of it, and how violently that company tends to react when it reports.

The calendar covers every US listing rather than only the names already held or
watched, because a calendar you can only look up is no use for finding anything
new. Holdings are marked, not filtered to.

Dates run across the top and are clicked directly; the companies reporting on
the selected day sit underneath, largest first, and opening one pulls up its
full record.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import streamlit as st

from analyzer import earnings as earn, interpret, store

from .common import (
    earnings_brief,
    escape,
    finding,
    fmt,
    gauge_grid,
    html,
    money,
    plain_summary,
    quote,
    stat,
    stat_grid,
)

# How many days the strip shows at once. Ten weekdays is a fortnight of trading
# and still fits one row without the cards becoming unreadable.
STRIP_DAYS = 10

# A peak day runs past three hundred companies, most of them too small to be
# why anyone opened this page.
PAGE_SIZE = 25

SIZE_FILTERS: dict[str, float] = {
    "Any size": 0.0,
    "Over $2B": 2e9,
    "Over $10B": 1e10,
    "Over $100B": 1e11,
}

SESSION_TONE = {earn.BEFORE: "warn", earn.AFTER: "buy", earn.UNKNOWN: "hold"}
SESSION_SHORT = {
    earn.BEFORE: "Before open", earn.AFTER: "After close", earn.UNKNOWN: "Time TBC",
}


def _weekdays(start: dt.date, count: int) -> list[dt.date]:
    """``count`` weekdays from ``start``. Markets do not report at weekends."""
    days, cursor = [], start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += dt.timedelta(days=1)
    return days


def _relative(days: int) -> str:
    if days < 0:
        return f"{abs(days)}d ago"
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days < 14:
        return f"in {days} days"
    return f"in {days // 7} weeks"


def _date_label(day: dt.date) -> str:
    return f"{day:%a} {day.day} {day:%b}"


def _followed() -> set[str]:
    """Everything held or watched, marked rather than filtered to."""
    owned = {p["symbol"] for p in store.load_portfolio() if p.get("symbol")}
    return owned | set(store.watchlist_symbols())


# --- The strip ------------------------------------------------------------


def _day_label(day: dt.date) -> str:
    """The face of a date card, as a button label.

    Drawn as the button rather than as cards above a row of buttons: a
    Streamlit button's label is markdown, so the card can be the hit target
    instead of merely sitting near one. Two trailing spaces are a markdown
    line break.
    """
    return f"{day:%a}  \n**{day.day}**  \n{day:%b}"


def _company_row(row: dict[str, Any], followed: bool) -> str:
    cap = row.get("market_cap")
    estimate = row.get("eps_estimate")
    session = row.get("session", earn.UNKNOWN)
    held = ("<span class='pill buy' style='margin-left:.4rem'>Following</span>"
            if followed else "")
    return (
        f"<div class='earn-card'>"
        f"<div class='earn-head'>"
        f"<span class='row-sym'>{escape(row['symbol'])}</span>{held}"
        f"<span class='pill {SESSION_TONE.get(session, 'hold')}'>"
        f"{escape(SESSION_SHORT.get(session, 'Time TBC'))}</span></div>"
        f"<div class='row-name'>{escape((row.get('name') or '')[:44])}</div>"
        f"<div class='earn-meta'>"
        f"Expected EPS <b>{fmt(estimate, 2) if estimate is not None else '—'}</b>"
        f"{f' · {cap / 1e9:,.0f}B market cap' if cap else ''}"
        f"</div></div>"
    )


# --- The detail panel -----------------------------------------------------


def _implied_block(brief: dict[str, Any]) -> None:
    """What the options market is pricing, said precisely."""
    implied = brief.get("implied")
    typical = (brief.get("stats") or {}).get("typical_move_pct")

    if not implied:
        html(finding(
            "note", "No options-based expectation available",
            "Either this name has no listed options, or no expiry covers the "
            "report yet. Its own history below is the better guide."
        ))
        return

    if implied.get("basis") == "event":
        html(stat_grid([
            stat("Options expect", fmt(implied["move_pct"], 1, "%"),
                 sub=f"either way, by {implied['expiry']}"),
            stat("It usually moves", fmt(typical, 1, "%") if typical else "—",
                 sub="median of recent reports"),
            stat("Break-even", fmt(implied["straddle"], 2),
                 sub=f"straddle at the {fmt(implied['strike'], 0)} strike"),
        ]))
        return

    # The straddle spans far more than the report, so calling it an earnings
    # expectation would overstate it several times over - NVIDIA's reads 14%
    # against a company that typically moves 3% on results.
    html(finding(
        "note",
        f"Options price a {implied['move_pct']:.1f}% range — but over "
        f"{implied['days_to_expiry']} days, not this report",
        f"The nearest expiry covering the report is {implied['expiry']}, "
        f"{implied['gap_days']} days after it, so that premium is mostly "
        f"ordinary movement between now and then rather than the report "
        f"itself. A usable earnings expectation only exists once weekly "
        f"options list around the date — roughly three weeks out. Until then, "
        f"its own history below is the honest guide."
    ))


def _history_table(brief: dict[str, Any]) -> None:
    events = brief.get("history") or []
    if not events:
        st.info("No reported quarters on record for this symbol.")
        return

    rows = []
    for e in events:
        surprise, move = e.get("surprise_pct"), e.get("move_pct")
        s_tone = "up" if (surprise or 0) > 0 else "down" if surprise is not None else ""
        m_tone = "up" if (move or 0) > 0 else "down" if move is not None else ""
        rows.append(
            f"<tr><td>{escape(e['date'] or '—')}</td>"
            f"<td class='num'>{fmt(e.get('eps_estimate'), 2)}</td>"
            f"<td class='num'>{fmt(e.get('reported_eps'), 2)}</td>"
            f"<td class='num {s_tone}'>{fmt(surprise, 1, '%', signed=True)}</td>"
            f"<td class='num {m_tone}'>{fmt(move, 1, '%', signed=True)}</td></tr>"
        )
    html(
        "<table class='earn-table'><thead><tr>"
        "<th>Reported</th><th>Expected EPS</th><th>Actual</th>"
        "<th>Surprise</th><th>Next-day move</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    st.caption(
        "The move is the first full session after the report — the reaction a "
        "holder actually experiences. A beat with a fall means the beat was "
        "already priced in, which is the single most common surprise for "
        "someone new to holding through earnings."
    )


def _detail(symbol: str, listing: dict[str, Any] | None = None) -> None:
    with st.spinner(f"Pulling {symbol}'s earnings record…"):
        brief = earnings_brief(symbol)

    event, stats = brief.get("event"), brief.get("stats") or {}
    st.markdown(f"## {symbol}")
    if listing and listing.get("name"):
        st.caption(listing["name"])

    # Nasdaq states the session outright; Yahoo only implies it from a
    # timestamp it often gets wrong, so the listing wins where they disagree.
    session = (listing or {}).get("session") or (event or {}).get("session")
    price = quote(symbol)
    html(stat_grid([
        stat("Reports", _date_label(dt.date.fromisoformat((listing or event)["date"]))
             if (listing or event) else "—",
             sub=_relative((event or {}).get("days_away", 0)) if event else ""),
        stat("Session", SESSION_SHORT.get(session, "Time TBC"),
             sub=earn.SESSION_NOTE.get(session, "")),
        stat("Expected EPS",
             fmt((listing or {}).get("eps_estimate")
                 if (listing or {}).get("eps_estimate") is not None
                 else (event or {}).get("eps_estimate"), 2)),
        stat("Price now", money(price) if price else "—"),
    ]))

    readings = interpret.read_many([
        ("days_to_earnings", (event or {}).get("days_away")),
        ("earnings_move_pct", stats.get("typical_move_pct")),
        ("beat_rate_pct", stats.get("beat_rate_pct")),
        ("eps_spread_pct", (event or {}).get("eps_spread_pct")),
        ("implied_vs_typical", brief.get("implied_vs_typical")),
    ])
    if readings:
        html(plain_summary(interpret.summarise(readings, f"{symbol}'s report")))
        html(gauge_grid(readings))

    st.subheader("How much could it move")
    _implied_block(brief)

    st.subheader("What happened last time")
    _history_table(brief)

    if stats.get("up_rate_pct") is not None:
        html(stat_grid([
            stat("Quarters on record", stats.get("quarters")),
            stat("Rose afterwards", fmt(stats["up_rate_pct"], 0, "%"),
                 sub="of recent reports"),
            stat("Biggest reaction",
                 fmt(stats.get("largest_move_pct"), 1, "%", signed=True)),
            stat("Average surprise",
                 fmt(stats.get("avg_surprise_pct"), 1, "%", signed=True)),
        ]))


# --- Page -----------------------------------------------------------------


def render() -> None:
    st.markdown("# Earnings")

    offset = st.session_state.get("_earn_offset", 0)
    days = _weekdays(dt.date.today() + dt.timedelta(days=offset), STRIP_DAYS)
    chosen = st.session_state.get("_earn_date") or days[0].isoformat()
    # Paging past the selected day leaves it selected but off the strip; keep
    # the selection inside the window the reader can see.
    if chosen not in {d.isoformat() for d in days}:
        chosen = days[0].isoformat()

    with st.container(key="cal_strip"):
        cols = st.columns([0.6] + [1] * STRIP_DAYS + [0.6],
                          vertical_alignment="center")
        with cols[0]:
            if st.button("‹", key="earn_prev", width="stretch", help="Earlier"):
                st.session_state["_earn_offset"] = offset - STRIP_DAYS
                st.session_state.pop("_earn_date", None)
                st.rerun()
        for i, day in enumerate(days):
            iso = day.isoformat()
            with cols[i + 1]:
                if st.button(_day_label(day), key=f"earn_d_{iso}",
                             width="stretch",
                             type="primary" if iso == chosen else "secondary"):
                    st.session_state["_earn_date"] = iso
                    st.session_state.pop("_earn_symbol", None)
                    st.session_state.pop("_earn_limit", None)
                    st.rerun()
        with cols[-1]:
            if st.button("›", key="earn_next", width="stretch", help="Later"):
                st.session_state["_earn_offset"] = offset + STRIP_DAYS
                st.session_state.pop("_earn_date", None)
                st.rerun()
    st.caption(
        f"{_relative((days[0] - dt.date.today()).days).capitalize()} — "
        f"{_date_label(days[-1])}. Weekdays only."
    )

    controls = st.columns([2, 2, 3], vertical_alignment="bottom")
    with controls[0]:
        size = st.selectbox("Company size", list(SIZE_FILTERS), index=1)
    with controls[1]:
        mine_only = st.toggle("Only what I follow", value=False)

    with st.spinner(f"Loading {_date_label(dt.date.fromisoformat(chosen))}…"):
        listings = earn.market_day(chosen)

    if not listings:
        html("<div class='muted'>Nothing scheduled for this date. Weekends and "
             "market holidays are empty, and the calendar thins out between "
             "reporting seasons.</div>")
        return

    followed = _followed()
    floor = SIZE_FILTERS[size]
    shown = [
        r for r in listings
        if (r.get("market_cap") or 0) >= floor
        and (not mine_only or r["symbol"] in followed)
    ]
    mine_count = sum(1 for r in listings if r["symbol"] in followed)

    html(plain_summary(
        f"{len(listings)} companies report on "
        f"{_date_label(dt.date.fromisoformat(chosen))}"
        f"{f', {mine_count} of them names you follow' if mine_count else ''}. "
        f"{len(shown)} shown at this size filter, largest first."
    ))

    if not shown:
        st.info("Nothing at this size on this date. Try a smaller floor.")
        return

    limit = st.session_state.get("_earn_limit", PAGE_SIZE)
    for row in shown[:limit]:
        left, right = st.columns([4, 1])
        with left:
            html(_company_row(row, row["symbol"] in followed))
        with right:
            if st.button("Details", key=f"open_{row['symbol']}_{chosen}",
                         width="stretch"):
                st.session_state["_earn_symbol"] = row["symbol"]
                st.rerun()

    if len(shown) > limit:
        if st.button(f"Show {min(PAGE_SIZE, len(shown) - limit)} more "
                     f"of {len(shown)}", width="stretch"):
            st.session_state["_earn_limit"] = limit + PAGE_SIZE
            st.rerun()

    if selected := st.session_state.get("_earn_symbol"):
        st.divider()
        listing = next((r for r in listings if r["symbol"] == selected), None)
        _detail(selected, listing)

    with st.expander("How to read this page"):
        st.markdown(
            "**Before the open** means the reaction lands in that day's "
            "session. **After the close** — which most large companies choose "
            "— means it lands the next morning, usually as a gap, so a stop "
            "set the evening before does not protect you at the price you set "
            "it.\n\n"
            "**Expected EPS** is the analyst consensus. Beating it is normal: "
            "most companies guide so that they can. What moves a stock is the "
            "*size* of the beat against what was already priced in, and the "
            "outlook management gives alongside it.\n\n"
            "The list covers every US listing reporting that day, ordered by "
            "size, so it is a place to find names as well as to check your "
            "own. Anything you hold or watch is marked **Following**."
        )
