"""Earnings view: what is reporting, when, and what it usually does.

Earnings are the one scheduled event that overrides everything else on a chart.
A stock can look clean on every indicator and still fall 12% the morning after
a report, so this page answers the questions that actually precede a decision:
what is coming in the next few weeks, whether it lands before the open or after
the close, what is expected of it, and how violently this particular company
tends to react when it reports.

The dates run across the top; the companies reporting on the selected date sit
underneath; opening one pulls up its full record.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

import streamlit as st

from analyzer import earnings as earn, interpret, store

from .common import (
    REFRESH_SECONDS,
    earnings_brief,
    earnings_events,
    earnings_pending,
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

# Matches the watchlist and portfolio: one batch of new lookups per pass, so a
# large universe cannot exhaust the memory a free container has.
LOOKUP_BATCH = 40

# How far ahead the calendar looks. Beyond a quarter the dates are provisional
# and mostly noise.
HORIZON_DAYS = 95

SESSION_TONE = {earn.BEFORE: "warn", earn.AFTER: "buy", earn.UNKNOWN: "hold"}
SESSION_SHORT = {
    earn.BEFORE: "Before open", earn.AFTER: "After close", earn.UNKNOWN: "Time TBC",
}


def _universe() -> tuple[str, ...]:
    """Everything held or watched, which is what a calendar should cover."""
    owned = [p["symbol"] for p in store.load_portfolio() if p.get("symbol")]
    watched = store.watchlist_symbols()
    return tuple(sorted(set(owned) | set(watched)))


def _date_label(iso: str) -> str:
    day = dt.date.fromisoformat(iso)
    return f"{day:%a} {day.day} {day:%b}"


def _relative(days: int) -> str:
    if days <= 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days < 14:
        return f"in {days} days"
    if days < 60:
        return f"in {days // 7} weeks"
    return f"in {days // 30} months"


# --- The strip ------------------------------------------------------------


def _strip(by_date: dict[str, list[dict]]) -> str:
    """A visual summary of the calendar, above the selector."""
    cells = []
    today = dt.date.today()
    for iso in sorted(by_date):
        events = by_date[iso]
        day = dt.date.fromisoformat(iso)
        away = (day - today).days
        owned = sum(1 for e in events if e.get("owned"))
        cells.append(
            f"<div class='cal-day'>"
            f"<div class='cal-dow'>{escape(f'{day:%a}')}</div>"
            f"<div class='cal-num'>{day.day}</div>"
            f"<div class='cal-mon'>{escape(f'{day:%b}')}</div>"
            f"<div class='cal-count'>{len(events)}"
            f"{' · ' + str(owned) + ' held' if owned else ''}</div>"
            f"<div class='cal-away'>{escape(_relative(away))}</div>"
            f"</div>"
        )
    return f"<div class='cal-strip'>{''.join(cells)}</div>"


def _company_card(event: dict[str, Any]) -> str:
    """One company on the selected date."""
    session = event.get("session", earn.UNKNOWN)
    estimate = event.get("eps_estimate")
    spread = event.get("eps_spread_pct")
    held = ("<span class='pill buy' style='margin-left:.4rem'>Held</span>"
            if event.get("owned") else "")
    disputed = (
        "<div class='row-sub'>Yahoo lists two dates a day apart for this one</div>"
        if event.get("date_disputed") else ""
    )
    return (
        f"<div class='earn-card'>"
        f"<div class='earn-head'>"
        f"<span class='row-sym'>{escape(event['symbol'])}</span>{held}"
        f"<span class='pill {SESSION_TONE.get(session, 'hold')}'>"
        f"{escape(SESSION_SHORT.get(session, 'Time TBC'))}</span></div>"
        f"<div class='earn-meta'>"
        f"Expected EPS <b>{fmt(estimate, 2) if estimate is not None else '—'}</b>"
        f"{f' · analysts differ by {spread:.0f}%' if spread is not None else ''}"
        f"</div>{disputed}</div>"
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


def _detail(symbol: str) -> None:
    with st.spinner(f"Pulling {symbol}'s earnings record…"):
        brief = earnings_brief(symbol)

    event, stats = brief.get("event"), brief.get("stats") or {}
    st.markdown(f"## {symbol}")

    if event:
        price = quote(symbol)
        html(stat_grid([
            stat("Reports", _date_label(event["date"]),
                 sub=_relative(event["days_away"])),
            stat("Session", SESSION_SHORT.get(event["session"], "Time TBC"),
                 sub=earn.SESSION_NOTE.get(event["session"], "")),
            stat("Expected EPS", fmt(event.get("eps_estimate"), 2),
                 sub=(f"range {fmt(event.get('eps_low'), 2)}–"
                      f"{fmt(event.get('eps_high'), 2)}"
                      if event.get("eps_low") is not None else "")),
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


@st.fragment(run_every=REFRESH_SECONDS)
def _calendar(universe: tuple[str, ...]) -> None:
    outstanding = len(earnings_pending(universe))
    if outstanding:
        with st.spinner(
            f"Checking {min(outstanding, LOOKUP_BATCH)} of {len(universe)} "
            f"symbols for scheduled reports…"
        ):
            events = earnings_events(universe, limit=LOOKUP_BATCH)
    else:
        events = earnings_events(universe)

    remaining = earnings_pending(universe)
    if remaining:
        st.info(
            f"{len(remaining)} of {len(universe)} symbols still to check — "
            "done in batches so a large list cannot exhaust the server."
        )
        if st.button(f"Check the next {min(len(remaining), LOOKUP_BATCH)}",
                     type="primary"):
            st.rerun()

    owned = set(p["symbol"] for p in store.load_portfolio() if p.get("symbol"))
    horizon = dt.date.today() + dt.timedelta(days=HORIZON_DAYS)
    by_date: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if dt.date.fromisoformat(event["date"]) > horizon:
            continue
        event["owned"] = event["symbol"] in owned
        by_date[event["date"]].append(event)

    if not by_date:
        html("<div class='muted'>Nothing you follow has a scheduled report in "
             "the next three months. Funds, currencies and crypto never will — "
             "only individual companies report.</div>")
        return

    held_count = sum(1 for e in events if e["symbol"] in owned)
    soonest = min(by_date)
    days = (dt.date.fromisoformat(soonest) - dt.date.today()).days
    html(plain_summary(
        f"{len(events)} scheduled reports across {len(by_date)} dates"
        f"{f', {held_count} of them companies you hold' if held_count else ''}. "
        f"The next is {_relative(days)}."
    ))

    html(_strip(by_date))

    dates = sorted(by_date)
    chosen = st.segmented_control(
        "Date", dates, format_func=_date_label, default=dates[0],
        key="_earn_date",
    )
    if not chosen:
        chosen = dates[0]

    st.subheader(f"Reporting {_date_label(chosen)}")
    for event in sorted(by_date[chosen], key=lambda e: e["symbol"]):
        left, right = st.columns([4, 1])
        with left:
            html(_company_card(event))
        with right:
            if st.button("Details", key=f"open_{event['symbol']}",
                         width="stretch"):
                st.session_state["_earn_symbol"] = event["symbol"]

    if selected := st.session_state.get("_earn_symbol"):
        st.divider()
        _detail(selected)


def render() -> None:
    st.markdown("# Earnings")

    universe = _universe()
    if not universe:
        html(
            "<div class='muted'>Nothing to schedule yet. Import a portfolio or "
            "add watchlist symbols and their reporting dates appear here.</div>"
        )
        return

    st.caption(
        f"{len(universe)} symbols you hold or follow · refreshes hourly"
    )
    _calendar(universe)

    with st.expander("How to read this page"):
        st.markdown(
            "**Before the open** means the reaction lands in that day's "
            "session. **After the close** — which most large companies choose — "
            "means it lands the next morning, usually as a gap, so a stop set "
            "the evening before does not protect you at the price you set it.\n\n"
            "**Expected EPS** is the analyst consensus. Beating it is normal: "
            "most companies guide so that they can. What moves a stock is the "
            "*size* of the beat against what was already priced in, and the "
            "outlook management gives alongside it.\n\n"
            "**How much could it move** compares what options are pricing "
            "against what this company has actually done on its recent "
            "reports. That comparison only exists close to the date — an "
            "option expiring weeks after the report is mostly pricing ordinary "
            "time, not the event."
        )
