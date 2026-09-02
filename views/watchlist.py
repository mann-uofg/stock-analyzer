"""Watchlist view: track symbols and rank them by holding horizon."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyzer import horizon, portfolio as pf, store

from .common import (
    escape,
    REFRESH_SECONDS,
    coerce_numeric,
    dash_column,
    finding,
    fmt,
    holding_rows,
    html,
    pill,
    screen,
    spark_series,
    sparkline,
)

HORIZON_LABELS = {
    "near_term": "Near term — days to weeks",
    "long_term": "Long term — quarters to years",
}


def _add_form() -> None:
    with st.sidebar:
        st.subheader("Add symbol")
        with st.form("add_watch", clear_on_submit=True):
            symbol = st.text_input("Symbol", placeholder="e.g. AMD").strip().upper()
            note = st.text_input("Note", placeholder="optional")
            if st.form_submit_button("Add", type="primary", width="stretch"):
                # Apply the same qualification the importer uses, so a bare
                # coin ticker does not quietly track an unrelated ETF.
                resolved = pf.resolve_symbol(symbol) if symbol else symbol
                changed, message = store.add_to_watchlist(resolved, note)
                if changed and resolved != symbol:
                    message += f" ({symbol} resolves to a different security)"
                if changed:
                    st.session_state["_watch_msg"] = message
                    st.rerun()
                else:
                    st.warning(message)

        entries = store.load_watchlist()
        if entries:
            st.subheader("Remove")
            target = st.selectbox("Symbol", [e["symbol"] for e in entries],
                                  label_visibility="collapsed")
            if st.button("Remove", width="stretch"):
                store.remove_from_watchlist(target)
                st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def _table(symbols: tuple[str, ...], horizon_key: str, notes: dict[str, str]) -> None:
    """Ranked watchlist. Re-runs itself hourly without a page reload."""
    # A large list takes a visible amount of time on its first pass, so say so
    # rather than showing an empty page that looks broken.
    with st.spinner(
        f"Analysing {len(symbols)} symbols… first run takes about "
        f"{max(5, len(symbols) // 2)} seconds, then it is cached."
    ):
        rows = screen(symbols)
    ok = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]

    if not ok:
        st.warning("None of the watchlist symbols could be analysed.")
        for row in failed:
            st.caption(f"{row['symbol']}: {row['error']}")
        return

    ranked = horizon.rank(ok, horizon_key)

    frame = pd.DataFrame([
        {
            "Symbol": r["symbol"],
            "Price": r.get("price"),
            "Chg %": r.get("change_pct"),
            "Verdict": r.get("verdict"),
            "Near": r.get("near_term_score"),
            "Long": r.get("long_term_score"),
            "Better as": r.get("bias"),
            "RSI": r.get("rsi"),
            "1M %": r.get("perf_1m"),
            "1Y %": r.get("perf_1y"),
            "Fwd P/E": r.get("forward_pe"),
            "Earnings in": r.get("days_to_earnings"),
            "Note": notes.get(r["symbol"], ""),
        }
        for r in ranked
    ])
    frame = coerce_numeric(frame, ["Price", "Chg %", "Near", "Long", "RSI",
                                   "1M %", "1Y %", "Fwd P/E", "Earnings in"])
    # Crypto has no P/E and no earnings date; render those as a dash rather
    # than letting the grid print "None".
    frame = dash_column(frame, "Fwd P/E", "{:.1f}")
    frame = dash_column(frame, "Earnings in", "{:.0f}", " d")

    sparks = spark_series(tuple(r["symbol"] for r in ranked))
    rich = []
    for row in ranked:
        change = row.get("change_pct")
        tone = "up" if (change or 0) >= 0 else "down"
        score = row.get(
            "near_term_score" if horizon_key == "near_term" else "long_term_score"
        )
        rich.append({
            "weight_pct": score,
            "cells": (
                f"<div><div class='row-sym'>{escape(row['symbol'])}</div>"
                f"<div class='row-name'>{escape((row.get('name') or '')[:22])}</div></div>"
                f"<div>{sparkline(sparks.get(row['symbol']))}</div>"
                f"<div><div class='row-num'>{fmt(row.get('price'))}</div>"
                f"<div class='row-sub {tone}'>{fmt(change, 2, '%', signed=True)}</div></div>"
                f"<div><div class='row-num'>{fmt(row.get('near_term_score'), 0)}</div>"
                f"<div class='row-sub'>near</div></div>"
                f"<div><div class='row-num'>{fmt(row.get('long_term_score'), 0)}</div>"
                f"<div class='row-sub'>long</div></div>"
                f"<div style='text-align:right'>{pill(row.get('verdict'))}</div>"
            ),
        })
    html(holding_rows(rich, [
        ("Symbol", "l"), ("30 days", "l"), ("Price", "r"),
        ("Near", "r"), ("Long", "r"), ("Call", "r"),
    ]))

    with st.expander("Full detail — every column, sortable"):
        st.dataframe(
          frame, width="stretch", hide_index=True,
          column_config={
              "Price": st.column_config.NumberColumn(format="%.2f"),
              "Chg %": st.column_config.NumberColumn(format="%+.2f"),
              "Near": st.column_config.ProgressColumn(
                  "Near", format="%.0f", min_value=0, max_value=100,
                  help="Momentum, trend, volume and positioning — days to weeks."),
              "Long": st.column_config.ProgressColumn(
                  "Long", format="%.0f", min_value=0, max_value=100,
                  help="Fundamentals and primary trend — quarters to years."),
              "RSI": st.column_config.NumberColumn(format="%.0f"),
              "1M %": st.column_config.NumberColumn(format="%+.1f"),
              "1Y %": st.column_config.NumberColumn(format="%+.1f"),
              "Fwd P/E": st.column_config.TextColumn("Fwd P/E"),
              "Earnings in": st.column_config.TextColumn("Earnings in"),
          },
        )

    top = ranked[0]
    cards = [finding("good", f"Top of the list — {top['symbol']}",
                     top.get("horizon_summary") or "")]

    imminent = [r for r in ranked
                if isinstance(r.get("days_to_earnings"), (int, float))
                and 0 <= r["days_to_earnings"] <= 14]
    if imminent:
        cards.append(finding(
            "note", "Reporting within two weeks",
            ", ".join(f"{r['symbol']} in {int(r['days_to_earnings'])} days"
                      for r in imminent),
        ))
    html("".join(cards))

    if failed:
        with st.expander(f"{len(failed)} symbol(s) could not be analysed"):
            for row in failed:
                st.caption(f"{row['symbol']}: {row['error']}")

    with st.expander("How the two scores differ"):
        st.markdown(
            "**Near term** weights momentum, trend, volume and options positioning. "
            "Valuation carries no weight — it does not move a stock over a fortnight.\n\n"
            "**Long term** weights fundamentals and the primary trend. Today's RSI "
            "carries almost none — it is noise at that horizon.\n\n"
            "A name scoring high on one and low on the other is telling you *when* "
            "it is interesting, not just whether."
        )


def render() -> None:
    st.markdown("# Watchlist")
    _add_form()

    if message := st.session_state.pop("_watch_msg", None):
        st.success(message)

    entries = store.load_watchlist()
    if not entries:
        html(
            "<div class='muted'>Nothing tracked yet. Add a symbol from the sidebar, "
            "or from the Research page.</div>"
        )
        return

    symbols = tuple(e["symbol"] for e in entries)
    notes = {e["symbol"]: e.get("note", "") for e in entries}

    head = st.columns([2, 1])
    with head[0]:
        horizon_key = st.radio(
            "Rank by", list(HORIZON_LABELS), horizontal=True,
            format_func=lambda k: HORIZON_LABELS[k], label_visibility="collapsed",
        )
    with head[1]:
        if st.button("Refresh now", width="stretch"):
            screen.clear()
            st.rerun()

    st.caption(f"{len(symbols)} symbols · analysed on load, refreshes hourly")
    if len(symbols) > 60:
        html(finding(
            "note", f"{len(symbols)} symbols on the watchlist",
            "They are fetched in parallel and cached for an hour, so the first "
            "load of the day is the slow one. If the list gets unwieldy, "
            "removing names you no longer follow is the fastest way to speed "
            "this page up.",
        ))
    _table(symbols, horizon_key, notes)
