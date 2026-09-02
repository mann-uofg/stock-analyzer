"""Portfolio view: import holdings, then analyse them like a watchlist.

On why there is no "Connect Wealthsimple" button, see the module docstring in
``analyzer/portfolio.py``. In short: no public API exists, and both workarounds
(credential-scraping clients, or third-party aggregators) conflict with keeping
your positions on this machine.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from analyzer import (
    benchmark,
    charts,
    horizon,
    performance,
    portfolio as pf,
    review,
    riskmodel,
    store,
)

from .common import (
    REFRESH_SECONDS,
    allocation_bar,
    finding,
    holding_rows,
    html,
    pill,
    ring,
    ring_row,
    sparkline,
    spark_series,
    split_sentence,
    stat,
    stat_grid,
    coerce_numeric,
    dash_column,
    fmt,
    fx_histories,
    fx_rates,
    histories,
    period_histories,
    money,
    portfolio_risk,
    quote,
    screen,
    since,
)


def _import_panel(show_heading: bool = True) -> None:
    # The heading is suppressed when this panel is nested inside the edit
    # expander on a page that has already printed it.
    if show_heading:
        st.markdown("# Portfolio")
        st.markdown(
            "<div class='muted'>Wealthsimple publishes no public API. The only "
            "programmatic routes need your account password and 2FA, or route "
            "your holdings through a third-party aggregator — both at odds with "
            "keeping this local. Export your holdings to CSV instead, or type "
            "them in.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

    tab_file, tab_manual = st.tabs(["Import a file", "Enter manually"])

    with tab_file:
        st.markdown(
            "In Wealthsimple: **Activity → Download CSV**, or any holdings export. "
            "Columns are matched by meaning, so most brokers' formats work as-is."
        )
        upload = st.file_uploader("Holdings CSV", type=["csv", "tsv", "txt"],
                                  label_visibility="collapsed")
        if upload is not None:
            positions, notes = pf.parse_holdings(upload)
            for note in notes:
                st.caption(note)
            if positions:
                preview = pd.DataFrame([p.to_dict() for p in positions])
                st.dataframe(preview, width="stretch", hide_index=True)
                if st.button("Save these positions", type="primary"):
                    store.save_portfolio([p.to_dict() for p in positions])
                    # Leave edit mode, otherwise saving drops the user back
                    # into the importer with the file still attached.
                    st.session_state.pop("_pf_edit", None)
                    st.rerun()
            else:
                st.error("No positions found. Check the file has a symbol column, "
                         "or use the manual tab.")

    with tab_manual:
        existing = store.load_portfolio()
        template = pd.DataFrame(
            existing or [{"symbol": "", "quantity": None, "avg_cost": None, "account": ""}]
        )
        for column in ("symbol", "quantity", "avg_cost", "account"):
            if column not in template.columns:
                template[column] = None
        edited = st.data_editor(
            template[["symbol", "quantity", "avg_cost", "account"]],
            num_rows="dynamic", width="stretch", hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", required=True),
                "quantity": st.column_config.NumberColumn("Shares", format="%.4f"),
                "avg_cost": st.column_config.NumberColumn("Avg cost", format="%.2f"),
                "account": st.column_config.TextColumn("Account"),
            },
        )
        if st.button("Save positions", type="primary", key="save_manual"):
            rows = []
            for _, r in edited.iterrows():
                symbol = str(r.get("symbol") or "").strip().upper()
                quantity = r.get("quantity")
                # NaN is truthy, so `quantity or 0` would let a blank cell
                # through as NaN; test for a real number instead. A negative
                # quantity is a legitimate short.
                if not symbol or not pd.notna(quantity) or float(quantity) == 0:
                    continue
                rows.append({
                    "symbol": symbol,
                    "quantity": float(quantity),
                    "avg_cost": float(r["avg_cost"]) if pd.notna(r.get("avg_cost")) else None,
                    "account": (str(r.get("account")).strip() or None)
                    if pd.notna(r.get("account")) else None,
                })
            if not rows:
                st.error("Nothing to save — every row needs a symbol and a quantity.")
            else:
                store.save_portfolio(rows)
                st.session_state.pop("_pf_edit", None)
                st.rerun()


def _value_chart(holdings: list[dict], base_currency: str | None,
                 present: list[str]) -> None:
    """Basket value over a selectable window, optionally against a benchmark."""
    st.header("Performance")

    controls = st.columns([3, 1.4])
    with controls[0]:
        window = st.radio(
            "Window", list(performance.PERIODS), horizontal=True,
            label_visibility="collapsed", key="perf_window",
        )
    with controls[1]:
        compare_to = st.selectbox(
            "Compare", ["None", "SPY", "QQQ", "XEQT.TO"],
            label_visibility="collapsed", key="perf_compare",
        )

    period, interval = performance.PERIODS[window]
    symbols = tuple(h["symbol"] for h in holdings)

    with st.spinner(f"Building {window} history…"):
        price_history = period_histories(symbols, period, interval)
        currencies = tuple(sorted({c for c in present if c and c != base_currency}))
        rates = (
            fx_histories(currencies, base_currency, period, interval)
            if currencies and base_currency else {}
        )
        built = performance.build(holdings, price_history, rates, base_currency)

    series = built["series"]
    if series.empty or len(series) < 2:
        st.info(
            f"Not enough overlapping history for the {window} window. "
            "A recently listed holding limits how far back the whole basket "
            "can be valued."
        )
        return

    stats = performance.summarise(series)
    change_tone = "up" if (stats["change"] or 0) >= 0 else "down"
    html(stat_grid([
        stat(f"Value {base_currency or ''}".strip(), money(stats["end_value"])),
        stat(f"Change ({window})", money(stats["change"]),
             fmt(stats["change_pct"], 2, "%", signed=True), tone=change_tone),
        stat("Period high", money(stats["high"])),
        stat("Period low", money(stats["low"])),
    ]))

    comparisons: dict[str, Any] = {}
    if compare_to != "None":
        bench = period_histories((compare_to,), period, interval).get(compare_to)
        bench_series = performance.close_series(bench)
        if bench_series is not None:
            comparisons[compare_to] = bench_series.reindex(
                series.index, method="ffill"
            ).dropna()

    st.plotly_chart(
        charts.portfolio_value(series, comparisons, base_currency or "",
                               intraday=window in performance.INTRADAY),
        width="stretch", config={"displaylogo": False, "scrollZoom": False},
    )

    start = built["start"]
    caption = (
        f"Value of the holdings you own today, at today's share counts, from "
        f"{start:%d %b %Y} onward"
        + (" — converted at each date's own exchange rate." if rates else ".")
        + " Because the export carries no purchase dates, this shows how the "
        "basket behaved, not what your account was actually worth."
    )
    if built["missing"]:
        caption += (
            " Excluded for lack of history at this interval: "
            + ", ".join(built["missing"]) + "."
        )
    html(f"<div class='muted'>{caption}</div>")

    # A four-month "1Y" chart looks broken unless the cause is named. Skipped
    # for intraday, where a differing start merely reflects that crypto trades
    # overnight and equities do not.
    if built.get("limited_by") and window not in performance.INTRADAY:
        html(finding(
            "note",
            f"Window starts {start:%d %b %Y}, limited by {built['limited_by']}",
            f"{built['limited_by']} has no price history before then, and the "
            "basket can only be valued on dates every holding traded. Remove it "
            "to see the rest of the book over a longer window.",
        ))


@st.fragment(run_every=REFRESH_SECONDS)
def _analysis(positions: tuple[dict, ...]) -> None:
    """Value the book and analyse every holding. Re-runs hourly."""
    holdings = [dict(p) for p in positions]
    symbols = tuple(h["symbol"] for h in holdings)

    quotes = {symbol: quote(symbol) for symbol in symbols}

    # A book holding both USD and CAD lines cannot be totalled until it is in
    # one currency. The base is whichever currency holds the most positions.
    currencies = {h["symbol"]: (h.get("currency") or "").upper() for h in holdings}
    present = [c for c in currencies.values() if c]
    base_currency = max(set(present), key=present.count) if present else None

    rates: dict[str, float] = {}
    missing_fx: list[str] = []
    if base_currency and len(set(present)) > 1:
        resolved = fx_rates(tuple(sorted(set(present))), base_currency)
        for symbol, ccy in currencies.items():
            rate = resolved.get(ccy)
            if ccy and rate is None and ccy != base_currency:
                missing_fx.append(ccy)
            rates[symbol] = rate if rate is not None else 1.0

    rows = pf.value_positions(holdings, quotes, rates)
    extra = portfolio_risk(symbols)
    summary = pf.summarise(
        rows,
        betas={s: extra.get(s, {}).get("beta") for s in symbols},
        sectors={s: extra.get(s, {}).get("sector") for s in symbols},
    )

    suffix = f" {base_currency}" if base_currency else ""
    pnl = summary.get("unrealised_pnl")
    html(stat_grid([
        stat(f"Market value{suffix}", money(summary.get("total_value"))),
        stat(f"Cost basis{suffix}", money(summary.get("total_cost"))),
        stat(f"Unrealised P/L{suffix}", money(pnl),
             fmt(summary.get("unrealised_pnl_pct"), 2, "%", signed=True),
             tone="up" if (pnl or 0) >= 0 else "down"),
        stat("Weighted beta", fmt(summary.get("weighted_beta"), 2),
             sub="vs benchmark, 5y monthly"),
        stat("Effective positions", fmt(summary.get("effective_positions"), 1),
             sub=f"of {summary.get('positions')} held"),
    ]))

    if base_currency and len(set(present)) > 1:
        others = ", ".join(sorted(set(present) - {base_currency}))
        st.caption(
            f"Totals converted to {base_currency} at today's spot rate "
            f"({others} → {base_currency}). Per-position values below stay in "
            "their own currency."
        )
    if missing_fx:
        st.warning(
            "No exchange rate available for " + ", ".join(sorted(set(missing_fx)))
            + " — those positions are counted at 1:1, so the total is understated "
              "or overstated by the true rate."
        )


    analysed = {r["symbol"]: r for r in screen(symbols) if not r.get("error")}

    # --- Book-level read, above the detail ---------------------------------
    book = review.compute(rows, summary, analysed, base_currency)
    st.header("What this book is telling you")

    html(ring_row([
        ring(book.get("near_term_score"), "Near term", "days to weeks"),
        ring(book.get("long_term_score"), "Long term", "quarters to years"),
    ]))

    html("".join(
        finding(f["level"], f["headline"], f["detail"]) for f in book["findings"]
    ))

    _value_chart(holdings, base_currency, present)

    st.header("Positions")

    frame = pd.DataFrame([
        {
            "Symbol": r["symbol"],
            "Shares": r["quantity"],
            "Avg cost": r.get("avg_cost"),
            "Price": r.get("price"),
            "Value": r.get("market_value"),
            "Weight %": r.get("weight_pct"),
            "P/L": r.get("unrealised_pnl"),
            "P/L %": r.get("unrealised_pnl_pct"),
            "Verdict": (analysed.get(r["symbol"]) or {}).get("verdict"),
            "Near": (analysed.get(r["symbol"]) or {}).get("near_term_score"),
            "Long": (analysed.get(r["symbol"]) or {}).get("long_term_score"),
            "Earnings in": (analysed.get(r["symbol"]) or {}).get("days_to_earnings"),
        }
        for r in sorted(rows, key=lambda x: -(x.get("market_value") or 0))
    ])
    # Rich rows first: symbol, trend, price, weight and verdict at a glance.
    sparks = spark_series(symbols)
    ordered = sorted(rows, key=lambda x: -(x.get("market_value_base") or 0))
    rich = []
    for row in ordered:
        info = analysed.get(row["symbol"]) or {}
        pnl_pct = row.get("unrealised_pnl_pct")
        tone = "up" if (pnl_pct or 0) >= 0 else "down"
        rich.append({
            "weight_pct": row.get("weight_pct"),
            "cells": (
                f"<div><div class='row-sym'>{row['symbol']}</div>"
                f"<div class='row-name'>{(info.get('name') or '')[:22]}</div></div>"
                f"<div>{sparkline(sparks.get(row['symbol']))}</div>"
                f"<div><div class='row-num'>{fmt(row.get('price'))}</div>"
                f"<div class='row-sub'>{row.get('currency') or ''}</div></div>"
                f"<div><div class='row-num'>{fmt(row.get('weight_pct'), 1, '%')}</div>"
                f"<div class='row-sub'>{money(row.get('market_value'))}</div></div>"
                f"<div><div class='row-num {tone}'>{money(row.get('unrealised_pnl'))}</div>"
                f"<div class='row-sub {tone}'>{fmt(pnl_pct, 1, '%', signed=True)}</div></div>"
                f"<div style='text-align:right'>{pill(info.get('verdict'))}</div>"
            ),
        })
    html(holding_rows(rich, [
        ("Holding", "l"), ("30 days", "l"), ("Price", "r"),
        ("Weight", "r"), ("Unrealised", "r"), ("Call", "r"),
    ]))

    frame = coerce_numeric(frame, ["Shares", "Avg cost", "Price", "Value",
                                   "Weight %", "P/L", "P/L %", "Near", "Long",
                                   "Earnings in"])
    frame = dash_column(frame, "Earnings in", "{:.0f}", " d")

    with st.expander("Full detail — every column, sortable"):
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={
                "Shares": st.column_config.NumberColumn(format="%.4g"),
                "Avg cost": st.column_config.NumberColumn(format="%.2f"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Value": st.column_config.NumberColumn(format="%.2f"),
                "Weight %": st.column_config.NumberColumn(format="%.1f"),
                "P/L": st.column_config.NumberColumn(format="%+.2f"),
                "P/L %": st.column_config.NumberColumn(format="%+.1f"),
                "Near": st.column_config.ProgressColumn(
                    "Near", format="%.0f", min_value=0, max_value=100),
                "Long": st.column_config.ProgressColumn(
                    "Long", format="%.0f", min_value=0, max_value=100),
                "Earnings in": st.column_config.TextColumn("Earnings in"),
            },
        )

    # --- Benchmark, correlation and stress --------------------------------
    price_histories = histories(symbols)
    betas = {s: extra.get(s, {}).get("beta") for s in symbols}

    st.divider()
    bench_tab, risk_tab = st.tabs(["Versus the index", "Correlation & stress"])

    with bench_tab:
        bench_histories = histories(benchmark.DEFAULT_BENCHMARKS)
        comparison = benchmark.compare(
            rows, price_histories, bench_histories,
            actual_pnl_pct=summary.get("unrealised_pnl_pct"),
        )
        html("".join(
            finding("note", *split_sentence(line))
            for line in comparison["commentary"]
        ))
        table = pd.DataFrame(comparison["table"]).rename(
            columns={"period": "Period", "portfolio_pct": "Your holdings",
                     "coverage_pct": "Coverage %"}
        )
        st.dataframe(
            coerce_numeric(table, [c for c in table.columns if c != "Period"]),
            width="stretch", hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%+.1f")
                for c in table.columns if c not in ("Period", "Coverage %")
            } | {"Coverage %": st.column_config.NumberColumn(format="%.0f")},
        )
        st.caption(comparison["note"])

    with risk_tab:
        corr = riskmodel.correlation_matrix(price_histories)
        stressed = riskmodel.stress(rows, betas)

        if stressed.get("scenarios"):
            cols = st.columns(len(stressed["scenarios"]))
            for col, scenario in zip(cols, stressed["scenarios"]):
                col.metric(
                    scenario["name"],
                    money(scenario["resulting_value"]),
                    fmt(scenario["portfolio_move_pct"], 1, "%", signed=True),
                    help=scenario["detail"],
                )
            st.caption(
                f"Portfolio beta {fmt(stressed.get('portfolio_beta'), 2)} over "
                f"{fmt(stressed.get('covered_pct'), 0, '%')} of the book. "
                + stressed["note"]
            )

        diversification = riskmodel.diversification_findings(rows, corr)
        if diversification:
            html("".join(
                finding(f["level"], f["headline"], f["detail"])
                for f in diversification
            ))

        heat = charts.correlation_heatmap(corr)
        if heat is not None:
            st.caption(
                "Daily-return correlation, past year. Red means they move "
                "together; blue means they move against each other."
            )
            st.plotly_chart(heat, width="stretch",
                            config={"displayModeBar": False})
        else:
            st.info("Not enough overlapping history to compute correlations.")

    st.divider()
    allocation = summary.get("sector_allocation_pct") or {}

    # The book as one bar: every holding, ordered by size.
    st.header("Allocation")
    html(allocation_bar([
        (r["symbol"], r.get("weight_pct") or 0) for r in ordered
        if (r.get("weight_pct") or 0) > 0
    ]))

    left, right = st.columns([2, 3])
    with left:
        if allocation:
            st.plotly_chart(
                charts.donut(allocation, f"{len(rows)} holdings"),
                width="stretch", config={"displayModeBar": False},
            )
        else:
            st.caption("No sector data available.")
    with right:
        st.subheader("What to act on")
        candidates = horizon.rank(
            [analysed[s] for s in symbols if s in analysed], "near_term"
        )
        weak = [c for c in candidates if (c.get("near_term_score") or 100) < 40]
        strong = [c for c in candidates if (c.get("near_term_score") or 0) >= 65]

        if strong:
            st.success("Strongest near-term: "
                       + ", ".join(f"{c['symbol']} ({c['near_term_score']:.0f})"
                                   for c in strong[:4]))
        if weak:
            st.error("Weakest near-term: "
                     + ", ".join(f"{c['symbol']} ({c['near_term_score']:.0f})"
                                 for c in weak[:4]))
        soon = [c for c in candidates
                if isinstance(c.get("days_to_earnings"), (int, float))
                and 0 <= c["days_to_earnings"] <= 14]
        if soon:
            st.info("Reporting within two weeks: "
                    + ", ".join(f"{c['symbol']} ({int(c['days_to_earnings'])}d)"
                                for c in soon))
        if not (strong or weak or soon):
            st.caption("Nothing in the book is at an extreme right now.")

    missing = [s for s in symbols if quotes.get(s) is None]
    if missing:
        st.caption("No price found for: " + ", ".join(missing)
                   + " — check the symbol matches the exchange listing "
                     "(Canadian listings usually need a .TO suffix).")


def render() -> None:
    positions = store.load_portfolio()

    if not positions:
        _import_panel()
        return

    # Holdings saved before crypto pairs were handled carry bare coin tickers,
    # which price against an unrelated security. Repair them once, in place.
    positions, repairs = pf.repair_symbols(positions)
    if repairs:
        store.save_portfolio(positions)
        st.info("Corrected crypto symbols: " + ", ".join(repairs)
                + " — a bare coin ticker resolves to a different security.")

    st.markdown("# Portfolio")
    head = st.columns([3, 1, 1])
    head[0].caption(
        f"{len(positions)} positions · imported "
        f"{since(store.last_updated(store.PORTFOLIO_FILE))} · refreshes hourly"
    )
    if head[1].button("Refresh now", width="stretch"):
        screen.clear()
        quote.clear()
        portfolio_risk.clear()
        st.rerun()
    if head[2].button("Replace holdings", width="stretch"):
        st.session_state["_pf_edit"] = True

    if st.session_state.get("_pf_edit"):
        with st.expander("Import or edit holdings", expanded=True):
            _import_panel(show_heading=False)
            if st.button("Close"):
                st.session_state.pop("_pf_edit", None)
                st.rerun()

    _analysis(tuple(positions))
