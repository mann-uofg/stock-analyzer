"""News view: one ranked stream across everything you own and watch.

The obvious design - a symbol picker and that symbol's headlines - is useless
at any real portfolio size. With a hundred names on a watchlist, checking each
one by hand is not a workflow, and the story that matters is never the one you
happened to click on.

So this page reads every symbol at once, merges the results, and ranks them by
how much they should change what you do today: how far the price actually
moved, who was speaking, whether you own the thing, and only then how recent it
is. A three-day-old headline that moved a position you hold 9% outranks a fresh
note on a name you merely follow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import streamlit as st

from analyzer import datafeed, newsfeed, store

from .common import REFRESH_SECONDS, finding, fmt, html, quote, stat, stat_grid

# The macro proxies. VIX is the market's own read on how frightened it is.
MARKET_SYMBOLS = ("^GSPC", "^IXIC", "^VIX")
MARKET_LABELS = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX"}


def _one_symbol(symbol: str) -> tuple[str, list[dict[str, Any]]]:
    """Headlines for a symbol, each carrying its realised price reaction."""
    try:
        history = datafeed.price_history(symbol, period="3mo")
    except Exception:
        history = None
    try:
        raw = datafeed.news(symbol)
    except Exception:
        return symbol, []
    return symbol, newsfeed.build(raw, history, symbol)["items"]


@st.cache_data(show_spinner=False, ttl=900)
def _all_news(symbols: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Every symbol's feed, fetched in parallel.

    Serially this is a minute-plus for a large watchlist, which is precisely
    the friction that made the old per-symbol page unusable.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if not symbols:
        return out
    workers = max(1, min(8, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_symbol, s) for s in symbols]
        for future in as_completed(futures):
            try:
                symbol, items = future.result()
            except Exception:
                continue
            if items:
                out[symbol] = items
    return out


@st.cache_data(show_spinner=False, ttl=900)
def _market_pulse() -> list[dict]:
    """Index levels and the volatility gauge, as context for the headlines."""
    out = []
    for symbol in MARKET_SYMBOLS:
        try:
            history = datafeed.price_history(symbol, period="1mo")
            close = history["Close"].astype(float).dropna()
            if len(close) < 2:
                continue
            last, prior = float(close.iloc[-1]), float(close.iloc[-2])
            out.append({
                "symbol": symbol, "label": MARKET_LABELS.get(symbol, symbol),
                "level": last, "change_pct": (last / prior - 1) * 100,
                "month_pct": (last / float(close.iloc[0]) - 1) * 100,
            })
        except Exception:
            continue
    return out


def _portfolio_weights() -> tuple[set[str], dict[str, float]]:
    """Which symbols are held, and how large each position is."""
    positions = store.load_portfolio()
    values: dict[str, float] = {}
    for position in positions:
        price = quote(position["symbol"])
        if price and position.get("quantity"):
            values[position["symbol"]] = abs(price * position["quantity"])
    total = sum(values.values())
    weights = ({s: v / total * 100 for s, v in values.items()} if total else {})
    return set(p["symbol"] for p in positions), weights


def _headline(item: dict) -> str:
    """One story: which tickers, what moved, who said it, how long ago."""
    move = item.get("move_pct")
    if move is None:
        move_html = "<span class='row-sub'>no price reference</span>"
    elif newsfeed.is_attributable(item):
        tone = "up" if move >= 0 else "down"
        window = "today" if item.get("same_session") else f"{item.get('sessions')}d"
        move_html = (
            f"<span class='{tone}' style='font-weight:700'>{move:+.2f}%</span>"
            f"<span class='row-sub'> since · {window}</span>"
        )
    else:
        # Too far back to call a reaction; shown as drift, in neutral ink, so
        # it is not read as this headline's doing.
        move_html = (
            f"<span class='row-sub'>{move:+.1f}% over the "
            f"{item.get('sessions')} sessions since — drift, not a reaction</span>"
        )

    age = item.get("age_hours")
    when = (
        "just now" if age is not None and age < 1
        else f"{age:.0f}h ago" if age is not None and age < 48
        else f"{age / 24:.0f}d ago" if age is not None else ""
    )

    tickers = "".join(
        f"<span class='pill {'buy' if item.get('owned') else 'hold'}' "
        f"style='margin-right:.3rem'>{s}</span>"
        for s in item["symbols"][:4]
    )
    extra = (f"<span class='row-sub'>+{len(item['symbols']) - 4} more</span>"
             if len(item["symbols"]) > 4 else "")

    who = f" · <b>{item['matched']}</b>" if item.get("matched") else ""
    link = item.get("link")
    title = (
        f"<a href='{link}' target='_blank' style='color:inherit;"
        f"text-decoration:none'>{item['title']}</a>" if link else item["title"]
    )
    summary = (
        f"<div class='finding-detail'>{item['summary'][:170]}…</div>"
        if item.get("summary") else ""
    )
    level = ("critical" if item.get("bucket") == "policy"
             else "warning" if abs(move or 0) >= 4 else "note")

    return (
        f"<div class='finding {level}'><div class='finding-body'>"
        f"<div style='margin-bottom:.35rem'>{tickers}{extra}</div>"
        f"<div class='finding-head'>{title}</div>{summary}"
        f"<div class='row-sub' style='text-align:left;margin-top:.4rem'>"
        f"{move_html} · {item.get('publisher') or 'unknown'} · {when}{who}</div>"
        f"</div></div>"
    )


@st.fragment(run_every=REFRESH_SECONDS)
def _stream(symbols: tuple[str, ...], owned: set[str],
            weights: dict[str, float], limit: int) -> None:
    with st.spinner(f"Reading {len(symbols)} feeds…"):
        per_symbol = _all_news(symbols)

    if not per_symbol:
        html("<div class='muted'>No headlines came back for anything you follow. "
             "Yahoo's feed is intermittent for smaller listings.</div>")
        return

    feed = newsfeed.aggregate(per_symbol, owned, weights)

    html(stat_grid([
        stat("Stories", feed["total"], sub=f"across {feed['symbols_covered']} symbols"),
        stat("On your holdings", len(feed["held"]),
             tone="down" if feed["held"] else ""),
        stat("From decision makers", feed["decision_maker_count"],
             sub="policy, executives, analysts"),
        stat("Moved 4%+", len(feed["movers"]),
             tone="down" if feed["movers"] else ""),
    ]))

    if feed["macro"]:
        st.header("Market-wide — affects everything you hold")
        html("".join(_headline(i) for i in feed["macro"][:4]))

    if feed["movers"]:
        st.header("Talk that moved money")
        st.caption(
            "Ranked by the size of the move since the session before "
            "publication. Coincidence and cause are not separable here, but "
            "this is where to look first."
        )
        html("".join(_headline(i) for i in feed["movers"][:8]))

    if feed["held"]:
        st.header("Your holdings")
        st.caption("Ranked by position size, speaker reach and price reaction.")
        html("".join(_headline(i) for i in feed["held"][:limit]))

    if feed["watched"]:
        st.header("Watchlist")
        html("".join(_headline(i) for i in feed["watched"][:limit]))

    with st.expander("How this is ranked"):
        st.markdown(
            "Every symbol you own or watch is read at once and the results are "
            "merged — the same wire story filed against six tickers appears "
            "once, tagged with all six.\n\n"
            "Ranking is by **how much a story should change what you do**, "
            "which is deliberately not recency:\n\n"
            "- **How far the price actually moved** since the session before "
            "publication, saturating around 12% so one huge mover cannot bury "
            "the rest of the page.\n"
            "- **Who was speaking** — a policy maker reprices everything, an "
            "executive moves a sector, an analyst moves a name for a session.\n"
            "- **Whether you own it**, weighted by position size.\n"
            "- **Recency**, last and weakest. A three-day-old headline that "
            "moved a real position outranks a fresh note on a name you only "
            "follow."
        )


def render() -> None:
    st.markdown("# News")

    watchlist = store.watchlist_symbols()
    owned, weights = _portfolio_weights()
    universe = sorted(set(watchlist) | owned)

    with st.sidebar:
        st.subheader("Feed")
        scope = st.radio(
            "Scope", ["Everything", "Holdings only", "Watchlist only"],
            label_visibility="collapsed",
        )
        limit = st.slider("Stories per section", 5, 40, 12)
        if st.button("Refresh feed", width="stretch"):
            _all_news.clear()
            _market_pulse.clear()
            st.rerun()

    # --- Market weather ---------------------------------------------------
    pulse = _market_pulse()
    if pulse:
        cards = []
        for entry in pulse:
            tone = "up" if entry["change_pct"] >= 0 else "down"
            if entry["symbol"] == "^VIX":
                # VIX is inverted: rising fear is a falling market, so the
                # colour is flipped to keep green meaning "good for you".
                tone = "down" if entry["change_pct"] >= 0 else "up"
            cards.append(stat(
                entry["label"], fmt(entry["level"], 2),
                fmt(entry["change_pct"], 2, "%", signed=True), tone=tone,
                sub=fmt(entry["month_pct"], 1, "% over a month", signed=True),
            ))
        html(stat_grid(cards))

        vix = next((e for e in pulse if e["symbol"] == "^VIX"), None)
        if vix:
            level = vix["level"]
            if level >= 28:
                html(finding(
                    "critical", f"VIX at {level:.1f} — the market is frightened",
                    "Above about 28, correlations converge and good names fall "
                    "with bad ones. Position size matters more than stock "
                    "picking until this subsides.",
                ))
            elif level >= 20:
                html(finding(
                    "warning", f"VIX at {level:.1f} — elevated nerves",
                    "Options are expensive and moves are wider than usual. This "
                    "is the range where a policy headline does real damage.",
                ))
            else:
                html(finding(
                    "good", f"VIX at {level:.1f} — calm",
                    "Volatility is subdued, so a surprise headline has more room "
                    "to move things than the recent range suggests.",
                ))

    if not universe:
        html("<div class='muted'>Add holdings or watchlist symbols and this "
             "page fills itself — there is nothing to pick from a dropdown.</div>")
        return

    if scope == "Holdings only":
        symbols = tuple(sorted(owned))
    elif scope == "Watchlist only":
        symbols = tuple(sorted(set(watchlist)))
    else:
        symbols = tuple(universe)

    st.caption(
        f"{len(symbols)} symbols · one merged stream, ranked by impact · "
        "refreshes hourly"
    )
    _stream(symbols, owned, weights, limit)
