"""Shared rendering helpers and cached analysis wrappers.

Kept deliberately small: formatting, caching, and the handful of primitives the
three views share. Anything view-specific lives in the view itself.
"""

from __future__ import annotations

import re
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from analyzer import (  # noqa: E402
    datafeed,
    earnings,
    engine,
    horizon,
    llm,
    risk,
    store,
)
from analyzer.config import BENCHMARKS  # noqa: E402
from analyzer.datafeed import DataError  # noqa: E402

# Analyses refresh on the hour, matching the auto-refresh cadence.
REFRESH_SECONDS = 3600

VERDICT_COLOUR = {
    "STRONG BUY": "#1f9d55",
    "BUY": "#4a9d6a",
    "HOLD": "#8a8578",
    "SELL": "#c2603f",
    "STRONG SELL": "#a83232",
}


# --- Formatting -----------------------------------------------------------


def fmt(value: Any, digits: int = 2, suffix: str = "", signed: bool = False) -> str:
    """Format a number, or an em dash when genuinely unavailable."""
    if value is None or isinstance(value, bool):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    return f"{number:{'+' if signed else ''},.{digits}f}{suffix}"


def money(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"${fmt(value, digits)}"


def big(value: Any) -> str:
    """Compact display for large figures."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    for cutoff, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= cutoff:
            return f"{number / cutoff:,.2f}{unit}"
    return f"{number:,.0f}"


def md_safe(text: Any) -> str:
    """Escape characters Streamlit's markdown would otherwise interpret.

    Model prose is full of dollar amounts, and a pair of them on one line turns
    everything between into LaTeX math.
    """
    return str(text).replace("\\", "\\\\").replace("$", r"\$")


def coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Force columns to a numeric dtype.

    A column holding both floats and ``None`` lands as object dtype, and the
    grid then prints the literal string "None" in every gap. Coercing turns
    those into NaN, which renders as an empty cell.
    """
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def dash_column(frame: pd.DataFrame, column: str, spec: str = "{:.1f}",
                suffix: str = "") -> pd.DataFrame:
    """Pre-format a sparse numeric column as text, with an em dash for gaps.

    Streamlit's grid prints missing numbers as the literal string "None" in
    every numeric configuration, which reads as a bug rather than as absent
    data. Columns where absence is normal - a coin has no P/E, an ETF has no
    earnings date - are therefore rendered as text. Columns that are always
    populated stay numeric so they remain sortable.
    """
    if column not in frame.columns:
        return frame
    frame[column] = frame[column].map(
        lambda v: "—" if v is None or pd.isna(v) else spec.format(v) + suffix
    )
    return frame


def since(iso: str | None) -> str:
    """Human-readable age of an ISO timestamp."""
    if not iso:
        return "never"
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return "unknown"
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - stamp).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def _escape(text: Any) -> str:
    """HTML-escape a value before it is interpolated into markup.

    Quotes are escaped as well as angle brackets: these components are built as
    f-strings with single-quoted attributes, so an apostrophe in a company name
    would otherwise close the attribute early - which is both a rendering bug
    and the opening an injected handler needs.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# Public alias: views escape untrusted text with this.
escape = _escape


# Everything rendered through these components is third-party text: headlines
# and company names from Yahoo, and ticker symbols typed by whoever is using
# the app. None of it is trustworthy, and Streamlit's unsafe_allow_html means
# an unescaped angle bracket is executable markup.
def safe_href(url: Any) -> str | None:
    """Return a link only if it is a plain http(s) URL.

    Blocks ``javascript:``, ``data:`` and similar schemes, and rejects anything
    containing a quote or angle bracket that could terminate the attribute and
    inject a new one.
    """
    if not url:
        return None
    candidate = str(url).strip()
    if not candidate.lower().startswith(("http://", "https://")):
        return None
    if any(ch in candidate for ch in "\"'<>` \n\r\t"):
        return None
    return _escape(candidate)


def verdict_chip(verdict: str | None, subtitle: str = "") -> str:
    """The headline call, coloured by direction rather than by strength."""
    tone = "hold"
    if verdict and "BUY" in verdict:
        tone = "buy"
    elif verdict and "SELL" in verdict:
        tone = "sell"
    sub = f"<div class='verdict-meta'>{_escape(subtitle)}</div>" if subtitle else ""
    return (
        f"<div style='text-align:right'>"
        f"<span class='verdict {tone}'>{_escape(verdict or '—')}</span>{sub}</div>"
    )


def _sign_of(text: Any) -> str:
    """Direction implied by a formatted figure, wherever the sign sits.

    ``money()`` produces "$-1,094.64", so testing only the first character
    misses the minus entirely and a loss renders in neutral white. The sign is
    found after any currency symbol.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    stripped = text.strip().lstrip("$€£¥ ")
    if stripped.startswith(("-", "−")):
        return "down"
    if stripped.startswith("+"):
        return "up"
    return ""


def stat(label: str, value: Any, delta: Any = None, sub: str = "",
         tone: str = "", small: bool = False) -> str:
    """One figure with its label.

    ``tone`` colours the value as well as the delta - a negative P/L printed in
    neutral white while only its percentage is red reads as an oversight. When
    tone is not given it is inferred from whichever of the two carries a sign.
    """
    if tone == "":
        tone = _sign_of(delta) or _sign_of(value)

    delta_html = (
        f"<div class='stat-delta {tone}'>{_escape(delta)}</div>" if delta else ""
    )
    sub_html = f"<div class='stat-sub'>{_escape(sub)}</div>" if sub else ""
    size = " sm" if small else ""
    # Only colour the value when it actually carries a sign; a price or a share
    # count is not a gain or a loss.
    value_tone = f" {tone}" if tone and _sign_of(value) else ""
    return (
        f"<div class='stat'><div class='stat-label'>{_escape(label)}</div>"
        f"<div class='stat-value{size}{value_tone}'>{_escape(value)}</div>"
        f"{delta_html}{sub_html}</div>"
    )


def stat_grid(cards: list[str]) -> str:
    return f"<div class='stat-grid'>{''.join(cards)}</div>"


def meter(value: float | None, label: str, suffix: str = "/100") -> str:
    """A 0-100 bar that grows on mount, so magnitude registers before the digits."""
    if value is None:
        return (
            f"<div class='meter'><div class='meter-head'><span>{_escape(label)}</span>"
            f"<span class='meter-val'>—</span></div>"
            f"<div class='meter-track'></div></div>"
        )
    pct = max(0.0, min(100.0, float(value)))
    colour = "var(--gain)" if pct >= 65 else "var(--warn)" if pct >= 40 else "var(--loss)"
    return (
        f"<div class='meter'><div class='meter-head'><span>{_escape(label)}</span>"
        f"<span class='meter-val'>{pct:.0f}{suffix}</span></div>"
        f"<div class='meter-track'><div class='meter-fill' "
        f"style='width:{pct:.1f}%;background:{colour}'></div></div></div>"
    )


_ZONE_FILL = {
    "good": "var(--zone-good)", "warn": "var(--zone-warn)",
    "bad": "var(--zone-bad)", "neutral": "var(--zone-neutral)",
}


def gauge(reading: Any, show_plain: bool = True) -> str:
    """A metric shown with the scale it belongs on.

    The number alone is only useful to someone who already knows the bands, so
    the track is painted with them: green where the reading would be reassuring,
    amber and red where it would not, and a marker at the value. The intent is
    that the picture answers "is this good?" before the digits are even read.
    """
    if reading is None:
        return ""

    # Colour the whole scale first, then drop the marker on it.
    stops = []
    for start, end, tone in (reading.zones or []):
        fill = _ZONE_FILL.get(tone, _ZONE_FILL["neutral"])
        stops.append(f"{fill} {start * 100:.2f}%, {fill} {end * 100:.2f}%")
    track_bg = (f"background:linear-gradient(90deg,{','.join(stops)})"
                if stops else "")

    marker = ""
    if reading.position is not None:
        marker = (f"<div class='gauge-marker' "
                  f"style='left:{reading.position * 100:.2f}%'></div>")

    lo, hi = reading.scale
    ends = (
        f"<div class='gauge-ends'><span>{lo:g}</span><span>{hi:g}</span></div>"
        if reading.position is not None else ""
    )
    plain = (f"<div class='gauge-plain'>{_escape(reading.plain)}</div>"
             if show_plain and reading.plain else "")
    note = (f"<div class='gauge-note'>{_escape(reading.note)}</div>"
            if reading.note else "")

    return (
        f"<div class='gauge'>"
        f"<div class='gauge-head'>"
        f"<span class='gauge-label'>{_escape(reading.label)}</span>"
        f"<span class='gauge-value num'>{_escape(reading.display)}</span></div>"
        f"<div class='gauge-track' style='{track_bg}'>{marker}</div>"
        f"{ends}"
        f"<div class='gauge-verdict {_escape(reading.tone)}'>"
        f"{_escape(reading.verdict)}</div>"
        f"{plain}{note}</div>"
    )


def gauge_grid(readings: list[Any], show_plain: bool = True) -> str:
    cards = [gauge(r, show_plain) for r in readings if r is not None]
    return f"<div class='gauge-grid'>{''.join(cards)}</div>"


def plain_summary(text: str, tone: str = "neutral") -> str:
    """The one-line, jargon-free reading that heads a panel."""
    return (f"<div class='plain-summary {_escape(tone)}'>{_escape(text)}</div>")


def split_sentence(text: str) -> tuple[str, str]:
    """Split off the first sentence, ignoring decimal points.

    A naive split on "." cuts "+154.6%" into "+154." and "6%", so the boundary
    must be a period followed by whitespace and a capital letter.
    """
    match = re.search(r"(?<=[)\w%])\.\s+(?=[A-Z])", text)
    if not match:
        return text.strip(), ""
    return text[: match.start() + 1].strip(), text[match.end():].strip()


def finding(level: str, headline: str, detail: str = "") -> str:
    """A single observation, keyed by severity rather than by colour alone."""
    detail_html = (
        f"<div class='finding-detail'>{_escape(detail)}</div>" if detail else ""
    )
    return (
        f"<div class='finding {level}'><div class='finding-body'>"
        f"<div class='finding-head'>{_escape(headline)}</div>{detail_html}"
        f"</div></div>"
    )


def sparkline(values: Any, width: int = 118, height: int = 30,
              tone: str | None = None) -> str:
    """An inline SVG trend line for a sequence of prices.

    Drawn as raw SVG rather than a chart library: a hundred of these on one
    page must cost nothing, and they need to sit inside table cells and cards
    where a Plotly figure cannot go. Colour follows direction, and the line
    draws itself once via a stroke-dash animation.
    """
    series = [float(v) for v in (values or []) if v == v and v is not None]
    if len(series) < 2:
        return f"<svg class='spark' width='{width}' height='{height}'></svg>"

    low, high = min(series), max(series)
    span = (high - low) or 1.0
    step = width / (len(series) - 1)
    pad = 2.5
    usable = height - pad * 2

    points = [
        (i * step, pad + usable - ((v - low) / span) * usable)
        for i, v in enumerate(series)
    ]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{line} L{width:.1f},{height} L0,{height} Z"

    rising = series[-1] >= series[0]
    colour = tone or ("var(--gain)" if rising else "var(--loss)")
    fill = "rgba(69,193,122,.16)" if rising else "rgba(242,100,90,.16)"
    ident = f"sg{abs(hash((len(series), series[0], series[-1]))) % 100000}"

    return (
        f"<svg class='spark' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' preserveAspectRatio='none'>"
        f"<defs><linearGradient id='{ident}' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='{fill}'/>"
        f"<stop offset='100%' stop-color='transparent'/></linearGradient></defs>"
        f"<path class='area' d='{area}' fill='url(#{ident})'/>"
        f"<path class='line' d='{line}' stroke='{colour}'/></svg>"
    )


def ring(value: float | None, label: str, caption: str = "") -> str:
    """A circular gauge whose sweep animates from zero on mount."""
    if value is None:
        pct, text, colour = 0.0, "—", "var(--text-faint)"
    else:
        pct = max(0.0, min(100.0, float(value)))
        text = f"{pct:.0f}"
        colour = ("var(--gain)" if pct >= 65 else
                  "var(--warn)" if pct >= 40 else "var(--loss)")
    cap = f"<div class='ring-cap'>{_escape(caption)}</div>" if caption else ""
    return (
        f"<div class='ring-item'>"
        f"<div class='ring' style='--pct:{pct:.0f};--ring-col:{colour}'>"
        f"<span class='ring-num' style='color:{colour}'>{text}</span></div>"
        f"<div><div class='ring-label'>{_escape(label)}</div>{cap}</div></div>"
    )


def ring_row(rings: list[str]) -> str:
    return f"<div class='ring-row'>{''.join(rings)}</div>"


# Distinct hues for allocation segments, ordered so neighbours stay separable.
SEGMENT_COLOURS = (
    "#6aa5ff", "#45c17a", "#e0a83c", "#c77dff", "#4dd0c1",
    "#f2645a", "#8b95a8", "#f0a5c0", "#9ad36b", "#ffb27a",
)


def allocation_bar(items: list[tuple[str, float]], min_label_pct: float = 7.0) -> str:
    """One stacked bar showing how the whole book divides up."""
    total = sum(max(v, 0) for _, v in items)
    if total <= 0:
        return ""
    segments = []
    for i, (label, value) in enumerate(items):
        share = max(value, 0) / total * 100
        if share <= 0:
            continue
        colour = SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)]
        text = _escape(label) if share >= min_label_pct else ""
        segments.append(
            f"<div class='alloc-seg' style='width:{share:.3f}%;background:{colour};"
            f"animation-delay:{i * 0.04:.2f}s' "
            f"title='{_escape(label)} — {share:.1f}%'>{text}</div>"
        )
    return f"<div class='alloc'>{''.join(segments)}</div>"


def holding_rows(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """A styled table where each row can carry a sparkline and a weight bar."""
    header = "".join(
        f"<div class='row-num'>{_escape(title)}</div>" if align == "r"
        else f"<div>{_escape(title)}</div>"
        for title, align in columns
    )
    out = [f"<div class='rows'><div class='row row-head'>{header}</div>"]
    for row in rows:
        weight = row.get("weight_pct")
        bar = (
            f"<div class='row-weight' style='width:{min(float(weight), 100):.2f}%'></div>"
            if weight else ""
        )
        out.append(f"<div class='row'>{row['cells']}{bar}</div>")
    out.append("</div>")
    return "".join(out)


def pill(text: str) -> str:
    tone = "hold"
    if text and "BUY" in text.upper():
        tone = "buy"
    elif text and "SELL" in text.upper():
        tone = "sell"
    return f"<span class='pill {tone}'>{_escape(text or '—')}</span>"


def html(markup: str) -> None:
    """Emit a component. Named `html` because each view already defines
    `render` as its page entry point."""
    st.markdown(markup, unsafe_allow_html=True)


# --- Cached analysis ------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def analyse(ticker: str, period: str = "5y", skip_options: bool = False) -> dict[str, Any]:
    return engine.analyse(ticker, period=period, skip_options=skip_options)


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    return datafeed.price_history(ticker, period=period)


@st.cache_data(show_spinner=False, ttl=300)
def quote(ticker: str) -> float | None:
    """Latest price only - cheap enough to refresh far more often."""
    fast = datafeed.fast_quote(ticker)
    price = fast.get("last_price")
    if price:
        return float(price)
    try:
        return float(datafeed.price_history(ticker, period="1mo")["Close"].iloc[-1])
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def synthesise(payload: dict[str, Any], authority: bool, _key: str) -> dict[str, Any]:
    return llm.synthesise(payload, numeric_authority=authority)


def _screen_one(symbol: str, period: str) -> dict[str, Any]:
    """Analyse a single symbol into a screen row. Never raises."""
    try:
        payload = engine.analyse(symbol, period=period, skip_options=True)
    except DataError as exc:
        return {"symbol": symbol, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "error": f"{exc.__class__.__name__}: {exc}"}
    return _screen_row(symbol, payload)


# Screen results are cached per symbol rather than per watchlist.
#
# They used to be memoised on the whole tuple of symbols, which meant adding
# one name changed the cache key and re-analysed every other name from scratch.
# Adding the twentieth symbol cost twenty analyses, so the work grew with the
# square of the list: every entry felt like a full reload, and a long enough
# watchlist could not finish inside the memory a free container has. Once the
# watchlist began persisting, that turned into a loop no refresh could escape,
# because the same doomed screen ran again on every load.
#
# Keyed by (symbol, period) and read on the main thread, so a new name costs
# exactly one analysis and the rest are hits. A plain dict rather than
# st.cache_data because the fan-out below runs in threads, which have no script
# context to reach Streamlit's cache through. The contents are public market
# data, so sharing one cache across visitors is safe and saves everyone work.
_SCREEN_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_SCREEN_LOCK = threading.Lock()


def _screen_cached(symbol: str, period: str) -> dict[str, Any] | None:
    with _SCREEN_LOCK:
        entry = _SCREEN_CACHE.get((symbol, period))
    if entry is None:
        return None
    stored_at, row = entry
    if time.time() - stored_at > REFRESH_SECONDS:
        return None
    return row


def _screen_store(symbol: str, period: str, row: dict[str, Any]) -> None:
    # An error row is deliberately not cached: a symbol that failed on a
    # network blip should be retried on the next load, not held as broken for
    # an hour.
    if row.get("error"):
        return
    with _SCREEN_LOCK:
        _SCREEN_CACHE[(symbol, period)] = (time.time(), row)


def _screen_clear() -> None:
    with _SCREEN_LOCK:
        _SCREEN_CACHE.clear()


def screen_pending(symbols: tuple[str, ...], period: str = "2y") -> list[str]:
    """Which symbols would still need analysing, for progress reporting."""
    return [s for s in symbols if _screen_cached(s, period) is None]


def screen(symbols: tuple[str, ...], period: str = "2y",
           limit: int | None = None) -> list[dict[str, Any]]:
    """Analyse many symbols for the watchlist and portfolio tables.

    Only symbols that are not already cached are fetched, so adding a name to
    a long watchlist costs one analysis rather than a full rescreen.

    ``limit`` caps how many *new* analyses one call performs. A free container
    has about a gigabyte, and a long watchlist analysed in a single pass can
    exhaust it - which killed the process mid-render, and once the watchlist
    persisted, did so again on every reload. Capping the batch keeps the page
    renderable and lets the rest be picked up on the next pass, since results
    accumulate in the cache.

    The fetch is parallel because every call here is network-bound waiting on
    Yahoo; a serial loop costs roughly six seconds per symbol, which made a
    hundred-name watchlist take over ten minutes and simply look broken.

    Options are skipped: chains are the slowest call by far and contribute
    little to a horizon ranking. One bad symbol yields an error row rather than
    aborting the whole screen.
    """
    if not symbols:
        return []

    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for symbol in symbols:
        row = _screen_cached(symbol, period)
        if row is None:
            missing.append(symbol)
        else:
            results[symbol] = row

    if limit is not None and limit >= 0:
        missing = missing[:limit]

    if missing:
        # Modest pool: Yahoo rate-limits aggressively, and past about a dozen
        # concurrent requests the failures cost more than the parallelism
        # saves. A shared host gets fewer still - Community Cloud allows about
        # a gigabyte, and eight threads each holding several years of OHLCV
        # will exhaust it.
        ceiling = 4 if store.is_shared_host() else 8
        workers = max(1, min(ceiling, len(missing)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_screen_one, symbol, period): symbol
                for symbol in missing
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {"symbol": symbol,
                           "error": f"{exc.__class__.__name__}: {exc}"}
                results[symbol] = row
                _screen_store(symbol, period, row)

    # Preserve the caller's ordering rather than completion order.
    return [results[s] for s in symbols if s in results]


# Callers clear the screen cache to force a refresh; keep that spelling working
# now that this is no longer an st.cache_data function.
screen.clear = _screen_clear  # type: ignore[attr-defined]


# --- Earnings calendar ----------------------------------------------------
#
# Cached per symbol and batched for the same reasons as screening above: the
# calendar spans everything held and watched, so a hundred names means a
# hundred network calls, and doing them all in one pass is what exhausts a
# free container.
_EARNINGS_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}
_EARNINGS_LOCK = threading.Lock()


def _earnings_cached(symbol: str) -> tuple[bool, dict[str, Any] | None]:
    """``(hit, event)``. A symbol with no scheduled report caches as None."""
    with _EARNINGS_LOCK:
        entry = _EARNINGS_CACHE.get(symbol)
    if entry is None:
        return False, None
    stored_at, event = entry
    if time.time() - stored_at > REFRESH_SECONDS:
        return False, None
    return True, event


def _earnings_one(symbol: str) -> dict[str, Any] | None:
    try:
        return earnings.upcoming(symbol)
    except Exception:
        # A symbol with no earnings at all - an ETF, a currency pair - is the
        # normal case here, not an error worth surfacing.
        return None


def earnings_pending(symbols: tuple[str, ...]) -> list[str]:
    return [s for s in symbols if not _earnings_cached(s)[0]]


def earnings_events(symbols: tuple[str, ...],
                    limit: int | None = None) -> list[dict[str, Any]]:
    """The next scheduled report for each symbol that has one."""
    if not symbols:
        return []

    found: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    for symbol in symbols:
        hit, event = _earnings_cached(symbol)
        if hit:
            found[symbol] = event
        else:
            missing.append(symbol)

    if limit is not None and limit >= 0:
        missing = missing[:limit]

    if missing:
        workers = max(1, min(4 if store.is_shared_host() else 8, len(missing)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_earnings_one, s): s for s in missing}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    event = future.result()
                except Exception:
                    event = None
                found[symbol] = event
                with _EARNINGS_LOCK:
                    _EARNINGS_CACHE[symbol] = (time.time(), event)

    return [found[s] for s in symbols if found.get(s)]


def earnings_clear() -> None:
    with _EARNINGS_LOCK:
        _EARNINGS_CACHE.clear()


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def earnings_brief(symbol: str) -> dict[str, Any]:
    """The full detail for one company. Only the opened company pays for this."""
    return earnings.brief(symbol)


def _screen_row(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one analysis into the row shape the tables expect."""
    verdict = payload.get("verdict") or {}
    tech = payload.get("technical") or {}
    quote_block = payload.get("quote") or {}
    horizons = horizon.compute(payload)
    fundamentals = (payload.get("fundamental") or {}).get("valuation") or {}
    profile = (payload.get("fundamental") or {}).get("profile") or {}
    earnings = (payload.get("fundamental") or {}).get("earnings") or {}

    return {
        "symbol": symbol,
        "name": profile.get("name"),
        "sector": profile.get("sector"),
        "price": quote_block.get("spot"),
        "change_pct": quote_block.get("change_pct"),
        "verdict": verdict.get("verdict"),
        "score": verdict.get("score_0_100"),
        "conviction": verdict.get("conviction_pct"),
        "near_term_score": horizons.get("near_term_score"),
        "long_term_score": horizons.get("long_term_score"),
        "bias": horizons.get("bias"),
        "horizon_summary": horizons.get("summary"),
        "rsi": (tech.get("momentum") or {}).get("rsi_14"),
        "trend": (tech.get("moving_averages") or {}).get("alignment"),
        "perf_1m": (tech.get("performance_pct") or {}).get("1m"),
        "perf_1y": (tech.get("performance_pct") or {}).get("1y"),
        "forward_pe": fundamentals.get("forward_pe"),
        "days_to_earnings": earnings.get("days_to_earnings"),
        "error": None,
    }


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def histories(symbols: tuple[str, ...], period: str = "2y") -> dict[str, Any]:
    """Price history per symbol, for correlation and benchmark work."""
    out: dict[str, Any] = {}
    for symbol in symbols:
        try:
            out[symbol] = datafeed.price_history(symbol, period=period)
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def spark_series(symbols: tuple[str, ...], period: str = "1mo") -> dict[str, list]:
    """Closing prices per symbol, thinned to what a 118px sparkline can show."""
    out: dict[str, list] = {}
    for symbol in symbols:
        try:
            history = datafeed.price_history(symbol, period=period)
        except Exception:
            continue
        if history is None or history.empty or "Close" not in history:
            continue
        closes = history["Close"].astype(float).dropna()
        if len(closes) < 2:
            continue
        # More points than pixels is wasted markup; keep about 40.
        step = max(1, len(closes) // 40)
        out[symbol] = [float(v) for v in closes.iloc[::step]]
    return out


@st.cache_data(show_spinner=False, ttl=600)
def period_histories(symbols: tuple[str, ...], period: str, interval: str
                     ) -> dict[str, Any]:
    """Price history at a specific period/interval, for the value chart.

    Cached for ten minutes rather than an hour: the intraday windows are the
    whole point of this fetch, and an hour-old 5-minute series is not intraday.
    """
    out: dict[str, Any] = {}
    for symbol in symbols:
        try:
            out[symbol] = datafeed.price_history(symbol, period=period,
                                                 interval=interval)
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False, ttl=600)
def fx_histories(currencies: tuple[str, ...], base: str, period: str,
                 interval: str) -> dict[str, Any]:
    """Historical FX series so past values convert at past rates."""
    out: dict[str, Any] = {}
    for currency in currencies:
        if not currency or currency == base:
            continue
        for pair in (f"{currency}{base}=X",):
            try:
                history = datafeed.price_history(pair, period=period,
                                                 interval=interval)
                if history is not None and not history.empty:
                    out[currency] = history
            except Exception:
                continue
    return out


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def fx_rates(currencies: tuple[str, ...], base: str) -> dict[str, float | None]:
    """Multipliers converting each currency into ``base``."""
    return {ccy: datafeed.fx_rate(ccy, base) for ccy in currencies if ccy}


@st.cache_data(show_spinner=False, ttl=REFRESH_SECONDS)
def portfolio_risk(symbols: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """5y monthly beta and sector per symbol, for the portfolio rollup.

    Measured against the first configured benchmark, so changing BENCHMARKS
    changes what the portfolio's beta is relative to.
    """
    benchmark_symbol = BENCHMARKS[0]
    out: dict[str, dict[str, Any]] = {}
    try:
        benchmark = datafeed.price_history(benchmark_symbol, period="5y")
    except Exception:
        benchmark = None

    for symbol in symbols:
        entry: dict[str, Any] = {"beta": None, "sector": None}
        try:
            info = datafeed.ticker_info(symbol)
            entry["sector"] = info.get("sector")
        except Exception:
            pass
        if benchmark is not None:
            try:
                asset = datafeed.price_history(symbol, period="5y")
                entry["beta"] = risk.monthly_beta(asset, benchmark).get("beta")
            except Exception:
                pass
        out[symbol] = entry
    return out


def refresh_note(key: str) -> None:
    """Record and display when this view last recomputed."""
    stamp = datetime.now(timezone.utc)
    st.session_state[f"_refreshed_{key}"] = stamp
    st.caption(
        f"Updated {stamp.astimezone().strftime('%H:%M')} · refreshes hourly"
    )
