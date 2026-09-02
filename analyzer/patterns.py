"""Chart and candlestick pattern detection, with a measured edge.

A pattern name on its own is close to worthless. "Bull flag" tells you what the
last few weeks looked like, not what happens next, and the published hit rates
for these formations come from books, other decades, and other instruments.

So every pattern detected here is also **backtested on this ticker's own
history**: how many times has it appeared, and how often was price higher 5,
10 and 20 sessions later? That converts folklore into a number you can size a
position against - and quite often the honest answer is "this setup has no
edge in this name", which is exactly the thing worth knowing before acting.

Two families are detected:

* **Candlestick** patterns - one to three bars, precise definitions, plentiful
  historical occurrences, so their hit rates are measurable.
* **Structural** patterns - double tops, head and shoulders, triangles, flags.
  These are rarer, so they carry levels (trigger, target, invalidation) rather
  than a hit rate; a handful of past occurrences cannot support a statistic.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

# Forward windows used to measure whether a signal led anywhere.
HORIZONS = (5, 10, 20)

# Below this many past occurrences, a hit rate is noise rather than evidence.
MIN_OCCURRENCES = 8


# --- Primitives -----------------------------------------------------------


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["Close"] - df["Open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["High"] - df["Low"]).replace(0, np.nan)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df[["Close", "Open"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["Close", "Open"]].min(axis=1) - df["Low"]


def _bullish(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def pivots(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs and lows: an extreme of its +/- window neighbourhood."""
    span = window * 2 + 1
    highs = df["High"][df["High"] == df["High"].rolling(span, center=True).max()]
    lows = df["Low"][df["Low"] == df["Low"].rolling(span, center=True).min()]
    return highs.dropna(), lows.dropna()


def _trend(df: pd.DataFrame, lookback: int = 20) -> float:
    """Percentage change over the lookback, used to set pattern context."""
    close = df["Close"]
    if len(close) <= lookback:
        return 0.0
    prior = float(close.iloc[-lookback - 1])
    return (float(close.iloc[-1]) / prior - 1) * 100 if prior else 0.0


# --- Candlestick definitions ----------------------------------------------
#
# Each returns a boolean Series: True on bars where the pattern completes.
# Definitions are deliberately strict - loose ones fire constantly and their
# measured edge collapses toward the base rate.


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bear = ~_bullish(df).shift(1).fillna(False).astype(bool)
    engulfs = (df["Close"] >= df["Open"].shift(1)) & (df["Open"] <= df["Close"].shift(1))
    return _bullish(df) & prev_bear & engulfs & (_body(df) > _body(df).shift(1))


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_bull = _bullish(df).shift(1).fillna(False).astype(bool)
    engulfs = (df["Close"] <= df["Open"].shift(1)) & (df["Open"] >= df["Close"].shift(1))
    return (~_bullish(df)) & prev_bull & engulfs & (_body(df) > _body(df).shift(1))


def hammer(df: pd.DataFrame) -> pd.Series:
    """Long lower wick, small body, in a downtrend - rejection of lower prices."""
    downtrend = df["Close"] < df["Close"].rolling(10).mean()
    return (
        (_lower_shadow(df) > 2 * _body(df))
        & (_upper_shadow(df) < 0.35 * _body(df).clip(lower=1e-9))
        & (_body(df) / _range(df) < 0.4)
        & downtrend
    ).fillna(False)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    uptrend = df["Close"] > df["Close"].rolling(10).mean()
    return (
        (_upper_shadow(df) > 2 * _body(df))
        & (_lower_shadow(df) < 0.35 * _body(df).clip(lower=1e-9))
        & (_body(df) / _range(df) < 0.4)
        & uptrend
    ).fillna(False)


def morning_star(df: pd.DataFrame) -> pd.Series:
    """Down bar, small indecisive bar, then a strong up bar closing into bar one."""
    big_down = (~_bullish(df).shift(2).fillna(False).astype(bool)) & (
        _body(df).shift(2) > _body(df).rolling(20).mean().shift(2)
    )
    small = _body(df).shift(1) < 0.5 * _body(df).shift(2)
    strong_up = _bullish(df) & (
        df["Close"] > (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    )
    return (big_down & small & strong_up).fillna(False)


def evening_star(df: pd.DataFrame) -> pd.Series:
    big_up = _bullish(df).shift(2).fillna(False).astype(bool) & (
        _body(df).shift(2) > _body(df).rolling(20).mean().shift(2)
    )
    small = _body(df).shift(1) < 0.5 * _body(df).shift(2)
    strong_down = (~_bullish(df)) & (
        df["Close"] < (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    )
    return (big_up & small & strong_down).fillna(False)


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    bull = _bullish(df)
    rising = (df["Close"] > df["Close"].shift(1)) & (
        df["Close"].shift(1) > df["Close"].shift(2)
    )
    solid = _body(df) > 0.5 * _range(df)
    return (bull & bull.shift(1) & bull.shift(2) & rising & solid).fillna(False)


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    bear = ~_bullish(df)
    falling = (df["Close"] < df["Close"].shift(1)) & (
        df["Close"].shift(1) < df["Close"].shift(2)
    )
    solid = _body(df) > 0.5 * _range(df)
    return (bear & bear.shift(1) & bear.shift(2) & falling & solid).fillna(False)


def gap_up(df: pd.DataFrame) -> pd.Series:
    """Opens above the prior high and holds it - unfilled demand."""
    return ((df["Low"] > df["High"].shift(1)) & _bullish(df)).fillna(False)


def gap_down(df: pd.DataFrame) -> pd.Series:
    return ((df["High"] < df["Low"].shift(1)) & (~_bullish(df))).fillna(False)


CANDLESTICKS: dict[str, tuple[Callable[[pd.DataFrame], pd.Series], str, str]] = {
    "Bullish engulfing": (bullish_engulfing, "bullish",
                          "Buyers took back the whole of the previous session's loss."),
    "Bearish engulfing": (bearish_engulfing, "bearish",
                          "Sellers erased the whole of the previous session's gain."),
    "Hammer": (hammer, "bullish",
               "Price was pushed down hard and closed back near the top - the low was rejected."),
    "Shooting star": (shooting_star, "bearish",
                      "Price ran up and closed back near the low - the high was rejected."),
    "Morning star": (morning_star, "bullish",
                     "A down bar, hesitation, then decisive buying - a classic bottom sequence."),
    "Evening star": (evening_star, "bearish",
                     "An up bar, hesitation, then decisive selling - a classic top sequence."),
    "Three white soldiers": (three_white_soldiers, "bullish",
                             "Three strong closes in a row; steady accumulation."),
    "Three black crows": (three_black_crows, "bearish",
                          "Three weak closes in a row; steady distribution."),
    "Gap up": (gap_up, "bullish", "Opened above yesterday's high and stayed there."),
    "Gap down": (gap_down, "bearish", "Opened below yesterday's low and stayed there."),
}


# --- Measuring whether a signal actually led anywhere ----------------------


def measure_edge(df: pd.DataFrame, signal: pd.Series,
                 horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    """Forward returns after every past occurrence of a signal.

    The comparison that matters is against the stock's own base rate over the
    same window. A setup that is "up 60% of the time" in a name that rises 58%
    of the time regardless has told you nothing.
    """
    close = df["Close"].astype(float)
    out: dict[str, Any] = {"occurrences": int(signal.sum()), "horizons": {}}
    if out["occurrences"] == 0:
        return out

    for horizon in horizons:
        forward = (close.shift(-horizon) / close - 1) * 100
        after = forward[signal.fillna(False)].dropna()
        base = forward.dropna()
        if len(after) == 0 or len(base) == 0:
            continue
        out["horizons"][horizon] = {
            "samples": int(len(after)),
            "hit_rate": float((after > 0).mean() * 100),
            "median_return": float(after.median()),
            "mean_return": float(after.mean()),
            "base_rate": float((base > 0).mean() * 100),
            "base_median": float(base.median()),
            # The only figure that matters: how much better than doing nothing.
            "edge": float((after > 0).mean() * 100 - (base > 0).mean() * 100),
        }
    return out


# --- Structural patterns ---------------------------------------------------


def _double(df: pd.DataFrame, highs: pd.Series, lows: pd.Series,
            lookback: int = 120) -> dict[str, Any] | None:
    """Double top or double bottom: two extremes at a similar level."""
    recent_h = highs.iloc[-4:] if len(highs) >= 2 else highs
    recent_l = lows.iloc[-4:] if len(lows) >= 2 else lows
    price = float(df["Close"].iloc[-1])
    atr = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])
    tolerance = max(atr * 1.2, price * 0.02)

    if len(recent_h) >= 2:
        a, b = float(recent_h.iloc[-2]), float(recent_h.iloc[-1])
        if abs(a - b) <= tolerance and price < min(a, b):
            between = df["Low"].loc[recent_h.index[-2]:recent_h.index[-1]]
            neckline = float(between.min()) if not between.empty else price
            return {
                "name": "Double top",
                "direction": "bearish",
                "stage": "confirmed" if price < neckline else "forming",
                "trigger": round(neckline, 2),
                "target": round(neckline - (max(a, b) - neckline), 2),
                "invalidation": round(max(a, b), 2),
                "detail": (
                    f"Price failed twice near {max(a, b):,.2f}. A close below the "
                    f"{neckline:,.2f} neckline completes the pattern; the measured "
                    "objective is the height of the pattern projected down."
                ),
            }

    if len(recent_l) >= 2:
        a, b = float(recent_l.iloc[-2]), float(recent_l.iloc[-1])
        if abs(a - b) <= tolerance and price > max(a, b):
            between = df["High"].loc[recent_l.index[-2]:recent_l.index[-1]]
            neckline = float(between.max()) if not between.empty else price
            return {
                "name": "Double bottom",
                "direction": "bullish",
                "stage": "confirmed" if price > neckline else "forming",
                "trigger": round(neckline, 2),
                "target": round(neckline + (neckline - min(a, b)), 2),
                "invalidation": round(min(a, b), 2),
                "detail": (
                    f"Price held twice near {min(a, b):,.2f}. A close above the "
                    f"{neckline:,.2f} neckline completes the pattern."
                ),
            }
    return None


def _head_and_shoulders(df: pd.DataFrame, highs: pd.Series,
                        lows: pd.Series) -> dict[str, Any] | None:
    """Three peaks, the middle highest, with a roughly level neckline."""
    price = float(df["Close"].iloc[-1])
    if len(highs) >= 3:
        l, h, r = (float(v) for v in highs.iloc[-3:])
        shoulders_even = abs(l - r) <= max(l, r) * 0.05
        if h > l and h > r and shoulders_even:
            troughs = lows[(lows.index > highs.index[-3]) & (lows.index < highs.index[-1])]
            if not troughs.empty:
                neckline = float(troughs.mean())
                return {
                    "name": "Head and shoulders",
                    "direction": "bearish",
                    "stage": "confirmed" if price < neckline else "forming",
                    "trigger": round(neckline, 2),
                    "target": round(neckline - (h - neckline), 2),
                    "invalidation": round(h, 2),
                    "detail": (
                        f"Three peaks with the middle highest at {h:,.2f}. A close "
                        f"below {neckline:,.2f} completes it."
                    ),
                }
    if len(lows) >= 3:
        l, h, r = (float(v) for v in lows.iloc[-3:])
        shoulders_even = abs(l - r) <= max(l, r) * 0.05
        if h < l and h < r and shoulders_even:
            peaks = highs[(highs.index > lows.index[-3]) & (highs.index < lows.index[-1])]
            if not peaks.empty:
                neckline = float(peaks.mean())
                return {
                    "name": "Inverse head and shoulders",
                    "direction": "bullish",
                    "stage": "confirmed" if price > neckline else "forming",
                    "trigger": round(neckline, 2),
                    "target": round(neckline + (neckline - h), 2),
                    "invalidation": round(h, 2),
                    "detail": (
                        f"Three troughs with the middle lowest at {h:,.2f}. A close "
                        f"above {neckline:,.2f} completes it."
                    ),
                }
    return None


def _triangle(df: pd.DataFrame, highs: pd.Series,
              lows: pd.Series) -> dict[str, Any] | None:
    """Converging or flat-edged trendlines through the recent swings."""
    if len(highs) < 3 or len(lows) < 3:
        return None

    recent_h, recent_l = highs.iloc[-3:], lows.iloc[-3:]
    hx = np.arange(len(recent_h))
    lx = np.arange(len(recent_l))
    h_slope = float(np.polyfit(hx, recent_h.to_numpy(dtype=float), 1)[0])
    l_slope = float(np.polyfit(lx, recent_l.to_numpy(dtype=float), 1)[0])

    price = float(df["Close"].iloc[-1])
    scale = price * 0.004  # small enough a slope to call "flat"
    resistance = float(recent_h.max())
    support = float(recent_l.min())

    if abs(h_slope) < scale and l_slope > scale:
        name, direction = "Ascending triangle", "bullish"
        detail = (
            f"Highs are capped near {resistance:,.2f} while lows keep rising - "
            "supply at a fixed level being absorbed. Resolution is usually upward."
        )
    elif abs(l_slope) < scale and h_slope < -scale:
        name, direction = "Descending triangle", "bearish"
        detail = (
            f"Lows are holding near {support:,.2f} while highs keep falling - "
            "demand at a fixed level being worn down. Resolution is usually downward."
        )
    elif h_slope < -scale and l_slope > scale:
        name, direction = "Symmetrical triangle", "neutral"
        detail = (
            "Range is narrowing from both sides. Direction is undecided; the "
            "breakout side is the signal, not the pattern itself."
        )
    else:
        return None

    return {
        "name": name, "direction": direction, "stage": "forming",
        "trigger": round(resistance if direction != "bearish" else support, 2),
        "target": None,
        "invalidation": round(support if direction != "bearish" else resistance, 2),
        "detail": detail,
    }


def _flag(df: pd.DataFrame) -> dict[str, Any] | None:
    """A sharp move (the pole) followed by a tight drift against it."""
    if len(df) < 40:
        return None
    close = df["Close"].astype(float)

    pole = (float(close.iloc[-11]) / float(close.iloc[-31]) - 1) * 100
    recent = close.iloc[-10:]
    drift = (float(recent.iloc[-1]) / float(recent.iloc[0]) - 1) * 100
    tightness = float(recent.std() / recent.mean() * 100)

    if abs(pole) < 12 or tightness > 4.5:
        return None
    if pole > 0 and -6 < drift <= 2:
        return {
            "name": "Bull flag", "direction": "bullish", "stage": "forming",
            "trigger": round(float(df["High"].iloc[-10:].max()), 2),
            "target": None,
            "invalidation": round(float(df["Low"].iloc[-10:].min()), 2),
            "detail": (
                f"A {pole:+.0f}% run followed by a quiet {drift:+.1f}% drift. "
                "Consolidation after a strong move usually resolves in the "
                "direction of the move, on a break of the flag high."
            ),
        }
    if pole < 0 and -2 <= drift < 6:
        return {
            "name": "Bear flag", "direction": "bearish", "stage": "forming",
            "trigger": round(float(df["Low"].iloc[-10:].min()), 2),
            "target": None,
            "invalidation": round(float(df["High"].iloc[-10:].max()), 2),
            "detail": (
                f"A {pole:+.0f}% fall followed by a weak {drift:+.1f}% bounce. "
                "A break of the flag low resumes the decline."
            ),
        }
    return None


def _range_break(df: pd.DataFrame) -> dict[str, Any] | None:
    """Compression into a tight band, then a decisive exit from it."""
    if len(df) < 60:
        return None
    close = df["Close"].astype(float)
    window = close.iloc[-40:-1]
    high, low = float(window.max()), float(window.min())
    width = (high - low) / low * 100 if low else 0
    price = float(close.iloc[-1])

    if width > 14:
        return None
    if price > high:
        return {
            "name": "Range breakout", "direction": "bullish", "stage": "confirmed",
            "trigger": round(high, 2), "target": round(high + (high - low), 2),
            "invalidation": round(low, 2),
            "detail": (
                f"Price spent weeks between {low:,.2f} and {high:,.2f}, a {width:.1f}% "
                "band, and has now closed above it. The measured objective is the "
                "range height projected upward."
            ),
        }
    if price < low:
        return {
            "name": "Range breakdown", "direction": "bearish", "stage": "confirmed",
            "trigger": round(low, 2), "target": round(low - (high - low), 2),
            "invalidation": round(high, 2),
            "detail": (
                f"Price spent weeks between {low:,.2f} and {high:,.2f} and has now "
                "closed below it."
            ),
        }
    return None


STRUCTURAL = (_double, _head_and_shoulders, _triangle, _flag, _range_break)


# --- Public entry point ----------------------------------------------------


def analyse(df: pd.DataFrame, recent_bars: int = 3) -> dict[str, Any]:
    """Patterns active now, each with whatever evidence supports it."""
    out: dict[str, Any] = {
        "candlesticks": [], "structural": [], "bias": None,
        "bias_detail": None, "trend_20d_pct": None,
    }
    if df is None or len(df) < 60:
        out["bias_detail"] = "Not enough price history to read patterns."
        return out

    out["trend_20d_pct"] = _trend(df)

    # --- Candlesticks firing in the last few sessions ---
    for name, (fn, direction, meaning) in CANDLESTICKS.items():
        try:
            signal = fn(df).fillna(False)
        except Exception:
            continue
        if not signal.iloc[-recent_bars:].any():
            continue

        fired = signal.iloc[-recent_bars:]
        when = fired[fired].index[-1]
        edge = measure_edge(df, signal)

        # What the textbook says versus what this ticker actually did. When
        # they disagree, the measurement wins - that disagreement is the most
        # useful thing on the page, because it says the pattern does not work
        # here regardless of what it is supposed to mean.
        horizon = edge["horizons"].get(20)
        measured = None
        agrees = None
        if horizon and edge["occurrences"] >= MIN_OCCURRENCES:
            measured = "bullish" if horizon["edge"] > 0 else "bearish"
            agrees = measured == direction

        out["candlesticks"].append({
            "name": name,
            "direction": direction,
            "measured_direction": measured,
            "agrees_with_history": agrees,
            "meaning": meaning,
            "date": when.date().isoformat() if hasattr(when, "date") else str(when),
            "bars_ago": int(len(df) - 1 - df.index.get_loc(when)),
            "edge": edge,
            "reliable": bool(
                horizon and edge["occurrences"] >= MIN_OCCURRENCES
                and abs(horizon["edge"]) >= 5
            ),
        })

    # --- Structural formations ---
    highs, lows = pivots(df)
    for detector in STRUCTURAL:
        try:
            found = detector(df, highs, lows) if detector in (
                _double, _head_and_shoulders, _triangle
            ) else detector(df)
        except Exception:
            continue
        if found:
            out["structural"].append(found)

    # --- Net read ---
    score = 0.0
    for pattern in out["structural"]:
        weight = 2.0 if pattern["stage"] == "confirmed" else 1.0
        score += weight * (1 if pattern["direction"] == "bullish"
                           else -1 if pattern["direction"] == "bearish" else 0)
    contradictions = 0
    for candle in out["candlesticks"]:
        # Where history is available it decides the direction, not the label.
        # A pattern with no measured edge contributes almost nothing.
        if candle["reliable"] and candle["measured_direction"]:
            weight, facing = 1.0, candle["measured_direction"]
            if candle["agrees_with_history"] is False:
                contradictions += 1
        else:
            weight, facing = 0.3, candle["direction"]
        score += weight * (1 if facing == "bullish" else -1)
    out["contradictions"] = contradictions

    if score >= 1.5:
        out["bias"] = "bullish"
    elif score <= -1.5:
        out["bias"] = "bearish"
    elif out["candlesticks"] or out["structural"]:
        out["bias"] = "mixed"

    out["bias_score"] = round(score, 2)
    if not out["candlesticks"] and not out["structural"]:
        out["bias_detail"] = (
            "No recognised pattern is active. That is a normal state - most of "
            "the time a chart is not forming anything worth naming."
        )
    else:
        counts = (f"{len(out['structural'])} structural, "
                  f"{len(out['candlesticks'])} candlestick")
        detail = f"Net read from {counts}."
        if contradictions:
            detail += (
                f" {contradictions} of them historically resolved the opposite way "
                "on this ticker, and are scored by what actually happened rather "
                "than by what the pattern is supposed to mean."
            )
        out["bias_detail"] = detail
    return out
