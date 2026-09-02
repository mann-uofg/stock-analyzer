"""Technical indicator computation.

Built on the ``ta`` library (pure pandas/numpy, no pinned transitive deps).
Every public function tolerates short or gappy history and returns ``None``
for values it cannot compute rather than raising.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from .config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    CCI_PERIOD,
    EMA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    SMA_PERIODS,
    STOCH_PERIOD,
    STOCH_SMOOTH,
    VOLUME_LOOKBACK,
    VOLUME_SPIKE_MULTIPLE,
    VWAP_PERIOD,
)


def _last(series: pd.Series | None) -> float | None:
    """Final finite value of a series, or None."""
    if series is None or len(series) == 0:
        return None
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _safe(fn, *args, **kwargs):
    """Run an indicator, swallowing failures caused by insufficient history."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _crossover_state(fast: float | None, slow: float | None) -> str | None:
    if fast is None or slow is None:
        return None
    return "above" if fast > slow else "below"


def _detect_cross(fast: pd.Series, slow: pd.Series, lookback: int | None = None) -> dict[str, Any]:
    """Find the most recent fast/slow crossover and how long ago it happened.

    The whole series is searched by default and ``bars_ago`` is always
    reported, so callers can judge relevance themselves - ``scoring`` only
    rewards a cross inside the last 30 bars, but an older one is still worth
    surfacing.
    """
    pair = pd.concat([fast, slow], axis=1).dropna()
    if len(pair) < 2:
        return {"event": None, "bars_ago": None}

    diff = pair.iloc[:, 0] - pair.iloc[:, 1]
    sign = np.sign(diff)
    changes = sign.diff()
    window = changes if lookback is None else changes.iloc[-lookback:]
    crosses = window[window != 0].dropna()
    crosses = crosses[crosses != 0]

    if crosses.empty:
        return {"event": None, "bars_ago": None}

    last_idx = crosses.index[-1]
    bars_ago = int(len(pair) - 1 - pair.index.get_loc(last_idx))
    event = "golden_cross" if crosses.iloc[-1] > 0 else "death_cross"
    return {"event": event, "bars_ago": bars_ago}


def support_resistance(df: pd.DataFrame, window: int = 10, max_levels: int = 3) -> dict[str, list]:
    """Swing-based support and resistance levels relative to the last close.

    Uses fractal pivots (a bar whose high/low is the extreme of its
    +/- ``window`` neighbourhood), then keeps the nearest levels on each side.
    """
    if len(df) < window * 2 + 1:
        return {"support": [], "resistance": []}

    highs, lows = df["High"], df["Low"]
    price = float(df["Close"].iloc[-1])

    pivot_high = highs[(highs == highs.rolling(window * 2 + 1, center=True).max())]
    pivot_low = lows[(lows == lows.rolling(window * 2 + 1, center=True).min())]

    res = sorted({round(float(v), 2) for v in pivot_high.dropna() if v > price})
    sup = sorted({round(float(v), 2) for v in pivot_low.dropna() if v < price}, reverse=True)

    return {"support": sup[:max_levels], "resistance": res[:max_levels]}


def compute(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the full technical panel for an OHLCV frame.

    ``df`` must carry Open/High/Low/Close/Volume columns indexed by date.
    """
    if df is None or df.empty or len(df) < 2:
        return {"error": "insufficient price history"}

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[-1])
    out: dict[str, Any] = {"price": price, "bars": int(len(df))}

    # --- Moving averages & crossovers ---
    sma: dict[str, float | None] = {}
    ema: dict[str, float | None] = {}
    sma_series: dict[int, pd.Series] = {}
    for period in SMA_PERIODS:
        s = _safe(lambda p=period: SMAIndicator(close, p).sma_indicator())
        if s is not None:
            sma_series[period] = s
        sma[f"sma_{period}"] = _last(s)
    for period in EMA_PERIODS:
        e = _safe(lambda p=period: EMAIndicator(close, p).ema_indicator())
        ema[f"ema_{period}"] = _last(e)

    out["moving_averages"] = {
        **sma,
        **ema,
        "price_vs_sma20": _crossover_state(price, sma.get("sma_20")),
        "price_vs_sma50": _crossover_state(price, sma.get("sma_50")),
        "price_vs_sma200": _crossover_state(price, sma.get("sma_200")),
    }

    if 50 in sma_series and 200 in sma_series:
        cross = _detect_cross(sma_series[50], sma_series[200])
        # The standing regime matters even when the crossover itself is old,
        # so report both rather than only a recent event.
        s50_last, s200_last = _last(sma_series[50]), _last(sma_series[200])
        if s50_last is not None and s200_last is not None:
            cross["state"] = "golden" if s50_last > s200_last else "death"
        else:
            cross["state"] = None
        out["moving_averages"]["golden_death_cross"] = cross
    else:
        out["moving_averages"]["golden_death_cross"] = {
            "event": None, "bars_ago": None, "state": None
        }

    # Stacked MAs are a cleaner trend read than any single crossover.
    s20, s50, s200 = sma.get("sma_20"), sma.get("sma_50"), sma.get("sma_200")
    if None not in (s20, s50, s200):
        if s20 > s50 > s200:
            alignment = "bullish_stacked"
        elif s20 < s50 < s200:
            alignment = "bearish_stacked"
        else:
            alignment = "mixed"
    else:
        alignment = None
    out["moving_averages"]["alignment"] = alignment

    # --- Momentum ---
    rsi = _safe(lambda: RSIIndicator(close, RSI_PERIOD).rsi())
    macd_obj = _safe(lambda: MACD(close, MACD_SLOW, MACD_FAST, MACD_SIGNAL))
    stoch = _safe(lambda: StochasticOscillator(high, low, close, STOCH_PERIOD, STOCH_SMOOTH))
    cci = _safe(lambda: CCIIndicator(high, low, close, CCI_PERIOD).cci())

    rsi_val = _last(rsi)
    macd_line = _last(macd_obj.macd()) if macd_obj is not None else None
    macd_sig = _last(macd_obj.macd_signal()) if macd_obj is not None else None
    macd_hist = _last(macd_obj.macd_diff()) if macd_obj is not None else None

    out["momentum"] = {
        "rsi_14": rsi_val,
        "rsi_state": (
            None
            if rsi_val is None
            else "overbought"
            if rsi_val >= 70
            else "oversold"
            if rsi_val <= 30
            else "neutral"
        ),
        "macd": macd_line,
        "macd_signal": macd_sig,
        "macd_histogram": macd_hist,
        "macd_state": _crossover_state(macd_line, macd_sig),
        "stoch_k": _last(stoch.stoch()) if stoch is not None else None,
        "stoch_d": _last(stoch.stoch_signal()) if stoch is not None else None,
        "cci_20": _last(cci),
    }

    # --- Volatility & channels ---
    bb = _safe(lambda: BollingerBands(close, BB_PERIOD, BB_STD))
    atr_val = _last(_safe(lambda: AverageTrueRange(high, low, close, ATR_PERIOD).average_true_range()))
    adx_obj = _safe(lambda: ADXIndicator(high, low, close, ADX_PERIOD))
    adx_val = _last(adx_obj.adx()) if adx_obj is not None else None

    out["volatility"] = {
        "bb_upper": _last(bb.bollinger_hband()) if bb is not None else None,
        "bb_middle": _last(bb.bollinger_mavg()) if bb is not None else None,
        "bb_lower": _last(bb.bollinger_lband()) if bb is not None else None,
        "bb_percent_b": _last(bb.bollinger_pband()) if bb is not None else None,
        "bb_bandwidth": _last(bb.bollinger_wband()) if bb is not None else None,
        "atr_14": atr_val,
        "atr_percent": (atr_val / price * 100) if atr_val and price else None,
        "adx_14": adx_val,
        "adx_state": (
            None
            if adx_val is None
            else "strong_trend"
            if adx_val >= 25
            else "weak_trend"
            if adx_val >= 20
            else "ranging"
        ),
        "plus_di": _last(adx_obj.adx_pos()) if adx_obj is not None else None,
        "minus_di": _last(adx_obj.adx_neg()) if adx_obj is not None else None,
    }

    # --- Volume ---
    obv = _safe(lambda: OnBalanceVolumeIndicator(close, vol).on_balance_volume())
    vwap = _safe(
        lambda: VolumeWeightedAveragePrice(high, low, close, vol, VWAP_PERIOD)
        .volume_weighted_average_price()
    )

    obv_series = obv.dropna() if obv is not None else None
    obv_trend = None
    if obv_series is not None and len(obv_series) > VOLUME_LOOKBACK:
        recent = obv_series.iloc[-VOLUME_LOOKBACK:]
        obv_trend = "rising" if recent.iloc[-1] > recent.iloc[0] else "falling"

    avg_vol = float(vol.iloc[-VOLUME_LOOKBACK:].mean()) if len(vol) >= VOLUME_LOOKBACK else None
    last_vol = float(vol.iloc[-1])
    ratio = (last_vol / avg_vol) if avg_vol else None

    out["volume"] = {
        "last_volume": last_vol,
        "avg_volume_20d": avg_vol,
        "volume_ratio": ratio,
        "volume_spike": bool(ratio and ratio >= VOLUME_SPIKE_MULTIPLE),
        "obv": _last(obv),
        "obv_trend_20d": obv_trend,
        "vwap_14": _last(vwap),
        "price_vs_vwap": _crossover_state(price, _last(vwap)),
    }

    # --- Structure ---
    out["levels"] = support_resistance(df)

    # --- Recent performance ---
    perf = {}
    for label, days in (("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)):
        if len(close) > days:
            prior = float(close.iloc[-days - 1])
            if prior:
                perf[label] = (price / prior - 1) * 100
    out["performance_pct"] = perf

    if len(close) >= 252:
        year = close.iloc[-252:]
        hi, lo = float(year.max()), float(year.min())
        out["range_52w"] = {
            "high": hi,
            "low": lo,
            "pct_from_high": (price / hi - 1) * 100 if hi else None,
            "pct_from_low": (price / lo - 1) * 100 if lo else None,
        }

    return out
