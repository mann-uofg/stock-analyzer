"""Deterministic scoring and trade construction.

Design decision: the verdict, the conviction score and every price level are
computed **here, in Python** - not by the language model. LLMs are unreliable
at arithmetic, and a stop-loss that is wrong by a decimal place is a real
financial loss. The model's job (see ``llm.py``) is to explain these numbers,
never to produce them.

A consequence worth stating: this module is fully functional with no LLM
installed at all.

Each bucket returns a score in [-1, 1] together with the human-readable reasons
that produced it, so every point of the composite is auditable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    ATR_STOP_MULTIPLE,
    MAX_STOP_ATR,
    MIN_RISK_REWARD,
    SCORE_WEIGHTS,
    VERDICT_BANDS,
)

Bucket = tuple[float, list[str]]


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _get(d: dict | None, *path, default=None):
    """Nested lookup that tolerates missing branches."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


# --- Individual buckets ---------------------------------------------------


def score_trend(tech: dict) -> Bucket:
    score, notes = 0.0, []
    ma = tech.get("moving_averages", {})

    alignment = ma.get("alignment")
    if alignment == "bullish_stacked":
        score += 0.40
        notes.append("MAs stacked bullishly (20 > 50 > 200)")
    elif alignment == "bearish_stacked":
        score -= 0.40
        notes.append("MAs stacked bearishly (20 < 50 < 200)")

    for key, weight, label in (
        ("price_vs_sma200", 0.25, "200-day"),
        ("price_vs_sma50", 0.15, "50-day"),
        ("price_vs_sma20", 0.10, "20-day"),
    ):
        state = ma.get(key)
        if state == "above":
            score += weight
            notes.append(f"Price above {label} SMA")
        elif state == "below":
            score -= weight
            notes.append(f"Price below {label} SMA")

    cross = ma.get("golden_death_cross", {}) or {}
    if cross.get("state") == "golden":
        score += 0.10
        if cross.get("event") == "golden_cross" and (cross.get("bars_ago") or 999) < 30:
            score += 0.10
            notes.append(f"Golden cross {cross['bars_ago']} bars ago")
        else:
            notes.append("50/200 in golden-cross regime")
    elif cross.get("state") == "death":
        score -= 0.10
        if cross.get("event") == "death_cross" and (cross.get("bars_ago") or 999) < 30:
            score -= 0.10
            notes.append(f"Death cross {cross['bars_ago']} bars ago")
        else:
            notes.append("50/200 in death-cross regime")

    # ADX gates trend conviction: a weak trend should not score like a strong one.
    vol = tech.get("volatility", {})
    adx, pdi, ndi = vol.get("adx_14"), vol.get("plus_di"), vol.get("minus_di")
    if adx is not None and adx < 20:
        score *= 0.7
        notes.append(f"ADX {adx:.1f} - trend is weak/ranging, signal discounted")
    elif adx is not None and adx >= 25 and pdi is not None and ndi is not None:
        direction = 0.15 if pdi > ndi else -0.15
        score += direction
        notes.append(
            f"ADX {adx:.1f} confirms a strong {'up' if pdi > ndi else 'down'}trend"
        )

    return _clamp(score), notes


def score_momentum(tech: dict) -> Bucket:
    score, notes = 0.0, []
    mom = tech.get("momentum", {})

    rsi = mom.get("rsi_14")
    if rsi is not None:
        if rsi >= 70:
            score -= 0.15
            notes.append(f"RSI {rsi:.1f} overbought - stretched, pullback risk")
        elif rsi >= 55:
            score += 0.30
            notes.append(f"RSI {rsi:.1f} in bullish territory")
        elif rsi <= 30:
            score += 0.15
            notes.append(f"RSI {rsi:.1f} oversold - mean-reversion potential")
        elif rsi <= 45:
            score -= 0.30
            notes.append(f"RSI {rsi:.1f} in bearish territory")

    macd_state, hist = mom.get("macd_state"), mom.get("macd_histogram")
    if macd_state == "above":
        score += 0.30
        notes.append("MACD above signal line")
    elif macd_state == "below":
        score -= 0.30
        notes.append("MACD below signal line")
    if hist is not None:
        score += 0.10 if hist > 0 else -0.10
        notes.append(f"MACD histogram {'positive' if hist > 0 else 'negative'} ({hist:.2f})")

    k = mom.get("stoch_k")
    if k is not None:
        if k >= 80:
            score -= 0.10
            notes.append(f"Stochastic {k:.0f} overbought")
        elif k <= 20:
            score += 0.10
            notes.append(f"Stochastic {k:.0f} oversold")

    cci = mom.get("cci_20")
    if cci is not None:
        if cci > 100:
            score += 0.15
            notes.append(f"CCI {cci:.0f} confirms strength")
        elif cci < -100:
            score -= 0.15
            notes.append(f"CCI {cci:.0f} confirms weakness")

    return _clamp(score), notes


def score_volatility(tech: dict, risk_panel: dict) -> Bucket:
    """Volatility is a risk/positioning read, so it scores mildly by design."""
    score, notes = 0.0, []
    vol = tech.get("volatility", {})

    pct_b = vol.get("bb_percent_b")
    if pct_b is not None:
        if pct_b > 1:
            score -= 0.30
            notes.append(f"Price above upper Bollinger band (%B {pct_b:.2f}) - extended")
        elif pct_b < 0:
            score += 0.30
            notes.append(f"Price below lower Bollinger band (%B {pct_b:.2f}) - washed out")
        elif pct_b > 0.8:
            score -= 0.10
            notes.append(f"Upper Bollinger quartile (%B {pct_b:.2f})")
        elif pct_b < 0.2:
            score += 0.10
            notes.append(f"Lower Bollinger quartile (%B {pct_b:.2f})")

    atr_pct = vol.get("atr_percent")
    if atr_pct is not None and atr_pct > 5:
        score -= 0.25
        notes.append(f"ATR {atr_pct:.1f}% of price - elevated daily range, size down")

    hv_pctile = _get(risk_panel, "volatility", "hv_percentile_1y")
    if hv_pctile is not None:
        if hv_pctile > 80:
            score -= 0.20
            notes.append(f"Realised vol in {hv_pctile:.0f}th percentile - unusually turbulent")
        elif hv_pctile < 20:
            score += 0.15
            notes.append(f"Realised vol in {hv_pctile:.0f}th percentile - calm regime")

    dd = risk_panel.get("max_drawdown_1y_pct")
    if dd is not None and dd < -35:
        score -= 0.15
        notes.append(f"1y max drawdown {dd:.1f}% - historically punishing")

    return _clamp(score), notes


def score_volume(tech: dict) -> Bucket:
    score, notes = 0.0, []
    v = tech.get("volume", {})

    if v.get("obv_trend_20d") == "rising":
        score += 0.35
        notes.append("OBV rising - accumulation")
    elif v.get("obv_trend_20d") == "falling":
        score -= 0.35
        notes.append("OBV falling - distribution")

    if v.get("price_vs_vwap") == "above":
        score += 0.25
        notes.append("Price above 14-day VWAP")
    elif v.get("price_vs_vwap") == "below":
        score -= 0.25
        notes.append("Price below 14-day VWAP")

    ratio = v.get("volume_ratio")
    if v.get("volume_spike"):
        # A spike confirms whatever the trend already is; direction comes from OBV.
        direction = 1 if v.get("obv_trend_20d") == "rising" else -1
        score += 0.30 * direction
        notes.append(f"Volume spike {ratio:.1f}x the 20-day average")
    elif ratio is not None and ratio < 0.6:
        score -= 0.10
        notes.append(f"Volume {ratio:.1f}x average - conviction is thin")

    return _clamp(score), notes


def score_fundamental(fund: dict) -> Bucket:
    score, notes = 0.0, []
    val = fund.get("valuation", {})
    earn = fund.get("earnings", {})
    cons = fund.get("consensus", {})

    peg = val.get("peg_ratio")
    if peg is not None and peg > 0:
        if peg < 1:
            score += 0.30
            notes.append(f"PEG {peg:.2f} - growth is cheap relative to earnings")
        elif peg > 3:
            score -= 0.25
            notes.append(f"PEG {peg:.2f} - expensive versus growth")

    trailing, forward = val.get("trailing_pe"), val.get("forward_pe")
    if forward is not None and forward > 0 and trailing is not None:
        if trailing <= 0:
            # A negative trailing P/E means the company lost money over the last
            # twelve months. A positive forward P/E therefore marks an expected
            # return to profit - comparing the two as magnitudes would read that
            # turnaround as deterioration.
            score += 0.20
            notes.append(
                f"Forward P/E {forward:.1f} against a loss-making trailing year - "
                "consensus expects a return to profit"
            )
        elif forward < trailing * 0.8:
            score += 0.20
            notes.append(
                f"Forward P/E {forward:.1f} well below trailing {trailing:.1f} - "
                "strong earnings growth priced in"
            )
        elif forward > trailing:
            score -= 0.15
            notes.append(f"Forward P/E {forward:.1f} above trailing {trailing:.1f} - earnings expected to fall")
    elif forward is not None and forward < 0:
        score -= 0.20
        notes.append(
            f"Forward P/E {forward:.1f} - consensus expects losses to continue"
        )

    fcf_yield = val.get("fcf_yield_pct")
    if fcf_yield is not None:
        if fcf_yield > 5:
            score += 0.20
            notes.append(f"FCF yield {fcf_yield:.1f}% - strong cash generation")
        elif fcf_yield < 1:
            score -= 0.10
            notes.append(f"FCF yield {fcf_yield:.1f}% - little cash return at this price")

    rev_growth = val.get("revenue_yoy_growth_pct")
    if rev_growth is not None:
        if rev_growth > 20:
            score += 0.25
            notes.append(f"Revenue +{rev_growth:.0f}% YoY")
        elif rev_growth < 0:
            score -= 0.25
            notes.append(f"Revenue {rev_growth:.0f}% YoY - contracting")

    beat_rate = earn.get("beat_rate_pct")
    avg_surprise = earn.get("avg_eps_surprise_pct")
    if beat_rate is not None and avg_surprise is not None:
        if beat_rate >= 75 and avg_surprise > 0:
            score += 0.20
            notes.append(f"Beat EPS in {beat_rate:.0f}% of last quarters (avg +{avg_surprise:.1f}%)")
        elif beat_rate <= 25:
            score -= 0.20
            notes.append(f"Missed EPS in {100 - beat_rate:.0f}% of last quarters")

    targets = cons.get("price_targets") or {}
    mean_target, current = targets.get("mean"), targets.get("current")
    if mean_target and current:
        upside = (mean_target / current - 1) * 100
        if upside > 15:
            score += 0.20
            notes.append(f"Analyst mean target implies {upside:+.0f}% upside")
        elif upside < -5:
            score -= 0.20
            notes.append(f"Trading above analyst mean target ({upside:+.0f}%)")

    return _clamp(score), notes


def score_options(opts: dict) -> Bucket:
    score, notes = 0.0, []
    if not opts.get("available"):
        return 0.0, ["No option data available for this symbol"]

    pcr = opts.get("put_call_ratio", {})
    ratio = pcr.get("open_interest")
    basis = "open interest"
    if ratio is None:
        ratio = pcr.get("volume")
        basis = "volume"

    if ratio is not None:
        if ratio < 0.7:
            score += 0.35
            notes.append(f"Put/call ratio {ratio:.2f} ({basis}) - call-heavy positioning")
        elif ratio > 1.3:
            score -= 0.35
            notes.append(f"Put/call ratio {ratio:.2f} ({basis}) - put-heavy/hedged")

    gamma = opts.get("gamma_exposure", {})
    if gamma.get("gamma_squeeze_flag"):
        score += 0.30
        notes.append(
            f"Gamma squeeze setup: call gamma clustered at {gamma.get('peak_call_strike')}"
        )
    net = gamma.get("net_gamma")
    if net is not None and net != 0:
        score += 0.15 if net > 0 else -0.15
        notes.append(f"Net NTM gamma {'call' if net > 0 else 'put'}-skewed")

    iv_hv = _get(opts, "iv_context", "iv_hv_ratio")
    if iv_hv is not None:
        if iv_hv > 1.3:
            score -= 0.15
            notes.append(f"IV/HV {iv_hv:.2f} - options rich, premium selling favoured")
        elif iv_hv < 0.9:
            score += 0.10
            notes.append(f"IV/HV {iv_hv:.2f} - options cheap versus realised vol")

    return _clamp(score), notes


# --- Trade construction ---------------------------------------------------


def build_trade_setup(
    price: float, tech: dict, direction: str, atr: float | None,
    range_52w: dict | None = None,
) -> dict[str, Any]:
    """Derive entry, stop and targets enforcing the minimum risk/reward.

    Long setups anchor the stop below the nearest support (or an ATR multiple,
    whichever is further from price, so noise does not trigger it). Targets
    prefer real resistance; if the nearest resistance fails the R/R floor, the
    target is pushed out to exactly satisfy it and flagged as synthetic.
    """
    levels = tech.get("levels", {})
    support = levels.get("support") or []
    resistance = levels.get("resistance") or []

    if not atr or atr <= 0 or not price or price <= 0:
        return {"valid": False, "reason": "ATR or price unavailable - cannot size risk"}

    if direction == "long":
        entry_low, entry_high = price - 0.5 * atr, price + 0.25 * atr

        # The ATR stop is the noise floor: never risk less than this, or normal
        # daily range stops us out. Structure only *widens* it, and only when
        # the level is close enough to be worth respecting.
        atr_stop = entry_low - ATR_STOP_MULTIPLE * atr
        support_stop = (max(support) * 0.995) if support else None
        stop = atr_stop
        if support_stop is not None and support_stop < entry_low:
            if (entry_low - support_stop) <= MAX_STOP_ATR * atr:
                stop = min(atr_stop, support_stop)

        risk = entry_low - stop
        if risk <= 0:
            return {"valid": False, "reason": "computed non-positive risk"}

        min_target = entry_high + MIN_RISK_REWARD * risk
        viable = [r for r in resistance if r > entry_high]
        target1 = viable[0] if viable else min_target
        synthetic = False
        if (target1 - entry_high) / risk < MIN_RISK_REWARD:
            target1, synthetic = min_target, True
        target2 = (
            viable[1] if len(viable) > 1 and viable[1] > target1 else entry_high + 3.5 * risk
        )
    else:
        entry_low, entry_high = price - 0.25 * atr, price + 0.5 * atr

        atr_stop = entry_high + ATR_STOP_MULTIPLE * atr
        resistance_stop = (min(resistance) * 1.005) if resistance else None
        stop = atr_stop
        if resistance_stop is not None and resistance_stop > entry_high:
            if (resistance_stop - entry_high) <= MAX_STOP_ATR * atr:
                stop = max(atr_stop, resistance_stop)

        risk = stop - entry_high
        if risk <= 0:
            return {"valid": False, "reason": "computed non-positive risk"}

        min_target = entry_low - MIN_RISK_REWARD * risk
        viable = [s for s in support if s < entry_low]
        target1 = viable[0] if viable else min_target
        synthetic = False
        if (entry_low - target1) / risk < MIN_RISK_REWARD:
            target1, synthetic = min_target, True
        target2 = (
            viable[1] if len(viable) > 1 and viable[1] < target1 else entry_low - 3.5 * risk
        )

    anchor = entry_high if direction == "long" else entry_low
    reward1 = abs(target1 - anchor)
    reward2 = abs(target2 - anchor)

    # A target beyond the yearly extreme is reachable only on a genuine
    # breakout, which is a materially different bet - surface it rather than
    # presenting the level as routine.
    breakout_required = False
    if range_52w:
        high, low = range_52w.get("high"), range_52w.get("low")
        if direction == "long" and high:
            breakout_required = target1 > high
        elif direction == "short" and low:
            breakout_required = target1 < low

    return {
        "valid": True,
        "direction": direction,
        "target_1_requires_52w_breakout": breakout_required,
        "entry_low": round(float(min(entry_low, entry_high)), 2),
        "entry_high": round(float(max(entry_low, entry_high)), 2),
        "stop_loss": round(float(stop), 2),
        "risk_per_share": round(float(risk), 2),
        "risk_pct": round(float(risk / price * 100), 2),
        "target_1": round(float(target1), 2),
        "target_2": round(float(target2), 2),
        "risk_reward_t1": round(float(reward1 / risk), 2),
        "risk_reward_t2": round(float(reward2 / risk), 2),
        "target_1_synthetic": synthetic,
        "atr_used": round(float(atr), 2),
        "basis": (
            f"Stop = {ATR_STOP_MULTIPLE}x ATR beyond entry, widened to clear the "
            "nearest structural level; targets require at least "
            f"{MIN_RISK_REWARD}:1 reward-to-risk."
        ),
    }


# --- Composite ------------------------------------------------------------


def compute(
    tech: dict, risk_panel: dict, opts: dict, fund: dict, price: float
) -> dict[str, Any]:
    """Blend every bucket into a verdict, a conviction score and a trade plan."""
    buckets: dict[str, Bucket] = {
        "trend": score_trend(tech),
        "momentum": score_momentum(tech),
        "volatility": score_volatility(tech, risk_panel),
        "volume": score_volume(tech),
        "fundamental": score_fundamental(fund),
        "options": score_options(opts),
    }

    # An ETF, a fund or a coin has no fundamentals, and a symbol with no listed
    # options has no positioning data. Scoring those buckets as 0.0 is not
    # neutral - it silently spends 25% of the weight on nothing and drags every
    # such instrument toward HOLD. Weights are renormalised over the buckets
    # that actually had data, which is also what horizon.py does.
    valuation = fund.get("valuation") or {}
    fundamental_available = bool(
        (fund.get("earnings") or {}).get("history")
        or any(
            valuation.get(key) is not None
            for key in ("trailing_pe", "forward_pe", "peg_ratio", "fcf_yield_pct",
                        "revenue_yoy_growth_pct", "price_to_sales", "ev_to_ebitda")
        )
    )
    unavailable = set()
    if not fundamental_available:
        unavailable.add("fundamental")
    if not opts.get("available"):
        unavailable.add("options")

    effective_weights = {
        name: weight for name, weight in SCORE_WEIGHTS.items() if name not in unavailable
    }
    total_weight = sum(effective_weights.values()) or 1.0
    effective_weights = {n: w / total_weight for n, w in effective_weights.items()}

    composite = sum(
        weight * buckets[name][0] for name, weight in effective_weights.items()
    )
    composite = _clamp(composite)
    score_100 = (composite + 1) * 50

    verdict = next(label for threshold, label in VERDICT_BANDS if score_100 >= threshold)

    # Conviction reflects agreement between buckets and the completeness of the
    # data, not merely the size of the score. Directionally split signals should
    # not produce a confident call.
    directional = [
        v for name, (v, _) in buckets.items()
        if name != "volatility" and name not in unavailable
    ]
    dispersion = float(np.std(directional)) if directional else 0.0
    agreement = _clamp(1 - dispersion, 0.0, 1.0)

    penalties: list[str] = []
    completeness = 1.0
    if not opts.get("available"):
        completeness -= 0.10
        penalties.append("no option data")
    if not fund.get("earnings", {}).get("history"):
        completeness -= 0.10
        penalties.append("no earnings history")
    if tech.get("bars", 0) < 252:
        completeness -= 0.10
        penalties.append("under 1y of price history")

    strength = abs(score_100 - 50) / 50  # 0 at neutral, 1 at an extreme
    conviction = 100 * strength * agreement * max(completeness, 0.5)
    conviction = float(np.clip(conviction, 0, 100))

    # --- Catalyst risk ---
    #
    # ``warnings`` is everything worth showing the user. ``event_risks`` is the
    # subset that genuinely argues against holding a position, and is the only
    # part the narrative layer may cite as directional counter-evidence - a
    # data-completeness notice is not a bear case.
    warnings_out: list[str] = []
    event_risks: list[str] = []

    def _risk(message: str) -> None:
        warnings_out.append(message)
        event_risks.append(message)

    days_to_earnings = _get(fund, "earnings", "days_to_earnings")
    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        conviction *= 0.75
        _risk(f"Earnings in {days_to_earnings} days - binary event risk; conviction discounted 25%")
    elif days_to_earnings is not None and days_to_earnings <= 21:
        _risk(f"Earnings in {days_to_earnings} days - size positions accordingly")

    avg_move = _get(fund, "earnings", "avg_abs_post_earnings_move_pct")
    if avg_move and days_to_earnings is not None and days_to_earnings <= 21:
        _risk(f"Historical post-earnings move averages {avg_move:.1f}% in absolute terms")

    if penalties:
        warnings_out.append("Reduced data completeness: " + ", ".join(penalties))

    direction = "long" if composite >= 0 else "short"
    atr = _get(tech, "volatility", "atr_14")
    setup = build_trade_setup(price, tech, direction, atr, tech.get("range_52w"))
    if setup.get("target_1_requires_52w_breakout"):
        _risk(
            "First target sits beyond the 52-week extreme - it requires a "
            "breakout, not a continuation."
        )
    if verdict == "HOLD":
        warnings_out.insert(
            0,
            f"Verdict is HOLD - the {direction} setup below is indicative of the "
            "marginal lean only, not an actionable signal.",
        )

    return {
        "verdict": verdict,
        "score_0_100": round(score_100, 1),
        "conviction_pct": round(conviction, 1),
        "composite_raw": round(composite, 4),
        "agreement": round(agreement, 3),
        "buckets": {
            name: {
                "score": round(value, 3),
                # The effective weight after renormalisation, so the
                # contributions shown actually sum to the composite.
                "weight": round(effective_weights.get(name, 0.0), 3),
                "contribution": round(effective_weights.get(name, 0.0) * value, 4),
                "available": name not in unavailable,
                "reasons": (
                    ["No data for this instrument - excluded from the score"]
                    if name in unavailable
                    else reasons or ["No signal - readings are mid-range"]
                ),
            }
            for name, (value, reasons) in buckets.items()
        },
        "trade_setup": setup,
        "warnings": warnings_out,
        "event_risks": event_risks,
    }
