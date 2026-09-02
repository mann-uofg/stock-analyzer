"""Option chain analysis, Black-Scholes Greeks, and implied-volatility solving.

Greeks are computed locally with scipy rather than via py_vollib: the closed
form is short, has no fragile transitive dependencies, and lets us control the
degenerate cases (expiry today, zero IV, missing quotes) explicitly. The
implementation is verified against py_vollib to 1e-9 in ``tests/``.

DATA-QUALITY NOTE
-----------------
Yahoo's free option endpoint has degraded: for many symbols it now returns
``openInterest == 0``, ``bid == ask == 0`` and a quantized placeholder
``impliedVolatility`` (values such as 0.00001, 0.03126, 0.12501 - powers of two,
not market vol). Only ``lastPrice`` and ``volume`` remain trustworthy.

This module therefore:
  * validates Yahoo's IV and, when implausible, **solves IV numerically** from
    the traded price by inverting Black-Scholes with Brent's method;
  * reports open-interest metrics as *unavailable* rather than emitting a
    misleading zero, falling back to volume-weighted gamma;
  * surfaces a ``data_quality`` block so the report never implies more
    precision than the underlying feed supports.

Conventions:
  * ``vega``  is per 1 percentage-point change in IV.
  * ``theta`` is per calendar day.
  * ``T``     is in years, ACT/365.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .config import IV_LOOKBACK_DAYS, NTM_BAND

# Bounds outside which a quoted IV is treated as a feed artefact.
_IV_MIN_PLAUSIBLE = 0.02   # 2% annualised
_IV_MAX_PLAUSIBLE = 5.00   # 500% annualised


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, kind: str = "c", q: float = 0.0
) -> float:
    """Black-Scholes-Merton price for a European option."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
        return float(intrinsic)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if kind == "c":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, kind: str = "c", q: float = 0.0
) -> dict[str, float | None]:
    """Analytic Black-Scholes-Merton price and Greeks.

    Returns ``None`` values when the inputs are degenerate (non-positive spot,
    strike, time, or volatility) rather than emitting inf/nan.
    """
    kind = kind.lower()[0]
    empty = {"price": None, "delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or not all(np.isfinite([S, K, T, r, sigma])):
        return empty

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t / 100.0  # per 1% vol

    if kind == "c":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        ) / 365.0
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        ) / 365.0
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    kind: str = "c", q: float = 0.0,
) -> float | None:
    """Invert Black-Scholes for sigma via Brent's method.

    Returns ``None`` when the price violates no-arbitrage bounds (which is
    common for stale last-trade prints) or the solver fails to bracket a root.
    """
    kind = kind.lower()[0]
    if not all(np.isfinite([market_price, S, K, T, r])) or market_price <= 0 or T <= 0 or S <= 0:
        return None

    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "c":
        lower_bound, upper_bound = max(S * disc_q - K * disc_r, 0.0), S * disc_q
    else:
        lower_bound, upper_bound = max(K * disc_r - S * disc_q, 0.0), K * disc_r

    # A price at or below intrinsic carries no time value: sigma is undefined.
    tol = 1e-6
    if market_price <= lower_bound + tol or market_price >= upper_bound - tol:
        return None

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, kind, q) - market_price

    try:
        lo, hi = 1e-4, _IV_MAX_PLAUSIBLE
        if objective(lo) * objective(hi) > 0:
            return None
        return float(brentq(objective, lo, hi, xtol=1e-6, maxiter=100))
    except (ValueError, RuntimeError):
        return None


def _years_to_expiry(expiry: str) -> float:
    """ACT/365 year fraction to expiry, floored just above zero."""
    try:
        exp = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    except ValueError:
        return 0.0
    days = (exp - dt.date.today()).days
    # Same-day expiry still has intraday value; treat as a few hours.
    return max(days, 0.25) / 365.0


def _pick_expiry(expiries: list[str], min_days: int = 20, max_days: int = 60) -> str | None:
    """Choose a liquid monthly-ish expiry: first one at least ``min_days`` out."""
    if not expiries:
        return None
    today = dt.date.today()
    dated = []
    for e in expiries:
        try:
            d = (dt.datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except ValueError:
            continue
        dated.append((d, e))
    if not dated:
        return None
    dated.sort()
    for days, e in dated:
        if min_days <= days <= max_days:
            return e
    for days, e in dated:
        if days >= min_days:
            return e
    return dated[-1][1]  # everything is near-dated; take the furthest


def _clean_chain(
    df: pd.DataFrame, spot: float, T: float, r: float, kind: str
) -> pd.DataFrame:
    """Normalise a raw yfinance option frame and repair implied volatility.

    Adds ``mid``, ``iv_used`` and ``iv_source`` columns. ``iv_source`` is one of
    ``yahoo`` (quoted value was plausible), ``solved`` (recovered from the
    traded price) or ``none``.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for col in ("strike", "lastPrice", "bid", "ask", "impliedVolatility",
                "openInterest", "volume"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["openInterest"] = out["openInterest"].fillna(0)
    out["volume"] = out["volume"].fillna(0)

    # A mid price is far more trustworthy than lastPrice - but Yahoo now
    # frequently returns bid=ask=0, in which case lastPrice is all we have.
    has_quote = (out["bid"] > 0) & (out["ask"] > 0)
    out["mid"] = np.where(has_quote, (out["bid"] + out["ask"]) / 2, out["lastPrice"])

    iv_used: list[float | None] = []
    iv_source: list[str] = []
    for _, row in out.iterrows():
        quoted = row["impliedVolatility"]
        price = row["mid"]
        strike = row["strike"]
        has_price = pd.notna(price) and price > 0 and pd.notna(strike)

        # Prefer IV solved from the traded price over Yahoo's quoted value.
        #
        # Solving is cheap and strictly more trustworthy: it is derived from
        # observed market data through the same model that then produces the
        # Greeks, so price and Greeks stay mutually consistent. Validating the
        # quoted value by repricing instead does not work - for deep in- or
        # out-of-the-money strikes vega is tiny, so even a placeholder
        # volatility reprices within tolerance while yielding badly wrong
        # Greeks. The quote is therefore only a fallback for strikes we cannot
        # solve (no price, or a price outside the no-arbitrage bounds).
        solved = (
            implied_volatility(float(price), spot, float(strike), T, r, kind)
            if has_price
            else None
        )
        if solved is not None:
            iv_used.append(solved)
            iv_source.append("solved")
        elif pd.notna(quoted) and _IV_MIN_PLAUSIBLE <= quoted <= _IV_MAX_PLAUSIBLE:
            iv_used.append(float(quoted))
            iv_source.append("yahoo")
        else:
            iv_used.append(None)
            iv_source.append("none")

    out["iv_used"] = iv_used
    out["iv_source"] = iv_source
    return out.dropna(subset=["strike"])


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float | None:
    """Average of the call and put IV at the strike nearest spot."""
    ivs = []
    for frame in (calls, puts):
        if frame.empty or "iv_used" not in frame.columns:
            continue
        valid = frame[frame["iv_used"].notna()]
        if valid.empty:
            continue
        idx = (valid["strike"] - spot).abs().idxmin()
        ivs.append(float(valid.loc[idx, "iv_used"]))
    return float(np.mean(ivs)) if ivs else None


def analyse(
    ticker: str,
    spot: float,
    expiries: list[str],
    chain_loader: Callable[[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    risk_free: float,
    hv_series: pd.Series | None = None,
    hv_current: float | None = None,
) -> dict[str, Any]:
    """Build the derivatives panel.

    ``chain_loader`` is a callable ``(ticker, expiry) -> (calls, puts)`` so this
    module stays independent of the data layer and is trivially testable.
    """
    if not expiries:
        return {"available": False, "reason": "no listed options for this symbol"}

    expiry = _pick_expiry(expiries)
    if expiry is None:
        return {"available": False, "reason": "no usable expiry dates"}

    raw_calls, raw_puts = chain_loader(ticker, expiry)
    if (raw_calls is None or raw_calls.empty) and (raw_puts is None or raw_puts.empty):
        return {"available": False, "reason": f"empty option chain for {expiry}"}

    T = _years_to_expiry(expiry)
    dte = max((dt.datetime.strptime(expiry, "%Y-%m-%d").date() - dt.date.today()).days, 0)

    calls = _clean_chain(raw_calls, spot, T, risk_free, "c")
    puts = _clean_chain(raw_puts, spot, T, risk_free, "p")

    if calls.empty and puts.empty:
        return {"available": False, "reason": f"unusable option chain for {expiry}"}

    # --- Feed quality assessment -----------------------------------------
    frames = [f for f in (calls, puts) if not f.empty]
    total_oi = float(sum(f["openInterest"].sum() for f in frames))
    total_vol = float(sum(f["volume"].sum() for f in frames))
    quote_frac = float(
        np.mean([((f["bid"] > 0) & (f["ask"] > 0)).mean() for f in frames])
    ) if frames else 0.0
    sources = pd.concat([f["iv_source"] for f in frames]) if frames else pd.Series(dtype=str)
    solved_frac = float((sources == "solved").mean()) if len(sources) else 0.0
    iv_missing_frac = float((sources == "none").mean()) if len(sources) else 1.0

    oi_available = total_oi > 0
    volume_available = total_vol > 0
    data_quality = {
        "open_interest_available": oi_available,
        "volume_available": volume_available,
        "quotes_available_fraction": quote_frac,
        "iv_solved_fraction": solved_frac,
        "iv_unavailable_fraction": iv_missing_frac,
        "notes": [],
    }
    if not oi_available and volume_available:
        data_quality["notes"].append(
            "Yahoo returned zero open interest across the chain; OI-based "
            "put/call ratio and gamma exposure are unavailable. Volume-based "
            "equivalents are reported instead."
        )
    elif not oi_available and not volume_available:
        # With neither measure of size, positioning cannot be inferred at all -
        # weighting by a column of zeros would report a confident 0.0 rather
        # than an absence.
        data_quality["notes"].append(
            "Neither open interest nor volume was returned for this expiry, so "
            "put/call ratios and gamma positioning cannot be computed. The "
            "chain is priced but shows no trading activity."
        )
    if solved_frac > 0.5:
        data_quality["notes"].append(
            "Quoted implied volatility was implausible; IV was solved "
            "numerically from traded prices via Black-Scholes inversion."
        )
    if quote_frac < 0.1:
        data_quality["notes"].append(
            "Bid/ask spreads were not returned; last traded price was used, "
            "which may be stale outside market hours."
        )

    lo, hi = spot * (1 - NTM_BAND), spot * (1 + NTM_BAND)

    def _ntm(frame: pd.DataFrame, kind: str) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        band = frame[(frame["strike"] >= lo) & (frame["strike"] <= hi)]
        if band.empty:
            band = frame.iloc[(frame["strike"] - spot).abs().argsort()[:8]]
        rows = []
        for _, r in band.sort_values("strike").iterrows():
            iv = r["iv_used"] if pd.notna(r["iv_used"]) else None
            greeks = (
                black_scholes_greeks(spot, float(r["strike"]), T, risk_free, float(iv), kind)
                if iv
                else {k: None for k in ("price", "delta", "gamma", "theta", "vega", "rho")}
            )
            rows.append(
                {
                    "strike": float(r["strike"]),
                    "mid": None if pd.isna(r["mid"]) else float(r["mid"]),
                    "bid": None if pd.isna(r["bid"]) else float(r["bid"]),
                    "ask": None if pd.isna(r["ask"]) else float(r["ask"]),
                    "iv_pct": float(iv) * 100 if iv else None,
                    "iv_source": r["iv_source"],
                    "open_interest": int(r["openInterest"]),
                    "volume": int(r["volume"]),
                    "model_price": greeks["price"],
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "theta": greeks["theta"],
                    "vega": greeks["vega"],
                }
            )
        return rows

    ntm_calls, ntm_puts = _ntm(calls, "c"), _ntm(puts, "p")

    # --- Put/Call ratios across the whole expiry, not just the NTM band ---
    call_oi = float(calls["openInterest"].sum()) if not calls.empty else 0.0
    put_oi = float(puts["openInterest"].sum()) if not puts.empty else 0.0
    call_vol = float(calls["volume"].sum()) if not calls.empty else 0.0
    put_vol = float(puts["volume"].sum()) if not puts.empty else 0.0

    atm_iv = _atm_iv(calls, puts, spot)
    atm_iv_pct = atm_iv * 100 if atm_iv else None

    # --- IV context ---
    #
    # Free data sources expose no historical implied-vol surface, so a true
    # 52-week IV Rank is not computable. We instead rank current ATM IV inside
    # the trailing realised-volatility distribution and report the IV/HV
    # premium. Both are labelled as proxies so they are not mistaken for
    # broker-grade IV Rank.
    iv_rank = iv_percentile = iv_hv_ratio = None
    if atm_iv_pct is not None and hv_series is not None and len(hv_series) >= 60:
        window = hv_series.iloc[-IV_LOOKBACK_DAYS:]
        lo_hv, hi_hv = float(window.min()), float(window.max())
        if hi_hv > lo_hv:
            iv_rank = float(np.clip((atm_iv_pct - lo_hv) / (hi_hv - lo_hv), 0, 1) * 100)
        iv_percentile = float((window < atm_iv_pct).mean() * 100)
    if atm_iv_pct is not None and hv_current:
        iv_hv_ratio = atm_iv_pct / hv_current

    # --- Gamma / open-interest concentration ---
    #
    # Dealer gamma exposure is normally OI * gamma * 100 shares. With OI
    # unavailable we weight by traded volume instead, which captures the same
    # positioning signal for the current session, and say so explicitly.
    weight_field = "open_interest" if oi_available else "volume"

    def _total_gamma(rows: list[dict[str, Any]]) -> float:
        return float(sum((r["gamma"] or 0) * r[weight_field] * 100 for r in rows))

    call_gamma, put_gamma = _total_gamma(ntm_calls), _total_gamma(ntm_puts)

    def _peak(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates = [r for r in rows if r[weight_field] > 0]
        return max(candidates, key=lambda r: r[weight_field], default=None)

    max_call, max_put = _peak(ntm_calls), _peak(ntm_puts)

    # Flag when near-the-money call positioning dwarfs puts and clusters just
    # above spot - the mechanical setup for a squeeze.
    call_weight = call_oi if oi_available else call_vol
    put_weight = put_oi if oi_available else put_vol
    squeeze_flag = bool(
        max_call
        and call_weight > 0
        and put_weight > 0
        and (call_weight / put_weight) > 1.5
        and call_gamma > put_gamma * 1.5
        and spot <= max_call["strike"] <= spot * 1.05
    )

    return {
        "available": True,
        "expiry": expiry,
        "days_to_expiry": dte,
        "time_to_expiry_years": T,
        "spot_used": spot,
        "risk_free_used_pct": risk_free * 100,
        "atm_iv_pct": atm_iv_pct,
        "data_quality": data_quality,
        "iv_context": {
            "hv_30d_pct": hv_current,
            "iv_hv_ratio": iv_hv_ratio,
            "iv_rank_proxy_pct": iv_rank,
            "iv_percentile_proxy_pct": iv_percentile,
            "note": (
                "IV rank/percentile are proxies computed against the trailing "
                "1y realised-volatility distribution; free data has no IV history."
            ),
        },
        "put_call_ratio": {
            "open_interest": (put_oi / call_oi) if call_oi else None,
            "volume": (put_vol / call_vol) if call_vol else None,
            "call_oi_total": call_oi if oi_available else None,
            "put_oi_total": put_oi if oi_available else None,
            "call_volume_total": call_vol,
            "put_volume_total": put_vol,
        },
        "gamma_exposure": {
            "weighted_by": weight_field,
            "ntm_call_gamma": call_gamma,
            "ntm_put_gamma": put_gamma,
            "net_gamma": call_gamma - put_gamma,
            "peak_call_strike": max_call["strike"] if max_call else None,
            "peak_call_weight": max_call[weight_field] if max_call else None,
            "peak_put_strike": max_put["strike"] if max_put else None,
            "peak_put_weight": max_put[weight_field] if max_put else None,
            "gamma_squeeze_flag": squeeze_flag,
        },
        "near_the_money": {"calls": ntm_calls, "puts": ntm_puts},
    }
