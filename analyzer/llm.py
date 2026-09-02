"""LLM synthesis, via Ollama Cloud or a local Ollama.

Three tiers, tried in order:

1. **Ollama Cloud** when ``OLLAMA_API_KEY`` is set - a frontier-scale model on
   their hardware, free tier, and nothing for the machine to compute. This is
   what makes the tool usable on a laptop: a 9B running locally pinned the GPU
   for minutes per note and turned the fans on.
2. **Local Ollama** when no key is set but the daemon is running. Slower and
   hotter, but entirely offline.
3. **A deterministic narrative** built from the same scored evidence, so a
   complete report is produced even with no model at all.

The privacy trade is explicit and belongs to whoever sets the key: with the
cloud provider the analysis payload - tickers, positions, prices - is sent to
ollama.com. Without a key nothing leaves the machine.

Whatever the model returns, its numbers are validated before they are shown.
A larger model is not a reason to trust a stop-loss unchecked.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import requests

from .config import MIN_RISK_REWARD, SETTINGS, VERDICT_BANDS
from .prompts import (
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_AUTHORITY,
    REPAIR_TEMPLATE,
    SYSTEM_PROMPT_AUTHORITY,
    SYSTEM_PROMPT_NARRATIVE,
    build_user_prompt,
)

VALID_VERDICTS = {label for _, label in VERDICT_BANDS}

# A level further than this from spot is a decimal-point error, not a thesis.
_MAX_LEVEL_DEVIATION = 0.50


class LLMUnavailable(RuntimeError):
    """Raised internally when no local model can serve the request."""


def provider() -> str:
    """Which backend will be used: 'cloud', 'local', or 'none'."""
    if SETTINGS.ollama_cloud_key:
        return "cloud"
    try:
        resp = requests.get(f"{SETTINGS.ollama_host}/api/tags", timeout=6)
        resp.raise_for_status()
        if resp.json().get("models"):
            return "local"
    except requests.RequestException:
        pass
    return "none"


def available() -> tuple[bool, str]:
    """Whether a model can be reached, and which one.

    Cloud is preferred when a key is present: it is both far stronger than a
    9B and costs the machine nothing, which is the whole reason for its
    existence here. Local remains the fallback so the tool still works with no
    key and no network.
    """
    if SETTINGS.ollama_cloud_key:
        return True, f"Ollama Cloud · {SETTINGS.ollama_cloud_model}"

    # A busy daemon can take a couple of seconds to answer even this, so the
    # budget is generous - a false "unavailable" silently downgrades the whole
    # report to the deterministic path.
    try:
        resp = requests.get(f"{SETTINGS.ollama_host}/api/tags", timeout=20)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.RequestException as exc:
        return False, (
            f"No OLLAMA_API_KEY set and local Ollama is not reachable at "
            f"{SETTINGS.ollama_host} ({exc.__class__.__name__})"
        )

    if not models:
        return False, "Ollama is running but has no models installed"

    wanted = SETTINGS.ollama_model
    # Accept an exact match or the same model under a different tag.
    if wanted in models or any(m.split(":")[0] == wanted.split(":")[0] for m in models):
        return True, f"local · {wanted}"
    return False, f"Model '{wanted}' not installed. Available: {', '.join(models)}"


def _call_cloud(system: str, user: str, schema: dict[str, Any]) -> str:
    """Ollama Cloud, via its OpenAI-compatible endpoint.

    A hosted frontier model needs no repair loop for arithmetic the way the
    local 9B did, but the response is still validated downstream - the size of
    the model is not a reason to trust a stop-loss unchecked.
    """
    response = requests.post(
        f"{SETTINGS.ollama_cloud_host}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {SETTINGS.ollama_cloud_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": SETTINGS.ollama_cloud_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Ask for JSON explicitly; the schema still governs validation.
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": SETTINGS.ollama_num_predict,
        },
        timeout=SETTINGS.llm_timeout,
    )
    if response.status_code == 401:
        raise ValueError("Ollama Cloud rejected the API key (401)")
    if response.status_code == 429:
        raise ValueError(
            "Ollama Cloud quota reached (429) - the free tier resets on a "
            "rolling window; the local model or the deterministic narrative "
            "covers the gap"
        )
    response.raise_for_status()

    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("Ollama Cloud returned no choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise ValueError("Ollama Cloud returned an empty message")

    # Some models wrap JSON in a fenced block despite response_format.
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    return content


def _call_ollama(system: str, user: str, schema: dict[str, Any]) -> str:
    payload = {
        "model": SETTINGS.ollama_model,
        "system": system,
        "prompt": user,
        "stream": False,
        "think": SETTINGS.ollama_think,
        "format": schema,
        "options": {
            "temperature": 0.2,      # analytical, not creative
            "top_p": 0.9,
            "num_ctx": SETTINGS.ollama_num_ctx,
            "num_predict": SETTINGS.ollama_num_predict,
        },
    }
    resp = requests.post(
        f"{SETTINGS.ollama_host}/api/generate",
        json=payload,
        timeout=SETTINGS.llm_timeout,
    )
    resp.raise_for_status()
    body = resp.json()

    # Ollama reports why generation stopped. "length" means we ran out of
    # output budget and the JSON is truncated - worth naming explicitly, since
    # the alternative is an opaque JSONDecodeError.
    if body.get("done_reason") == "length":
        raise ValueError(
            f"model output hit the {SETTINGS.ollama_num_predict}-token budget and was "
            "truncated; raise OLLAMA_NUM_PREDICT"
        )
    return body.get("response", "")


def _coerce_list(value: Any, n: int = 3) -> list[str]:
    """Normalise a model-supplied list field to exactly ``n`` strings."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = []
    return items[:n]


def _f(value: Any) -> float | None:
    """Coerce to a finite float, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def validate_numbers(
    parsed: dict[str, Any], price: float, repair: bool = False
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check a model-proposed trade plan before it can reach the user.

    Returns ``(plan, [])`` when every invariant holds, or ``(None, issues)``
    describing what failed. This is the safety net that makes handing the model
    numeric authority defensible: a transposed digit or an inverted stop is
    caught here rather than on a broker ticket.

    Two classes of problem are treated differently:

    * **Structural** - a wrong verdict label, an inverted stop, targets out of
      order, a level implausibly far from spot. These mean the plan is
      incoherent, and it is always rejected.
    * **Arithmetic** - the reward-to-risk floor. Small local models write a
      sound structural thesis (entry and stop read off real levels) and then
      misplace the target by a few percent. With ``repair=True`` the target is
      snapped out to exactly meet the floor, preserving the model's entry and
      stop and recording the change in ``adjustments``.
    """
    issues: list[str] = []

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        issues.append(f"verdict {verdict!r} is not one of the five allowed values")

    conviction = _f(parsed.get("conviction_pct"))
    if conviction is None or not 0 <= conviction <= 100:
        issues.append(f"conviction_pct {parsed.get('conviction_pct')!r} outside 0-100")

    direction = str(parsed.get("direction", "")).strip().lower()
    if direction not in {"long", "short"}:
        issues.append(f"direction {direction!r} is not long/short")

    levels: dict[str, float] = {}
    for key in ("entry_low", "entry_high", "stop_loss", "target_1", "target_2"):
        value = _f(parsed.get(key))
        if value is None or value <= 0:
            issues.append(f"{key} is missing or non-positive")
        else:
            levels[key] = value

    if issues:
        return None, issues

    # Every level must sit within a sane distance of spot.
    for key, value in levels.items():
        if price > 0 and abs(value / price - 1) > _MAX_LEVEL_DEVIATION:
            issues.append(
                f"{key} {value:,.2f} is more than "
                f"{_MAX_LEVEL_DEVIATION:.0%} away from the price {price:,.2f}"
            )

    lo, hi = levels["entry_low"], levels["entry_high"]
    stop, t1, t2 = levels["stop_loss"], levels["target_1"], levels["target_2"]

    if lo > hi:
        issues.append(f"entry_low {lo:,.2f} exceeds entry_high {hi:,.2f}")

    if direction == "long":
        if not stop < lo:
            issues.append(f"long stop {stop:,.2f} is not below entry_low {lo:,.2f}")
        if not hi < t1 < t2:
            issues.append(
                f"long targets out of order: entry_high {hi:,.2f}, "
                f"t1 {t1:,.2f}, t2 {t2:,.2f}"
            )
        risk, reward = hi - stop, t1 - hi
    else:
        if not stop > hi:
            issues.append(f"short stop {stop:,.2f} is not above entry_high {hi:,.2f}")
        if not lo > t1 > t2:
            issues.append(
                f"short targets out of order: entry_low {lo:,.2f}, "
                f"t1 {t1:,.2f}, t2 {t2:,.2f}"
            )
        risk, reward = stop - lo, lo - t1

    if issues:
        return None, issues

    if risk <= 0:
        return None, [f"non-positive risk per share ({risk:,.2f})"]

    adjustments: list[str] = []
    anchor = hi if direction == "long" else lo
    rr1 = reward / risk

    if rr1 < MIN_RISK_REWARD - 1e-6:
        if not repair:
            issues.append(
                f"reward:risk on target_1 is {rr1:.2f}:1, below the "
                f"{MIN_RISK_REWARD}:1 floor"
            )
            return None, issues

        original = t1
        t1 = (
            anchor + MIN_RISK_REWARD * risk
            if direction == "long"
            else anchor - MIN_RISK_REWARD * risk
        )
        adjustments.append(
            f"target_1 moved from {original:,.2f} to {t1:,.2f} to meet the "
            f"{MIN_RISK_REWARD}:1 floor (model's plan gave {rr1:.2f}:1); its "
            "entry and stop are unchanged"
        )
        reward = abs(t1 - anchor)
        rr1 = reward / risk

        # Keep the second target beyond the first after the adjustment.
        if (direction == "long" and t2 <= t1) or (direction == "short" and t2 >= t1):
            t2 = anchor + 3.5 * risk if direction == "long" else anchor - 3.5 * risk
            adjustments.append(f"target_2 moved to {t2:,.2f} to stay beyond target_1")

    rr2 = abs(t2 - anchor) / risk

    return {
        "valid": True,
        "direction": direction,
        "entry_low": round(lo, 2),
        "entry_high": round(hi, 2),
        "stop_loss": round(stop, 2),
        "risk_per_share": round(risk, 2),
        "risk_pct": round(risk / price * 100, 2) if price else None,
        "target_1": round(t1, 2),
        "target_2": round(t2, 2),
        "risk_reward_t1": round(rr1, 2),
        "risk_reward_t2": round(rr2, 2),
        "basis": str(parsed.get("numbers_rationale", "")).strip()
        or "Levels set by the local model from the supplied evidence.",
        "author": "llm",
        "adjustments": adjustments,
    }, []


def synthesise(payload: dict[str, Any], numeric_authority: bool = True) -> dict[str, Any]:
    """Produce the narrative report, and optionally the numbers with it.

    Always returns a usable dict. ``source`` records where the prose came from;
    ``numbers_source`` records who owns the verdict and price levels.
    """
    engine_verdict = payload.get("verdict", {}) or {}
    engine_setup = dict(engine_verdict.get("trade_setup") or {})
    engine_setup.setdefault("author", "engine")
    engine_block = {
        "numbers_source": "engine",
        "numbers_issues": [],
        "verdict": engine_verdict.get("verdict"),
        "conviction_pct": engine_verdict.get("conviction_pct"),
        "trade_setup": engine_setup,
        "engine_verdict": engine_verdict.get("verdict"),
        "engine_conviction_pct": engine_verdict.get("conviction_pct"),
        "engine_trade_setup": engine_setup,
    }

    ok, detail = available()
    if not ok:
        return {**_fallback(payload), **engine_block,
                "source": "deterministic", "llm_note": detail}

    using = provider()
    system = SYSTEM_PROMPT_AUTHORITY if numeric_authority else SYSTEM_PROMPT_NARRATIVE
    schema = OUTPUT_SCHEMA_AUTHORITY if numeric_authority else OUTPUT_SCHEMA
    base_prompt = build_user_prompt(payload, numeric_authority)
    price = payload.get("quote", {}).get("spot") or 0.0

    parsed: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    issues: list[str] = []
    last_error: Exception | None = None

    # Two failure modes share this retry budget:
    #   * a truncated response - the model sometimes reasons at length inside a
    #     string field until it exhausts the token budget, leaving invalid JSON;
    #   * a plan that misses the reward-to-risk floor, where handing the
    #     specific failure back usually fixes it.
    # Both are transient, so a retry is worth far more than an immediate
    # downgrade to the deterministic narrative.
    attempts = (2 if using == 'cloud' else 3) if numeric_authority else 2
    for _ in range(attempts):
        prompt = base_prompt
        if issues:
            prompt += REPAIR_TEMPLATE.format(
                issues="\n".join(f"  - {i}" for i in issues)
            )
        try:
            raw = (
                _call_cloud(system, prompt, schema) if using == "cloud"
                else _call_ollama(system, prompt, schema)
            )
            candidate = json.loads(raw)
            if not isinstance(candidate, dict):
                raise ValueError("model returned a non-object")
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if parsed is not None:
                break  # keep the earlier prose; the retry merely failed to improve it
            continue

        parsed = candidate
        if not numeric_authority:
            break

        plan, issues = validate_numbers(parsed, price)
        if plan is not None:
            break

    if parsed is None:
        return {
            **_fallback(payload),
            **engine_block,
            "source": "deterministic",
            "llm_note": (
                f"Local model call failed ({last_error.__class__.__name__}: {last_error})"
                if last_error
                else "Local model returned no usable response"
            ),
        }

    # Both honest attempts missed the reward floor. Rather than discard a
    # structurally sound plan over arithmetic, snap the target to the floor and
    # disclose the change; a plan that is structurally broken still fails here.
    if numeric_authority and plan is None:
        plan, issues = validate_numbers(parsed, price, repair=True)

    numbers = dict(engine_block)
    if numeric_authority:
        if plan is not None:
            numbers.update(
                {
                    "numbers_source": "llm",
                    "numbers_issues": [],
                    "verdict": str(parsed["verdict"]).strip().upper(),
                    "conviction_pct": round(float(parsed["conviction_pct"]), 1),
                    "trade_setup": plan,
                }
            )
        else:
            # Still wrong after a correction pass - keep the engine's numbers
            # and record exactly why, rather than silently substituting.
            numbers["numbers_source"] = "engine (model plan rejected)"
            numbers["numbers_issues"] = issues

    assert parsed is not None  # the no-parse path returns above
    return {
        **numbers,
        "executive_summary": str(parsed.get("executive_summary", "")).strip(),
        "technical_summary": str(parsed.get("technical_summary", "")).strip(),
        "risk_volatility_assessment": str(parsed.get("risk_volatility_assessment", "")).strip(),
        "fundamental_earnings_thesis": str(parsed.get("fundamental_earnings_thesis", "")).strip(),
        "options_positioning": str(parsed.get("options_positioning", "")).strip(),
        "bull_case": _coerce_list(parsed.get("bull_case")),
        "bear_case": _coerce_list(parsed.get("bear_case")),
        "trade_commentary": str(parsed.get("trade_commentary", "")).strip(),
        "key_risk": str(parsed.get("key_risk", "")).strip(),
        "numbers_rationale": str(parsed.get("numbers_rationale", "")).strip() or None,
        "source": (f"ollama-cloud:{SETTINGS.ollama_cloud_model}" if using == "cloud"
                   else f"ollama-local:{SETTINGS.ollama_model}"),
        "llm_note": None,
    }


# --- Deterministic fallback ----------------------------------------------


def _top_reasons(verdict: dict, positive: bool, limit: int = 3) -> list[str]:
    """Pull the strongest supporting or opposing reasons from the buckets."""
    buckets = verdict.get("buckets") or {}
    ranked = sorted(
        buckets.items(),
        key=lambda kv: kv[1]["contribution"],
        reverse=positive,
    )
    out: list[str] = []
    for _, bucket in ranked:
        aligned = bucket["score"] > 0 if positive else bucket["score"] < 0
        if not aligned:
            continue
        for reason in bucket["reasons"]:
            if reason not in out:
                out.append(reason)
            if len(out) >= limit:
                return out
    return out


def _counter_signals(payload: dict[str, Any], direction: str, limit: int = 3) -> list[str]:
    """Evidence arguing AGAINST the verdict.

    When every bucket leans the same way, ranking bucket scores yields nothing
    for the opposing case - yet real caveats (event risk, negative alpha, a
    weak trend reading, rich options) almost always exist. This surfaces them
    so the note is never one-sided.
    """
    verdict = payload.get("verdict", {})
    tech = payload.get("technical", {})
    risk_panel = payload.get("risk", {})
    opts = payload.get("options", {})
    out: list[str] = []

    # Signals that already scored against the verdict come first.
    for _, bucket in (verdict.get("buckets") or {}).items():
        aligned_against = bucket["score"] < 0 if direction == "long" else bucket["score"] > 0
        if aligned_against:
            out.extend(r for r in bucket["reasons"] if r not in out)

    # Only genuine event risks count as directional counter-evidence; the
    # scoring engine separates those from presentational warnings for us.
    out.extend(r for r in (verdict.get("event_risks") or []) if r not in out)

    one_year: dict = {}
    benchmark_symbol = None
    for symbol, windows in (risk_panel.get("benchmarks") or {}).items():
        one_year, benchmark_symbol = windows.get("1y", {}) or {}, symbol
        break  # one benchmark is enough for the narrative

    alpha, beta = one_year.get("alpha_annual_pct"), one_year.get("beta")
    adx = tech.get("volatility", {}).get("adx_14")
    rsi = tech.get("momentum", {}).get("rsi_14")
    rng = tech.get("range_52w") or {}
    iv_hv = _get_nested(opts, "iv_context", "iv_hv_ratio")

    if direction == "long":
        if alpha is not None and alpha < 0:
            out.append(
                f"Negative 1y Jensen's alpha ({alpha:.1f}%) vs {benchmark_symbol}: the "
                "name has underperformed its beta-implied return"
            )
        if beta is not None and beta > 1.5:
            out.append(
                f"High beta {beta:.2f} vs {benchmark_symbol} - amplifies any broad market drawdown"
            )
        if adx is not None and adx < 20:
            out.append(f"ADX {adx:.1f} shows no established trend - breakouts often fail here")
        if rsi is not None and rsi >= 70:
            out.append(f"RSI {rsi:.1f} is overbought - poor entry into strength")
        if rng.get("pct_from_high") is not None and rng["pct_from_high"] > -10:
            out.append(
                f"Price is only {abs(rng['pct_from_high']):.1f}% below its 52-week high - "
                "limited room before prior supply"
            )
        if iv_hv is not None and iv_hv > 1.2:
            out.append(f"IV/HV {iv_hv:.2f} - hedging or call-buying is expensive here")
    else:
        # Mirror image: what could invalidate a short.
        if alpha is not None and alpha > 0:
            out.append(
                f"Positive 1y Jensen's alpha ({alpha:+.1f}%) vs {benchmark_symbol} - "
                "the name has been rewarding holders despite the setup"
            )
        if adx is not None and adx < 20:
            out.append(
                f"ADX {adx:.1f} indicates a range, not a downtrend - shorts get squeezed "
                "in chop"
            )
        if rsi is not None and rsi <= 30:
            out.append(f"RSI {rsi:.1f} is oversold - elevated risk of a reflex bounce")
        if rng.get("pct_from_low") is not None and rng["pct_from_low"] < 10:
            out.append(
                f"Price is only {rng['pct_from_low']:.1f}% above its 52-week low - "
                "chasing weakness into established demand"
            )
        if iv_hv is not None and iv_hv < 0.9:
            out.append(f"IV/HV {iv_hv:.2f} - cheap options make protective calls easy to own")

    seen: list[str] = []
    for item in out:
        if item not in seen:
            seen.append(item)
    return seen[:limit]


def _get_nested(d: dict | None, *path: str) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _ordinal(n: float) -> str:
    """1 -> 1st, 2 -> 2nd, 71 -> 71st."""
    i = int(round(n))
    if 10 <= i % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def _fallback(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the narrative from scored evidence when no local model is available."""
    verdict = payload.get("verdict", {})
    tech = payload.get("technical", {})
    fund = payload.get("fundamental", {})
    risk_panel = payload.get("risk", {})
    opts = payload.get("options", {})
    meta = payload.get("meta", {})

    ticker = meta.get("ticker", "?")
    price = payload.get("quote", {}).get("spot")
    ma = tech.get("moving_averages", {})
    mom = tech.get("momentum", {})
    vol = tech.get("volatility", {})
    val = fund.get("valuation", {})
    earn = fund.get("earnings", {})
    levels = tech.get("levels", {})

    def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
        return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "n/a"

    exec_summary = (
        f"{ticker} scores {verdict.get('score_0_100')}/100 on the composite model, "
        f"yielding a {verdict.get('verdict')} with {verdict.get('conviction_pct')}% conviction "
        f"(signal agreement {verdict.get('agreement')}). "
        f"Last price {_fmt(price)}, trend alignment '{ma.get('alignment')}'."
    )

    technical = (
        f"Price {_fmt(price)} sits {ma.get('price_vs_sma50')} the 50-day "
        f"({_fmt(ma.get('sma_50'))}) and {ma.get('price_vs_sma200')} the 200-day "
        f"({_fmt(ma.get('sma_200'))}). RSI {_fmt(mom.get('rsi_14'), 1)} "
        f"({mom.get('rsi_state')}), MACD {mom.get('macd_state')} its signal with a "
        f"{_fmt(mom.get('macd_histogram'))} histogram. ADX {_fmt(vol.get('adx_14'), 1)} "
        f"indicates {vol.get('adx_state')}. Support {levels.get('support')}, "
        f"resistance {levels.get('resistance')}."
    )

    beta_bits = []
    for symbol, windows in (risk_panel.get("benchmarks") or {}).items():
        one_year = windows.get("1y", {})
        if one_year.get("beta") is not None:
            beta_bits.append(
                f"beta {one_year['beta']:.2f} vs {symbol} "
                f"(alpha {_fmt(one_year.get('alpha_annual_pct'), 1, '%')})"
            )
    hv_pctile = risk_panel.get("volatility", {}).get("hv_percentile_1y")
    risk_text = (
        f"1y {', '.join(beta_bits) if beta_bits else 'beta unavailable'}. "
        f"30-day realised vol {_fmt(risk_panel.get('volatility', {}).get('hv_30d_annual_pct'), 1, '%')} "
        f"({_ordinal(hv_pctile) if hv_pctile is not None else 'n/a'} percentile of the past year). "
        f"Max 1y drawdown {_fmt(risk_panel.get('max_drawdown_1y_pct'), 1, '%')}. "
        f"ATR {_fmt(vol.get('atr_14'))} ({_fmt(vol.get('atr_percent'), 1, '%')} of price)."
    )

    fundamental = (
        f"Trailing P/E {_fmt(val.get('trailing_pe'), 1)}, forward P/E "
        f"{_fmt(val.get('forward_pe'), 1)}, PEG {_fmt(val.get('peg_ratio'))}, "
        f"P/S {_fmt(val.get('price_to_sales'), 1)}, EV/EBITDA "
        f"{_fmt(val.get('ev_to_ebitda'), 1)}, FCF yield "
        f"{_fmt(val.get('fcf_yield_pct'), 2, '%')}. "
        f"EPS beat rate {_fmt(earn.get('beat_rate_pct'), 0, '%')} over the last "
        f"{len(earn.get('history') or [])} quarters (avg surprise "
        f"{_fmt(earn.get('avg_eps_surprise_pct'), 1, '%')}). "
        f"Next earnings {earn.get('next_earnings_date')} "
        f"({earn.get('days_to_earnings')} days), historical absolute post-earnings move "
        f"{_fmt(earn.get('avg_abs_post_earnings_move_pct'), 1, '%')}."
    )

    if opts.get("available"):
        pcr = opts.get("put_call_ratio", {})
        gamma = opts.get("gamma_exposure", {})
        ratio = pcr.get("open_interest")
        basis = "open interest"
        if ratio is None:
            ratio, basis = pcr.get("volume"), "volume"
        options_text = (
            f"{opts.get('expiry')} expiry ({opts.get('days_to_expiry')}d): ATM IV "
            f"{_fmt(opts.get('atm_iv_pct'), 1, '%')} against 30d realised "
            f"{_fmt(opts.get('iv_context', {}).get('hv_30d_pct'), 1, '%')} "
            f"(IV/HV {_fmt(opts.get('iv_context', {}).get('iv_hv_ratio'))}). "
            f"Put/call {_fmt(ratio)} by {basis}. Net NTM gamma "
            f"{_fmt(gamma.get('net_gamma'), 0)}, weighted by {gamma.get('weighted_by')}. "
            f"Squeeze flag: {gamma.get('gamma_squeeze_flag')}."
        )
    else:
        options_text = f"Option data unavailable: {opts.get('reason')}."

    setup = verdict.get("trade_setup", {}) or {}
    if setup.get("valid"):
        trade_text = (
            f"{setup['direction'].upper()} between {setup['entry_low']} and "
            f"{setup['entry_high']}, stop {setup['stop_loss']} "
            f"(risk {setup['risk_pct']}% of price). Targets {setup['target_1']} "
            f"({setup['risk_reward_t1']}:1) and {setup['target_2']} "
            f"({setup['risk_reward_t2']}:1). {setup['basis']} "
            f"The thesis is invalidated on a close beyond the stop."
        )
    else:
        trade_text = f"No trade setup could be constructed: {setup.get('reason')}."

    warnings_list = verdict.get("warnings") or []
    key_risk = (
        warnings_list[0]
        if warnings_list
        else "Composite signals may be mutually correlated, overstating independent confirmation."
    )

    direction = (verdict.get("trade_setup") or {}).get("direction", "long")
    bull = _top_reasons(verdict, positive=True)
    bear = _top_reasons(verdict, positive=False)

    # With a one-sided score the opposing list comes back empty; fill it from
    # explicit counter-evidence so the note always presents both sides.
    if direction == "long" and len(bear) < 3:
        bear += [c for c in _counter_signals(payload, "long") if c not in bear]
    elif direction == "short" and len(bull) < 3:
        bull += [c for c in _counter_signals(payload, "short") if c not in bull]

    return {
        "executive_summary": exec_summary,
        "technical_summary": technical,
        "risk_volatility_assessment": risk_text,
        "fundamental_earnings_thesis": fundamental,
        "options_positioning": options_text,
        "bull_case": bull[:3] or ["No materially bullish signals."],
        "bear_case": bear[:3] or ["No materially bearish signals."],
        "trade_commentary": trade_text,
        "key_risk": key_risk,
    }
