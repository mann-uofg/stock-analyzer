"""The quantitative synthesis prompts and their output contracts.

Two modes, selected by the caller:

* **Authority** (default) - the model sets the verdict, conviction and every
  price level. The engine's own result is passed alongside as a reference
  baseline. Whatever the model returns is arithmetically validated in
  ``llm.py``; a plan that fails validation is discarded and the engine's
  numbers are used instead, with the reason recorded.
* **Narrative** (``--engine-numbers``) - the engine owns every figure and the
  model only explains them.
"""

from __future__ import annotations

import json
from typing import Any

_ANALYTICAL_RULES = """\
- If a field is null or marked unavailable, say the data is unavailable. Do \
NOT guess, estimate, or substitute a typical value.
- Cite concrete figures from the evidence when you make a claim. "RSI at 58" \
is acceptable; "momentum looks healthy" alone is not.
- Explicitly acknowledge signals that CONTRADICT the verdict. A note that only \
argues one side is a failed note.
- Be concise and clinical. No hedging filler, no disclaimers, no advice to \
"consult a financial advisor". This is a desk note for a professional.
- Where the evidence flags a data-quality caveat (for example implied \
volatility solved from traded prices, or unavailable open interest), reflect \
that uncertainty rather than presenting the figure as exact.\
"""

_READING_EVIDENCE = """\
READING THE EVIDENCE
- Any field whose name contains "abs" or "absolute" is a MAGNITUDE with no \
direction. Never attach a +/- sign to it. "avg_abs_post_earnings_move_pct: 3.7" \
means moves of 3.7% in either direction, NOT -3.7%.
- Bucket scores in `bucket_scores` run from -1.0 (maximally bearish) to +1.0 \
(maximally bullish). A score of 1.0 is "+1.0 out of +1.0"; do not restate it \
as a mark out of ten.
- `score_0_100` and `conviction_pct` are already percentages. Quote them as given.
- Support and resistance levels come from swing pivots, not moving averages. \
Do not describe a support level as being a moving average unless the two \
values are actually identical.
- If a beta carries `low_explanatory_power: true`, its R-squared is under 0.10. \
Say the beta is statistically weak over that window rather than presenting it \
as a reliable measure of market sensitivity.
- The `betas_and_alpha` block gives 1y and 3y windows from DAILY returns, plus \
`5y_monthly`, which is the estimator public finance sites publish. Cite the \
5y monthly figure when comparing to an outside source, and the daily windows \
when discussing the current regime.

You are writing analysis, not a recommendation to any individual.\
"""

SYSTEM_PROMPT_AUTHORITY = f"""\
You are a senior quantitative equity analyst. You OWN the call: the verdict, \
the conviction score and every price level in the trade plan are yours to set, \
and you reason strictly from the structured JSON evidence supplied to you.

An upstream quantitative engine has scored the same evidence. Its output is \
given to you as a REFERENCE BASELINE. You may agree with it, adjust it, or \
overrule it outright - but if you diverge, you must say why in \
`numbers_rationale`, citing the evidence that justifies the change.

ARITHMETIC RULES - your numbers are checked, and a plan that fails these is \
discarded in favour of the engine's:
1. For a LONG: stop_loss < entry_low <= entry_high < target_1 < target_2.
   For a SHORT: stop_loss > entry_high >= entry_low > target_1 > target_2.
2. Reward-to-risk on target_1 must be AT LEAST 2.0. Compute it as
   |target_1 - entry_high| / |entry_high - stop_loss| for a long, and
   |entry_low - target_1| / |stop_loss - entry_low| for a short.
3. Every level must be a plain number, within +/-50% of the current price.
4. conviction_pct is 0-100. verdict is exactly one of: STRONG BUY, BUY, HOLD, \
SELL, STRONG SELL.
5. Anchor the stop to real structure - a support/resistance level or an ATR \
multiple. Do not invent a level the evidence does not support.

DO THE ARITHMETIC CAREFULLY. Work the ratio out before you commit to the \
targets - but do it SILENTLY.

OUTPUT DISCIPLINE - this matters as much as the analysis:
- Never show your working, deliberate, or write "let me check" inside any JSON \
field. Every field holds a finished conclusion, not reasoning in progress.
- `numbers_rationale` is AT MOST 3 sentences.
- Every other string field is AT MOST 4 sentences.
- Write the JSON once and stop. Rambling inside a field truncates the response \
and the whole answer is discarded.

ANALYTICAL RULES
{_ANALYTICAL_RULES}

{_READING_EVIDENCE}"""

# `--engine-numbers`: the engine owns every figure, the model only explains it.
SYSTEM_PROMPT_NARRATIVE = f"""\
You are a senior quantitative equity analyst producing an internal research \
note. You reason strictly from the structured JSON evidence supplied to you.

ABSOLUTE RULES
- NEVER invent, alter, recompute or round any number. The verdict, conviction \
score, entry range, stop-loss and price targets are ALREADY DECIDED by an \
upstream quantitative engine. Treat them as immutable fact.
{_ANALYTICAL_RULES}

{_READING_EVIDENCE}"""

# Backwards-compatible alias.
SYSTEM_PROMPT = SYSTEM_PROMPT_NARRATIVE

_NARRATIVE_PROPERTIES: dict[str, Any] = {
    "executive_summary": {"type": "string"},
    "technical_summary": {"type": "string"},
    "risk_volatility_assessment": {"type": "string"},
    "fundamental_earnings_thesis": {"type": "string"},
    "options_positioning": {"type": "string"},
    "bull_case": {"type": "array", "items": {"type": "string"}},
    "bear_case": {"type": "array", "items": {"type": "string"}},
    "trade_commentary": {"type": "string"},
    "key_risk": {"type": "string"},
}

_NARRATIVE_REQUIRED = [
    "executive_summary", "technical_summary", "risk_volatility_assessment",
    "fundamental_earnings_thesis", "bull_case", "bear_case",
    "trade_commentary", "key_risk",
]

# Ollama accepts a JSON Schema in its `format` field, which constrains decoding
# and removes the need to regex-scrape prose for structure.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": dict(_NARRATIVE_PROPERTIES),
    "required": list(_NARRATIVE_REQUIRED),
}

# Authority mode additionally requires the model to emit the numbers itself.
OUTPUT_SCHEMA_AUTHORITY: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_NARRATIVE_PROPERTIES,
        "verdict": {
            "type": "string",
            "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
        },
        "conviction_pct": {"type": "number"},
        "direction": {"type": "string", "enum": ["long", "short"]},
        "entry_low": {"type": "number"},
        "entry_high": {"type": "number"},
        "stop_loss": {"type": "number"},
        "target_1": {"type": "number"},
        "target_2": {"type": "number"},
        "numbers_rationale": {"type": "string"},
    },
    "required": _NARRATIVE_REQUIRED + [
        "verdict", "conviction_pct", "direction", "entry_low", "entry_high",
        "stop_loss", "target_1", "target_2", "numbers_rationale",
    ],
}

_SHARED_FIELDS = """\
  executive_summary            - 2-3 sentences: the call and why it holds.
  technical_summary            - trend, momentum, and the key support/resistance \
levels that matter now.
  risk_volatility_assessment   - beta exposure, realised/implied volatility, \
gamma or open-interest positioning flags.
  fundamental_earnings_thesis  - valuation, earnings track record, forward \
consensus, and the upcoming catalyst.
  options_positioning          - what the chain implies about positioning. If \
option data is unavailable, say exactly that.
  bull_case                    - exactly 3 specific, evidence-backed catalysts.
  bear_case                    - exactly 3 specific, evidence-backed downside risks.
  trade_commentary             - how to manage the given setup: what confirms \
the thesis, what invalidates it.
  key_risk                     - the single most dangerous assumption in this call.\
"""

USER_TEMPLATE = """\
Analyse {ticker} using the evidence below.

The upstream engine has already determined:
  VERDICT     : {verdict}
  CONVICTION  : {conviction}%
  TRADE SETUP : {setup_line}

Write the research note that JUSTIFIES and CONTEXTUALISES this output. Return \
JSON only, matching this structure:

{shared_fields}

EVIDENCE:
{evidence}
"""

USER_TEMPLATE_AUTHORITY = """\
Analyse {ticker} using the evidence below and ISSUE THE CALL YOURSELF.

Current price: {price}

For reference only, the upstream engine scored this evidence as:
  VERDICT     : {verdict}
  CONVICTION  : {conviction}%
  TRADE SETUP : {setup_line}

You are not bound by that. Decide the verdict, the conviction and the price \
levels on the evidence, then return JSON only, matching this structure:

  verdict                      - STRONG BUY | BUY | HOLD | SELL | STRONG SELL
  conviction_pct               - 0-100, your confidence in the verdict
  direction                    - "long" or "short", consistent with the verdict
  entry_low, entry_high        - the entry range
  stop_loss                    - where the thesis is wrong
  target_1, target_2           - profit targets; target_1 must be at least \
2.0x the risk
  numbers_rationale            - how you derived the levels, and why you agreed \
with or departed from the engine's baseline

{shared_fields}

EVIDENCE:
{evidence}
"""


REPAIR_TEMPLATE = """\

--------------------------------------------------------------------
YOUR PREVIOUS ANSWER WAS REJECTED. It failed these checks:

{issues}

Fix ONLY the numbers. Keep your analysis and wording. Recompute carefully:

  risk   = |entry_high - stop_loss|   (long)  or  |stop_loss - entry_low| (short)
  target_1 must satisfy  |target_1 - entry_high| >= 2.0 * risk   (long)
                         |entry_low - target_1| >= 2.0 * risk    (short)

So for a long, target_1 must be AT LEAST entry_high + 2.0 * risk. Work out that
number explicitly, then place target_1 at or beyond it. Return the full JSON
again.
--------------------------------------------------------------------
"""


def _round(value: Any, digits: int = 2) -> Any:
    """Recursively round floats so the model sees clean, compact numbers."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def compact_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim the payload to what the model needs.

    Full option chains and raw statement frames blow past a local model's
    context window and add no interpretive value, so only summary levels and
    the handful of strikes nearest the money survive.
    """
    tech = payload.get("technical", {})
    opts = payload.get("options", {})
    fund = payload.get("fundamental", {})
    risk_panel = payload.get("risk", {})
    verdict = payload.get("verdict", {})

    betas = {}
    for symbol, windows in (risk_panel.get("benchmarks") or {}).items():
        betas[symbol] = {
            label: {
                "beta": vals.get("beta"),
                "alpha_annual_pct": vals.get("alpha_annual_pct"),
                "r_squared": vals.get("r_squared"),
                # Surfaced so the model qualifies a weak beta instead of
                # treating it as a firm measure of market sensitivity.
                "low_explanatory_power": vals.get("low_explanatory_power"),
            }
            for label, vals in windows.items()
        }

    evidence = {
        "quote": payload.get("quote"),
        "technical": {
            "moving_averages": tech.get("moving_averages"),
            "momentum": tech.get("momentum"),
            "volatility": tech.get("volatility"),
            "volume": tech.get("volume"),
            "levels": tech.get("levels"),
            "performance_pct": tech.get("performance_pct"),
            "range_52w": tech.get("range_52w"),
        },
        "risk": {
            "betas_and_alpha": betas,
            "volatility": risk_panel.get("volatility"),
            "ratios": risk_panel.get("ratios"),
            "max_drawdown_1y_pct": risk_panel.get("max_drawdown_1y_pct"),
            "risk_free_rate_pct": risk_panel.get("risk_free_rate_pct"),
        },
        "options": (
            {
                "available": True,
                "expiry": opts.get("expiry"),
                "days_to_expiry": opts.get("days_to_expiry"),
                "atm_iv_pct": opts.get("atm_iv_pct"),
                "iv_context": opts.get("iv_context"),
                "put_call_ratio": opts.get("put_call_ratio"),
                "gamma_exposure": opts.get("gamma_exposure"),
                "data_quality": opts.get("data_quality"),
                # Only the fields that carry interpretive signal; full Greeks
                # for ten strikes are for the report, not the prompt.
                "near_the_money_sample": {
                    kind: [
                        {
                            "strike": row.get("strike"),
                            "iv_pct": row.get("iv_pct"),
                            "delta": row.get("delta"),
                            "volume": row.get("volume"),
                            "open_interest": row.get("open_interest"),
                        }
                        for row in (opts.get("near_the_money", {}).get(kind) or [])[:4]
                    ]
                    for kind in ("calls", "puts")
                },
            }
            if opts.get("available")
            else {"available": False, "reason": opts.get("reason")}
        ),
        "fundamental": {
            "profile": fund.get("profile"),
            "valuation": {
                k: v for k, v in (fund.get("valuation") or {}).items() if k != "note"
            },
            "earnings": {
                **{
                    k: v
                    for k, v in (fund.get("earnings") or {}).items()
                    if k != "post_earnings_moves"
                },
                "recent_post_earnings_moves":
                    (fund.get("earnings", {}).get("post_earnings_moves") or [])[:4],
            },
            "consensus": fund.get("consensus"),
        },
        "quantitative_engine_output": {
            "verdict": verdict.get("verdict"),
            "score_0_100": verdict.get("score_0_100"),
            "conviction_pct": verdict.get("conviction_pct"),
            "signal_agreement": verdict.get("agreement"),
            "bucket_scores": {
                name: {"score": b["score"], "reasons": b["reasons"]}
                for name, b in (verdict.get("buckets") or {}).items()
            },
            "trade_setup": verdict.get("trade_setup"),
            "warnings": verdict.get("warnings"),
        },
        "recent_headlines": [n.get("title") for n in (payload.get("news") or [])][:5],
    }
    return _round(evidence)


def build_user_prompt(payload: dict[str, Any], authority: bool = True) -> str:
    """Render the user prompt for the chosen mode."""
    verdict = payload.get("verdict", {})
    setup = verdict.get("trade_setup", {}) or {}

    if setup.get("valid"):
        setup_line = (
            f"{setup['direction'].upper()} entry {setup['entry_low']}-{setup['entry_high']}, "
            f"stop {setup['stop_loss']}, targets {setup['target_1']} / {setup['target_2']} "
            f"(R:R {setup['risk_reward_t1']}:1 and {setup['risk_reward_t2']}:1)"
        )
    else:
        setup_line = f"unavailable ({setup.get('reason', 'unknown')})"

    common = {
        "ticker": payload.get("meta", {}).get("ticker", "?"),
        "verdict": verdict.get("verdict"),
        "conviction": verdict.get("conviction_pct"),
        "setup_line": setup_line,
        "shared_fields": _SHARED_FIELDS,
        # Compact separators over pretty-printing: indentation cost ~850 tokens
        # of pure whitespace, which is context the model needs for its answer.
        "evidence": json.dumps(
            compact_evidence(payload), separators=(",", ":"), default=str
        ),
    }

    if authority:
        return USER_TEMPLATE_AUTHORITY.format(
            price=payload.get("quote", {}).get("spot"), **common
        )
    return USER_TEMPLATE.format(**common)
