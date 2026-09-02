"""Benchmark cloud models on this app's actual job.

Shared by the command-line script and the in-app button, because the key lives
in different places depending on how you run it: a ``.env`` file locally, or
Streamlit secrets once deployed - where there is no shell to run a script from.

What is measured is deliberately narrow. General leaderboards do not predict
performance here; this job needs valid JSON across a nineteen-field schema and
arithmetic that clears a 2:1 reward-to-risk floor. Each candidate receives the
exact prompt the app sends, on a real payload, and its answer goes through the
same validator that guards the live trade plan - with repair disabled, so the
result reflects what the model got right unaided.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests

from .config import SETTINGS
from .llm import validate_numbers
from .prompts import (
    OUTPUT_SCHEMA_AUTHORITY,
    SYSTEM_PROMPT_AUTHORITY,
    build_user_prompt,
)

# Ordered cheapest-first. Measured on 2026-09-02: everything above
# nemotron-3-super returned HTTP 402 on the free tier, so the paid models are
# kept last - they are worth re-testing only if the plan changes.
CANDIDATES: tuple[str, ...] = (
    "gpt-oss:20b",
    "gpt-oss:120b",
    "nemotron-3-nano:30b",
    "nemotron-3-super",
    "gemma4:31b",
    # Paid tiers as of the last run; listed so the benchmark reports it
    # explicitly rather than leaving you to wonder.
    "glm-5.3-flash",
    "glm-5.3",
    "deepseek-v4-flash:0731",
)

REQUIRED_FIELDS: list[str] = OUTPUT_SCHEMA_AUTHORITY["required"]


def run_one(model: str, system: str, user: str, price: float,
            timeout: int | None = None) -> dict[str, Any]:
    """One model, one attempt. No retries - retries would mask unreliability."""
    result: dict[str, Any] = {
        "model": model, "ok": False, "seconds": None, "valid_json": False,
        "fields": 0, "arithmetic": False, "risk_reward": None, "note": "",
        "unavailable": False,
    }
    if not SETTINGS.ollama_cloud_key:
        result["note"] = "no API key set"
        return result

    started = time.time()
    try:
        response = requests.post(
            f"{SETTINGS.ollama_cloud_host}/v1/chat/completions",
            headers={"Authorization": f"Bearer {SETTINGS.ollama_cloud_key}",
                     "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": SETTINGS.ollama_num_predict,
            },
            timeout=timeout or SETTINGS.llm_timeout,
        )
    except requests.RequestException as exc:
        result["note"] = exc.__class__.__name__
        return result

    result["seconds"] = round(time.time() - started, 1)

    if response.status_code == 402:
        # Not a failure of the model - it is simply not on this plan. Recorded
        # separately so it never counts against a candidate's reliability.
        result["unavailable"] = True
        result["note"] = "not on the free tier (402)"
        return result
    if response.status_code == 429:
        result["unavailable"] = True
        result["note"] = "quota reached (429)"
        return result
    if response.status_code != 200:
        result["note"] = f"HTTP {response.status_code}: {response.text[:90]}"
        return result

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        result["note"] = "no message content"
        return result

    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        result["note"] = f"invalid JSON ({exc.msg})"
        return result

    result["valid_json"] = True
    result["fields"] = sum(
        1 for f in REQUIRED_FIELDS if parsed.get(f) not in (None, "")
    )

    plan, issues = validate_numbers(parsed, price, repair=False)
    if plan:
        result["arithmetic"] = True
        result["risk_reward"] = plan["risk_reward_t1"]
    else:
        result["note"] = issues[0][:80] if issues else "validation failed"

    result["ok"] = result["valid_json"] and result["fields"] == len(REQUIRED_FIELDS)
    return result


def compare(payload: dict[str, Any], models: tuple[str, ...] = CANDIDATES,
            timeout: int | None = None,
            on_result: Callable[[dict[str, Any]], None] | None = None
            ) -> list[dict[str, Any]]:
    """Score every candidate against one analysed ticker.

    Run serially on purpose: concurrent requests would compete for the same
    quota and distort the latency figure, which is half the reason to measure.
    """
    system = SYSTEM_PROMPT_AUTHORITY
    user = build_user_prompt(payload, authority=True)
    price = payload.get("quote", {}).get("spot") or 0.0

    rows = []
    for model in models:
        row = run_one(model, system, user, price, timeout)
        rows.append(row)
        if on_result:
            on_result(row)
    return rows


def recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a winner, and say why in terms the user can check.

    The fastest model that produced complete JSON *and* sound arithmetic wins:
    on a GPU-time-metered free tier, extra capability beyond "correct" costs
    quota for nothing.
    """
    passing = [r for r in rows if r["ok"] and r["arithmetic"]]
    # Models your plan cannot reach were never really candidates, so they are
    # excluded from the denominator - "1 of 8" reads as though seven models
    # failed when six were never tried.
    tried = [r for r in rows if not r.get("unavailable")]
    blocked = [r for r in rows if r.get("unavailable")]
    if passing:
        best = min(passing, key=lambda r: r["seconds"] or 1e9)
        blocked_note = (
            f" {len(blocked)} {'was' if len(blocked) == 1 else 'were'} "
            "unavailable on your plan and never ran."
            if blocked else ""
        )
        return {
            "model": best["model"],
            "reason": (
                f"Fastest model that produced complete JSON and arithmetic that "
                f"passed unaided ({best['seconds']}s, R:R {best['risk_reward']}:1). "
                f"{len(passing)} of {len(tried)} reachable candidates managed "
                f"both.{blocked_note}"
            ),
            "passing": [r["model"] for r in passing],
        }

    # Nothing was fully correct; fall back to whatever at least parsed.
    parsed = [r for r in rows if r["valid_json"]]
    if parsed:
        best = min(parsed, key=lambda r: r["seconds"] or 1e9)
        return {
            "model": best["model"],
            "reason": (
                "No candidate got the arithmetic right unaided, so this is "
                "simply the quickest that returned usable JSON. The app's "
                "validator catches the maths and falls back to engine numbers, "
                "so the trade plan stays sound either way."
            ),
            "passing": [],
        }
    return {
        "model": None,
        "reason": (
            "No candidate returned usable JSON. Check the key, or raise "
            "OLLAMA_NUM_PREDICT if answers are being truncated."
        ),
        "passing": [],
    }
