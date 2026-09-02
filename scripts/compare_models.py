#!/usr/bin/env python3
"""Benchmark Ollama Cloud models on this app's actual job.

    python scripts/compare_models.py --ticker NVDA
    python scripts/compare_models.py --models glm-5.3,deepseek-v4-flash:0731

Model leaderboards measure general ability. This measures the three things
that actually decide whether a model is usable *here*, on a real payload from
your own data:

1. **Valid JSON** across a nineteen-field schema. A model that writes beautiful
   prose but fences it in markdown or truncates mid-string is worthless to a
   program.
2. **Arithmetic that survives validation** - the reward-to-risk floor, stop on
   the correct side of entry, targets in order. This is precisely where the
   local 9B failed, repeatedly and confidently.
3. **Latency and quota cost.** Ollama Cloud's free tier meters GPU time, so the
   largest model is not automatically the right one for something you run many
   times a day.

Nothing here is guesswork: each candidate gets the same prompt the app sends,
and its answer goes through the same validator that guards the real trade plan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from analyzer import engine, llm  # noqa: E402
from analyzer.config import SETTINGS  # noqa: E402
from analyzer.prompts import (  # noqa: E402
    OUTPUT_SCHEMA_AUTHORITY,
    SYSTEM_PROMPT_AUTHORITY,
    build_user_prompt,
)

# A spread of current cloud models: two frontier, two mid, two light. Edit
# freely - the point is the measurement, not this particular slate.
DEFAULT_CANDIDATES = (
    "deepseek-v4-pro:0813",
    "glm-5.3",
    "kimi-k2.6",
    "qwen3.5:397b",
    "deepseek-v4-flash:0731",
    "glm-5.3-flash",
    "nemotron-3-super",
    "gpt-oss:120b",
)

REQUIRED_FIELDS = OUTPUT_SCHEMA_AUTHORITY["required"]


def run_one(model: str, system: str, user: str, price: float,
            timeout: int) -> dict:
    """One model, one attempt, no retries - retries would hide unreliability."""
    result = {
        "model": model, "ok": False, "seconds": None, "json": False,
        "fields": 0, "arithmetic": False, "rr": None, "note": "",
    }
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
            timeout=timeout,
        )
    except requests.RequestException as exc:
        result["note"] = f"{exc.__class__.__name__}"
        return result

    result["seconds"] = round(time.time() - started, 1)

    if response.status_code != 200:
        body = response.text[:120].replace("\n", " ")
        result["note"] = f"HTTP {response.status_code}: {body}"
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

    result["json"] = True
    result["fields"] = sum(1 for f in REQUIRED_FIELDS if parsed.get(f) not in (None, ""))

    # The real validator, with repair off: we want to know whether the model
    # got the arithmetic right unaided, not whether we can patch it.
    plan, issues = llm.validate_numbers(parsed, price, repair=False)
    if plan:
        result["arithmetic"] = True
        result["rr"] = plan["risk_reward_t1"]
    else:
        result["note"] = issues[0][:74] if issues else "validation failed"

    result["ok"] = result["json"] and result["fields"] == len(REQUIRED_FIELDS)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--models", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    if not SETTINGS.ollama_cloud_key:
        print("Set OLLAMA_API_KEY first (see .env.example).")
        return 2

    print(f"Building the {args.ticker} payload the app would send…")
    payload = engine.analyse(args.ticker, skip_options=True)
    system = SYSTEM_PROMPT_AUTHORITY
    user = build_user_prompt(payload, authority=True)
    price = payload["quote"]["spot"]
    print(f"  prompt ≈{(len(system) + len(user)) // 4:,} tokens · "
          f"{args.ticker} at {price:,.2f}\n")

    header = (f"{'model':<26}{'ok':>4}{'json':>6}{'fields':>8}"
              f"{'maths':>7}{'R:R':>6}{'secs':>7}  note")
    print(header)
    print("-" * len(header))

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        row = run_one(model, system, user, price, args.timeout)
        rows.append(row)
        print(
            f"{row['model']:<26}"
            f"{'Y' if row['ok'] else 'n':>4}"
            f"{'Y' if row['json'] else 'n':>6}"
            f"{row['fields']}/{len(REQUIRED_FIELDS):<6}"
            f"{'Y' if row['arithmetic'] else 'n':>7}"
            f"{(f'{row['rr']:.2f}' if row['rr'] else '—'):>6}"
            f"{(row['seconds'] if row['seconds'] is not None else '—'):>7}"
            f"  {row['note']}"
        )

    usable = [r for r in rows if r["ok"] and r["arithmetic"]]
    print()
    if usable:
        fastest = min(usable, key=lambda r: r["seconds"] or 1e9)
        print(f"Complete JSON *and* sound arithmetic: "
              f"{', '.join(r['model'] for r in usable)}")
        print(f"Fastest of those: {fastest['model']} at {fastest['seconds']}s — "
              "on a GPU-time-metered free tier, the quickest passing model is "
              "usually the right default.")
        print(f"\nSet it with:  OLLAMA_CLOUD_MODEL={fastest['model']}")
    else:
        print("Nothing passed both checks. The app's validator catches this and "
              "falls back to engine numbers, so it degrades safely — but try "
              "raising OLLAMA_NUM_PREDICT, or a larger candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
