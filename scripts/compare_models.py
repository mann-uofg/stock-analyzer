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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from analyzer import engine, modelbench  # noqa: E402
from analyzer.config import SETTINGS  # noqa: E402

# A spread of current cloud models: two frontier, two mid, two light. Edit
# freely - the point is the measurement, not this particular slate.
DEFAULT_CANDIDATES = modelbench.CANDIDATES

REQUIRED_FIELDS = modelbench.REQUIRED_FIELDS


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
    system = modelbench.SYSTEM_PROMPT_AUTHORITY
    user = modelbench.build_user_prompt(payload, authority=True)
    price = payload["quote"]["spot"]
    print(f"  prompt ≈{(len(system) + len(user)) // 4:,} tokens · "
          f"{args.ticker} at {price:,.2f}\n")

    header = (f"{'model':<26}{'ok':>4}{'json':>6}{'fields':>8}"
              f"{'maths':>7}{'R:R':>6}{'secs':>7}  note")
    print(header)
    print("-" * len(header))

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        row = modelbench.run_one(model, system, user, price, args.timeout)
        rows.append(row)
        print(
            f"{row['model']:<26}"
            f"{'Y' if row['ok'] else 'n':>4}"
            f"{'Y' if row['valid_json'] else 'n':>6}"
            f"{row['fields']}/{len(REQUIRED_FIELDS):<6}"
            f"{'Y' if row['arithmetic'] else 'n':>7}"
            f"{(f"{row['risk_reward']:.2f}" if row['risk_reward'] else '—'):>6}"
            f"{(row['seconds'] if row['seconds'] is not None else '—'):>7}"
            f"  {row['note']}"
        )

    verdict = modelbench.recommend(rows)
    print()
    print(verdict["reason"])
    if verdict["model"]:
        print(f"\nSet it with:  OLLAMA_CLOUD_MODEL={verdict['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
