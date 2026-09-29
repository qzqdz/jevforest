#!/usr/bin/env python3
"""Sealed N4 live Jev eval (OpenRouter). Never writes the API key."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# load gitignored .env if present
env_path = ROOT / ".env"
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

from jevforest.eval.synthesis import eval_task  # noqa: E402
from jevforest.providers import LiveJevProvider  # noqa: E402


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing", file=sys.stderr)
        return 2
    report = eval_task(
        ROOT / "examples" / "tasks" / "tickets",
        jev=LiveJevProvider(),
        shot=0,
        seed=0,
        run_search=False,
    )
    report["provider"] = "live:~typesafe/jev-latest"
    out = ROOT / "results" / "n4_live_ticket_routing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, m in report["methods"].items():
        t = m["test"]
        print(
            f"  {name:16} acc={t['accuracy_all']:.3f} cov={t['coverage']:.3f} "
            f"answered={t['answered_accuracy']:.3f} n={t['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
