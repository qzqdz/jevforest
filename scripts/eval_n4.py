#!/usr/bin/env python3
"""N4 sealed eval on examples/tasks/tickets with the stub provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevforest.eval.synthesis import eval_task  # noqa: E402
from jevforest.providers import StubKeywordProvider  # noqa: E402


def main() -> int:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for shot in (0, 1, 3, 5):
        for seed in (0, 1):
            report = eval_task(
                ROOT / "examples" / "tasks" / "tickets",
                jev=StubKeywordProvider(),
                shot=shot,  # type: ignore[arg-type]
                seed=seed,
            )
            reports.append(report)
            print(
                f"shot={shot} seed={seed}  "
                f"author={report['methods']['author_once']['test']['accuracy_all']:.3f}  "
                f"search={report['methods']['search']['test']['accuracy_all']:.3f}"
            )
    payload = {"task": "ticket_routing", "provider": "stub_keyword", "runs": reports}
    path = out_dir / "n4_ticket_routing.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
