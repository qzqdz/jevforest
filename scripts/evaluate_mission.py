#!/usr/bin/env python3
"""Frozen evaluator for mission jevforest-afa-reliability-public."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
MISSION = ROOT / ".omc" / "autoresearch" / "jevforest-afa-reliability-public"
RANDOM_ACC5 = 0.746
CUBE_ACC3_MIN = 0.95


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--cube-json", default=str(ROOT / "results" / "sample_cube_round1_baseline.json"))
    p.add_argument("--miniboone-json", default=str(ROOT / "results" / "sample_miniboone_round1_baseline.json"))
    p.add_argument("--notes", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    pytest = subprocess.run(
        [str(PY), "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    pytest_ok = pytest.returncode == 0

    cube = _load(Path(args.cube_json))
    mb = _load(Path(args.miniboone_json))
    cube_n = len(cube.get("seeds") or [])
    mb_n = len(mb.get("seeds") or [])

    cube_acc3 = next(r["acc_mean"] for r in cube["aggregate"] if r["budget"] == 3)
    mb_acc5 = next(r["acc_mean"] for r in mb["aggregate"] if r["budget"] == 5)
    mb_acc5_seed0 = next(
        next(c["accuracy"] for c in s["curve"] if c["budget"] == 5)
        for s in mb["per_seed"]
        if s["seed"] == 0
    )

    results_md = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_have_numbers = (
        "sample_miniboone" in results_md
        and "Acc@5" in results_md
        and "±" in results_md
    )
    no_overclaim = "尚未实现" in readme or "not implemented" in readme.lower() or "planned, not implemented" in (ROOT / "README_EN.md").read_text(encoding="utf-8").lower()
    tracked_secrets = False
    for pat in ("sk-or-v1-", "OPENROUTER_API_KEY=sk-"):
        for f in (ROOT / "README.md", ROOT / "README_EN.md", ROOT / ".env.example"):
            if f.is_file() and pat in f.read_text(encoding="utf-8"):
                tracked_secrets = True

    gates = {
        "pytest": pytest_ok,
        "cube_acc3_mean": cube_acc3 >= CUBE_ACC3_MIN,
        "miniboone_acc5_vs_random": mb_acc5_seed0 >= RANDOM_ACC5,
        "sampling_cube_seeds": cube_n >= 3,
        "sampling_miniboone_seeds": mb_n >= 3,
        "results_from_json": docs_have_numbers,
        "readme_no_overclaim": no_overclaim,
        "no_secrets": not tracked_secrets,
    }
    score_parts = {
        "pytest": 1.0 if pytest_ok else 0.0,
        "cube_acc3": _clip01(cube_acc3),
        "mb_acc5": _clip01((mb_acc5 - 0.70) / 0.15),
        "sampling": 1.0 if cube_n >= 3 and mb_n >= 3 else 0.0,
        "docs": 1.0 if docs_have_numbers and no_overclaim and not tracked_secrets else 0.0,
    }
    score = sum(score_parts.values()) / len(score_parts)
    passed = all(gates.values())
    payload = {
        "pass": passed,
        "score": round(score, 4),
        "iteration": args.iteration,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "score_parts": score_parts,
        "metrics": {
            "cube_acc3_mean": cube_acc3,
            "cube_n_seeds": cube_n,
            "miniboone_acc5_mean": mb_acc5,
            "miniboone_acc5_seed0": mb_acc5_seed0,
            "miniboone_n_seeds": mb_n,
            "pytest_returncode": pytest.returncode,
        },
        "artifacts": {
            "cube_json": args.cube_json,
            "miniboone_json": args.miniboone_json,
        },
        "notes": args.notes,
        "pytest_tail": (pytest.stdout or pytest.stderr)[-500:],
    }
    out = Path(
        args.out
        or (
            MISSION
            / "runs"
            / "20260928-public"
            / "evaluations"
            / f"iteration-{args.iteration:04d}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pass": passed, "score": payload["score"], "gates": gates}, indent=2))
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
