"""N2: frozen-evaluator propose / evaluate / select / revise. Candidates cannot edit the scorer."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from collections.abc import Callable
from typing import Any

from jevforest.artifact import freeze_jevclass
from jevforest.author import author_once
from jevforest.contract import TaskContract
from jevforest.runtime.pipeline import PipelineExecutionError, PipelineRuntime
from jevforest.spec.validate import spec_hash, validate_spec

Evaluator = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def evaluator_hash(fn: Evaluator) -> str:
    src = inspect.getsource(fn)
    return hashlib.sha256(src.encode()).hexdigest()


def accuracy_evaluator(jevclass: dict[str, Any], rows: list[dict[str, Any]], *, jev: Any) -> dict[str, Any]:
    """Label accuracy on provided rows. Abstain counts as incorrect unless label is None."""
    rt = PipelineRuntime(jevclass["spec"], jev, max_jev_calls=int(jevclass["contract"]["run_budget"]["max_jev_calls"]))
    correct = 0
    abstain = 0
    n = 0
    for row in rows:
        obs = {"text": row["text"]} if "text" in row else row
        try:
            out = rt.run(obs)
        except PipelineExecutionError:
            n += 1
            continue
        n += 1
        if out["abstain"]:
            abstain += 1
            continue
        if out["output"] == row.get("label"):
            correct += 1
    acc = correct / n if n else 0.0
    coverage = (n - abstain) / n if n else 0.0
    return {
        "n": n,
        "accuracy_all": acc,
        "coverage": coverage,
        "abstain_rate": abstain / n if n else 0.0,
        "answered_accuracy": (correct / (n - abstain)) if (n - abstain) else 0.0,
    }


def _mutate(spec: dict[str, Any], i: int) -> dict[str, Any]:
    s = copy.deepcopy(spec)
    if i % 4 == 1:
        s["abstain_when"] = "False"
        s.pop("abstain_when", None)
    elif i % 4 == 2:
        s["nodes"] = [n for n in s["nodes"] if n.get("id") != "urgent"]
        s.pop("abstain_when", None)
    elif i % 4 == 3:
        # rebuild from seed labels if present
        s["abstain_when"] = "False"
    else:
        s.pop("abstain_when", None)
    if "abstain_when" in s and s["abstain_when"] == "False":
        s.pop("abstain_when", None)
    return validate_spec(s)


def search(
    contract: TaskContract,
    dev_rows: list[dict[str, Any]],
    *,
    jev: Any,
    evaluator: Evaluator = accuracy_evaluator,
    seed_spec: dict[str, Any] | None = None,
    min_coverage: float = 0.5,
) -> dict[str, Any]:
    t0 = time.time()
    ev_hash = evaluator_hash(evaluator)
    budget = contract.build_budget
    max_iter = int(budget.get("max_iterations", 4))
    max_calls_note = int(budget.get("max_jev_calls", 40))
    log: list[dict[str, Any]] = []
    if seed_spec is None:
        seed = author_once(contract.goal, labels=contract.labels, task_id=contract.task_id, oracle=False)
        candidates = [seed["spec"]]
    else:
        candidates = [validate_spec(seed_spec)]

    best: dict[str, Any] | None = None
    best_score = -1.0
    calls = 0
    for i in range(max_iter):
        spec = candidates[min(i, len(candidates) - 1)] if i == 0 else _mutate(candidates[0], i)
        frozen = freeze_jevclass(
            contract=contract,
            spec=spec,
            verification="valid",
            source={"method": "search", "iteration": i, "evaluator_hash": ev_hash},
            evaluator_hash=ev_hash,
        )
        # wrap evaluator so it receives jev
        metrics = evaluator(frozen, dev_rows, jev=jev) if "jev" in inspect.signature(evaluator).parameters else evaluator(frozen, dev_rows)
        calls += int(metrics.get("n") or 0)
        # score: accuracy_all with coverage floor
        cov = float(metrics.get("coverage") or 0)
        acc = float(metrics.get("accuracy_all") or 0)
        score = acc if cov >= min_coverage else -1.0
        entry = {
            "iteration": i,
            "spec_hash": spec_hash(spec),
            "metrics": metrics,
            "score": score,
            "kept": False,
        }
        if score > best_score:
            best_score = score
            best = frozen
            entry["kept"] = True
        log.append(entry)
        if time.time() - t0 > float(budget.get("max_seconds", 120)):
            break
        if calls >= max_calls_note:
            break

    if best is None:
        best = freeze_jevclass(
            contract=contract,
            spec=candidates[0],
            verification="unverified",
            source={"method": "search", "failed": True, "evaluator_hash": ev_hash},
            evaluator_hash=ev_hash,
        )
        best["verification"] = "unverified"
    else:
        best = dict(best)
        best["verification"] = "verified"
        best["source"] = {
            "method": "search",
            "evaluator_hash": ev_hash,
            "log": log,
            "jev_calls_billed": calls,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    return {"jevclass": best, "log": log, "evaluator_hash": ev_hash, "jev_calls_billed": calls}
