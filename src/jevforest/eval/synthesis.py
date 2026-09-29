"""N4 sealed synthesis eval. Test labels are scoring-only; search never sees them."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jevforest.author import author_once
from jevforest.contract import TaskContract
from jevforest.data.shots import assert_disjoint, load_jsonl, take_k
from jevforest.runtime.pipeline import PipelineRuntime
from jevforest.search import accuracy_evaluator, evaluator_hash, search

Shot = Literal[0, 1, 3, 5]


def _digest_rows(rows: list[dict[str, Any]]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_task_dir(task_dir: str | Path) -> dict[str, Any]:
    root = Path(task_dir)
    contract = TaskContract.from_json(json.loads((root / "contract.json").read_text(encoding="utf-8")))
    support = load_jsonl(root / "support.jsonl")
    dev = load_jsonl(root / "dev.jsonl")
    test = load_jsonl(root / "test.jsonl")
    assert_disjoint(support, dev, test)
    return {
        "contract": contract,
        "support": support,
        "dev": dev,
        "test": test,
        "support_hash": _digest_rows(support),
        "dev_hash": _digest_rows(dev),
        "test_hash": _digest_rows(test),
        "sealed": True,
    }


def run_split(jevclass: dict[str, Any], rows: list[dict[str, Any]], jev: Any) -> dict[str, Any]:
    return accuracy_evaluator(jevclass, rows, jev=jev)


def eval_task(
    task_dir: str | Path,
    *,
    jev: Any,
    shot: Shot = 0,
    seed: int = 0,
    min_coverage: float = 0.5,
    run_search: bool = True,
) -> dict[str, Any]:
    pack = load_task_dir(task_dir)
    contract: TaskContract = pack["contract"]
    support_k = take_k(pack["support"], shot, seed=seed)
    # baselines
    authored = author_once(
        contract.goal,
        examples=support_k,
        labels=contract.labels,
        task_id=contract.task_id,
        oracle=False,
    )
    direct = author_once(
        contract.goal,
        examples=support_k,
        labels=contract.labels,
        task_id=contract.task_id + "_direct",
        oracle=False,
    )
    # "direct" uses the generic single-choice template by tweaking goal? Keep same author for structure;
    # the runtime still asks Jev once per instance. Distinction is search vs author-once vs author-once.

    methods: dict[str, Any] = {
        "author_once": authored,
    }
    if run_search:
        found = search(contract, pack["dev"], jev=jev, seed_spec=authored["spec"], min_coverage=min_coverage)
        methods["search"] = found["jevclass"]
        search_log = found
    else:
        search_log = None

    report_methods = {}
    for name, jc in methods.items():
        dev_m = run_split(jc, pack["dev"], jev)
        test_m = run_split(jc, pack["test"], jev)
        # coverage floor: below min_coverage the test run is marked fail
        test_pass = (test_m["coverage"] >= min_coverage) and (test_m["accuracy_all"] >= 0)
        report_methods[name] = {
            "verification": jc.get("verification"),
            "spec_hash": jc.get("spec_hash"),
            "jevclass_hash": jc.get("jevclass_hash"),
            "dev": dev_m,
            "test": test_m,
            "test_meets_coverage": test_pass,
        }

    return {
        "task_id": contract.task_id,
        "shot": shot,
        "seed": seed,
        "min_coverage": min_coverage,
        "evaluator_hash": evaluator_hash(accuracy_evaluator),
        "support_hash": pack["support_hash"],
        "dev_hash": pack["dev_hash"],
        "test_hash": pack["test_hash"],
        "n_support_used": len(support_k),
        "methods": report_methods,
        "search_log": None if search_log is None else {"log": search_log["log"], "jev_calls_billed": search_log["jev_calls_billed"]},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sealed": True,
        "note": "test labels used only for this sealed report; search saw dev only. stub_keyword accuracy is pipeline consistency, not live Jev quality.",
    }
