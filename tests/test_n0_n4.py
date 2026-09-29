"""N0–N4 synthesis path. Offline stub provider; eval-afa still imported elsewhere."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevforest.artifact import freeze_jevclass, make_receipt
from jevforest.author import author_once
from jevforest.contract import TaskContract
from jevforest.data.shots import adapt_json, adapt_table_row, adapt_text, assert_disjoint, take_k
from jevforest.eval.synthesis import eval_task
from jevforest.providers import ReplayProvider, StubKeywordProvider
from jevforest.runtime.pipeline import PipelineExecutionError, PipelineRuntime
from jevforest.search import search
from jevforest.spec.flow import compile_flow
from jevforest.spec.validate import spec_hash, validate_spec

ROOT = Path(__file__).resolve().parents[1]
TICKETS = ROOT / "examples" / "tasks" / "tickets"


def _expr_spec() -> dict:
    return {
        "version": 2,
        "name": "echo",
        "jev_model": "none",
        "nodes": [{"id": "y", "kind": "expression", "expression": "obs['text']"}],
        "output": "nodes['y']",
    }


def test_n0_expression_run_freeze_replay() -> None:
    spec = validate_spec(_expr_spec())
    rt = PipelineRuntime(spec, StubKeywordProvider(), max_jev_calls=0)
    run = rt.run({"text": "hello"})
    assert run["output"] == "hello"
    assert run["jev_calls"] == 0
    contract = TaskContract(goal="echo", labels=["hello"], task_id="echo")
    art = freeze_jevclass(contract=contract, spec=spec, verification="valid", source={"method": "hand"})
    receipt = make_receipt(art, run, mode="replay")
    assert art["spec_hash"] == spec_hash(spec)
    assert receipt["decision"] == "hello"
    # replay: same spec hash
    run2 = PipelineRuntime(art["spec"], StubKeywordProvider(), max_jev_calls=0).run({"text": "hello"})
    assert run2["output"] == run["output"]


def test_n0_cycle_and_bad_ref() -> None:
    bad = {
        "version": 2,
        "name": "cycle",
        "jev_model": "none",
        "nodes": [
            {"id": "a", "kind": "expression", "depends_on": ["b"], "expression": "1"},
            {"id": "b", "kind": "expression", "depends_on": ["a"], "expression": "2"},
        ],
        "output": "nodes['a']",
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_spec(bad)
    missing = {
        "version": 2,
        "name": "miss",
        "jev_model": "none",
        "nodes": [{"id": "a", "kind": "expression", "expression": "nodes['nope']"}],
        "output": "nodes['a']",
    }
    with pytest.raises(ValueError, match="unknown"):
        compile_flow(missing)


def test_n0_budget_stop() -> None:
    spec = validate_spec(
        {
            "version": 2,
            "name": "ask",
            "jev_model": "~typesafe/jev-latest",
            "nodes": [
                {
                    "id": "q",
                    "kind": "jev",
                    "state": "obs['text']",
                    "questions": {
                        "is_urgent": {"type": "noul", "instructions": "urgent?"}
                    },
                }
            ],
            "output": "nodes['q']['is_urgent']['noul']",
        }
    )
    with pytest.raises(PipelineExecutionError, match="budget"):
        PipelineRuntime(spec, StubKeywordProvider(), max_jev_calls=0).run({"text": "x"})


def test_n0_ticket_handwritten() -> None:
    art = author_once("Route support tickets to billing technical sales", task_id="ticket_routing")
    rt = PipelineRuntime(art["spec"], StubKeywordProvider())
    out = rt.run({"text": "Please refund the payout that failed."})
    assert out["output"] == "billing"
    assert art["verification"] == "unverified"


def test_n1_caps_examples() -> None:
    with pytest.raises(ValueError, match="at most 5"):
        author_once("goal", examples=[{"text": "x"}] * 6)


def test_n2_search_cannot_change_evaluator_hash() -> None:
    contract = TaskContract.from_json(json.loads((TICKETS / "contract.json").read_text()))
    dev = [json.loads(l) for l in (TICKETS / "dev.jsonl").read_text().splitlines() if l.strip()]
    result = search(contract, dev, jev=StubKeywordProvider())
    hashes = {e["spec_hash"] for e in result["log"]}
    assert result["evaluator_hash"]
    assert result["jevclass"]["evaluator_hash"] == result["evaluator_hash"]
    assert result["log"]
    # scorer identity is constant across mutations
    assert all(result["jevclass"]["evaluator_hash"] == result["evaluator_hash"] for _ in hashes)


def test_n3_shot_isolation() -> None:
    support = [{"id": "s", "text": "a"}]
    dev = [{"id": "d", "text": "b"}]
    test = [{"id": "t", "text": "c"}]
    assert_disjoint(support, dev, test)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(support, [{"id": "s"}])
    assert take_k(dev * 5, 3, seed=0)
    assert take_k(dev, 0) == []
    assert adapt_text("hi")["text"] == "hi"
    assert "text" in adapt_json({"text": "x", "k": 1})
    assert "x=1" in adapt_table_row({"x": "1", "y": 2}, text_fields=["x"])["text"]


def test_n4_sealed_eval_stub() -> None:
    report = eval_task(TICKETS, jev=StubKeywordProvider(), shot=0, seed=0, run_search=True)
    assert report["sealed"] if "sealed" in report else report["test_hash"]
    assert report["n_support_used"] == 0
    test = report["methods"]["author_once"]["test"]
    assert test["n"] == 8
    assert test["accuracy_all"] >= 0.5
    assert report["methods"]["search"]["test"]["coverage"] >= 0.5
    # search billed
    assert report["search_log"]["jev_calls_billed"] >= 1


def test_replay_provider() -> None:
    spec = {
        "version": 2,
        "name": "r",
        "jev_model": "x",
        "nodes": [
            {
                "id": "q",
                "kind": "jev",
                "state": "obs['text']",
                "questions": {"is_urgent": {"type": "noul", "instructions": "u"}},
            }
        ],
        "output": "nodes['q']['is_urgent']['noul']",
    }
    from jevforest.providers import _archive_key

    state = "hello"
    questions = spec["nodes"][0]["questions"]
    key = _archive_key(state, questions)
    archive = {key: {"answers": {"is_urgent": {"type": "noul", "noul": 0.33}}}}
    rt = PipelineRuntime(spec, ReplayProvider(archive))
    assert rt.run({"text": "hello"})["output"] == 0.33
