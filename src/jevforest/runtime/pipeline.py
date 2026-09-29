"""Sequential v2 DAG runtime. Jev is injected; never created here."""

from __future__ import annotations

import time
from typing import Any

from jevforest.spec.expr import evaluate_expression, json_guard
from jevforest.spec.flow import compile_flow, topological_order
from jevforest.spec.validate import validate_spec


class PipelineExecutionError(ValueError):
    def __init__(self, message: str, *, partial_result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.partial_result = partial_result or {}


class PipelineRuntime:
    def __init__(self, spec: dict[str, Any], jev: Any, *, max_jev_calls: int = 8) -> None:
        self.spec = validate_spec(spec)
        self.jev = jev
        self.max_jev_calls = int(max_jev_calls)
        self.dependencies = compile_flow(self.spec)
        self.order = topological_order(self.dependencies)
        self._nodes = {n["id"]: n for n in self.spec["nodes"]}

    def run(self, obs: Any, memory: dict[str, Any] | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        json_guard(obs)
        json_guard(memory or {})
        outputs: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        jev_calls = 0
        for nid in self.order:
            node = self._nodes[nid]
            visible = {d: outputs[d] for d in self.dependencies[nid]}
            ctx = {"obs": obs, "nodes": visible, "memory": memory or {}}
            started = time.perf_counter()
            try:
                if node["kind"] == "expression":
                    result = evaluate_expression(node["expression"], ctx)
                    kind = "expression"
                else:
                    if jev_calls >= self.max_jev_calls:
                        raise PipelineExecutionError(
                            f"run budget exceeded: max_jev_calls={self.max_jev_calls}",
                            partial_result={"nodes": outputs, "trace": trace},
                        )
                    state = evaluate_expression(node.get("state", "obs"), ctx)
                    raw = self.jev.decide(state, node["questions"])
                    jev_calls += 1
                    result = raw.get("answers", raw)
                    kind = "jev"
                outputs[nid] = result
                trace.append(
                    {
                        "id": nid,
                        "kind": kind,
                        "ok": True,
                        "output": result,
                        "ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
            except PipelineExecutionError:
                raise
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "id": nid,
                    "kind": node["kind"],
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ms": round((time.perf_counter() - started) * 1000, 3),
                }
                trace.append(rec)
                raise PipelineExecutionError(
                    f"node {nid} failed: {exc}",
                    partial_result={"nodes": outputs, "trace": trace},
                ) from exc

        ctx_all = {"obs": obs, "nodes": outputs, "memory": memory or {}}
        abstain = False
        if self.spec.get("abstain_when"):
            abstain = bool(evaluate_expression(self.spec["abstain_when"], ctx_all))
        output = evaluate_expression(self.spec["output"], ctx_all)
        updated = memory or {}
        if self.spec.get("memory_update"):
            updated = evaluate_expression(self.spec["memory_update"], ctx_all)
        stop = "abstain" if abstain else "output"
        return {
            "output": None if abstain else output,
            "abstain": abstain,
            "memory": updated,
            "nodes": outputs,
            "trace": trace,
            "jev_calls": jev_calls,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "stop_reason": stop,
        }
