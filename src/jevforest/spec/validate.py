"""JevHarness v2-compatible pipeline spec validation (expression + jev only)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from jevforest.spec.expr import json_guard, parse_expression
from jevforest.spec.flow import compile_flow


def validate_questions(questions: Any, *, strict: bool = False) -> dict[str, Any]:
    json_guard(questions)
    if not isinstance(questions, dict) or not questions or len(questions) > 255:
        raise ValueError("questions required")
    for qid, q in questions.items():
        if not isinstance(q, dict) or not isinstance(qid, str) or not qid:
            raise ValueError("invalid question")
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            raise ValueError(f"unknown question fields: {sorted(extra)}")
        if not isinstance(q.get("instructions"), str) or not q["instructions"]:
            raise ValueError("instructions required")
        t = q.get("type")
        c = q.get("criteria")
        if t not in ("choice", "score", "noul"):
            raise ValueError("unknown question type")
        if t == "choice" and (
            not isinstance(c, dict) or not 1 <= len(c) <= 255 or not all(isinstance(v, str) for v in c.values())
        ):
            raise ValueError("invalid choice criteria")
        if t == "score" and (not isinstance(c, list) or not 2 <= len(c) <= 10 or not all(isinstance(v, str) for v in c)):
            raise ValueError("invalid score criteria")
        if strict and t == "noul" and "criteria" in q:
            raise ValueError("noul questions have no criteria")
    return copy.deepcopy(questions)


def validate_spec(spec: Any) -> dict[str, Any]:
    json_guard(spec)
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    version = spec.get("version")
    if version not in (2,):
        raise ValueError("jevforest N0 accepts pipeline version 2 only")
    allowed = {"version", "name", "jev_model", "nodes", "output", "memory_update", "abstain_when"}
    extra = set(spec) - allowed
    if extra:
        raise ValueError(f"unknown pipeline fields: {sorted(extra)}")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("pipeline name required")
    if not isinstance(spec.get("jev_model"), str) or not spec["jev_model"]:
        raise ValueError("jev_model required")
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or len(nodes) > 64:
        raise ValueError("nodes must be a list of at most 64")
    ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("node must be an object")
        ident = node.get("id")
        kind = node.get("kind")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("node IDs must be unique")
        if kind == "expression":
            if set(node) - {"id", "kind", "expression", "depends_on"}:
                raise ValueError("unknown expression node fields")
            parse_expression(node.get("expression") or "None")
        elif kind == "jev":
            if set(node) - {"id", "kind", "state", "questions", "depends_on"}:
                raise ValueError("unknown Jev node fields")
            parse_expression(node.get("state", "obs"))
            validate_questions(node.get("questions"))
        else:
            raise ValueError(f"unknown node kind: {kind!r} (v3 python nodes are out of scope)")
        ids.add(ident)
    parse_expression(spec.get("output") or "None")
    if "memory_update" in spec:
        parse_expression(spec["memory_update"])
    if "abstain_when" in spec:
        parse_expression(spec["abstain_when"])
    compile_flow(spec)
    if len(json.dumps(spec)) > 200_000:
        raise ValueError("pipeline too large")
    return copy.deepcopy(spec)


def spec_hash(spec: dict[str, Any]) -> str:
    validated = validate_spec(spec)
    blob = json.dumps(validated, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
