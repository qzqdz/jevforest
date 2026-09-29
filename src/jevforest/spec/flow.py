"""v2 DAG compile: explicit depends_on ∪ static nodes['id'] references."""

from __future__ import annotations

import ast
from typing import Any

from jevforest.spec.expr import parse_expression

NODES_NAME = "nodes"


def expression_dependencies(expression: str, *, strict: bool = True) -> set[str]:
    tree = parse_expression(expression)
    deps: set[str] = set()
    consumed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == NODES_NAME:
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                deps.add(sl.value)
                consumed.add(id(node.value))
            elif strict:
                raise ValueError("dynamic nodes[...] access is not allowed")
    if strict:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == NODES_NAME and id(node) not in consumed:
                raise ValueError("bare 'nodes' reference is not allowed")
    return deps


def _node_expressions(node: dict[str, Any]) -> list[str]:
    kind = node.get("kind")
    if kind == "expression":
        return [str(node.get("expression", "None"))]
    if kind == "jev":
        return [str(node.get("state", "obs"))]
    return []


def compile_flow(spec: dict[str, Any]) -> dict[str, list[str]]:
    nodes = spec.get("nodes") or []
    ids: list[str] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"node {i} must be a dict")
        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            raise ValueError(f"node {i} must have a string id")
        if nid in ids:
            raise ValueError(f"duplicate node id: {nid}")
        ids.append(nid)
    idset = set(ids)
    deps: dict[str, set[str]] = {nid: set() for nid in ids}
    for node in nodes:
        nid = node["id"]
        explicit = node.get("depends_on") or []
        if not isinstance(explicit, list) or not all(isinstance(x, str) for x in explicit):
            raise ValueError(f"node {nid}: depends_on must be a list of strings")
        for d in explicit:
            if d not in idset:
                raise ValueError(f"node {nid}: unknown depends_on {d!r}")
            if d == nid:
                raise ValueError(f"node {nid}: self-dependency")
            deps[nid].add(d)
        for expr in _node_expressions(node):
            for d in expression_dependencies(expr, strict=True):
                if d not in idset:
                    raise ValueError(f"node {nid}: unknown nodes[{d!r}]")
                if d == nid:
                    raise ValueError(f"node {nid}: self-reference")
                deps[nid].add(d)
    # cycle
    visiting: set[str] = set()
    seen: set[str] = set()

    def walk(n: str) -> None:
        if n in seen:
            return
        if n in visiting:
            raise ValueError(f"cycle involving {n}")
        visiting.add(n)
        for d in deps[n]:
            walk(d)
        visiting.remove(n)
        seen.add(n)

    for nid in ids:
        walk(nid)
    return {nid: sorted(deps[nid]) for nid in ids}


def topological_order(dependencies: dict[str, list[str]]) -> list[str]:
    pending = {k: set(v) for k, v in dependencies.items()}
    order: list[str] = []
    ready = [k for k, v in pending.items() if not v]
    ready.sort()
    while ready:
        n = ready.pop(0)
        order.append(n)
        for k, v in pending.items():
            if n in v:
                v.remove(n)
                if not v and k not in order and k not in ready:
                    ready.append(k)
                    ready.sort()
    if len(order) != len(dependencies):
        raise ValueError("cycle in dependency graph")
    return order
