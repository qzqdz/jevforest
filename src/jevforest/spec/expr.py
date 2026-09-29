"""Bounded JSON-safe expression interpreter (JevHarness v2 subset)."""

from __future__ import annotations

import ast
import math
import statistics
from typing import Any

MAX_ITEMS = 10_000
MAX_TEXT = 100_000
MAX_JSON_VISITS = 100_000
MAX_JSON_TEXT = 1_000_000


def json_guard(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    if budget is None:
        budget = [MAX_JSON_VISITS, MAX_JSON_TEXT]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("total JSON element limit exceeded")
    if depth > 30:
        raise ValueError("JSON nesting limit exceeded")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, int) and value.bit_length() > 1024:
            raise ValueError("integer too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite number")
    elif isinstance(value, str):
        if len(value) > MAX_TEXT:
            raise ValueError("text too long")
        budget[1] -= len(value)
        if budget[1] < 0:
            raise ValueError("total JSON text limit exceeded")
    elif isinstance(value, (list, dict)):
        if len(value) > MAX_ITEMS:
            raise ValueError("collection too large")
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for k, v in items:
            if isinstance(value, dict) and not isinstance(k, str):
                raise ValueError("JSON keys must be strings")
            json_guard(v, depth + 1, budget)
    else:
        raise ValueError("only JSON values are allowed")
    return value


def _last(seq: Any, default: Any = 0) -> Any:
    return seq[-1] if seq else default


def _get(obj: Any, key: Any, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


FUNCS = {
    "min": min,
    "max": max,
    "abs": abs,
    "sum": sum,
    "len": len,
    "mean": statistics.mean,
    "last": _last,
    "get": _get,
    "clip": lambda x, lo, hi: min(hi, max(lo, x)),
}
_ALLOWED = (
    ast.Expression,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
)


def parse_expression(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression or len(expression) > 12_000:
        raise ValueError("invalid expression length")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid expression syntax") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 512:
        raise ValueError("expression too complex")
    for node in nodes:
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"obs", "nodes", "memory", "unknown", *FUNCS}:
            raise ValueError(f"unknown expression name: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords
        ):
            raise ValueError("disallowed function call")
    return tree  # type: ignore[return-value]


def evaluate_expression(expression: str, context: dict[str, Any]) -> Any:
    tree = parse_expression(expression)
    json_guard(context)

    def visit(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant):
            return json_guard(n.value)
        if isinstance(n, ast.Name):
            if n.id == "unknown":
                return None
            if n.id in ("obs", "nodes", "memory"):
                return context[n.id]
            raise ValueError("function name is not a value")
        if isinstance(n, (ast.List, ast.Tuple)):
            return json_guard([visit(x) for x in n.elts])
        if isinstance(n, ast.Dict):
            return json_guard({visit(k): visit(v) for k, v in zip(n.keys, n.values) if k is not None})
        if isinstance(n, ast.Attribute):
            obj = visit(n.value)
            if not isinstance(obj, dict):
                raise ValueError("attribute access requires an object")
            return json_guard(obj.get(n.attr))
        if isinstance(n, ast.Subscript):
            val = visit(n.value)
            sl = n.slice
            key = visit(sl)
            try:
                return json_guard(val[key])
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("subscript failed") from exc
        if isinstance(n, ast.IfExp):
            return visit(n.body if visit(n.test) else n.orelse)
        if isinstance(n, ast.UnaryOp):
            x = visit(n.operand)
            if isinstance(n.op, ast.Not):
                return not x
            return json_guard(-x if isinstance(n.op, ast.USub) else +x)
        if isinstance(n, ast.BoolOp):
            x: Any = True if isinstance(n.op, ast.And) else False
            for child in n.values:
                x = visit(child)
                if isinstance(n.op, ast.And) and not x:
                    return x
                if isinstance(n.op, ast.Or) and x:
                    return x
            return x
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add):
                x = a + b
            elif isinstance(n.op, ast.Sub):
                x = a - b
            elif isinstance(n.op, ast.Mult):
                x = a * b
            elif isinstance(n.op, ast.Div):
                x = a / b
            else:
                x = a % b
            return json_guard(x)
        if isinstance(n, ast.Compare):
            left = visit(n.left)
            for op, right_node in zip(n.ops, n.comparators):
                right = visit(right_node)
                if isinstance(op, ast.Eq):
                    yes = left == right
                elif isinstance(op, ast.NotEq):
                    yes = left != right
                elif isinstance(op, ast.Lt):
                    yes = left < right
                elif isinstance(op, ast.LtE):
                    yes = left <= right
                elif isinstance(op, ast.Gt):
                    yes = left > right
                elif isinstance(op, ast.GtE):
                    yes = left >= right
                elif isinstance(op, ast.In):
                    yes = left in right
                elif isinstance(op, ast.NotIn):
                    yes = left not in right
                elif isinstance(op, ast.Is):
                    yes = left is right
                else:
                    yes = left is not right
                if not yes:
                    return False
                left = right
            return True
        if isinstance(n, ast.Call):
            assert isinstance(n.func, ast.Name)
            return json_guard(FUNCS[n.func.id](*(visit(x) for x in n.args)))
        raise ValueError("unsupported expression")

    try:
        return json_guard(visit(tree))
    except (KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ValueError(f"expression failed: {type(exc).__name__}") from exc
