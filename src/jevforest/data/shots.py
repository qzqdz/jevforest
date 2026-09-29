"""N3: k-shot isolation. Support / dev / test never mix."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def take_k(rows: list[dict[str, Any]], k: Literal[0, 1, 3, 5], *, seed: int = 0) -> list[dict[str, Any]]:
    if k == 0:
        return []
    if k not in (1, 3, 5):
        raise ValueError("shot k must be 0, 1, 3 or 5")
    if len(rows) < k:
        raise ValueError(f"need {k} support rows, have {len(rows)}")
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:k]]


def assert_disjoint(*splits: list[dict[str, Any]], key: str = "id") -> None:
    seen: set[Any] = set()
    for split in splits:
        ids = [r.get(key, json.dumps(r, sort_keys=True)) for r in split]
        for i in ids:
            if i in seen:
                raise ValueError(f"split leakage on {key}={i!r}")
            seen.add(i)


def adapt_text(text: str, *, source: str = "user") -> dict[str, Any]:
    return {"text": text, "evidence": [{"source": source, "kind": "text"}]}


def adapt_json(obj: dict[str, Any], *, source: str = "user") -> dict[str, Any]:
    text = obj.get("text")
    if text is None:
        text = json.dumps(obj, ensure_ascii=False)
    return {"text": str(text), "payload": obj, "evidence": [{"source": source, "kind": "json"}]}


def adapt_table_row(row: dict[str, Any], *, text_fields: list[str] | None = None, source: str = "table") -> dict[str, Any]:
    fields = text_fields or [k for k, v in row.items() if isinstance(v, str)]
    text = " | ".join(f"{k}={row[k]}" for k in fields if k in row)
    return {"text": text, "payload": row, "evidence": [{"source": source, "kind": "table", "fields": fields}]}
