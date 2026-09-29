"""TaskContract: frozen task description. Structural valid ≠ behavior verified."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class TaskContract:
    goal: str
    labels: list[str]
    input_schema: dict[str, str] = field(default_factory=lambda: {"text": "string"})
    unknown_policy: str = "abstain"
    forbidden: list[str] = field(default_factory=list)
    success: str = "label matches independent oracle on the evaluation split"
    evidence: str = "independent labels; Jev answers are not the oracle"
    build_budget: dict[str, int] = field(
        default_factory=lambda: {"max_iterations": 4, "max_jev_calls": 40, "max_seconds": 120}
    )
    run_budget: dict[str, int] = field(default_factory=lambda: {"max_jev_calls": 8, "max_ms": 30_000})
    assumptions: list[str] = field(default_factory=list)
    task_id: str = "untitled"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TaskContract":
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in payload.items() if k in known}
        return cls(**data)

    def digest(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()


Verification = Literal["valid", "verified", "unverified"]
