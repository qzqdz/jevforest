"""Receipt + freeze helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from jevforest.contract import TaskContract, Verification
from jevforest.spec.validate import spec_hash, validate_spec


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def freeze_jevclass(
    *,
    contract: TaskContract,
    spec: dict[str, Any],
    verification: Verification,
    source: dict[str, Any],
    evaluator_hash: str | None = None,
) -> dict[str, Any]:
    spec = validate_spec(spec)
    sh = spec_hash(spec)
    artifact = {
        "kind": "JevClass",
        "version": "0.1.0",
        "contract": contract.to_json(),
        "contract_hash": contract.digest(),
        "spec": spec,
        "spec_hash": sh,
        "evaluator_hash": evaluator_hash,
        "verification": verification,
        "source": source,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact["jevclass_hash"] = _digest(
        {k: artifact[k] for k in ("kind", "contract_hash", "spec_hash", "evaluator_hash", "verification")}
    )
    return artifact


def make_receipt(
    jevclass: dict[str, Any],
    run: dict[str, Any],
    *,
    mode: str = "fresh",
    input_ref: str | None = None,
) -> dict[str, Any]:
    return {
        "jevclass_hash": jevclass.get("jevclass_hash"),
        "spec_hash": jevclass.get("spec_hash"),
        "verification": jevclass.get("verification"),
        "mode": mode,
        "input_ref": input_ref,
        "decision": run.get("output"),
        "abstain": bool(run.get("abstain")),
        "stop_reason": run.get("stop_reason"),
        "jev_calls": run.get("jev_calls"),
        "elapsed_ms": run.get("elapsed_ms"),
        "trace": run.get("trace"),
        "receipt_hash": _digest(
            {
                "jevclass_hash": jevclass.get("jevclass_hash"),
                "decision": run.get("output"),
                "abstain": run.get("abstain"),
                "mode": mode,
            }
        ),
    }
