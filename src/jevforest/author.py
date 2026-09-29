"""Author-once: goal + context + 0–5 examples → validated JevClass (unverified unless oracle)."""

from __future__ import annotations

from typing import Any

from jevforest.artifact import freeze_jevclass
from jevforest.contract import TaskContract
from jevforest.spec.validate import validate_spec

DEFAULT_MODEL = "~typesafe/jev-latest"


def ticket_routing_spec(*, jev_model: str = DEFAULT_MODEL, abstain_lo: float = 0.45, abstain_hi: float = 0.55) -> dict[str, Any]:
    return {
        "version": 2,
        "name": "ticket_routing",
        "jev_model": jev_model,
        "nodes": [
            {
                "id": "dept",
                "kind": "jev",
                "state": "obs['text']",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team should handle this message?",
                        "criteria": {
                            "billing": "Payments, invoicing, refunds, payouts",
                            "technical": "Bugs, outages, APIs, integrations",
                            "sales": "Pricing, upgrades, new accounts",
                            "other": "None of the above",
                        },
                    }
                },
            },
            {
                "id": "urgent",
                "kind": "jev",
                "state": "obs['text']",
                "questions": {
                    "is_urgent": {
                        "type": "noul",
                        "instructions": "Does this message convey urgency?",
                    }
                },
            },
            {
                "id": "decision",
                "kind": "expression",
                "depends_on": ["dept"],
                "expression": "nodes['dept']['department']['choice']",
            },
        ],
        "output": "nodes['decision']",
        "abstain_when": (
            f"(nodes['urgent']['is_urgent']['noul'] > {abstain_lo}) and "
            f"(nodes['urgent']['is_urgent']['noul'] < {abstain_hi})"
        ),
    }


def author_once(
    goal: str,
    *,
    context: dict[str, Any] | None = None,
    examples: list[dict[str, Any]] | None = None,
    labels: list[str] | None = None,
    jev_model: str = DEFAULT_MODEL,
    oracle: bool = False,
    evaluator_hash: str | None = None,
    task_id: str = "authored",
) -> dict[str, Any]:
    examples = list(examples or [])
    if len(examples) > 5:
        raise ValueError("N1/N3: at most 5 support examples")
    assumptions: list[str] = []
    g = goal.lower()
    if any(w in g for w in ("ticket", "queue", "routing", "billing", "department")):
        spec = ticket_routing_spec(jev_model=jev_model)
        labels = labels or ["billing", "technical", "sales", "other"]
        assumptions.append("used built-in ticket_routing template")
    else:
        labs = labels or ["yes", "no"]
        spec = {
            "version": 2,
            "name": "generic_choice",
            "jev_model": jev_model,
            "nodes": [
                {
                    "id": "ask",
                    "kind": "jev",
                    "state": "obs['text'] if isinstance(obs, dict) else obs",
                    "questions": {
                        "label": {
                            "type": "choice",
                            "instructions": goal,
                            "criteria": {lab: lab for lab in labs},
                        }
                    },
                }
            ],
            "output": "nodes['ask']['label']['choice']",
        }
        labels = labs
        assumptions.append("generic single-choice template; goal used as instructions")
        # isinstance is not allowed in expressions - fix
        spec["nodes"][0]["state"] = "obs['text']"

    validate_spec(spec)
    contract = TaskContract(
        goal=goal,
        labels=list(labels),
        assumptions=assumptions,
        task_id=task_id,
    )
    if context:
        contract.assumptions.append("context keys: " + ",".join(sorted(context)))
    contract.assumptions.append(f"support_examples={len(examples)}")
    verification = "verified" if oracle else "unverified"
    source = {
        "method": "author_once",
        "n_examples": len(examples),
        "template": spec["name"],
    }
    return freeze_jevclass(
        contract=contract,
        spec=spec,
        verification=verification,  # type: ignore[arg-type]
        source=source,
        evaluator_hash=evaluator_hash,
    )
