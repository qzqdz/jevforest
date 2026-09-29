"""Author-once: goal + context + 0–5 examples → validated JevClass (unverified unless oracle).

Jev does not generate free text. It only answers noul/choice/score, so authoring
picks a frozen template (and optional abstain band) from typed questions.
"""

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


def direct_choice_spec(goal: str, labels: list[str], *, jev_model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Per-instance baseline: one Choice question, no frozen multi-node program."""
    labs = labels or ["yes", "no"]
    return {
        "version": 2,
        "name": "direct_choice",
        "jev_model": jev_model,
        "nodes": [
            {
                "id": "ask",
                "kind": "jev",
                "state": "obs['text']",
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


def _pick_template(goal: str, jev: Any | None) -> tuple[str, dict[str, Any] | None]:
    questions = {
        "is_routing": {
            "type": "noul",
            "instructions": "Is this a routing / queue / department classification task?",
        },
        "template": {
            "type": "choice",
            "instructions": "Which frozen template should author the pipeline?",
            "criteria": {
                "ticket_routing": "Support tickets to billing/technical/sales/other",
                "direct_choice": "Single label choice using the goal as instructions",
            },
        },
    }
    if jev is None:
        g = goal.lower()
        name = "ticket_routing" if any(w in g for w in ("ticket", "queue", "routing", "billing", "department")) else "direct_choice"
        return name, None
    raw = jev.decide(goal, questions)
    answers = raw.get("answers") or {}
    tmpl = (answers.get("template") or {}).get("choice") or "direct_choice"
    if tmpl not in ("ticket_routing", "direct_choice"):
        tmpl = "direct_choice"
    return tmpl, raw


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
    jev: Any | None = None,
    template: str | None = None,
) -> dict[str, Any]:
    examples = list(examples or [])
    if len(examples) > 5:
        raise ValueError("N1/N3: at most 5 support examples")
    assumptions: list[str] = []
    jev_raw = None
    if template is None:
        template, jev_raw = _pick_template(goal, jev)
        assumptions.append("template chosen by Jev noul/choice" if jev_raw else "template chosen by goal keywords (no jev)")
    if template == "ticket_routing":
        spec = ticket_routing_spec(jev_model=jev_model)
        labels = labels or ["billing", "technical", "sales", "other"]
    else:
        labels = labels or ["yes", "no"]
        spec = direct_choice_spec(goal, list(labels), jev_model=jev_model)
        assumptions.append("direct_choice: goal used as Choice instructions")

    validate_spec(spec)
    contract = TaskContract(
        goal=goal,
        labels=list(labels),
        assumptions=assumptions,
        task_id=task_id,
    )
    if context:
        contract.assumptions.append("context keys: " + ",".join(sorted(str(k) for k in context)))
    contract.assumptions.append(f"support_examples={len(examples)}")
    verification = "verified" if oracle else "unverified"
    source = {
        "method": "author_once",
        "n_examples": len(examples),
        "template": spec["name"],
        "author_jev": jev_raw,
    }
    return freeze_jevclass(
        contract=contract,
        spec=spec,
        verification=verification,  # type: ignore[arg-type]
        source=source,
        evaluator_hash=evaluator_hash,
    )
