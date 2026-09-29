"""Turn a natural-language goal into a v2 Jev spec (no named task templates)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from jevforest.artifact import freeze_jevclass
from jevforest.contract import TaskContract
from jevforest.spec.validate import validate_spec
from jevforest.sdk.jev import ENV_API_KEY, DEFAULT_MODEL

AUTHOR_CHAT_MODEL = os.environ.get("JEVFOREST_AUTHOR_MODEL", "openai/gpt-4o-mini")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def extract_labels(goal: str) -> list[str]:
    """Pull class names out of the goal sentence (to A, B, or C / A / B / C)."""
    stop = {"the", "a", "an", "to", "or", "and", "queues", "queue", "labels", "classes", "into", "as"}
    m = re.search(r"\b(?:to|into|as)\s+(.+?)(?:\.|$)", goal, re.I)
    chunk = m.group(1) if m else goal
    parts = re.split(r",|/|\bor\b|\band\b", chunk, flags=re.I)
    labs: list[str] = []
    for p in parts:
        tok = re.sub(r"[^a-z0-9_]+", " ", p.lower()).strip()
        words = [w for w in tok.split() if w and w not in stop]
        if len(words) == 1:
            labs.append(words[0])
        elif 2 <= len(words) <= 4 and all(len(w) >= 3 for w in words):
            labs.extend(words)
    # drop leftovers like "otherqueues"
    labs = [x.replace("queues", "").replace("queue", "") for x in labs]
    labs = [x for x in labs if x and x not in stop]
    seen: list[str] = []
    for x in labs:
        if x not in seen:
            seen.append(x)
    return seen if len(seen) >= 2 else []


def spec_from_goal(
    goal: str,
    labels: list[str],
    *,
    jev_model: str = DEFAULT_MODEL,
    include_urgency: bool | None = None,
) -> dict[str, Any]:
    if include_urgency is None:
        include_urgency = bool(re.search(r"abstain|urgent|unclear", goal, re.I))
    labs = list(labels)
    criteria = {lab: f"Messages that belong to {lab}" for lab in labs}
    nodes: list[dict[str, Any]] = [
        {
            "id": "label",
            "kind": "jev",
            "state": "obs['text']",
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": goal,
                    "criteria": criteria,
                }
            },
        }
    ]
    spec: dict[str, Any] = {
        "version": 2,
        "name": "goal_authored",
        "jev_model": jev_model,
        "nodes": nodes,
        "output": "nodes['label']['label']['choice']",
    }
    if include_urgency:
        nodes.append(
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
            }
        )
        spec["abstain_when"] = (
            "(nodes['urgent']['is_urgent']['noul'] > 0.45) and "
            "(nodes['urgent']['is_urgent']['noul'] < 0.55)"
        )
    return spec


def spec_from_chat(goal: str, labels: list[str], examples: list[dict[str, Any]], *, jev_model: str) -> dict[str, Any] | None:
    key = os.environ.get(ENV_API_KEY, "")
    if not key:
        return None
    prompt = {
        "goal": goal,
        "labels": labels,
        "examples": examples[:5],
        "schema": "Return ONLY a JSON object with keys instructions (string) and criteria (object label->description).",
    }
    body = json.dumps(
        {
            "model": AUTHOR_CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You write Jev Choice questions. Reply with JSON only.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        CHAT_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None
    try:
        content = raw["choices"][0]["message"]["content"]
        if content.strip().startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content.strip())
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
        instructions = str(parsed.get("instructions") or goal)
        criteria = parsed.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            return None
        criteria = {str(k): str(v) for k, v in criteria.items()}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    spec = spec_from_goal(goal, list(criteria), jev_model=jev_model, include_urgency=None)
    spec["nodes"][0]["questions"]["label"]["instructions"] = instructions
    spec["nodes"][0]["questions"]["label"]["criteria"] = criteria
    spec["name"] = "goal_authored_chat"
    return spec


def direct_choice_spec(goal: str, labels: list[str], *, jev_model: str = DEFAULT_MODEL) -> dict[str, Any]:
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
    jev: Any | None = None,  # noqa: ARG001 — reserved; Jev cannot emit text specs
    template: str | None = None,  # noqa: ARG001
    use_chat: bool = True,
) -> dict[str, Any]:
    examples = list(examples or [])
    if len(examples) > 5:
        raise ValueError("N1/N3: at most 5 support examples")
    assumptions: list[str] = []
    labs = list(labels) if labels else extract_labels(goal)
    if len(labs) < 2:
        labs = ["yes", "no"]
        assumptions.append("could not parse labels from goal; defaulted to yes/no")
    else:
        assumptions.append(f"labels parsed from goal: {labs}")

    spec = None
    if use_chat:
        spec = spec_from_chat(goal, labs, examples, jev_model=jev_model)
        if spec is not None:
            assumptions.append(f"choice instructions authored by {AUTHOR_CHAT_MODEL}")
    if spec is None:
        spec = spec_from_goal(goal, labs, jev_model=jev_model)
        assumptions.append("spec compiled from goal text (no chat author)")

    validate_spec(spec)
    contract = TaskContract(goal=goal, labels=labs, assumptions=assumptions, task_id=task_id)
    if context:
        contract.assumptions.append("context keys: " + ",".join(sorted(str(k) for k in context)))
    contract.assumptions.append(f"support_examples={len(examples)}")
    return freeze_jevclass(
        contract=contract,
        spec=spec,
        verification="verified" if oracle else "unverified",  # type: ignore[arg-type]
        source={"method": "author_once", "n_examples": len(examples), "spec_name": spec["name"]},
        evaluator_hash=evaluator_hash,
    )
