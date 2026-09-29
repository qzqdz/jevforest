"""Jev answer providers: live OpenRouter, replay, and deterministic stub."""

from __future__ import annotations

from typing import Any, Protocol

from jevforest.sdk.jev import JevDecisionsClient


class JevProvider(Protocol):
    name: str

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        ...


class LiveJevProvider:
    name = "live"

    def __init__(self, client: JevDecisionsClient | None = None) -> None:
        self.client = client or JevDecisionsClient()

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self.client.decide(state, questions)


class ReplayProvider:
    """Play back archived answers keyed by (state_json, question_ids)."""

    name = "replay"

    def __init__(self, archive: dict[str, dict[str, Any]]) -> None:
        self.archive = archive

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        key = _archive_key(state, questions)
        if key not in self.archive:
            raise KeyError(f"no archived answer for {key}")
        return self.archive[key]


class StubKeywordProvider:
    """Offline stand-in for tests and sealed eval. Not a Jev model."""

    name = "stub_keyword"

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        text = _state_text(state).lower()
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            t = q.get("type")
            if t == "noul":
                urgent = any(w in text for w in ("urgent", "asap", "immediately", "down", "outage", "failing"))
                answers[qid] = {"type": "noul", "noul": 0.92 if urgent else 0.12}
            elif t == "choice":
                criteria = q.get("criteria") or {}
                picked = _pick_choice(text, list(criteria))
                probs = {k: (0.8 if k == picked else 0.2 / max(len(criteria) - 1, 1)) for k in criteria}
                answers[qid] = {"type": "choice", "choice": picked, "probabilities": probs}
            elif t == "score":
                levels = list(q.get("criteria") or ["low", "mid", "high"])
                idx = 2 if any(w in text for w in ("angry", "furious", "worst")) else 0
                answers[qid] = {"type": "score", "score": idx, "label": levels[min(idx, len(levels) - 1)]}
            else:
                raise ValueError(f"unknown question type {t}")
        return {"model": "stub_keyword", "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0}}


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        if "text" in state:
            return str(state["text"])
        return " ".join(str(v) for v in state.values())
    return str(state)


def _pick_choice(text: str, keys: list[str]) -> str:
    lowered = {k.lower(): k for k in keys}
    for needle, orig in lowered.items():
        if needle in text:
            return orig
    billing = ("refund", "invoice", "payout", "charge", "billing", "payment")
    tech = ("bug", "outage", "api", "error", "crash", "integration", "technical")
    sales = ("price", "upgrade", "plan", "sales", "quote")
    for words, name in ((billing, "billing"), (tech, "technical"), (sales, "sales")):
        if any(w in text for w in words) and name in lowered:
            return lowered[name]
    if "other" in lowered:
        return lowered["other"]
    return keys[0]


def _archive_key(state: Any, questions: dict[str, Any]) -> str:
    import json

    return json.dumps({"state": state, "q": sorted(questions)}, sort_keys=True)
