"""Offline tests for the Jev Decisions payload builder."""

from __future__ import annotations

import pytest

from jevforest.sdk.jev import (
    DEFAULT_MODEL,
    JevDecisionsClient,
    JevDecisionsError,
    build_decisions_request,
)


def test_build_decisions_request_shape() -> None:
    req = build_decisions_request(
        "payouts failing",
        {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this message convey urgency?",
                "criteria": {"true": "time-sensitive", "false": "no urgency"},
            }
        },
    )
    assert req["model"] == DEFAULT_MODEL
    assert req["state"] == "payouts failing"
    assert req["questions"]["is_urgent"]["type"] == "noul"


def test_empty_questions_rejected() -> None:
    with pytest.raises(ValueError):
        build_decisions_request("x", {})


def test_decide_without_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = JevDecisionsClient(api_key="")
    with pytest.raises(JevDecisionsError):
        client.decide("hello", {"q": {"type": "noul", "instructions": "yes?", "criteria": {"true": "y", "false": "n"}}})
