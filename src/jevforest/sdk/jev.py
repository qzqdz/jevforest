"""OpenRouter Decisions API client for TypeSafe Jev.

Jev does not generate chat text. It answers typed questions about a state
(noul / choice / score) and returns calibrated probabilities. The API key
must come from the environment — never commit it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "~typesafe/jev-latest"
DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "JEV_MODEL"


class JevDecisionsError(RuntimeError):
    """HTTP or schema error from the Decisions API."""


def build_decisions_request(
    state: Any,
    questions: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    if not questions:
        raise ValueError("questions must be a non-empty dict")
    return {"model": model, "state": state, "questions": questions}


class JevDecisionsClient:
    """Minimal client for POST /api/alpha/decisions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        url: str = DEFAULT_URL,
        timeout: float = 60.0,
        referer: str | None = None,
        title: str | None = None,
        retries: int = 4,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY, "")
        self.model = model or os.environ.get(ENV_MODEL, DEFAULT_MODEL)
        self.url = url
        self.timeout = float(timeout)
        self.referer = referer
        self.title = title
        self.retries = int(retries)

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise JevDecisionsError(
                f"{ENV_API_KEY} is not set. Copy .env.example to .env and add a key."
            )
        payload = build_decisions_request(state, questions, model=self.model)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        last: Exception | None = None
        for attempt in range(max(1, self.retries)):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    status = getattr(resp, "status", 200)
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise JevDecisionsError(f"expected object, got {type(parsed)}")
                return parsed
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last = JevDecisionsError(f"HTTP {exc.code}: {body[:500]}")
                if exc.code < 500 and exc.code != 429:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last = JevDecisionsError(f"request failed: {exc}")
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        assert last is not None
        raise last


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "ENV_API_KEY",
    "ENV_MODEL",
    "JevDecisionsClient",
    "JevDecisionsError",
    "build_decisions_request",
]
