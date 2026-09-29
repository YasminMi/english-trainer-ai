"""Reusable test doubles for providers and HTTP sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

from english_trainer.llm.base import ChatMessage, GenerationOptions, LLMUnavailableError


@dataclass
class FakeChatModel:
    """Returns scripted replies (or a function of the messages) and records every call."""

    replies: list[str] | Callable[[list[ChatMessage], str | None], str] = field(
        default_factory=list
    )
    name: str = "fake"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        self.calls.append({"messages": list(messages), "system": system, "options": options})
        if callable(self.replies):
            return self.replies(messages, system)
        if not self.replies:
            raise LLMUnavailableError("no scripted reply")
        return self.replies.pop(0)


class OfflineChatModel:
    name = "offline"

    def chat(self, *_args: object, **_kwargs: object) -> str:
        raise LLMUnavailableError("offline")


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession(requests.Session):
    """Routes requests to handlers keyed by URL suffix."""

    def __init__(self, handlers: dict[str, Callable[..., FakeResponse]]) -> None:
        super().__init__()
        self.handlers = handlers
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def _dispatch(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append((method, url, kwargs))
        for suffix, handler in self.handlers.items():
            if url.endswith(suffix) or suffix in url:
                return handler(**kwargs)
        raise requests.ConnectionError(f"no handler for {url}")

    def get(self, url: str | bytes, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._dispatch("GET", str(url), **kwargs)

    def post(self, url: str | bytes, data: Any = None, json: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        return self._dispatch("POST", str(url), json=json, **kwargs)
