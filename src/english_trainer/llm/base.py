"""Provider-agnostic contracts for text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached or returned no usable output."""


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    temperature: float = 0.7
    max_tokens: int = 300
    timeout: float = 120.0


class ChatModel(Protocol):
    """Anything that can answer a chat conversation.

    Implementations raise :class:`LLMUnavailableError` instead of returning
    ``None`` so callers cannot silently mistake an outage for an empty answer.
    """

    @property
    def name(self) -> str: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str: ...
