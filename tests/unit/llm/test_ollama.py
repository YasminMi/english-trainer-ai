from __future__ import annotations

from typing import Any

import pytest
import requests

from english_trainer.config import OllamaSettings
from english_trainer.llm.base import GenerationOptions, LLMUnavailableError
from english_trainer.llm.ollama import OllamaClient
from tests.fakes import FakeResponse, FakeSession


def _client(session: FakeSession) -> OllamaClient:
    return OllamaClient(
        OllamaSettings(url="http://ollama:11434", model="llama3.2"), session=session
    )


def test_chat_sends_system_prompt_and_options() -> None:
    session = FakeSession(
        {"/api/chat": lambda **_: FakeResponse({"message": {"content": "  Hi there!  "}})}
    )

    text = _client(session).chat(
        [{"role": "user", "content": "Hello"}],
        system="Be kind.",
        options=GenerationOptions(temperature=0.0, max_tokens=50, timeout=5),
    )

    assert text == "Hi there!"
    _, url, kwargs = session.requests[0]
    assert url == "http://ollama:11434/api/chat"
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "Be kind."}
    assert kwargs["json"]["options"] == {"temperature": 0.0, "num_predict": 50}
    assert kwargs["timeout"] == 5


@pytest.mark.parametrize(
    "handler",
    [
        lambda **_: FakeResponse({"message": {"content": ""}}),
        lambda **_: FakeResponse({}, status_code=500),
        lambda **_: FakeResponse(ValueError("not json")),
    ],
)
def test_chat_failures_raise_unavailable(handler: Any) -> None:
    with pytest.raises(LLMUnavailableError):
        _client(FakeSession({"/api/chat": handler})).chat([{"role": "user", "content": "Hi"}])


def test_connection_errors_raise_unavailable_with_context() -> None:
    def refuse(**_: Any) -> FakeResponse:
        raise requests.ConnectionError("refused")

    with pytest.raises(LLMUnavailableError, match=r"llama3\.2"):
        _client(FakeSession({"/api/chat": refuse})).chat([{"role": "user", "content": "Hi"}])


def test_is_available() -> None:
    assert _client(FakeSession({"/api/tags": lambda **_: FakeResponse({})})).is_available()
    assert not _client(FakeSession({})).is_available()
