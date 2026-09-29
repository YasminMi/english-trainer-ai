from __future__ import annotations

from typing import Any

import pytest
import requests

from english_trainer.llm.gateway import TextGenerator
from english_trainer.services.dictionary import OnlineDictionary, suggest_similar
from english_trainer.services.writing import TOPICS, WritingService, resolve_topic
from tests.fakes import FakeChatModel, FakeResponse, FakeSession


def test_resolve_topic_accepts_numbers_and_free_text() -> None:
    assert resolve_topic("1") == TOPICS[0]
    assert resolve_topic("99") == "99"
    assert resolve_topic("my trip") == "my trip"
    assert resolve_topic("  ") == "anything"


def test_writing_review_embeds_topic_and_text() -> None:
    model = FakeChatModel(replies=["AVALIACAO: OTIMO"])

    result = WritingService(TextGenerator([model])).review("I  like   pizza.", "food")

    assert result is not None and result.text == "AVALIACAO: OTIMO"
    prompt = model.calls[0]["messages"][0]["content"]
    assert "Topic: food" in prompt
    assert "Student wrote: I like pizza." in prompt


def test_writing_review_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="Write something"):
        WritingService(TextGenerator([FakeChatModel()])).review("   ", "food")


API_PAYLOAD = [
    {
        "phonetic": "/ʃeə/",
        "meanings": [
            {
                "partOfSpeech": "noun",
                "definitions": [{"definition": "A seat.", "example": "Sit on the chair."}],
            }
        ],
    }
]


def test_online_dictionary_parses_the_first_meaning() -> None:
    session = FakeSession({"/entries/en/chair": lambda **_: FakeResponse(API_PAYLOAD)})

    entry = OnlineDictionary(session=session).lookup(" Chair ")

    assert entry is not None
    assert entry.part_of_speech == "noun"
    assert entry.definition == "A seat."
    assert entry.example == "Sit on the chair."


@pytest.mark.parametrize(
    "response",
    [FakeResponse([], status_code=404), FakeResponse([]), FakeResponse([{"meanings": []}])],
)
def test_online_dictionary_returns_none_for_missing_words(response: FakeResponse) -> None:
    session = FakeSession({"/entries/en/": lambda **_: response})
    assert OnlineDictionary(session=session).lookup("zzz") is None


def test_online_dictionary_survives_network_errors() -> None:
    def fail(**_: Any) -> FakeResponse:
        raise requests.Timeout

    assert OnlineDictionary(session=FakeSession({"/entries/": fail})).lookup("chair") is None


def test_suggest_similar() -> None:
    assert suggest_similar("chiar", ["chair", "table", "cheer"]) == ["chair", "cheer"]
