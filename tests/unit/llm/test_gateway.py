from __future__ import annotations

from pathlib import Path

import pytest

from english_trainer.config import Settings
from english_trainer.llm.cache import ResponseCache
from english_trainer.llm.gateway import TextGenerator, build_text_generator, strip_reasoning
from english_trainer.llm.local_examples import LocalExamplesModel
from tests.fakes import FakeChatModel, OfflineChatModel


def test_falls_back_to_the_next_provider() -> None:
    backup = FakeChatModel(replies=["from backup"], name="backup")

    result = TextGenerator([OfflineChatModel(), backup]).generate("hello")

    assert result is not None
    assert result.text == "from backup"
    assert result.provider == "backup"


def test_returns_none_when_every_provider_fails() -> None:
    assert TextGenerator([OfflineChatModel()]).generate("hello") is None


def test_generate_is_deterministic_and_cached(tmp_path: Path) -> None:
    model = FakeChatModel(replies=["first answer"])
    generator = TextGenerator([model], cache=ResponseCache(tmp_path / "cache.json"))

    first = generator.generate("same prompt")
    second = generator.generate("same prompt")

    assert first is not None and second is not None
    assert second.text == "first answer"
    assert second.cached is True
    assert len(model.calls) == 1
    assert model.calls[0]["options"].temperature == 0.0


def test_chat_is_never_cached(tmp_path: Path) -> None:
    model = FakeChatModel(replies=["one", "two"])
    generator = TextGenerator([model], cache=ResponseCache(tmp_path / "cache.json"))
    messages = [{"role": "user", "content": "hi"}]

    assert generator.chat(messages).text == "one"  # type: ignore[union-attr, arg-type]
    assert generator.chat(messages).text == "two"  # type: ignore[union-attr, arg-type]


def test_reasoning_blocks_are_removed() -> None:
    model = FakeChatModel(replies=["<think>\nplanning...\n</think>\nFinal answer"])

    result = TextGenerator([model]).generate("q")

    assert result is not None
    assert result.text == "Final answer"
    assert strip_reasoning("no tags") == "no tags"


def test_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        TextGenerator([])


def test_provider_chain_follows_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("AI_LOCAL_FALLBACK", "1")

    names = [provider.name for provider in build_text_generator(Settings()).providers]

    assert names == ["ollama", "huggingface", "limited-local-examples"]
    assert [
        p.name for p in build_text_generator(Settings(ai={"local_fallback": False})).providers
    ] == [
        "ollama",
        "huggingface",
    ]


def test_local_examples_only_answer_sentence_generation() -> None:
    model = LocalExamplesModel()
    text = model.chat(
        [
            {
                "role": "user",
                "content": "Generate 2 sentences with verb to be in past tense. "
                "Output format: EN | PT | DICA",
            }
        ]
    )

    assert text.splitlines() == [
        "I was at home. | Eu estava em casa. | Use was com I",
        "She was tired. | Ela estava cansada. | Use was com she",
    ]
    assert TextGenerator([model]).generate("Evaluate this student's free writing") is None
