from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from english_trainer.config import Settings, get_settings


def test_defaults_match_documented_values() -> None:
    settings = Settings()

    assert settings.ollama.base_url == "http://localhost:11434"
    assert settings.ollama.model == "llama3.2"
    assert settings.rag.model == "gemma2:2b"
    assert settings.rag.embed_model == "embeddinggemma"
    assert settings.rag.min_score == pytest.approx(0.38)
    assert settings.rag.retrieval == "dense"
    assert settings.huggingface.enabled is False
    assert settings.ai.local_fallback is False


def test_environment_overrides_use_legacy_variable_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434/")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.35")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("AI_LOCAL_FALLBACK", "yes")

    settings = Settings()

    assert settings.ollama.base_url == "http://gpu-box:11434"
    assert settings.rag.min_score == pytest.approx(0.35)
    assert settings.huggingface.enabled is True
    assert settings.ai.local_fallback is True


def test_secrets_are_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret")

    assert "hf_super_secret" not in repr(Settings())


def test_retrieval_mode_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RETRIEVAL", "hybrid")
    assert Settings().rag.retrieval == "hybrid"
    monkeypatch.setenv("RAG_RETRIEVAL", "sparse")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("value", ["-0.1", "1.5", "not-a-number"])
def test_invalid_min_score_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RAG_MIN_SCORE", value)

    with pytest.raises(ValidationError):
        Settings()


def test_paths_are_derived_from_home(home: Path) -> None:
    paths = get_settings().paths

    assert paths.content_file == home / "data" / "content.json"
    assert paths.notes_dir == home / "notes"
    assert paths.rag_index_dir == home / "data" / "rag_index"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
