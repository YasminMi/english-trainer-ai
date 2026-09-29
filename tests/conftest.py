from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from english_trainer.config import get_settings

_PROVIDER_ENV = (
    "OLLAMA_URL",
    "OLLAMA_MODEL",
    "HF_TOKEN",
    "HF_MODEL",
    "AI_LOCAL_FALLBACK",
    "RAG_MODEL",
    "RAG_EMBED_MODEL",
    "RAG_MIN_SCORE",
    "RAG_TOP_K",
    "LOG_LEVEL",
    "LOG_JSON_FORMAT",
    "ENGLISH_TRAINER_HOME",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests independent from the developer's shell environment and .env file."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the application at an empty, temporary home directory."""
    monkeypatch.setenv("ENGLISH_TRAINER_HOME", str(tmp_path))
    get_settings.cache_clear()
    return tmp_path


@pytest.fixture(autouse=True)
def reset_package_logger() -> Iterator[None]:
    """Undo configure_logging() so caplog sees records in every test."""
    yield
    logger = logging.getLogger("english_trainer")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
