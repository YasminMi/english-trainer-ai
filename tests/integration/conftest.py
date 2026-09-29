from __future__ import annotations

import json
from pathlib import Path

import pytest

from english_trainer.app import Application, build_application
from english_trainer.config import Settings
from english_trainer.llm.gateway import TextGenerator
from english_trainer.services.coach import CoachService
from english_trainer.services.writing import WritingService
from tests.fakes import FakeChatModel

CONTENT = {
    "metadata": {"flashcards_categories": ["food"]},
    "sentences": [{"id": "s1", "en": "I am happy.", "pt": "Eu estou feliz.", "tags": ["to_be"]}],
    "flashcards": [{"id": "f1", "en": "apple", "pt": "maçã", "category": "food"}],
    "to_be_exercises": [
        {
            "id": "t1",
            "question": "She ___ ready.",
            "options": ["am", "is", "are"],
            "correct": "is",
            "pt": "Ela está pronta.",
            "explanation": "Use is with she.",
        }
    ],
}


@pytest.fixture
def model() -> FakeChatModel:
    return FakeChatModel(replies=lambda _messages, _system: "Great job!")


@pytest.fixture
def trainer(tmp_path: Path, model: FakeChatModel) -> Application:
    """A fully wired application over a temporary home and a scripted model."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "content.json").write_text(json.dumps(CONTENT), encoding="utf-8")
    app = build_application(Settings(home=tmp_path))
    generator = TextGenerator([model])
    app.text_generator = generator
    app.chat_generator = generator
    app.coach = CoachService(generator)
    app.writing = WritingService(generator)
    return app
