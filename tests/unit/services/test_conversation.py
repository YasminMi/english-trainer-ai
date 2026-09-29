from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from english_trainer.llm.gateway import TextGenerator
from english_trainer.services.conversation import (
    GREETING,
    REVIEW_PROMPT,
    SYSTEM_PROMPT,
    ConversationSession,
)
from tests.fakes import FakeChatModel


def test_session_tracks_learner_turns_and_uses_tutor_prompt() -> None:
    model = FakeChatModel(replies=["Hi! How are you?", "Nice! What do you do?"])
    session = ConversationSession(TextGenerator([model]))

    assert session.start() == "Hi! How are you?"
    assert session.send("I are fine") == "Nice! What do you do?"

    assert session.turns == 1
    assert session.history[0]["content"] == GREETING
    assert model.calls[0]["system"] == SYSTEM_PROMPT


def test_failed_reply_does_not_leave_an_orphan_message() -> None:
    model = FakeChatModel(replies=["Hello!"])
    session = ConversationSession(TextGenerator([model]))
    session.start()

    assert session.send("Are you there?") is None
    assert session.history[-1]["role"] == "assistant"
    assert session.turns == 0


def test_empty_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        ConversationSession(TextGenerator([FakeChatModel()])).send("  ")


def test_review_covers_only_learner_messages() -> None:
    model = FakeChatModel(replies=["Hello!", "Ok.", "Cool.", "[X] I are -> [OK] I am"])
    session = ConversationSession(TextGenerator([model]))
    session.start()
    session.send("I are fine")
    session.send("She go to school")

    review = session.review()

    assert review == "[X] I are -> [OK] I am"
    request = model.calls[-1]
    assert request["system"] == REVIEW_PROMPT
    content = request["messages"][0]["content"]
    assert "Message 1: I are fine" in content
    assert GREETING not in content


def test_review_needs_at_least_two_messages() -> None:
    session = ConversationSession(TextGenerator([FakeChatModel(replies=["Hi", "Ok"])]))
    session.start()
    session.send("Hello")
    assert session.review() is None


def test_save_writes_a_readable_transcript(tmp_path: Path) -> None:
    session = ConversationSession(TextGenerator([FakeChatModel(replies=["Hello!"])]))
    session.start()

    path = session.save(tmp_path, review="All good", now=datetime(2026, 6, 1, 9, 5, 7))

    assert path.name == "conversation_2026-06-01_09-05-07.txt"
    text = path.read_text(encoding="utf-8")
    assert "Miu: Hello!" in text
    assert "Grammar Review" in text
