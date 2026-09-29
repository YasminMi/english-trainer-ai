"""Run CLI flows end to end with scripted keyboard input."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from english_trainer.app import Application
from english_trainer.interfaces import cli
from english_trainer.interfaces.cli import assistant, practice, tools


def _script(monkeypatch: pytest.MonkeyPatch, *answers: str) -> None:
    """Feed ``answers`` to input(); afterwards behave like a closed stdin."""
    feed: Iterator[str] = iter(answers)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_to_be_trainer_records_answers(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _script(monkeypatch, "2", "", "0", "")

    practice.run_to_be_trainer(trainer)

    assert trainer.progress["to_be"]["t1"] == {"seen": 1, "correct": 1}
    assert "Good job" in capsys.readouterr().out


def test_add_sentence_is_saved_to_the_user_file(
    trainer: Application, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(monkeypatch, "1", "I study every day.", "Eu estudo todo dia.", "routine", "", "0")

    tools.add_content_menu(trainer)

    saved = trainer.content_repository.load_user()["sentences"]
    assert [s["en"] for s in saved] == ["I study every day."]


def test_duplicate_sentence_is_rejected(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _script(monkeypatch, "1", "I am happy", "x", "", "", "0")

    tools.add_content_menu(trainer)

    assert "already exists" in capsys.readouterr().out
    assert trainer.content_repository.load_user()["sentences"] == []


def test_free_writing_shows_ai_feedback(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _script(monkeypatch, "1", "I wake up at seven.", "", "", "0")

    assistant.run_free_writing(trainer)

    assert "Great job!" in capsys.readouterr().out
    assert trainer.progress["ai_uses"] == 1


def test_main_menu_exits_cleanly_and_saves(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trainer.progress.setdefault("profile", {})["onboarding_completed"] = True
    _script(monkeypatch, "99", "", "0")

    cli.run(trainer)

    assert "Bye!" in capsys.readouterr().out
    assert trainer.settings.paths.progress_file.exists()


def test_running_out_of_input_exits_instead_of_looping(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trainer.progress.setdefault("profile", {})["onboarding_completed"] = True
    _script(monkeypatch, "1")  # enters the to-be trainer, then stdin is exhausted

    cli.run(trainer)

    assert "Bye!" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["0", "-3", "abc", ""])
def test_onboarding_keeps_default_goal_for_invalid_answers(
    trainer: Application, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    _script(monkeypatch, answer)

    tools.onboarding(trainer)

    assert trainer.progress["daily"]["goal"] == 20
    assert trainer.progress["profile"]["onboarding_completed"] is True
