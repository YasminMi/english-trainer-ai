"""Drive the Textual app headlessly through its public screens."""

from __future__ import annotations

import asyncio

from textual.widgets import Button, Input, Static

from english_trainer.app import Application
from english_trainer.interfaces.tui.app import EnglishTrainerApp
from english_trainer.interfaces.tui.screens.home import HomeScreen, ProgressScreen
from english_trainer.interfaces.tui.screens.practice import FlashcardPracticeScreen, ToBeScreen


def _run(coro_factory):  # type: ignore[no-untyped-def]
    asyncio.run(coro_factory())


def test_home_lists_every_feature(trainer: Application) -> None:
    async def scenario() -> None:
        app = EnglishTrainerApp(trainer)
        async with app.run_test(size=(120, 50)):
            assert isinstance(app.screen, HomeScreen)
            labels = [str(button.label) for button in app.screen.query(Button)]
            assert len(labels) == 10
            assert any("Miu Grounded Tutor" in label for label in labels)
            assert any("1 exercícios" in label for label in labels)

    _run(scenario)


def test_to_be_answer_is_recorded_and_persisted(trainer: Application) -> None:
    async def scenario() -> None:
        app = EnglishTrainerApp(trainer)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.press("1")
            await pilot.pause()
            assert isinstance(app.screen, ToBeScreen)
            await pilot.click("#opt-1")  # "is"
            await pilot.pause()
            feedback = app.screen.query_one("#tobe-feedback", Static)
            assert "Correto" in str(feedback.render())

    _run(scenario)
    stored = trainer.progress_repository.load()
    assert stored["to_be"]["t1"] == {"seen": 1, "correct": 1}


def test_flashcard_flow_through_topic_menu(trainer: Application) -> None:
    async def scenario() -> None:
        app = EnglishTrainerApp(trainer)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.press("2")
            await pilot.pause()
            await pilot.click("#t_1")  # "Food" category
            await pilot.pause()
            assert isinstance(app.screen, FlashcardPracticeScreen)
            app.screen.query_one("#fc-input", Input).value = "maca"
            await pilot.press("enter")
            await pilot.pause()
            assert "Correto" in str(app.screen.query_one("#fc-feedback", Static).render())

    _run(scenario)
    assert trainer.progress["flashcards"]["f1"]["correct"] == 1


def test_progress_screen_renders_with_empty_history(trainer: Application) -> None:
    async def scenario() -> None:
        app = EnglishTrainerApp(trainer)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.press("7")
            await pilot.pause()
            assert isinstance(app.screen, ProgressScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    _run(scenario)
