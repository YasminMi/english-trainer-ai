import asyncio

from textual.widgets import Button

from english_trainer import tui_app


def test_home_screen_exposes_grounded_tutor(monkeypatch):
    content = {
        "sentences": [],
        "flashcards": [],
        "to_be_exercises": [],
        "metadata": {},
    }
    monkeypatch.setattr(tui_app, "load_content", lambda: content)
    monkeypatch.setattr(tui_app, "load_progress", lambda: {})
    monkeypatch.setattr(tui_app, "save_progress", lambda _progress: None)

    async def run_app():
        app = tui_app.EnglishTrainerApp()
        async with app.run_test(size=(120, 50)):
            button = app.screen.query_one("#brag", Button)
            assert "Miu Grounded Tutor" in str(button.label)

    asyncio.run(run_app())
