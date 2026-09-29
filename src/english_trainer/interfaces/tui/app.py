"""Textual application shell."""

from __future__ import annotations

from textual.app import App

from english_trainer.app import Application, build_application
from english_trainer.interfaces.tui.screens.home import HomeScreen


class EnglishTrainerApp(App[None]):
    CSS_PATH = "styles.tcss"
    TITLE = "English Trainer AI"

    def __init__(self, trainer: Application | None = None) -> None:
        super().__init__()
        self.trainer = trainer or build_application()

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def on_unmount(self) -> None:
        self.trainer.save_progress()
        self.trainer.close()


def run(trainer: Application | None = None) -> None:
    EnglishTrainerApp(trainer).run()
