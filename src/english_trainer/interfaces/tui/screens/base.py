"""Shared screen behaviour: typed access to the application and session scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from textual.binding import Binding, BindingType
from textual.screen import Screen

if TYPE_CHECKING:
    from english_trainer.app import Application
    from english_trainer.interfaces.tui.app import EnglishTrainerApp


class TrainerScreen(Screen[None]):
    """Base screen: ``Esc``/``q`` go back and ``self.trainer`` is the typed application."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,q", "back", "Voltar")]

    @property
    def trainer(self) -> Application:
        return cast("EnglishTrainerApp", self.app).trainer

    def action_back(self) -> None:
        self.app.pop_screen()


@dataclass
class SessionScore:
    correct: int = 0
    total: int = 0

    def add(self, correct: bool) -> None:
        self.total += 1
        self.correct += int(correct)

    @property
    def pct(self) -> int:
        return int(self.correct / self.total * 100) if self.total else 0

    def markup(self) -> str:
        if not self.total:
            return "[dim]Sessão: 0/0[/dim]"
        return f"[dim]Sessão:[/dim] [green]{self.correct}[/green]/{self.total} ({self.pct}%)"
