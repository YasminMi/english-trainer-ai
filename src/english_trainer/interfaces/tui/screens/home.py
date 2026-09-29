"""Home menu and progress statistics screens."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Footer, ProgressBar, Rule, Static

from english_trainer.domain.progress import compute_overall
from english_trainer.interfaces.tui.screens.assistant import (
    CoachScreen,
    ConversationScreen,
    DictionaryScreen,
    FreeWritingScreen,
    GroundedTutorScreen,
)
from english_trainer.interfaces.tui.screens.base import TrainerScreen
from english_trainer.interfaces.tui.screens.practice import (
    DictationPracticeScreen,
    FlashcardPracticeScreen,
    SentencePracticeScreen,
    ToBeScreen,
    TopicMenuScreen,
)


class HomeScreen(TrainerScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        *(Binding(key, f"open('{key}')", show=False) for key in "123456789"),
        Binding("r,R", "open('r')", show=False),
        Binding("q,Q", "app.quit", "Sair"),
    ]

    def _menu(self) -> list[tuple[str, str, Callable[[], Screen[None]]]]:
        content = self.trainer.content
        flashcard_categories = [
            str(c) for c in content["metadata"].get("flashcards_categories", []) or []
        ]
        return [
            (
                "1",
                f"To Be Trainer          {len(content['to_be_exercises'])} exercícios",
                ToBeScreen,
            ),
            (
                "2",
                f"Flashcards              {len(content['flashcards'])} cards",
                lambda: TopicMenuScreen(
                    "FLASHCARDS",
                    content["flashcards"],
                    FlashcardPracticeScreen,
                    all_label="Todas as palavras",
                    categories=flashcard_categories,
                ),
            ),
            (
                "3",
                f"Sentence Training        {len(content['sentences'])} frases",
                lambda: TopicMenuScreen(
                    "SENTENCE TRAINING",
                    content["sentences"],
                    SentencePracticeScreen,
                    all_label="Todas as frases",
                ),
            ),
            ("4", "Dictionary               local + online", DictionaryScreen),
            ("5", "Miu — AI Coach           gerar frases com IA", CoachScreen),
            ("6", "Conversation with Miu ★  conversar em inglês", ConversationScreen),
            ("7", "Progress                 estatísticas", ProgressScreen),
            (
                "8",
                "Dictation  🎵             ouça e escreva",
                lambda: TopicMenuScreen(
                    "DICTATION",
                    content["sentences"],
                    DictationPracticeScreen,
                    all_label="Todas as frases",
                ),
            ),
            ("9", "Free Writing  ✏           escreva com correção da IA", FreeWritingScreen),
            ("r", "Miu Grounded Tutor       RAG local com fontes", GroundedTutorScreen),
        ]

    def compose(self) -> ComposeResult:
        yield Static("  ◆  ENGLISH TRAINER AI", id="home-brand")
        yield Static("", id="home-stats")
        yield ProgressBar(total=100, show_percentage=False, id="home-pb")
        with Container(id="menu-wrap"):
            for key, label, _ in self._menu():
                yield Button(f"  {key.upper()}   {label}", id=f"nav-{key}", classes="menu-btn")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_stats()

    def on_screen_resume(self) -> None:
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        stats = compute_overall(self.trainer.progress)
        self.query_one("#home-stats", Static).update(
            f"  Progresso geral: {stats['accuracy']:.0f}%   "
            f"Total: {stats['total_seen']:.0f} tentativas   "
            f"Correto: {stats['total_correct']:.0f}"
        )
        self.query_one("#home-pb", ProgressBar).progress = stats["accuracy"]

    @on(Button.Pressed, ".menu-btn")
    def on_menu(self, event: Button.Pressed) -> None:
        self.action_open((event.button.id or "").removeprefix("nav-"))

    def action_open(self, key: str) -> None:
        for menu_key, _, factory in self._menu():
            if menu_key == key:
                self.app.push_screen(factory())
                return


class ProgressScreen(TrainerScreen):
    def compose(self) -> ComposeResult:
        stats = compute_overall(self.trainer.progress)
        yield Static("  ◆  PROGRESS — Estatísticas", classes="sec-header")
        with Container(id="prog-wrap"):
            yield Static("Resumo Geral", classes="prog-title")
            yield Static(
                f"[bold]Total:[/bold] {stats['total_seen']:.0f}   "
                f"[bold]Correto:[/bold] {stats['total_correct']:.0f}   "
                f"[bold]Precisão:[/bold] [green]{stats['accuracy']:.1f}%[/green]"
            )
            yield Rule()
            for name, mode in (
                ("To Be Trainer", "to_be"),
                ("Flashcards", "flashcards"),
                ("Sentences", "sentences"),
            ):
                seen, correct = stats[f"{mode}_seen"], stats[f"{mode}_correct"]
                with Horizontal(classes="prog-row"):
                    yield Static(name, classes="prog-name")
                    yield Static(f"{correct:.0f}/{seen:.0f}", classes="prog-count")
                    bar = ProgressBar(total=100, show_percentage=False, classes="prog-bar")
                    bar.progress = correct / seen * 100 if seen else 0
                    yield bar
            yield Rule()
            yield Static(
                f"[dim]Meta diária:[/dim]        {stats['today_attempts']:.0f}/"
                f"{stats['daily_goal']:.0f}   [dim]Sequência:[/dim] {stats['streak']:.0f} dia(s)"
            )
            yield Static(f"[dim]Revisões pendentes:[/dim] {stats['due_reviews_total']:.0f}")
            yield Static(f"[dim]Dictionary lookups:[/dim] {stats['dictionary_lookups']:.0f}")
            yield Static(f"[dim]Miu interactions:[/dim]   {stats['ai_uses']:.0f}")
            if sessions := self.trainer.progress.get("conversation_sessions", 0):
                turns = self.trainer.progress.get("conversation_turns", 0)
                yield Static(
                    f"[dim]Conversations:[/dim]      {sessions} sessões, {turns} mensagens"
                )
        yield Footer()
