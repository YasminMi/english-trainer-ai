"""Offline practice screens: topic picker, flashcards, sentences, to be drills and dictation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, ProgressBar, Rule, Static

from english_trainer.domain.content import available_sources, filter_items, note_label
from english_trainer.domain.hints import detect_tense_hint
from english_trainer.domain.matching import is_close_answer, normalize_pt_phrase
from english_trainer.domain.selection import item_key, weighted_choice
from english_trainer.domain.types import Flashcard, Sentence, ToBeExercise
from english_trainer.interfaces.tui.screens.base import SessionScore, TrainerScreen
from english_trainer.services.audio import RATE_NORMAL, RATE_SLOW, audio_available, speak

ScreenFactory = Callable[[list[Any], str], Screen[None]]


class TopicMenuScreen(TrainerScreen):
    """Lists "all items", optional categories and every Markdown note as buttons."""

    def __init__(
        self,
        title: str,
        items: Sequence[Any],
        open_practice: ScreenFactory,
        *,
        all_label: str = "Todos os itens",
        categories: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._title = title
        self._items = list(items)
        self._open = open_practice
        self._all_label = all_label
        self._choices: list[tuple[str, list[Any]]] = [(all_label, self._items)]
        for category in categories:
            label = category.replace("_", " ").title()
            self._choices.append((label, filter_items(self._items, category=category)))
        for note in available_sources(self._items):
            self._choices.append((note_label(note), filter_items(self._items, source_note=note)))

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  {self._title} — Escolha um tópico", classes="sec-header")
        with Container(id="menu-wrap"):
            for index, (label, items) in enumerate(self._choices):
                yield Button(
                    f"  {escape(label)}  ({len(items)})", id=f"t_{index}", classes="menu-btn"
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        label, items = self._choices[int((event.button.id or "t_0")[2:])]
        self.app.push_screen(self._open(items, label))


# --------------------------------------------------------------------------- flashcards


class FlashcardPracticeScreen(TrainerScreen):
    def __init__(self, cards: list[Flashcard], label: str) -> None:
        super().__init__()
        self._cards = cards
        self._label = label
        self._card: Flashcard | None = None
        self._score = SessionScore()
        self._awaiting_next = False

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  FLASHCARDS — {escape(self._label)}", classes="sec-header")
        with Horizontal(id="fc-layout"):
            with Vertical(id="fc-left"):
                yield Static("", id="fc-category", classes="muted")
                yield Static("", id="fc-word")
                yield Static("Qual é a tradução em português?", classes="muted")
                yield Input(placeholder="Digite a tradução...", id="fc-input")
                yield Static("", id="fc-feedback")
            with Vertical(id="fc-right"):
                yield Static("Este item", id="fc-item-title")
                yield Static("", id="fc-item-seen")
                yield Static("", id="fc-item-acc")
                yield Rule()
                yield Static("Sessão", classes="muted")
                yield Static("", id="fc-sess-label")
                yield ProgressBar(total=100, show_percentage=False, id="fc-sess-pb")
        yield Footer()

    def on_mount(self) -> None:
        self._next_card()

    def _next_card(self) -> None:
        if not self._cards:
            self.query_one("#fc-word", Static).update("[red]Nenhum card encontrado.[/red]")
            return
        self._card = card = weighted_choice(self._cards, self.trainer.progress, "flashcards")
        category = card.get("category") or note_label(card.get("source_note", ""))
        self.query_one("#fc-word", Static).update(f"[bold]{escape(card['en'])}[/bold]")
        self.query_one("#fc-category", Static).update(f"[dim]Categoria: {escape(category)}[/dim]")
        self.query_one("#fc-feedback", Static).update("")
        answer = self.query_one("#fc-input", Input)
        answer.clear()
        answer.focus()
        self._awaiting_next = False
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        if self._card is None:
            return
        stats = self.trainer.progress["flashcards"].get(item_key(self._card))
        seen, correct = (stats["seen"], stats["correct"]) if stats else (0, 0)
        accuracy = correct / seen * 100 if seen else 0
        self.query_one("#fc-item-seen", Static).update(f"[dim]Visto:[/dim] {seen}x")
        self.query_one("#fc-item-acc", Static).update(
            f"[dim]Acerto:[/dim] {correct}/{seen} ({accuracy:.0f}%)"
        )
        self.query_one("#fc-sess-label", Static).update(
            self._score.markup() if self._score.total else "[dim]Nenhuma tentativa ainda[/dim]"
        )
        self.query_one("#fc-sess-pb", ProgressBar).progress = self._score.pct

    @on(Input.Submitted, "#fc-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._awaiting_next:
            self._next_card()
            return
        if self._card is None or not event.value.strip():
            return
        correct = is_close_answer(event.value, self._card["pt"])
        self.trainer.record_result("flashcards", item_key(self._card), correct)
        self._score.add(correct)
        self.query_one("#fc-feedback", Static).update(
            "[green]✓ Correto![/green]"
            if correct
            else f"[red]✗ Quase![/red]  Correto: [bold]{escape(self._card['pt'])}[/bold]\n"
            "[dim]Enter para continuar[/dim]"
        )
        self._awaiting_next = True
        event.input.clear()
        self._refresh_stats()


# --------------------------------------------------------------------------- sentences


class SentencePracticeScreen(TrainerScreen):
    def __init__(self, sentences: list[Sentence], label: str) -> None:
        super().__init__()
        self._sentences = sentences
        self._label = label
        self._sentence: Sentence | None = None
        self._score = SessionScore()
        self._awaiting_next = False

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  SENTENCE TRAINING — {escape(self._label)}", classes="sec-header")
        with Container(id="st-wrap"):
            yield Static(self._score.markup(), id="st-session")
            yield Static("", id="st-sentence")
            yield Static(
                "[dim]Traduza para português OU repita a frase em inglês[/dim]", classes="muted"
            )
            yield Input(placeholder="Sua resposta...", id="st-input")
            yield Static("", id="st-feedback")
            yield Button(
                "  ✓  Minha resposta está correta",
                id="st-override",
                classes="menu-btn override-btn hidden",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._next_sentence()

    def _next_sentence(self) -> None:
        if not self._sentences:
            self.query_one("#st-sentence", Static).update("[red]Nenhuma frase encontrada.[/red]")
            return
        self._sentence = weighted_choice(self._sentences, self.trainer.progress, "sentences")
        self.query_one("#st-sentence", Static).update(
            f"[bold][yellow]{escape(self._sentence['en'])}[/yellow][/bold]"
        )
        self.query_one("#st-feedback", Static).update("")
        self.query_one("#st-override", Button).add_class("hidden")
        answer = self.query_one("#st-input", Input)
        answer.clear()
        answer.focus()
        self._awaiting_next = False

    @on(Input.Submitted, "#st-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._awaiting_next:
            self._next_sentence()
            return
        sentence = self._sentence
        if sentence is None or not event.value.strip():
            return
        expected_pt = normalize_pt_phrase(sentence["pt"])
        correct = is_close_answer(event.value, [sentence["pt"], expected_pt, sentence["en"]])
        self._record(correct)
        if correct:
            self.query_one("#st-feedback", Static).update("[green]✓ Correto![/green]")
        else:
            hint = detect_tense_hint(sentence["en"], event.value)
            hint_line = f"\n[yellow]{escape(hint)}[/yellow]" if hint else ""
            self.query_one("#st-feedback", Static).update(
                f"[red]✗ Quase![/red]{hint_line}\n"
                f"[dim]Esperado (PT):[/dim] [bold]{escape(expected_pt)}[/bold]\n"
                "[dim]Enter para continuar[/dim]"
            )
            self.query_one("#st-override", Button).remove_class("hidden")
        self._awaiting_next = True
        event.input.clear()

    @on(Button.Pressed, "#st-override")
    def on_override(self) -> None:
        """Let the learner accept an answer the matcher rejected (e.g. a valid synonym)."""
        self._record(True)
        self._next_sentence()

    def _record(self, correct: bool) -> None:
        if self._sentence is None:
            return
        self.trainer.record_result("sentences", item_key(self._sentence), correct)
        self._score.add(correct)
        self.query_one("#st-session", Static).update(self._score.markup())


# --------------------------------------------------------------------------- to be


class ToBeScreen(TrainerScreen):
    MAX_OPTIONS = 4

    def __init__(self) -> None:
        super().__init__()
        self._exercise: ToBeExercise | None = None
        self._score = SessionScore()
        self._awaiting_next = False

    def compose(self) -> ComposeResult:
        yield Static("  ◆  TO BE TRAINER — am / is / are / was / were", classes="sec-header")
        with Container(id="tobe-wrap"):
            yield Static(self._score.markup(), id="tobe-session")
            yield Static("", id="tobe-question")
            yield Static("", id="tobe-pt", classes="muted")
            for index in range(self.MAX_OPTIONS):
                yield Button("", id=f"opt-{index}", classes="opt-btn")
            yield Static("", id="tobe-feedback")
        yield Footer()

    def on_mount(self) -> None:
        self._next()

    def _next(self) -> None:
        exercises = self.trainer.content["to_be_exercises"]
        if not exercises:
            self.query_one("#tobe-question", Static).update(
                "[red]Nenhum exercício encontrado.[/red]"
            )
            return
        self._exercise = exercise = weighted_choice(exercises, self.trainer.progress, "to_be")
        self.query_one("#tobe-question", Static).update(
            f"[bold]{escape(exercise['question'])}[/bold]"
        )
        self.query_one("#tobe-pt", Static).update(f"[dim]{escape(exercise['pt'])}[/dim]")
        self.query_one("#tobe-feedback", Static).update("")
        for index in range(self.MAX_OPTIONS):
            button = self.query_one(f"#opt-{index}", Button)
            button.display = index < len(exercise["options"])
            if button.display:
                button.label = f"  {exercise['options'][index]}"
        self._awaiting_next = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("opt-") or self._exercise is None:
            return
        if self._awaiting_next:
            self._next()
            return

        exercise = self._exercise
        chosen = exercise["options"][int(button_id[4:])]
        correct = chosen == exercise["correct"]
        self.trainer.record_result("to_be", item_key(exercise), correct)
        self._score.add(correct)

        feedback = (
            "[green]✓ Correto![/green]"
            if correct
            else f"[red]✗  Correto: [bold]{escape(exercise['correct'])}[/bold][/red]"
        )
        if explanation := exercise.get("explanation"):
            feedback += f"\n[dim]{escape(explanation)}[/dim]"
        feedback += "\n[dim]Clique em qualquer opção para continuar[/dim]"
        self.query_one("#tobe-feedback", Static).update(feedback)
        self.query_one("#tobe-session", Static).update(self._score.markup())
        self._awaiting_next = True


# --------------------------------------------------------------------------- dictation


class DictationPracticeScreen(TrainerScreen):
    def __init__(self, sentences: list[Sentence], label: str) -> None:
        super().__init__()
        self._sentences = sentences
        self._label = label
        self._sentence: Sentence | None = None
        self._score = SessionScore()
        self._awaiting_next = False
        self._rate = RATE_NORMAL

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  DICTATION — {escape(self._label)}", classes="sec-header")
        with Container(id="dt-wrap"):
            yield Static(self._score.markup(), id="dt-session")
            yield Button("  🐇  Velocidade: Normal", id="dt-speed", classes="menu-btn")
            yield Static("🎵  Preparando áudio...", id="dt-status")
            yield Static("[dim]Ouça e escreva o que ouviu em inglês[/dim]", classes="muted")
            yield Input(placeholder="Type what you heard...", id="dt-input", disabled=True)
            yield Button("  ↺  Replay", id="dt-replay", classes="menu-btn", disabled=True)
            yield Static("", id="dt-feedback")
        yield Footer()

    def on_mount(self) -> None:
        self._next_sentence()

    def _next_sentence(self) -> None:
        if not self._sentences:
            self.query_one("#dt-status", Static).update("[red]Nenhuma frase encontrada.[/red]")
            return
        self._sentence = weighted_choice(self._sentences, self.trainer.progress, "sentences")
        self.query_one("#dt-feedback", Static).update("")
        self.query_one("#dt-input", Input).clear()
        self._awaiting_next = False
        self._start_playback("🎵  [yellow]Playing audio...[/yellow]  (aguarde)")

    def _start_playback(self, status: str) -> None:
        self.query_one("#dt-status", Static).update(status)
        self.query_one("#dt-input", Input).disabled = True
        self.query_one("#dt-replay", Button).disabled = True
        self._play_audio()

    @on(Button.Pressed, "#dt-speed")
    def on_toggle_speed(self) -> None:
        slow = self._rate == RATE_NORMAL
        self._rate = RATE_SLOW if slow else RATE_NORMAL
        self.query_one("#dt-speed", Button).label = (
            "  🐢  Velocidade: Devagar" if slow else "  🐇  Velocidade: Normal"
        )

    @work(thread=True, exclusive=True)
    def _play_audio(self) -> None:
        text = self._sentence["en"] if self._sentence else ""
        played = audio_available() and speak(text, rate=self._rate)
        self.app.call_from_thread(self._after_playback, text, played)

    def _after_playback(self, text: str, played: bool) -> None:
        self.query_one("#dt-replay", Button).disabled = False
        answer = self.query_one("#dt-input", Input)
        answer.disabled = False
        answer.focus()
        self.query_one("#dt-status", Static).update(
            "🎵  [dim]Pronto — digite o que ouviu[/dim]"
            if played
            else f"[yellow]Áudio indisponível.[/yellow]  Frase: [bold]{escape(text)}[/bold]"
        )

    @on(Button.Pressed, "#dt-replay")
    def on_replay(self) -> None:
        if not self._awaiting_next:
            self._start_playback("🎵  [yellow]Replaying...[/yellow]")

    @on(Input.Submitted, "#dt-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._awaiting_next:
            self._next_sentence()
            return
        sentence = self._sentence
        if sentence is None or not event.value.strip():
            return
        correct = is_close_answer(event.value, sentence["en"], strict=True)
        self.trainer.record_result("sentences", item_key(sentence), correct)
        self._score.add(correct)
        self.query_one("#dt-feedback", Static).update(
            "[green]✓ Correto![/green]"
            if correct
            else f"[red]✗  Era:[/red] [bold]{escape(sentence['en'])}[/bold]\n"
            f"[dim]PT: {escape(sentence['pt'])}[/dim]\n[dim]Enter para continuar[/dim]"
        )
        self.query_one("#dt-session", Static).update(self._score.markup())
        self._awaiting_next = True
        event.input.clear()
