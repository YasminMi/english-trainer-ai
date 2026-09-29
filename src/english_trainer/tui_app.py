"""
Textual TUI for English Trainer AI.

Design choices:
  Home       — Dark modern, progress header, menu tiles
  Flashcards — Split layout: word card (left) + item stats (right)
  Conversation — Classic terminal, [Miu] green / [You] white
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Input,
    ProgressBar,
    RichLog,
    Rule,
    Static,
)

from .conversation import SYSTEM_PROMPT, _check_ollama, _get_error_review, _save_conversation
from .data_access import find_in_dictionary, load_content, load_progress, save_progress
from .dictionary_api import fetch_dictionary_entry
from .flashcards import _filter as _fc_filter
from .flashcards import _weighted_choice as _w_card
from .hf_llm import DEFAULT_MODEL, generate_conversation_reply, generate_text
from .progress import (
    compute_overall,
    register_ai_use,
    register_dictionary_lookup,
    register_flashcard_result,
    register_sentence_result,
    register_to_be_result,
)
from .rag import RagService
from .to_be_trainer import _weighted_choice as _w_to_be
from .training import _detect_tense_hint
from .training import _weighted_choice as _w_sentence
from .utils import is_close_answer, normalize_pt_phrase

# CSS

APP_CSS = """
Screen      { background: #0d1117; color: #e6edf3; }
Footer      { background: #161b22; color: #8b949e; }
Rule        { color: #30363d; }

Input       { border: tall #30363d; background: #0d1117; color: #e6edf3; }
Input:focus { border: tall #00d4aa; }

Button              { background: #161b22; border: tall #30363d; color: #e6edf3; text-style: none; }
Button:hover        { background: #1c2128; border: tall #00d4aa; color: #00d4aa; text-style: none; }
Button:focus        { background: #1c2128; border: tall #00d4aa; text-style: none; }
Button.-active      { text-style: none; }
Button:focus-within { text-style: none; }

ProgressBar > .bar--bar      { color: #00d4aa; }
ProgressBar > .bar--complete { color: #3fb950; }

/* ── shared util classes ── */
.muted   { color: #8b949e; }
.correct { color: #3fb950; text-style: bold; }
.wrong   { color: #f85149; text-style: bold; }
.warn    { color: #e3b341; }

/* ── section header strip ── */
.sec-header {
    background: #161b22; color: #58a6ff; text-style: bold;
    padding: 0 2; height: 3; border-bottom: solid #30363d;
    content-align: left middle;
}

/* ────────── HOME ────────── */
#home-brand {
    background: #161b22; color: #00d4aa; text-style: bold;
    text-align: center; height: 3; content-align: center middle;
    border-bottom: solid #21262d;
}
#home-stats {
    background: #161b22; color: #8b949e;
    padding: 0 2; height: 2; border-bottom: solid #21262d;
}
#home-pb { width: 1fr; height: 1; background: #161b22; border-bottom: solid #30363d; }
#home-pb Bar { width: 1fr; }
#menu-wrap { padding: 1 8; height: 1fr; align: center top; overflow-y: auto; }
.menu-btn {
    width: 100%; height: 3; margin: 0 0 1 0;
    background: #161b22; border: tall #30363d;
    color: #e6edf3; content-align: left middle; padding: 0 2;
}
.menu-btn:hover { background: #1c2128; border: tall #00d4aa; color: #00d4aa; }
.menu-btn:focus { background: #1c2128; border: tall #00d4aa; }

/* ────────── FLASHCARD SPLIT ────────── */
#fc-layout   { height: 1fr; padding: 1 1 0 1; }
#fc-left     { width: 3fr; background: #161b22; border: tall #30363d; padding: 1 2; margin-right: 1; }
#fc-right    { width: 2fr; background: #161b22; border: tall #30363d; padding: 1 2; }
#fc-word {
    text-align: center; text-style: bold; color: #e6edf3;
    height: 7; border: double #00d4aa; background: #0d1117;
    content-align: center middle; margin: 1 0;
}
#fc-item-title { color: #58a6ff; text-style: bold; margin-bottom: 1; }
#fc-feedback   { min-height: 3; }

/* ────────── SENTENCE TRAINING ────────── */
#st-wrap     { padding: 2 4; }
#st-sentence {
    text-style: bold; color: #ffa657; padding: 1 2;
    border: tall #30363d; background: #161b22;
    margin: 1 0; min-height: 3;
}
#st-feedback { min-height: 4; }

/* ────────── TO BE ────────── */
#tobe-wrap   { padding: 2 4; }
#tobe-question {
    text-style: bold; color: #ffa657; padding: 1 2;
    border: tall #30363d; background: #161b22;
    margin: 1 0; text-align: center; min-height: 3;
}
#tobe-feedback { min-height: 3; text-align: center; }
.opt-btn { width: 100%; height: 3; margin: 0 0 1 0; }

/* ────────── DICTIONARY ────────── */
#dict-wrap   { padding: 2 4; }
#dict-result {
    background: #161b22; border: tall #30363d;
    padding: 1 2; min-height: 8; margin-top: 1;
}

/* ────────── CONVERSATION ────────── */
#conv-titlebar {
    background: #161b22; color: #58a6ff; text-style: bold;
    padding: 0 2; height: 3; border-bottom: solid #30363d;
    content-align: left middle;
}
#conv-log        { background: #0d1117; padding: 0 1; height: 1fr; }
#conv-status     { color: #8b949e; height: 1; padding: 0 2; background: #161b22; }
#conv-input-area { height: 5; background: #161b22; border-top: solid #30363d; padding: 1 2; }

/* ────────── PROGRESS ────────── */
#prog-wrap   { padding: 2 4; }
.prog-title  { color: #58a6ff; text-style: bold; margin-bottom: 1; }
.prog-row    { height: 3; margin-bottom: 1; }
.prog-name   { width: 22; content-align: left middle; }
.prog-count  { width: 14; content-align: right middle; color: #8b949e; }
.prog-bar    { width: 1fr; height: 1; }

/* ────────── DICTATION ────────── */
#dt-wrap     { padding: 2 4; }
#dt-status   {
    text-align: center; color: #00d4aa; text-style: bold;
    padding: 1 2; border: tall #30363d; background: #161b22;
    margin: 1 0; min-height: 3; content-align: center middle;
}
#dt-feedback { min-height: 4; }
#dt-replay   { width: 20; margin-top: 1; }

/* ────────── FREE WRITING ────────── */
#fw-wrap         { padding: 2 4; }
#fw-input-area   { margin-top: 1; }
#fw-feedback     {
    background: #161b22; border: tall #30363d;
    padding: 1 2; min-height: 8; margin-top: 1;
}
.hidden      { display: none; }
.override-btn { background: #0d2818; border: tall #3fb950; color: #3fb950; margin-top: 1; }

#rag-wrap     { padding: 1 3; }
#rag-status   { color: #8b949e; height: 2; margin-top: 1; }
#rag-answer   { border: tall #30363d; padding: 1 2; min-height: 8; height: 1fr; }
#rag-sources  { color: #8b949e; min-height: 5; margin-top: 1; }
"""

# HOME


class HomeScreen(Screen):
    BINDINGS = [
        Binding("1", "nav_1", "", show=False),
        Binding("2", "nav_2", "", show=False),
        Binding("3", "nav_3", "", show=False),
        Binding("4", "nav_4", "", show=False),
        Binding("5", "nav_5", "", show=False),
        Binding("6", "nav_6", "", show=False),
        Binding("7", "nav_7", "", show=False),
        Binding("8", "nav_8", "", show=False),
        Binding("9", "nav_9", "", show=False),
        Binding("r,R", "nav_rag", "", show=False),
        Binding("q,Q", "quit_app", "Sair"),
    ]

    def _stats_text(self) -> str:
        stats = compute_overall(self.app.progress)  # type: ignore[attr-defined]
        return (
            f"  Progresso geral: {stats['accuracy']:.0f}%   "
            f"Total: {stats['total_seen']} tentativas   "
            f"Correto: {stats['total_correct']}"
        )

    def compose(self) -> ComposeResult:
        c = self.app.content  # type: ignore[attr-defined]
        n_s = len(c.get("sentences", []))
        n_c = len(c.get("flashcards", []))
        n_e = len(c.get("to_be_exercises", []))

        yield Static("  ◆  ENGLISH TRAINER AI", id="home-brand")
        yield Static(self._stats_text(), id="home-stats")
        yield ProgressBar(total=100, show_percentage=False, id="home-pb")

        with Container(id="menu-wrap"):
            yield Button(
                f"  1   To Be Trainer          {n_e} exercícios", id="b1", classes="menu-btn"
            )
            yield Button(f"  2   Flashcards              {n_c} cards", id="b2", classes="menu-btn")
            yield Button(
                f"  3   Sentence Training        {n_s} frases", id="b3", classes="menu-btn"
            )
            yield Button(
                "  4   Dictionary               local + online", id="b4", classes="menu-btn"
            )
            yield Button(
                "  5   Miu — AI Coach           gerar frases / corrigir",
                id="b5",
                classes="menu-btn",
            )
            yield Button(
                "  6   Conversation with Miu ★  conversar em inglês", id="b6", classes="menu-btn"
            )
            yield Button("  7   Progress                 estatísticas", id="b7", classes="menu-btn")
            yield Button(
                "  8   Dictation  🎵             ouça e escreva", id="b8", classes="menu-btn"
            )
            yield Button(
                "  9   Free Writing  ✏           escreva com correção da IA",
                id="b9",
                classes="menu-btn",
            )
            yield Button(
                "  R   Miu Grounded Tutor       RAG local com fontes", id="brag", classes="menu-btn"
            )

        yield Footer()

    def on_mount(self) -> None:
        stats = compute_overall(self.app.progress)  # type: ignore[attr-defined]
        self.query_one("#home-pb", ProgressBar).progress = stats["accuracy"]

    def on_screen_resume(self) -> None:
        stats = compute_overall(self.app.progress)  # type: ignore[attr-defined]
        try:
            self.query_one("#home-stats", Static).update(self._stats_text())
            self.query_one("#home-pb", ProgressBar).progress = stats["accuracy"]
        except Exception:
            pass

    @on(Button.Pressed, "#b1")
    def action_nav_1(self) -> None:
        self.app.push_screen(ToBeScreen())

    @on(Button.Pressed, "#b2")
    def action_nav_2(self) -> None:
        self.app.push_screen(FlashcardMenuScreen())

    @on(Button.Pressed, "#b3")
    def action_nav_3(self) -> None:
        self.app.push_screen(SentenceMenuScreen())

    @on(Button.Pressed, "#b4")
    def action_nav_4(self) -> None:
        self.app.push_screen(DictionaryScreen())

    @on(Button.Pressed, "#b5")
    def action_nav_5(self) -> None:
        self.app.push_screen(AICoachScreen())

    @on(Button.Pressed, "#b6")
    def action_nav_6(self) -> None:
        self.app.push_screen(ConversationScreen())

    @on(Button.Pressed, "#b7")
    def action_nav_7(self) -> None:
        self.app.push_screen(ProgressScreen())

    @on(Button.Pressed, "#b8")
    def action_nav_8(self) -> None:
        self.app.push_screen(DictationMenuScreen())

    @on(Button.Pressed, "#b9")
    def action_nav_9(self) -> None:
        self.app.push_screen(FreeWritingScreen())

    @on(Button.Pressed, "#brag")
    def action_nav_rag(self) -> None:
        self.app.push_screen(RagTutorScreen())

    def action_quit_app(self) -> None:
        self.app.exit()


# FLASHCARD — menu


class FlashcardMenuScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        metadata = self.app.content.get("metadata", {})  # type: ignore[attr-defined]
        categories = metadata.get("flashcards_categories", [])
        cards = self.app.content.get("flashcards", [])  # type: ignore[attr-defined]
        notes = sorted({c.get("source_note") for c in cards if c.get("source_note")})

        yield Static("  ◆  FLASHCARDS — Escolha uma categoria", classes="sec-header")
        with Container(id="menu-wrap"):
            yield Button(f"  Todas as palavras  ({len(cards)} total)", id="all", classes="menu-btn")
            for cat in categories:
                n = sum(1 for c in cards if c.get("category") == cat)
                yield Button(
                    f"  {cat.replace('_', ' ').title()}  ({n})", id=f"c_{cat}", classes="menu-btn"
                )
            if notes:
                yield Static("  — Notas Markdown —", classes="muted")
                for i, note in enumerate(notes):
                    lbl = note.split("/")[-1].replace(".md", "").replace("_", " ").title()
                    n = sum(1 for c in cards if c.get("source_note") == note)
                    yield Button(f"  {lbl}  ({n})", id=f"n_{i}", classes="menu-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        cards = self.app.content.get("flashcards", [])  # type: ignore[attr-defined]
        notes = sorted({c.get("source_note") for c in cards if c.get("source_note")})
        if bid == "all":
            self.app.push_screen(FlashcardPracticeScreen(None, None))
        elif bid.startswith("c_"):
            self.app.push_screen(FlashcardPracticeScreen(bid[2:], None))
        elif bid.startswith("n_"):
            self.app.push_screen(FlashcardPracticeScreen(None, notes[int(bid[2:])]))

    def action_back(self) -> None:
        self.app.pop_screen()


# FLASHCARD — practice (split layout)


class FlashcardPracticeScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def __init__(self, category: str | None, source_note: str | None):
        super().__init__()
        self._category = category
        self._source_note = source_note
        self._card: dict[str, Any] = {}
        self._sess_ok = 0
        self._sess_total = 0
        self._feedback_mode = False

    def _label(self) -> str:
        if self._source_note:
            return self._source_note.split("/")[-1].replace(".md", "").replace("_", " ").title()
        return self._category.replace("_", " ").title() if self._category else "Todos"

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  FLASHCARDS — {self._label()}", classes="sec-header")
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
        cards = self.app.content.get("flashcards", [])  # type: ignore[attr-defined]
        filtered = _fc_filter(cards, self._category, self._source_note)
        if not filtered:
            self.query_one("#fc-word", Static).update("[red]Nenhum card encontrado.[/red]")
            return
        prog = self.app.progress.get("flashcards", {})  # type: ignore[attr-defined]
        self._card = _w_card(filtered, prog, progress=self.app.progress)  # type: ignore[attr-defined]
        en = self._card.get("en", "")
        cat = self._card.get("category") or (
            self._card.get("source_note", "").split("/")[-1].replace(".md", "")
        )
        self.query_one("#fc-word", Static).update(f"[bold]{en}[/bold]")
        self.query_one("#fc-category", Static).update(f"[dim]Categoria: {cat}[/dim]")
        self.query_one("#fc-feedback", Static).update("")
        inp = self.query_one("#fc-input", Input)
        inp.clear()
        inp.focus()
        self._refresh_item_stats()
        self._refresh_sess_stats()
        self._feedback_mode = False

    def _refresh_item_stats(self) -> None:
        cid = str(self._card.get("id", self._card.get("en", "")))
        stats = self.app.progress.get("flashcards", {}).get(cid, {"seen": 0, "correct": 0})  # type: ignore[attr-defined]
        seen = int(stats.get("seen", 0))
        correct = int(stats.get("correct", 0))
        acc = correct / seen * 100 if seen > 0 else 0
        self.query_one("#fc-item-seen", Static).update(f"[dim]Visto:[/dim] {seen}x")
        self.query_one("#fc-item-acc", Static).update(
            f"[dim]Acerto:[/dim] {correct}/{seen} ({acc:.0f}%)"
        )

    def _refresh_sess_stats(self) -> None:
        if self._sess_total == 0:
            self.query_one("#fc-sess-label", Static).update("[dim]Nenhuma tentativa ainda[/dim]")
            return
        pct = int(self._sess_ok / self._sess_total * 100)
        self.query_one("#fc-sess-label", Static).update(
            f"[green]{self._sess_ok}[/green]/{self._sess_total} ({pct}%)"
        )
        self.query_one("#fc-sess-pb", ProgressBar).progress = pct

    @on(Input.Submitted, "#fc-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._feedback_mode:
            self._next_card()
            return
        user = event.value.strip()
        if not user:
            return
        pt = self._card.get("pt", "")
        cid = str(self._card.get("id", self._card.get("en", "")))
        correct = is_close_answer(user, pt)
        register_flashcard_result(self.app.progress, cid, correct)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._sess_total += 1
        if correct:
            self._sess_ok += 1
            self.query_one("#fc-feedback", Static).update("[green]✓ Correto![/green]")
        else:
            self.query_one("#fc-feedback", Static).update(
                f"[red]✗ Quase![/red]  Correto: [bold]{pt}[/bold]\n[dim]Enter para continuar[/dim]"
            )
        self._feedback_mode = True
        self.query_one("#fc-input", Input).clear()
        self._refresh_item_stats()
        self._refresh_sess_stats()

    def action_back(self) -> None:
        self.app.pop_screen()


# SENTENCE TRAINING — menu


class SentenceMenuScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        notes = sorted({s.get("source_note") for s in sentences if s.get("source_note")})
        yield Static("  ◆  SENTENCE TRAINING — Escolha um tópico", classes="sec-header")
        with Container(id="menu-wrap"):
            yield Button(
                f"  Todas as frases  ({len(sentences)} total)", id="all", classes="menu-btn"
            )
            for i, note in enumerate(notes):
                lbl = note.split("/")[-1].replace(".md", "").replace("_", " ").title()
                n = sum(1 for s in sentences if s.get("source_note") == note)
                yield Button(f"  {lbl}  ({n})", id=f"n_{i}", classes="menu-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        notes = sorted({s.get("source_note") for s in sentences if s.get("source_note")})
        if bid == "all":
            self.app.push_screen(SentencePracticeScreen(None))
        elif bid.startswith("n_"):
            self.app.push_screen(SentencePracticeScreen(notes[int(bid[2:])]))

    def action_back(self) -> None:
        self.app.pop_screen()


# SENTENCE TRAINING — practice


class SentencePracticeScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def __init__(self, source_note: str | None):
        super().__init__()
        self._source_note = source_note
        self._sent: dict[str, Any] = {}
        self._sess_ok = 0
        self._sess_total = 0
        self._feedback_mode = False

    def _label(self) -> str:
        if self._source_note:
            return self._source_note.split("/")[-1].replace(".md", "").replace("_", " ").title()
        return "Todas as frases"

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  SENTENCE TRAINING — {self._label()}", classes="sec-header")
        with Container(id="st-wrap"):
            yield Static("[dim]Sessão: 0/0[/dim]", id="st-session")
            yield Static("", id="st-sentence")
            yield Static(
                "[dim]Traduz para Português OU repete a frase em Inglês[/dim]", classes="muted"
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
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        if self._source_note:
            sentences = [s for s in sentences if s.get("source_note") == self._source_note]
        if not sentences:
            self.query_one("#st-sentence", Static).update("[red]Nenhuma frase encontrada.[/red]")
            return
        prog = self.app.progress.get("sentences", {})  # type: ignore[attr-defined]
        self._sent = _w_sentence(sentences, prog, progress=self.app.progress)  # type: ignore[attr-defined]
        en = self._sent.get("en", "")
        self.query_one("#st-sentence", Static).update(f"[bold][yellow]{en}[/yellow][/bold]")
        self.query_one("#st-feedback", Static).update("")
        self.query_one("#st-override", Button).add_class("hidden")
        inp = self.query_one("#st-input", Input)
        inp.clear()
        inp.focus()
        self._feedback_mode = False

    def _update_session(self) -> None:
        if self._sess_total == 0:
            return
        pct = int(self._sess_ok / self._sess_total * 100)
        self.query_one("#st-session", Static).update(
            f"[dim]Sessão:[/dim] [green]{self._sess_ok}[/green]/{self._sess_total} ({pct}%)"
        )

    @on(Input.Submitted, "#st-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._feedback_mode:
            self._next_sentence()
            return
        user = event.value.strip()
        if not user:
            return
        en = self._sent.get("en", "")
        pt = self._sent.get("pt", "")
        pt_d = normalize_pt_phrase(pt)
        sid = str(self._sent.get("id", en))
        correct = is_close_answer(user, [pt, pt_d, en])
        register_sentence_result(self.app.progress, sid, correct)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._sess_total += 1
        if correct:
            self._sess_ok += 1
            self.query_one("#st-feedback", Static).update("[green]✓ Correto![/green]")
            self.query_one("#st-override", Button).add_class("hidden")
        else:
            hint = _detect_tense_hint(en, user)
            hint_line = f"\n[yellow]{hint}[/yellow]" if hint else ""
            self.query_one("#st-feedback", Static).update(
                f"[red]✗ Quase![/red]{hint_line}\n"
                f"[dim]Esperado (PT):[/dim] [bold]{pt_d}[/bold]\n"
                f"[dim]Enter para continuar[/dim]"
            )
            self.query_one("#st-override", Button).remove_class("hidden")
        self._feedback_mode = True
        self.query_one("#st-input", Input).clear()
        self._update_session()

    @on(Button.Pressed, "#st-override")
    def on_override(self) -> None:
        sid = str(self._sent.get("id", self._sent.get("en", "")))
        register_sentence_result(self.app.progress, sid, correct=True)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._sess_ok += 1
        self._update_session()
        self._next_sentence()

    def action_back(self) -> None:
        self.app.pop_screen()


# TO BE TRAINER


class ToBeScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def __init__(self):
        super().__init__()
        self._ex: dict[str, Any] = {}
        self._opts: list[str] = []
        self._sess_ok = 0
        self._sess_total = 0
        self._feedback_mode = False

    def compose(self) -> ComposeResult:
        yield Static("  ◆  TO BE TRAINER — am / is / are / was / were", classes="sec-header")
        with Container(id="tobe-wrap"):
            yield Static("[dim]Sessão: 0/0[/dim]", id="tobe-session")
            yield Static("", id="tobe-question")
            yield Static("", id="tobe-pt", classes="muted")
            yield Button("", id="opt-0", classes="opt-btn")
            yield Button("", id="opt-1", classes="opt-btn")
            yield Button("", id="opt-2", classes="opt-btn")
            yield Button("", id="opt-3", classes="opt-btn")
            yield Static("", id="tobe-feedback")
        yield Footer()

    def on_mount(self) -> None:
        self._next()

    def _next(self) -> None:
        exercises = self.app.content.get("to_be_exercises", [])  # type: ignore[attr-defined]
        if not exercises:
            self.query_one("#tobe-question", Static).update(
                "[red]Nenhum exercício encontrado.[/red]"
            )
            return
        self._ex = _w_to_be(exercises, self.app.progress)  # type: ignore[attr-defined]
        self._opts = self._ex.get("options", [])
        self.query_one("#tobe-question", Static).update(
            f"[bold]{self._ex.get('question', '')}[/bold]"
        )
        self.query_one("#tobe-pt", Static).update(f"[dim]{self._ex.get('pt', '')}[/dim]")
        self.query_one("#tobe-feedback", Static).update("")
        for i in range(4):
            btn = self.query_one(f"#opt-{i}", Button)
            if i < len(self._opts):
                btn.label = f"  {self._opts[i]}"
                btn.display = True
            else:
                btn.display = False
        self._feedback_mode = False
        self._update_session()

    def _update_session(self) -> None:
        if self._sess_total == 0:
            return
        pct = int(self._sess_ok / self._sess_total * 100)
        self.query_one("#tobe-session", Static).update(
            f"[dim]Sessão:[/dim] [green]{self._sess_ok}[/green]/{self._sess_total} ({pct}%)"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("opt-"):
            return
        if self._feedback_mode:
            self._next()
            return
        idx = int(bid[4:])
        if idx >= len(self._opts):
            return
        chosen = self._opts[idx]
        correct_ans = self._ex.get("correct", "")
        expl = self._ex.get("explanation", "")
        ex_id = str(self._ex.get("id", self._ex.get("question", "")))
        correct = chosen == correct_ans
        register_to_be_result(self.app.progress, ex_id, correct)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._sess_total += 1
        if correct:
            self._sess_ok += 1
            fb = "[green]✓ Correto![/green]"
        else:
            fb = f"[red]✗  Correto: [bold]{correct_ans}[/bold][/red]"
        if expl:
            fb += f"\n[dim]{expl}[/dim]"
        fb += "\n[dim]Clique em qualquer opção para continuar[/dim]"
        self.query_one("#tobe-feedback", Static).update(fb)
        self._feedback_mode = True
        self._update_session()

    def action_back(self) -> None:
        self.app.pop_screen()


# DICTIONARY


class DictionaryScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        yield Static("  ◆  DICTIONARY — local + online", classes="sec-header")
        with Container(id="dict-wrap"):
            yield Static(
                "[dim]Digite uma palavra em inglês e pressione Enter.[/dim]", classes="muted"
            )
            yield Input(placeholder="Ex: happy, tired, work...", id="dict-input")
            yield Static("[dim]Aguardando...[/dim]", id="dict-result")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#dict-input", Input).focus()

    @on(Input.Submitted, "#dict-input")
    def on_search(self, event: Input.Submitted) -> None:
        word = event.value.strip()
        if not word:
            return
        register_dictionary_lookup(self.app.progress)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self.query_one("#dict-result", Static).update("[dim]Buscando...[/dim]")
        self._do_search(word)

    @work(thread=True)
    def _do_search(self, word: str) -> None:
        content = self.app.content  # type: ignore[attr-defined]
        meaning = find_in_dictionary(content, word)
        if meaning:
            result = (
                f"[bold]{word}[/bold]\n"
                f"[green]Significado (PT):[/green] {meaning}\n"
                f"[dim]Fonte: dicionário local[/dim]"
            )
        else:
            entry = fetch_dictionary_entry(word.lower())
            if entry:
                lines = [f"[bold]{word}[/bold]  [dim](online)[/dim]"]
                if entry.get("phonetic"):
                    lines.append(f"Fonética: [cyan]{entry['phonetic']}[/cyan]")
                if entry.get("part_of_speech"):
                    lines.append(f"Tipo: [yellow]{entry['part_of_speech']}[/yellow]")
                if entry.get("definition"):
                    lines.append(f"Definição: {entry['definition']}")
                if entry.get("example"):
                    lines.append(f'Exemplo: [italic]"{entry["example"]}"[/italic]')
                result = "\n".join(lines)
            else:
                import difflib

                d = content.get("dictionary", {})
                close = difflib.get_close_matches(word.lower(), d.keys(), n=3, cutoff=0.6)
                if close:
                    sugg = "\n".join(f"  • [bold]{m}[/bold]: {d[m]}" for m in close)
                    result = f"[red]'{word}' não encontrado.[/red]\n\n[dim]Talvez:[/dim]\n{sugg}"
                else:
                    result = f"[red]'{word}' não encontrado.[/red]"

        self.app.call_from_thread(self.query_one("#dict-result", Static).update, result)

    def action_back(self) -> None:
        self.app.pop_screen()


# AI COACH


class AICoachScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        yield Static("  ◆  MIU — AI COACH — Geração de frases", classes="sec-header")
        with Container(id="st-wrap"):
            yield Static("[dim]Tempo verbal:[/dim]", classes="muted")
            yield Button("  Present tense", id="present", classes="menu-btn")
            yield Button("  Past tense", id="past", classes="menu-btn")
            yield Rule()
            yield Static("", id="coach-result")
        yield Footer()

    @on(Button.Pressed, "#present")
    def gen_present(self) -> None:
        self._generate("present")

    @on(Button.Pressed, "#past")
    def gen_past(self) -> None:
        self._generate("past")

    def _generate(self, tense: str) -> None:
        self.query_one("#coach-result", Static).update("[dim]Gerando frases com IA...[/dim]")
        self._do_generate(tense)

    @work(thread=True)
    def _do_generate(self, tense: str) -> None:
        import re

        from .ai_coach import (
            _fallback_pt,
            _fallback_sentences,
            _pick_targets,
            _prompt_generate_sentences,
        )

        targets = _pick_targets()
        prompt = _prompt_generate_sentences(targets, tense, 5)
        text = generate_text(prompt, max_new_tokens=220)

        lines: list[str] = []
        if text:
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln or "<think>" in ln.lower():
                    continue
                if re.search(r"\b(am|is|are|was|were)\b", ln.lower()):
                    lines.append(ln)

        used_builtin_fallback = False
        if not lines:
            lines = [f"{s}  |  {_fallback_pt(s)}" for s in _fallback_sentences(targets, tense, 5)]
            used_builtin_fallback = True

        output_parts = []
        for ln in lines[:5]:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 2:
                output_parts.append(
                    f"[bold][yellow]{parts[0]}[/yellow][/bold]\n  [dim]PT:[/dim] {parts[1]}"
                )
            else:
                output_parts.append(f"[bold][yellow]{ln}[/yellow][/bold]")

        result = "\n\n".join(output_parts) or "[red]Sem resposta da IA.[/red]"
        if used_builtin_fallback:
            result = (
                "[yellow]Exemplos locais limitados: nenhuma IA avaliou esta saída.[/yellow]\n\n"
                + result
            )
        self.app.call_from_thread(self.query_one("#coach-result", Static).update, result)

    def action_back(self) -> None:
        self.app.pop_screen()


# CONVERSATION — classic terminal, [Miu] green / [You] white


class ConversationScreen(Screen):
    BINDINGS = [
        Binding("escape", "end_conv", "Encerrar"),
        Binding("ctrl+s", "save_conv", "Salvar"),
    ]

    def __init__(self):
        super().__init__()
        self._history: list[dict[str, str]] = []
        self._busy = False
        self._turns = 0
        self._reviewing = False

    def compose(self) -> ComposeResult:
        yield Static(
            f"  ◆  CONVERSATION WITH MIU  —  {DEFAULT_MODEL}   |   Esc: encerrar   Ctrl+S: salvar",
            id="conv-titlebar",
        )
        yield RichLog(id="conv-log", markup=True, highlight=False, wrap=True)
        yield Static("", id="conv-status")
        with Container(id="conv-input-area"):
            yield Input(placeholder="Type in English and press Enter...", id="conv-input")
        yield Footer()

    def on_mount(self) -> None:
        if not _check_ollama():
            log = self.query_one("#conv-log", RichLog)
            log.write("[red][ERR][/red] Ollama não está rodando.")
            log.write("[dim]Inicie com:[/dim] [bold]ollama serve[/bold]")
            log.write("[dim]Baixe um modelo:[/dim] [bold]ollama pull llama3.2[/bold]")
            self.query_one("#conv-input", Input).disabled = True
            return
        self._set_busy(True)
        self._do_greet()

    @work(thread=True)
    def _do_greet(self) -> None:
        first_msg = {
            "role": "user",
            "content": "Hello! Let's start a conversation to practice my English.",
        }
        reply = generate_conversation_reply([first_msg], system_prompt=SYSTEM_PROMPT)
        self._history.append(first_msg)
        if reply:
            self._history.append({"role": "assistant", "content": reply})
            self.app.call_from_thread(self._write, "ai", reply)
        else:
            self.app.call_from_thread(self._write, "err", "Ollama não respondeu. Modelo carregado?")
        self.app.call_from_thread(self._set_busy, False)

    @work(thread=True, exclusive=True)
    def _do_reply(self) -> None:
        reply = generate_conversation_reply(self._history, system_prompt=SYSTEM_PROMPT)
        if reply:
            self._history.append({"role": "assistant", "content": reply})
            self.app.call_from_thread(self._write, "ai", reply)
        else:
            self.app.call_from_thread(self._write, "err", "Sem resposta (Ollama ainda carregando?)")
        self.app.call_from_thread(self._set_busy, False)

    @work(thread=True)
    def _do_review(self) -> None:
        self.app.call_from_thread(self._write, "sep", "─" * 44)
        self.app.call_from_thread(
            self._write, "sep", f"Conversa encerrada — {self._turns} mensagens."
        )
        self.app.call_from_thread(self._write, "sep", "Analisando erros gramaticais...")
        review = _get_error_review(self._history)
        self.app.call_from_thread(self._write, "sep", "─── Grammar Review ───")
        if review:
            for line in review.strip().split("\n"):
                if line.strip():
                    self.app.call_from_thread(self._write, "sep", line)
        else:
            self.app.call_from_thread(self._write, "sep", "(Sem resposta da IA para a revisão.)")
        self.app.call_from_thread(self._write, "sep", "─" * 44)
        self.app.progress["conversation_sessions"] = (
            int(self.app.progress.get("conversation_sessions", 0)) + 1
        )  # type: ignore[attr-defined]
        self.app.progress["conversation_turns"] = (
            int(self.app.progress.get("conversation_turns", 0)) + self._turns
        )  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._reviewing = False
        self._turns = 0
        self.app.call_from_thread(
            self.query_one("#conv-status", Static).update,
            "[dim]Pressione Esc para voltar ao menu.[/dim]",
        )

    def _write(self, role: str, text: str) -> None:
        log = self.query_one("#conv-log", RichLog)
        text = escape(text)
        if role == "ai":
            log.write(f"[green][Miu][/green] {text}\n")
        elif role == "user":
            log.write(f"[white][You][/white] {text}\n")
        elif role == "err":
            log.write(f"[red][ERR][/red] {text}\n")
        else:
            log.write(f"[dim]{text}[/dim]")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        inp = self.query_one("#conv-input", Input)
        inp.disabled = busy
        status = self.query_one("#conv-status", Static)
        if busy:
            status.update("[dim]AI está pensando...[/dim]")
        else:
            status.update("")
            inp.focus()

    @on(Input.Submitted, "#conv-input")
    def on_message(self, event: Input.Submitted) -> None:
        if self._busy:
            return
        text = event.value.strip()
        if not text:
            return
        self.query_one("#conv-input", Input).clear()
        self._write("user", text)
        self._history.append({"role": "user", "content": text})
        self._turns += 1
        self._set_busy(True)
        self._do_reply()

    def action_end_conv(self) -> None:
        if self._reviewing:
            return
        if self._turns == 0:
            self.app.pop_screen()
            return
        self._reviewing = True
        self._set_busy(True)
        self._do_review()

    def action_save_conv(self) -> None:
        if self._history:
            path = _save_conversation(self._history, None)
            self._write("sep", f"Salvo em: {path.name}")


# PROGRESS


class ProgressScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        stats = compute_overall(self.app.progress)  # type: ignore[attr-defined]
        prog = self.app.progress  # type: ignore[attr-defined]

        yield Static("  ◆  PROGRESS — Estatísticas", classes="sec-header")
        with Container(id="prog-wrap"):
            yield Static("Resumo Geral", classes="prog-title")
            yield Static(
                f"[bold]Total:[/bold] {stats['total_seen']}   "
                f"[bold]Correto:[/bold] {stats['total_correct']}   "
                f"[bold]Precisão:[/bold] [green]{stats['accuracy']:.1f}%[/green]"
            )
            yield Rule()

            modes = [
                ("To Be Trainer", stats["to_be_seen"], stats["to_be_correct"]),
                ("Flashcards", stats["flashcards_seen"], stats["flashcards_correct"]),
                ("Sentences", stats["sentences_seen"], stats["sentences_correct"]),
            ]
            for name, seen, correct in modes:
                acc = correct / seen * 100 if seen > 0 else 0
                with Horizontal(classes="prog-row"):
                    yield Static(name, classes="prog-name")
                    yield Static(f"{correct}/{seen}", classes="prog-count")
                    yield ProgressBar(total=100, show_percentage=False, classes="prog-bar")

            yield Rule()
            yield Static(f"[dim]Dictionary lookups:[/dim] {stats['dictionary_lookups']}")
            yield Static(f"[dim]Miu interactions:[/dim]   {stats['ai_uses']}")
            conv_s = int(prog.get("conversation_sessions", 0))
            conv_t = int(prog.get("conversation_turns", 0))
            if conv_s > 0:
                yield Static(
                    f"[dim]Conversations:[/dim]       {conv_s} sessões, {conv_t} mensagens"
                )

        yield Footer()

    def on_mount(self) -> None:
        stats = compute_overall(self.app.progress)  # type: ignore[attr-defined]
        vals = [
            stats["to_be_correct"] / max(stats["to_be_seen"], 1) * 100,
            stats["flashcards_correct"] / max(stats["flashcards_seen"], 1) * 100,
            stats["sentences_correct"] / max(stats["sentences_seen"], 1) * 100,
        ]
        for bar, val in zip(self.query(".prog-bar").results(ProgressBar), vals):
            bar.progress = val

    def action_back(self) -> None:
        self.app.pop_screen()


# DICTATION — menu


class DictationMenuScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def compose(self) -> ComposeResult:
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        notes = sorted({s.get("source_note") for s in sentences if s.get("source_note")})
        yield Static("  ◆  DICTATION — Ouça e escreva em inglês", classes="sec-header")
        with Container(id="menu-wrap"):
            yield Button(
                f"  Todas as frases  ({len(sentences)} total)", id="all", classes="menu-btn"
            )
            for i, note in enumerate(notes):
                lbl = note.split("/")[-1].replace(".md", "").replace("_", " ").title()
                n = sum(1 for s in sentences if s.get("source_note") == note)
                yield Button(f"  {lbl}  ({n})", id=f"n_{i}", classes="menu-btn")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        notes = sorted({s.get("source_note") for s in sentences if s.get("source_note")})
        if bid == "all":
            self.app.push_screen(DictationPracticeScreen(None))
        elif bid.startswith("n_"):
            self.app.push_screen(DictationPracticeScreen(notes[int(bid[2:])]))

    def action_back(self) -> None:
        self.app.pop_screen()


# DICTATION — practice


class DictationPracticeScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    def __init__(self, source_note: str | None):
        super().__init__()
        self._source_note = source_note
        self._sent: dict[str, Any] = {}
        self._sess_ok = 0
        self._sess_total = 0
        self._feedback_mode = False
        self._rate = "+0%"

    def _label(self) -> str:
        if self._source_note:
            return self._source_note.split("/")[-1].replace(".md", "").replace("_", " ").title()
        return "Todas as frases"

    def compose(self) -> ComposeResult:
        yield Static(f"  ◆  DICTATION — {self._label()}", classes="sec-header")
        with Container(id="dt-wrap"):
            yield Static("[dim]Sessão: 0/0[/dim]", id="dt-session")
            yield Button("  🐢  Velocidade: Normal", id="dt-speed", classes="menu-btn")
            yield Static("🎵  Preparando áudio...", id="dt-status")
            yield Static("[dim]Ouça e escreva o que ouviu em inglês[/dim]", classes="muted")
            yield Input(placeholder="Type what you heard...", id="dt-input", disabled=True)
            yield Button("  ↺  Replay", id="dt-replay", classes="menu-btn", disabled=True)
            yield Static("", id="dt-feedback")
        yield Footer()

    def on_mount(self) -> None:
        self._next_sentence()

    def _next_sentence(self) -> None:
        sentences = self.app.content.get("sentences", [])  # type: ignore[attr-defined]
        if self._source_note:
            sentences = [s for s in sentences if s.get("source_note") == self._source_note]
        if not sentences:
            self.query_one("#dt-status", Static).update("[red]Nenhuma frase encontrada.[/red]")
            return
        prog = self.app.progress.get("sentences", {})  # type: ignore[attr-defined]
        self._sent = _w_sentence(sentences, prog, progress=self.app.progress)  # type: ignore[attr-defined]
        self.query_one("#dt-status", Static).update(
            "🎵  [yellow]Playing audio...[/yellow]  (aguarde)"
        )
        self.query_one("#dt-feedback", Static).update("")
        inp = self.query_one("#dt-input", Input)
        inp.clear()
        inp.disabled = True
        self.query_one("#dt-replay", Button).disabled = True
        self._feedback_mode = False
        self._play_audio()

    @on(Button.Pressed, "#dt-speed")
    def on_toggle_speed(self) -> None:
        from .audio import RATE_NORMAL, RATE_SLOW

        if self._rate == RATE_NORMAL:
            self._rate = RATE_SLOW
            self.query_one("#dt-speed", Button).label = "  🐢  Velocidade: Devagar"
        else:
            self._rate = RATE_NORMAL
            self.query_one("#dt-speed", Button).label = "  🐇  Velocidade: Normal"

    @work(thread=True)
    def _play_audio(self) -> None:
        from .audio import audio_available, speak

        en = self._sent.get("en", "")
        success = speak(en, rate=self._rate) if audio_available() else False

        def _after() -> None:
            self.query_one("#dt-replay", Button).disabled = False
            inp = self.query_one("#dt-input", Input)
            inp.disabled = False
            inp.focus()
            if success:
                self.query_one("#dt-status", Static).update(
                    "🎵  [dim]Pronto — digite o que ouviu[/dim]"
                )
            else:
                self.query_one("#dt-status", Static).update(
                    f"[yellow]Áudio indisponível.[/yellow]  Frase: [bold]{en}[/bold]"
                )

        self.app.call_from_thread(_after)

    @on(Button.Pressed, "#dt-replay")
    def on_replay(self) -> None:
        if self._feedback_mode:
            return
        self.query_one("#dt-input", Input).disabled = True
        self.query_one("#dt-replay", Button).disabled = True
        self.query_one("#dt-status", Static).update("🎵  [yellow]Replaying...[/yellow]")
        self._play_audio()

    def _update_session(self) -> None:
        if self._sess_total == 0:
            return
        pct = int(self._sess_ok / self._sess_total * 100)
        self.query_one("#dt-session", Static).update(
            f"[dim]Sessão:[/dim] [green]{self._sess_ok}[/green]/{self._sess_total} ({pct}%)"
        )

    @on(Input.Submitted, "#dt-input")
    def on_answer(self, event: Input.Submitted) -> None:
        if self._feedback_mode:
            self._next_sentence()
            return
        user = event.value.strip()
        if not user:
            return
        en = self._sent.get("en", "")
        pt = self._sent.get("pt", "")
        sid = str(self._sent.get("id", en))
        correct = is_close_answer(user, en, strict=True)
        register_sentence_result(self.app.progress, sid, correct)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]
        self._sess_total += 1
        if correct:
            self._sess_ok += 1
            self.query_one("#dt-feedback", Static).update("[green]✓ Correto![/green]")
        else:
            self.query_one("#dt-feedback", Static).update(
                f"[red]✗  Era:[/red] [bold]{en}[/bold]\n"
                f"[dim]PT: {pt}[/dim]\n"
                f"[dim]Enter para continuar[/dim]"
            )
        self._feedback_mode = True
        self.query_one("#dt-input", Input).clear()
        self._update_session()

    def action_back(self) -> None:
        self.app.pop_screen()


# FREE WRITING


class FreeWritingScreen(Screen):
    BINDINGS = [Binding("escape,q", "back", "Voltar")]

    _TOPICS: list[str] = [
        "your daily routine",
        "your favorite food",
        "your family",
        "your job or studies",
        "a place you like",
        "your weekend",
        "your hobbies",
        "the weather today",
        "your goals for this year",
        "a movie or series you watched recently",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._topic = ""

    def compose(self) -> ComposeResult:
        yield Static("  ◆  FREE WRITING — Escreva e receba correção da IA", classes="sec-header")
        with Container(id="fw-wrap"):
            yield Static("[dim]Escolha um tópico:[/dim]", classes="muted")
            for i, t in enumerate(self._TOPICS):
                yield Button(f"  {i + 1:2}.  {t}", id=f"topic_{i}", classes="menu-btn")
            with Container(id="fw-input-area", classes="hidden"):
                yield Static("", id="fw-selected")
                yield Static(
                    "[dim]Escreva 1-2 frases em inglês sobre o tópico e pressione Enter[/dim]",
                    classes="muted",
                )
                yield Input(placeholder="Write in English...", id="fw-input")
                yield Static("", id="fw-feedback")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("topic_"):
            return
        self._topic = self._TOPICS[int(bid[6:])]
        for btn in self.query(".menu-btn").results(Button):
            btn.display = False
        area = self.query_one("#fw-input-area")
        area.remove_class("hidden")
        self.query_one("#fw-selected", Static).update(
            f"[dim]Tópico:[/dim] [bold][yellow]{self._topic}[/yellow][/bold]"
        )
        self.query_one("#fw-feedback", Static).update("")
        inp = self.query_one("#fw-input", Input)
        inp.clear()
        inp.focus()

    @on(Input.Submitted, "#fw-input")
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self._topic:
            return
        self.query_one("#fw-feedback", Static).update("[dim]Enviando para a IA...[/dim]")
        self.query_one("#fw-input", Input).disabled = True
        self._do_check(text, self._topic)

    @work(thread=True)
    def _do_check(self, text: str, topic: str) -> None:
        from .free_writing import _prompt

        response = generate_text(_prompt(text, topic), max_new_tokens=400)
        register_ai_use(self.app.progress)  # type: ignore[attr-defined]
        save_progress(self.app.progress)  # type: ignore[attr-defined]

        feedback = (
            escape(response.strip())
            if response
            else (
                "[red]IA indisponível.[/red] Nenhuma avaliação foi realizada.\n"
                "[dim]Verifique a configuração do Ollama ou do Hugging Face.[/dim]"
            )
        )

        def _update() -> None:
            self.query_one("#fw-feedback", Static).update(feedback)
            inp = self.query_one("#fw-input", Input)
            inp.disabled = False
            inp.clear()
            inp.focus()

        self.app.call_from_thread(_update)

    def action_back(self) -> None:
        self.app.pop_screen()


# GROUNDED TUTOR


class RagTutorScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Voltar")]

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("  ◆  MIU — GROUNDED STUDY TUTOR", classes="sec-header")
        with Container(id="rag-wrap"):
            yield Static(
                "Pergunte sobre as notas, exemplos ou erros anteriores. "
                "A resposta mostrará as fontes recuperadas.",
                markup=False,
            )
            yield Input(
                placeholder="Ex.: Por que 'She don't like coffee' está errado?", id="rag-input"
            )
            yield Static("", id="rag-status", markup=False)
            yield Static("", id="rag-answer", markup=False)
            yield Static("", id="rag-sources", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#rag-input", Input).focus()

    @on(Input.Submitted, "#rag-input")
    def submit_question(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self._busy:
            return
        self._busy = True
        event.input.disabled = True
        self.query_one("#rag-status", Static).update(
            "Buscando no material local e preparando uma resposta fundamentada..."
        )
        self._answer(question)

    @work(thread=True)
    def _answer(self, question: str) -> None:
        service = None
        try:
            service = RagService()
            result = service.ask(
                question,
                self.app.content,  # type: ignore[attr-defined]
                self.app.progress,  # type: ignore[attr-defined]
            )
            register_ai_use(self.app.progress)  # type: ignore[attr-defined]
            save_progress(self.app.progress)  # type: ignore[attr-defined]
        except Exception as exc:
            self.app.call_from_thread(self._show_error, str(exc))
        else:
            self.app.call_from_thread(self._show_result, result)
        finally:
            if service is not None:
                service.close()

    def _show_result(self, result: Any) -> None:
        self.query_one("#rag-answer", Static).update(result.text)
        source_lines = ["Fontes recuperadas:"]
        source_lines.extend(
            f"[{source.reference}] {source.source} — {source.title} ({source.score:.3f})"
            for source in result.sources
        )
        source_lines.append(f"Índice local: {result.indexed_chunks} chunks")
        self.query_one("#rag-sources", Static).update("\n".join(source_lines))
        self._finish("Resposta fundamentada no material local.")

    def _show_error(self, message: str) -> None:
        self.query_one("#rag-answer", Static).update(f"RAG indisponível: {message}")
        self.query_one("#rag-sources", Static).update("")
        self._finish("Nenhuma resposta foi gerada.")

    def _finish(self, status: str) -> None:
        self._busy = False
        self.query_one("#rag-status", Static).update(status)
        input_widget = self.query_one("#rag-input", Input)
        input_widget.disabled = False
        input_widget.clear()
        input_widget.focus()

    def action_back(self) -> None:
        self.app.pop_screen()


# APP


class EnglishTrainerApp(App):
    CSS = APP_CSS
    TITLE = "English Trainer AI"

    def __init__(self) -> None:
        super().__init__()
        self.content: dict[str, Any] = {}
        self.progress: dict[str, Any] = {}

    def on_mount(self) -> None:
        self.content = load_content()
        self.progress = load_progress()
        self.push_screen(HomeScreen())

    def on_unmount(self) -> None:
        save_progress(self.progress)


def run_tui() -> None:
    EnglishTrainerApp().run()
