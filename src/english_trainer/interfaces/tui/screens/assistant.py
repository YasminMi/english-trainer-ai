"""AI-assisted screens: dictionary, coach, conversation, free writing and grounded tutor."""

from __future__ import annotations

from typing import ClassVar

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.widgets import Button, Footer, Input, RichLog, Rule, Static

from english_trainer.domain.content import find_in_dictionary
from english_trainer.domain.progress import register_conversation, register_dictionary_lookup
from english_trainer.interfaces.tui.screens.base import TrainerScreen
from english_trainer.rag import RagAnswer, RagError
from english_trainer.services.coach import SentenceBatch, Tense
from english_trainer.services.conversation import ConversationSession
from english_trainer.services.dictionary import suggest_similar
from english_trainer.services.writing import TOPICS

# --------------------------------------------------------------------------- dictionary


class DictionaryScreen(TrainerScreen):
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
        register_dictionary_lookup(self.trainer.progress)
        self.trainer.save_progress()
        self.query_one("#dict-result", Static).update("[dim]Buscando...[/dim]")
        self._search(word)

    @work(thread=True, exclusive=True)
    def _search(self, word: str) -> None:
        self.app.call_from_thread(self.query_one("#dict-result", Static).update, self._lookup(word))

    def _lookup(self, word: str) -> str:
        safe_word = escape(word)
        if meaning := find_in_dictionary(self.trainer.content, word):
            return (
                f"[bold]{safe_word}[/bold]\n[green]Significado (PT):[/green] {escape(meaning)}\n"
                "[dim]Fonte: dicionário local[/dim]"
            )
        if entry := self.trainer.dictionary.lookup(word):
            lines = [f"[bold]{safe_word}[/bold]  [dim](online)[/dim]"]
            for label, value, style in (
                ("Fonética", entry.phonetic, "cyan"),
                ("Tipo", entry.part_of_speech, "yellow"),
                ("Definição", entry.definition, ""),
                ("Exemplo", entry.example, "italic"),
            ):
                if value:
                    text = f"[{style}]{escape(value)}[/{style}]" if style else escape(value)
                    lines.append(f"{label}: {text}")
            return "\n".join(lines)

        local = self.trainer.content["dictionary"]
        suggestions = suggest_similar(word, local)
        if not suggestions:
            return f"[red]'{safe_word}' não encontrado.[/red]"
        listing = "\n".join(
            f"  • [bold]{escape(m)}[/bold]: {escape(local[m])}" for m in suggestions
        )
        return f"[red]'{safe_word}' não encontrado.[/red]\n\n[dim]Talvez:[/dim]\n{listing}"


# --------------------------------------------------------------------------- coach


class CoachScreen(TrainerScreen):
    def compose(self) -> ComposeResult:
        yield Static("  ◆  MIU — AI COACH — Geração de frases", classes="sec-header")
        with Container(id="st-wrap"):
            yield Static("[dim]Tempo verbal:[/dim]", classes="muted")
            yield Button("  Present tense", id="present", classes="menu-btn")
            yield Button("  Past tense", id="past", classes="menu-btn")
            yield Rule()
            yield Static("", id="coach-result")
        yield Footer()

    @on(Button.Pressed, "#present, #past")
    def on_generate(self, event: Button.Pressed) -> None:
        self.query_one("#coach-result", Static).update("[dim]Gerando frases com IA...[/dim]")
        self._generate("past" if event.button.id == "past" else "present")

    @work(thread=True, exclusive=True)
    def _generate(self, tense: Tense) -> None:
        batch = self.trainer.coach.generate_sentences(tense, 5)
        self.trainer.record_ai_use()
        self.app.call_from_thread(self.query_one("#coach-result", Static).update, _render(batch))


def _render(batch: SentenceBatch) -> str:
    blocks = []
    for sentence in batch.sentences:
        block = f"[bold][yellow]{escape(sentence.en)}[/yellow][/bold]"
        if sentence.pt:
            block += f"\n  [dim]PT:[/dim] {escape(sentence.pt)}"
        if sentence.tip:
            block += f"\n  [dim]Dica:[/dim] {escape(sentence.tip)}"
        blocks.append(block)
    text = "\n\n".join(blocks) or "[red]Sem resposta da IA.[/red]"
    if batch.used_builtin_examples:
        text = (
            "[yellow]Exemplos locais limitados: nenhuma IA avaliou esta saída.[/yellow]\n\n" + text
        )
    return text


# --------------------------------------------------------------------------- conversation


class ConversationScreen(TrainerScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "end_conversation", "Encerrar"),
        Binding("ctrl+s", "save_conversation", "Salvar"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._session: ConversationSession | None = None
        self._busy = False
        self._reviewing = False

    def compose(self) -> ComposeResult:
        model = escape(self.trainer.settings.ollama.model)
        yield Static(
            f"  ◆  CONVERSATION WITH MIU  —  {model}   |   Esc: encerrar   Ctrl+S: salvar",
            id="conv-titlebar",
        )
        yield RichLog(id="conv-log", markup=True, highlight=False, wrap=True)
        yield Static("", id="conv-status")
        with Container(id="conv-input-area"):
            yield Input(placeholder="Type in English and press Enter...", id="conv-input")
        yield Footer()

    def on_mount(self) -> None:
        self._set_busy(True)
        self._greet()

    @work(thread=True)
    def _greet(self) -> None:
        if not self.trainer.ollama.is_available():
            for line in (
                "[red][ERR][/red] Ollama não está rodando.",
                "[dim]Inicie com:[/dim] [bold]ollama serve[/bold]",
                "[dim]Baixe um modelo:[/dim] [bold]ollama pull llama3.2[/bold]",
            ):
                self.app.call_from_thread(self.query_one("#conv-log", RichLog).write, line)
            self.app.call_from_thread(self._set_busy, False)
            return
        self._session = self.trainer.new_conversation()
        reply = self._session.start()
        self._show_reply(reply, "Ollama não respondeu. O modelo foi baixado?")

    @work(thread=True, exclusive=True)
    def _reply(self, text: str) -> None:
        if self._session is not None:
            self._show_reply(self._session.send(text), "Sem resposta (Ollama ainda carregando?)")

    def _show_reply(self, reply: str | None, error: str) -> None:
        self.app.call_from_thread(self._write, "ai" if reply else "err", reply or error)
        self.app.call_from_thread(self._set_busy, False)

    @work(thread=True)
    def _review(self) -> None:
        session = self._session
        if session is None:
            return
        for line in ("─" * 44, f"Conversa encerrada — {session.turns} mensagens.", "Analisando..."):
            self.app.call_from_thread(self._write, "sep", line)
        review = session.review()
        self.app.call_from_thread(self._write, "sep", "─── Grammar Review ───")
        for line in (review or "(Sem resposta da IA para a revisão.)").splitlines():
            if line.strip():
                self.app.call_from_thread(self._write, "sep", line)
        self.app.call_from_thread(self._write, "sep", "─" * 44)
        register_conversation(self.trainer.progress, session.turns)
        self.trainer.save_progress()
        self.app.call_from_thread(
            self.query_one("#conv-status", Static).update,
            "[dim]Pressione Esc para voltar ao menu.[/dim]",
        )
        self._session = None

    def _write(self, role: str, text: str) -> None:
        log = self.query_one("#conv-log", RichLog)
        prefix = {
            "ai": "[green][Miu][/green] ",
            "user": "[white][You][/white] ",
            "err": "[red][ERR][/red] ",
        }
        if role in prefix:
            log.write(f"{prefix[role]}{escape(text)}\n")
        else:
            log.write(f"[dim]{escape(text)}[/dim]")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        field = self.query_one("#conv-input", Input)
        field.disabled = busy or self._session is None
        self.query_one("#conv-status", Static).update(
            "[dim]Miu está pensando...[/dim]" if busy else ""
        )
        if not field.disabled:
            field.focus()

    @on(Input.Submitted, "#conv-input")
    def on_message(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self._busy or not text or self._session is None:
            return
        event.input.clear()
        self._write("user", text)
        self._set_busy(True)
        self._reply(text)

    def action_end_conversation(self) -> None:
        if self._reviewing:
            self.app.pop_screen()
            return
        if self._session is None or self._session.turns == 0:
            self.app.pop_screen()
            return
        self._reviewing = True
        self._set_busy(True)
        self._review()

    def action_save_conversation(self) -> None:
        if self._session and self._session.history:
            path = self._session.save(self.trainer.settings.paths.conversations_dir)
            self._write("sep", f"Salvo em: {path.name}")


# --------------------------------------------------------------------------- free writing


class FreeWritingScreen(TrainerScreen):
    def __init__(self) -> None:
        super().__init__()
        self._topic = ""

    def compose(self) -> ComposeResult:
        yield Static("  ◆  FREE WRITING — Escreva e receba correção da IA", classes="sec-header")
        with Container(id="fw-wrap"):
            yield Static("[dim]Escolha um tópico:[/dim]", classes="muted")
            for index, topic in enumerate(TOPICS):
                yield Button(f"  {index + 1:2}.  {topic}", id=f"topic_{index}", classes="menu-btn")
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
        button_id = event.button.id or ""
        if not button_id.startswith("topic_"):
            return
        self._topic = TOPICS[int(button_id.removeprefix("topic_"))]
        for button in self.query(".menu-btn").results(Button):
            button.display = False
        self.query_one("#fw-input-area").remove_class("hidden")
        self.query_one("#fw-selected", Static).update(
            f"[dim]Tópico:[/dim] [bold][yellow]{self._topic}[/yellow][/bold]"
        )
        field = self.query_one("#fw-input", Input)
        field.clear()
        field.focus()

    @on(Input.Submitted, "#fw-input")
    def on_submit(self, event: Input.Submitted) -> None:
        if not event.value.strip() or not self._topic:
            return
        self.query_one("#fw-feedback", Static).update("[dim]Enviando para a IA...[/dim]")
        event.input.disabled = True
        self._check(event.value, self._topic)

    @work(thread=True, exclusive=True)
    def _check(self, text: str, topic: str) -> None:
        result = self.trainer.writing.review(text, topic)
        self.trainer.record_ai_use()
        feedback = (
            escape(result.text)
            if result
            else "[red]IA indisponível.[/red] Nenhuma avaliação foi realizada.\n"
            "[dim]Verifique a configuração do Ollama ou do Hugging Face.[/dim]"
        )
        self.app.call_from_thread(self._show_feedback, feedback)

    def _show_feedback(self, feedback: str) -> None:
        self.query_one("#fw-feedback", Static).update(feedback)
        field = self.query_one("#fw-input", Input)
        field.disabled = False
        field.clear()
        field.focus()


# --------------------------------------------------------------------------- grounded tutor


class GroundedTutorScreen(TrainerScreen):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "back", "Voltar")]

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

    @work(thread=True, exclusive=True)
    def _answer(self, question: str) -> None:
        try:
            result = self.trainer.rag().ask(question, self.trainer.content, self.trainer.progress)
        except RagError as exc:
            self.app.call_from_thread(self._show_error, str(exc))
            return
        self.trainer.record_ai_use()
        self.app.call_from_thread(self._show_result, result)

    def _show_result(self, result: RagAnswer) -> None:
        self.query_one("#rag-answer", Static).update(result.text)
        lines = ["Fontes recuperadas:"]
        lines.extend(
            f"[{source.reference}] {source.source} — {source.title} ({source.score:.3f})"
            for source in result.sources
        )
        lines.append(f"Índice local: {result.indexed_chunks} chunks")
        self.query_one("#rag-sources", Static).update("\n".join(lines))
        self._finish("Resposta fundamentada no material local.")

    def _show_error(self, message: str) -> None:
        self.query_one("#rag-answer", Static).update(f"RAG indisponível: {message}")
        self.query_one("#rag-sources", Static).update("")
        self._finish("Nenhuma resposta foi gerada.")

    def _finish(self, status: str) -> None:
        self._busy = False
        self.query_one("#rag-status", Static).update(status)
        field = self.query_one("#rag-input", Input)
        field.disabled = False
        field.clear()
        field.focus()
