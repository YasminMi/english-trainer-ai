"""AI-assisted flows: coach, conversation, free writing and the grounded tutor."""

from __future__ import annotations

from english_trainer.app import Application
from english_trainer.domain.matching import clean_text
from english_trainer.domain.progress import register_conversation
from english_trainer.interfaces.cli.io import ask, choose, header, pause
from english_trainer.rag import RagError
from english_trainer.services.coach import Tense, Verdict
from english_trainer.services.writing import TOPICS, resolve_topic

OLLAMA_HELP = (
    "Ollama is not running.\n\n"
    "To start it:\n"
    "  1. Install Ollama: https://ollama.com\n"
    "  2. Run:            ollama serve\n"
    "  3. Pull a model:   ollama pull llama3.2\n"
)


def coach_menu(app: Application) -> None:
    options = [
        ("Generate new sentences (gerar frases)", "generate"),
        ("Check my translation (corrigir minha tradução)", "check"),
    ]
    while (action := choose("Miu — AI Coach", options)) is not None:
        if action == "generate":
            _generate_sentences(app)
        else:
            _check_translation(app)


def _generate_sentences(app: Application) -> None:
    header("Miu — Generate sentences")
    tense: Tense = "past" if ask("Choose tense (present/past): ").lower() == "past" else "present"
    count = ask("How many sentences? (default 5): ")
    batch = app.coach.generate_sentences(tense, int(count) if count.isdigit() else 5)
    app.record_ai_use()

    print("\n--- Generated sentences ---")
    if batch.used_builtin_examples:
        print("Limited built-in examples: no AI model evaluated this output.\n")
    for sentence in batch.sentences:
        print(sentence.en)
        if sentence.pt:
            print(f"PT: {sentence.pt}")
        if sentence.tip:
            print(f"Dica: {sentence.tip}")
        print()
    print("Tip: add good sentences with 'Add content' to practise them offline.")
    pause()


def _check_translation(app: Application) -> None:
    header("Miu — Check my translation")
    en = ask("EN sentence: ")
    if not en:
        return
    user_pt = ask("Your PT: ")
    expected = next(
        (s["pt"] for s in app.content["sentences"] if clean_text(s["en"]) == clean_text(en)),
        None,
    ) or ask("Expected PT (optional, press ENTER to skip): ")

    feedback = app.coach.check_translation(en, user_pt, expected or None)
    if feedback.verdict is Verdict.ACCEPTED_LOCALLY:
        pause("✅ Acceptable (local check).")
        return
    app.record_ai_use()
    if feedback.verdict is Verdict.UNAVAILABLE:
        pause("AI is not available right now. No evaluation was performed.")
        return
    print(f"\n--- AI feedback ---\n{feedback.text}\n-------------------")
    pause()


def run_conversation(app: Application) -> None:
    header("Conversation with Miu")
    print("Chat freely in English. Miu will correct your mistakes gently.")
    print("Type '0' or 'bye' to end the conversation.\n")
    if not app.ollama.is_available():
        pause(OLLAMA_HELP)
        return

    session = app.new_conversation()
    greeting = session.start()
    if greeting is None:
        pause("Could not get a reply. Is a model pulled in Ollama? (ollama pull llama3.2)")
        return
    print(f"Miu: {greeting}\n")

    while (message := ask("You: ")).lower() not in {"0", "exit", "quit", "bye"}:
        if not message:
            continue
        print("Miu is thinking...", end="\r")
        reply = session.send(message)
        print(f"Miu: {reply}\n" if reply else "Miu: (no response — is Ollama still running?)\n")

    if session.turns == 0:
        print("\nNo messages sent.")
        return

    print(f"\nConversation ended — {session.turns} message(s) sent.")
    print("\nAnalyzing your grammar mistakes...\n")
    review = session.review()
    if review:
        print(f"{'=' * 40}\n Grammar Review\n{'=' * 40}\n{review}\n{'=' * 40}")
    else:
        print("(Could not generate a grammar review.)")

    if ask("\nSave this conversation + review? (y/n): ").lower() == "y":
        path = session.save(app.settings.paths.conversations_dir, review)
        print(f"Saved to: {path}")

    register_conversation(app.progress, session.turns)
    app.save_progress()
    pause()


def run_free_writing(app: Application) -> None:
    while True:
        header("Free Writing")
        print("Write 2-4 sentences in English. The AI explains mistakes in Portuguese.")
        print("Type '0' to go back.\n")
        for index, topic in enumerate(TOPICS, start=1):
            print(f"  {index:2}) {topic}")
        choice = ask("\nTopic number (or write your own): ")
        if choice == "0":
            return
        topic = resolve_topic(choice)

        header(f"Free Writing — {topic}")
        print("Press ENTER on an empty line when done.\n")
        lines: list[str] = []
        while len(lines) < 4 and (line := ask("")):
            lines.append(line)
        if not lines:
            continue

        print("\n[Sending to AI...]\n")
        result = app.writing.review(" ".join(lines), topic)
        app.record_ai_use()
        print("--- AI Feedback ---")
        print(result.text if result else "AI is not available right now (is Ollama running?).")
        print("-------------------")
        pause()


def run_grounded_tutor(app: Application) -> None:
    header("Miu — Grounded Study Tutor")
    print("Ask about the local notes, examples or your recurring mistakes.")
    print("The first question may take longer while the local index is created.")
    print("Type :reindex to rebuild the index or 0 to return.\n")
    try:
        service = app.rag()
    except RagError as exc:
        pause(f"RAG unavailable: {exc}")
        return

    force_reindex = False
    while (question := ask("You: ")) != "0":
        if question == ":reindex":
            force_reindex = True
            print("The index will be rebuilt with the next question.")
            continue
        if not question:
            continue

        print("Searching the local study material...")
        try:
            result = service.ask(question, app.content, app.progress, force_reindex=force_reindex)
        except RagError as exc:
            print(f"\nUnable to answer: {exc}\n")
            continue
        finally:
            force_reindex = False

        app.record_ai_use()
        print(f"\nMiu:\n{result.text}\n\nSources:")
        for source in result.sources:
            print(
                f"[{source.reference}] {source.source} — {source.title} (score {source.score:.3f})"
            )
        print(f"\nLocal index: {result.indexed_chunks} chunks\n")
