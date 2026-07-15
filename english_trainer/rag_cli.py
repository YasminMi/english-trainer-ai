from __future__ import annotations

from typing import Any

from .progress import register_ai_use
from .rag import RagError, RagService
from .utils import clear_screen, pause, read_choice


def run_rag_tutor(content: dict[str, Any], progress: dict[str, Any]) -> None:
    clear_screen()
    print("=== Miu — Grounded Study Tutor ===")
    print("Ask about the local notes, examples or your recurring mistakes.")
    print("The first question may take longer while the local index is created.")
    print("Type :reindex to rebuild the index or 0 to return.\n")

    try:
        service = RagService()
    except RagError as exc:
        print(f"RAG unavailable: {exc}")
        pause()
        return
    except Exception as exc:
        print(f"RAG unavailable: {exc}")
        pause()
        return

    force_reindex = False
    while True:
        question = read_choice("You: ").strip()
        if question == "0":
            service.close()
            return
        if question == ":reindex":
            force_reindex = True
            print("The index will be rebuilt with the next question.")
            continue
        if not question:
            continue

        print("Searching the local study material...")
        try:
            result = service.ask(
                question,
                content,
                progress,
                force_reindex=force_reindex,
            )
        except RagError as exc:
            print(f"\nUnable to answer: {exc}\n")
            force_reindex = False
            continue

        force_reindex = False
        register_ai_use(progress)
        print(f"\nMiu:\n{result.text}\n")
        print("Sources:")
        for source in result.sources:
            print(
                f"[{source.reference}] {source.source} — {source.title} "
                f"(score {source.score:.3f})"
            )
        print(f"\nLocal index: {result.indexed_chunks} chunks\n")

