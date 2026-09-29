"""Prompt construction for grounded answers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from english_trainer.rag.models import SearchHit

SYSTEM_PROMPT = (
    "You are Miu, a personal AI English tutor using retrieval-augmented generation. "
    "Use only the supplied sources. Follow explicit grammar rules exactly, "
    "cite them, and never invent or generalize a rule."
)

MAX_CHUNK_CHARS = 900
MAX_DIFFICULT_ITEMS = 5

_PORTUGUESE_MARKERS = re.compile(
    r"\b(por que|porque|explique|como|qual|quando|erro|errado|frase|está|estão|não|você|é)\b"
)


def response_language(question: str) -> str:
    """Answer in Portuguese when the learner asks in Portuguese."""
    return "Brazilian Portuguese" if _PORTUGUESE_MARKERS.search(question.lower()) else "English"


def learner_context(content: Mapping[str, Any], progress: Mapping[str, Any]) -> str:
    """Summarize the learner's most-missed items so the answer can address them."""
    labels: dict[str, tuple[str, str]] = {}
    for kind, section, field in (
        ("sentence", "sentences", "en"),
        ("flashcard", "flashcards", "en"),
        ("to be", "to_be_exercises", "question"),
    ):
        for item in content.get(section, []):
            labels[str(item.get("id", ""))] = (kind, str(item.get(field, "")))

    difficult: list[tuple[float, int, str]] = []
    for section in ("sentences", "flashcards", "to_be"):
        for item_id, stats in progress.get(section, {}).items():
            seen = int(stats.get("seen", 0))
            errors = max(0, seen - int(stats.get("correct", 0)))
            if seen and errors:
                kind, label = labels.get(str(item_id), (section, str(item_id)))
                difficult.append(
                    (errors / seen, errors, f"{kind}: {label} ({errors}/{seen} errors)")
                )

    if not difficult:
        return "No recurring difficulty has been recorded yet."
    difficult.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines = "\n".join(f"- {item[2]}" for item in difficult[:MAX_DIFFICULT_ITEMS])
    return f"Recent difficult items:\n{lines}"


def format_sources(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(
        f"[S{index}] {hit.chunk.title}\n"
        f"Source: {hit.chunk.source}\n"
        f"Topic: {hit.chunk.topic}; Level: {hit.chunk.level}\n"
        f"{hit.chunk.text[:MAX_CHUNK_CHARS]}"
        for index, hit in enumerate(hits, start=1)
    )


def build_prompt(question: str, hits: Sequence[SearchHit], learner: str) -> str:
    return (
        f"Write the entire answer in {response_language(question)}.\n"
        "Use only the retrieved study material below for grammar rules and examples.\n"
        "Cite supporting sources inline as [S1], [S2], and so on.\n"
        "If the material is insufficient, say what is missing instead of inventing a rule.\n"
        "Keep the answer under 100 words and cite references without copying their metadata.\n"
        "When correcting a sentence, give the correction, the exact rule, "
        "and one related example.\n"
        "Do not describe an auxiliary as a conjugated form of the main verb.\n"
        "Finish with one English practice question about the same rule, without its answer.\n\n"
        f"Student context:\n{learner}\n\n"
        f"Retrieved study material:\n{format_sources(hits)}\n\n"
        f"Student question:\n{question}"
    )
