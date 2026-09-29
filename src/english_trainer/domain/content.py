"""Queries and edits over study content that do not depend on storage or UI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

from english_trainer.domain.matching import clean_text
from english_trainer.domain.types import Content, Flashcard, Sentence

T = TypeVar("T", bound=Mapping[str, Any])


class DuplicateEntryError(ValueError):
    """Raised when a user tries to add content that already exists."""


def merge_content(base: Content, extra: Mapping[str, Any]) -> None:
    """Append ``extra`` into ``base`` in place, section by section."""
    base["metadata"].update(extra.get("metadata", {}))
    base["sentences"].extend(extra.get("sentences", []))
    base["flashcards"].extend(extra.get("flashcards", []))
    base["to_be_exercises"].extend(extra.get("to_be_exercises", []))
    base["dictionary"].update(extra.get("dictionary", {}))


def find_in_dictionary(content: Content, word: str) -> str | None:
    key = " ".join(clean_text(word).replace("/", " ").split())
    return content.get("dictionary", {}).get(key)


def available_sources(items: Iterable[Mapping[str, Any]]) -> list[str]:
    """Distinct Markdown notes that contributed to ``items``, sorted for stable menus."""
    return sorted({str(item["source_note"]) for item in items if item.get("source_note")})


def note_label(source_note: str) -> str:
    """Human-friendly topic label: ``notes/english/to_be_a1.md`` -> ``To Be A1``."""
    return source_note.rsplit("/", 1)[-1].removesuffix(".md").replace("_", " ").title()


def filter_items(
    items: Sequence[T],
    *,
    category: str | None = None,
    source_note: str | None = None,
) -> list[T]:
    return [
        item
        for item in items
        if (category is None or item.get("category") == category)
        and (source_note is None or item.get("source_note") == source_note)
    ]


def add_sentence(
    content: Content,
    user_content: Content,
    *,
    en: str,
    pt: str,
    tags: Sequence[str] = (),
) -> Sentence:
    """Add a user sentence to both the live content and the user's private file."""
    en = en.strip()
    if not en:
        raise ValueError("An English sentence is required.")
    if clean_text(en) in {clean_text(item.get("en", "")) for item in content["sentences"]}:
        raise DuplicateEntryError("This sentence already exists.")

    entry: Sentence = {
        "id": f"user_s_{len(user_content['sentences']) + 1:06d}",
        "en": en,
        "pt": pt.strip(),
    }
    cleaned_tags = [tag.strip() for tag in tags if tag.strip()]
    if cleaned_tags:
        entry["tags"] = cleaned_tags
    content["sentences"].append(entry)
    user_content["sentences"].append(entry)
    return entry


def add_flashcard(
    content: Content,
    user_content: Content,
    *,
    en: str,
    pt: str,
    category: str | None = None,
) -> Flashcard:
    en = en.strip()
    if not en:
        raise ValueError("An English word is required.")
    if clean_text(en) in {clean_text(item.get("en", "")) for item in content["flashcards"]}:
        raise DuplicateEntryError("This word already exists in flashcards.")

    entry: Flashcard = {"id": f"user_w_{en.lower().replace(' ', '_')}", "en": en, "pt": pt.strip()}
    if category and category.strip():
        entry["category"] = category.strip()
    content["flashcards"].append(entry)
    user_content["flashcards"].append(entry)
    return entry


def add_dictionary_entry(
    content: Content, user_content: Content, *, word: str, meaning: str
) -> None:
    word, meaning = word.strip(), meaning.strip()
    if not word or not meaning:
        raise ValueError("Both the word and its meaning are required.")
    content["dictionary"][word] = meaning
    user_content["dictionary"][word] = meaning
