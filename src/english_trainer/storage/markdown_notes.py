"""Parse study notes written in Markdown into content items.

A note may contain ``## Sentences``, ``## Flashcards`` and ``## Dictionary``
sections; other sections (grammar explanations) are only used by the RAG index.

.. code-block:: markdown

    ## Sentences
    - en: She is at home now.
      pt: Ela está em casa agora.
      tags: to_be, place

    ## Flashcards
    - kitchen = cozinha
"""

from __future__ import annotations

import re
from pathlib import Path

from english_trainer.domain.matching import clean_text
from english_trainer.domain.types import Content, Flashcard, Sentence, empty_content


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(heading, body)`` pairs for every ``##`` section."""
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", clean_text(value).replace("/", "_"))
    return cleaned.strip("_") or "note"


def _parse_pairs(body: str) -> list[tuple[str, str]]:
    """Parse ``- english = portuguese`` bullet lines."""
    pairs = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("- ") and "=" in line:
            en, pt = line[2:].split("=", 1)
            pairs.append((en.strip(), pt.strip()))
    return pairs


def _parse_sentences(body: str, note_slug: str, source: str) -> list[Sentence]:
    """Parse ``- en:`` blocks; ``pt:`` and ``tags:`` may appear in any order."""
    blocks: list[dict[str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("- en:"):
            blocks.append({"en": line.split(":", 1)[1].strip()})
        elif blocks and line.startswith(("pt:", "tags:")):
            key, value = line.split(":", 1)
            blocks[-1][key] = value.strip()

    sentences: list[Sentence] = []
    for block in blocks:
        if not block.get("en") or not block.get("pt"):
            continue
        tags = [tag.strip() for tag in block.get("tags", "").split(",") if tag.strip()]
        sentences.append(
            {
                "id": f"md_{note_slug}_s_{len(sentences) + 1:03d}",
                "en": block["en"],
                "pt": block["pt"],
                "tags": ["markdown", note_slug, *tags],
                "source_note": source,
            }
        )
    return sentences


def parse_note(path: Path, *, notes_dir: Path, home: Path) -> Content:
    content = empty_content()
    source = path.relative_to(home).as_posix()
    note_slug = slugify(path.relative_to(notes_dir).with_suffix("").as_posix())

    for heading, body in split_sections(path.read_text(encoding="utf-8")):
        section = clean_text(heading)
        if section == "sentences":
            content["sentences"].extend(_parse_sentences(body, note_slug, source))
        elif section == "flashcards":
            cards: list[Flashcard] = [
                {
                    "id": f"md_{note_slug}_f_{index:03d}",
                    "en": en,
                    "pt": pt,
                    "category": note_slug,
                    "source_note": source,
                }
                for index, (en, pt) in enumerate(_parse_pairs(body), start=1)
            ]
            content["flashcards"].extend(cards)
        elif section == "dictionary":
            content["dictionary"].update(dict(_parse_pairs(body)))
    return content
