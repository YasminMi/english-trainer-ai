from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..data_access import BASE_DIR, NOTES_DIR
from .models import KnowledgeChunk

_GENERIC_TAGS = {
    "affirmative",
    "negative",
    "question",
    "present",
    "past",
    "future",
    "adjective",
    "noun",
    "word",
    "functional",
    "markdown",
}


def build_knowledge_chunks(
    content: dict[str, Any],
    notes_dir: Path = NOTES_DIR,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    chunks.extend(_chunks_from_markdown(notes_dir))
    chunks.extend(_chunks_from_explanations(content.get("to_be_exercises", [])))
    chunks.extend(_chunks_from_sentences(content.get("sentences", [])))
    chunks.extend(_chunks_from_flashcards(content.get("flashcards", [])))
    return _deduplicate(chunks)


def corpus_fingerprint(chunks: Iterable[KnowledgeChunk], embedding_model: str) -> str:
    digest = hashlib.sha256(embedding_model.encode("utf-8"))
    for chunk in sorted(chunks, key=lambda item: item.id):
        digest.update(chunk.id.encode("ascii"))
    return digest.hexdigest()


def _chunks_from_markdown(notes_dir: Path, max_chars: int = 1200) -> list[KnowledgeChunk]:
    if not notes_dir.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(notes_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        title, sections = _markdown_sections(text)
        source = _relative_source(path)
        topic = _slug(path.stem)
        level = _infer_level(f"{title} {path.as_posix()}")

        for section, body in sections:
            blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
            for index, grouped in enumerate(_group_text(blocks, max_chars), start=1):
                chunk_text = f"{title}\n{section}\n\n{grouped}"
                chunks.append(
                    _make_chunk(
                        text=chunk_text,
                        source=source,
                        title=f"{title} — {section}",
                        topic=topic,
                        level=level,
                        kind="note",
                        metadata={"section": section, "part": index},
                    )
                )
    return chunks


def _markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    title = "Study note"
    current = "Overview"
    lines: list[str] = []
    sections: list[tuple[str, str]] = []

    for raw_line in text.splitlines():
        if raw_line.startswith("# "):
            title = raw_line[2:].strip() or title
            continue
        if raw_line.startswith("## "):
            if lines:
                sections.append((current, "\n".join(lines).strip()))
            current = raw_line[3:].strip() or "Section"
            lines = []
            continue
        lines.append(raw_line)

    if lines:
        sections.append((current, "\n".join(lines).strip()))
    return title, [(heading, body) for heading, body in sections if body]


def _chunks_from_explanations(exercises: list[dict[str, Any]]) -> list[KnowledgeChunk]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exercise in exercises:
        explanation = str(exercise.get("explanation", "")).strip()
        if explanation:
            grouped[explanation].append(exercise)

    chunks: list[KnowledgeChunk] = []
    for explanation, examples in grouped.items():
        sample_lines = []
        for example in examples[:3]:
            question = str(example.get("question", "")).strip()
            correct = str(example.get("correct", "")).strip()
            pt = str(example.get("pt", "")).strip()
            sample_lines.append(f"Example: {question} → {correct} ({pt})")
        text = "To be grammar rule\n\n" + explanation
        if sample_lines:
            text += "\n\n" + "\n".join(sample_lines)
        chunks.append(
            _make_chunk(
                text=text,
                source="data/content.json#to_be_exercises",
                title="To be grammar rule",
                topic="to_be",
                level="A1",
                kind="rule",
                metadata={"example_count": len(examples)},
            )
        )
    return chunks


def _chunks_from_sentences(
    sentences: list[dict[str, Any]],
    group_size: int = 6,
    max_items_per_topic: int = 30,
) -> list[KnowledgeChunk]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sentence in sentences:
        if sentence.get("source_note"):
            continue
        tags = [str(tag) for tag in sentence.get("tags", [])]
        topic = _topic_from_tags(tags)
        level = _infer_level(" ".join(tags))
        source = (
            "data/user_content.json"
            if str(sentence.get("id", "")).startswith("user_")
            else "data/content.json#sentences"
        )
        grouped[(source, topic, level)].append(sentence)

    chunks: list[KnowledgeChunk] = []
    for (source, topic, level), items in sorted(grouped.items()):
        selected = _representative_sample(items, max_items_per_topic)
        for part, batch in enumerate(_batches(selected, group_size), start=1):
            lines = []
            for item in batch:
                tags = ", ".join(str(tag) for tag in item.get("tags", []))
                line = f"English: {item.get('en', '')}\nPortuguese: {item.get('pt', '')}"
                if tags:
                    line += f"\nTags: {tags}"
                lines.append(line)
            chunks.append(
                _make_chunk(
                    text=f"Examples about {topic}\n\n" + "\n\n".join(lines),
                    source=source,
                    title=f"Examples: {topic.replace('_', ' ')}",
                    topic=topic,
                    level=level,
                    kind="examples",
                    metadata={"part": part, "item_count": len(batch)},
                )
            )
    return chunks


def _chunks_from_flashcards(
    cards: list[dict[str, Any]], group_size: int = 10
) -> list[KnowledgeChunk]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        if card.get("source_note"):
            continue
        category = str(card.get("category") or "vocabulary")
        source = (
            "data/user_content.json"
            if str(card.get("id", "")).startswith("user_")
            else "data/content.json#flashcards"
        )
        grouped[(source, category)].append(card)

    chunks: list[KnowledgeChunk] = []
    for (source, category), items in sorted(grouped.items()):
        for part, batch in enumerate(_batches(items, group_size), start=1):
            pairs = [f"{item.get('en', '')} = {item.get('pt', '')}" for item in batch]
            chunks.append(
                _make_chunk(
                    text=f"Vocabulary: {category}\n\n" + "\n".join(pairs),
                    source=source,
                    title=f"Vocabulary: {category.replace('_', ' ')}",
                    topic=_slug(category),
                    level="beginner",
                    kind="vocabulary",
                    metadata={"part": part, "item_count": len(batch)},
                )
            )
    return chunks


def _make_chunk(
    *,
    text: str,
    source: str,
    title: str,
    topic: str,
    level: str,
    kind: str,
    metadata: dict[str, Any],
) -> KnowledgeChunk:
    normalized = re.sub(r"\s+", " ", text).strip()
    chunk_id = hashlib.sha256(f"{source}\n{title}\n{normalized}".encode()).hexdigest()
    return KnowledgeChunk(
        id=chunk_id,
        text=text.strip(),
        source=source,
        title=title,
        topic=topic,
        level=level,
        kind=kind,
        metadata=metadata,
    )


def _topic_from_tags(tags: list[str]) -> str:
    for tag in tags:
        normalized = _slug(tag)
        if normalized and normalized not in _GENERIC_TAGS:
            return normalized
    return "general_english"


def _infer_level(value: str) -> str:
    match = re.search(r"\b([ABC][12])\b", value.upper())
    return match.group(1) if match else "beginner"


def _relative_source(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "general"


def _group_text(blocks: list[str], max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        addition = len(block) + (2 if current else 0)
        if current and size + addition > max_chars:
            groups.append("\n\n".join(current))
            current = []
            size = 0
        current.append(block)
        size += addition
    if current:
        groups.append("\n\n".join(current))
    return groups


def _representative_sample(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def _batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _deduplicate(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    unique: dict[str, KnowledgeChunk] = {}
    for chunk in chunks:
        unique.setdefault(chunk.id, chunk)
    return sorted(unique.values(), key=lambda item: (item.source, item.topic, item.id))
