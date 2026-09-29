"""Value objects shared by indexing, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    """A retrievable unit of study material.

    ``id`` is a content hash, so an edited note produces a new chunk id and
    unchanged material keeps its id across re-indexing.
    """

    id: str
    text: str
    source: str
    title: str
    topic: str
    level: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Stable relevance label used by the evaluation dataset.

        Markdown chunks are labelled ``<path>#<section>``; JSON-derived chunks
        use their source anchor (``data/content.json#to_be_exercises``).
        """
        section = self.metadata.get("section")
        return f"{self.source}#{section}" if section else self.source

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "topic": self.topic,
            "level": self.level,
            "kind": self.kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> KnowledgeChunk:
        return cls(
            id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            source=str(payload["source"]),
            title=str(payload["title"]),
            topic=str(payload.get("topic", "general")),
            level=str(payload.get("level", "beginner")),
            kind=str(payload.get("kind", "content")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchHit:
    """A retrieved chunk with its final ranking score and per-signal diagnostics."""

    chunk: KnowledgeChunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None

    def with_score(self, score: float) -> SearchHit:
        return replace(self, score=score)


@dataclass(frozen=True)
class RagSource:
    reference: str
    source: str
    title: str
    topic: str
    score: float
