from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    text: str
    source: str
    title: str
    topic: str
    level: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

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
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class RagSource:
    reference: str
    source: str
    title: str
    topic: str
    score: float
