"""Keep the vector store in sync with the study material."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from english_trainer.rag.chunking import corpus_fingerprint
from english_trainer.rag.embeddings import Embedder
from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.storage.json_store import read_json_object, write_json_atomic

logger = logging.getLogger(__name__)


class KnowledgeStore(Protocol):
    def count(self) -> int: ...

    def replace(
        self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def search(self, vector: Sequence[float], limit: int = 5) -> list[SearchHit]: ...


class KnowledgeIndex:
    """Rebuilds the store only when the corpus or the embedding model changed.

    A manifest next to the store records the corpus fingerprint, so opening
    the tutor with unchanged notes costs one file read instead of re-embedding.
    """

    def __init__(
        self, embedder: Embedder, store: KnowledgeStore, directory: Path | None = None
    ) -> None:
        """``directory=None`` skips the manifest (in-memory stores are always rebuilt)."""
        self.embedder = embedder
        self.store = store
        self.manifest_path = directory / "manifest.json" if directory else None

    def sync(self, chunks: Sequence[KnowledgeChunk], *, force: bool = False) -> int:
        """Ensure the store holds exactly ``chunks``; return the indexed chunk count."""
        fingerprint = corpus_fingerprint(chunks, self.embedder.model)
        manifest = self._manifest()
        if (
            not force
            and manifest.get("fingerprint") == fingerprint
            and self.store.count() == len(chunks)
        ):
            return len(chunks)

        started = time.perf_counter()
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.store.replace(chunks, vectors)
        if self.manifest_path is not None:
            write_json_atomic(
                self.manifest_path,
                {
                    "fingerprint": fingerprint,
                    "embedding_model": self.embedder.model,
                    "chunk_count": len(chunks),
                },
            )
        logger.info(
            "rag.index_rebuilt",
            extra={
                "chunks": len(chunks),
                "model": self.embedder.model,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return len(chunks)

    def _manifest(self) -> dict[str, Any]:
        return read_json_object(self.manifest_path) if self.manifest_path else {}
