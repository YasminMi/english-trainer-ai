from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from english_trainer.rag.models import KnowledgeChunk, SearchHit


class VectorStoreError(RuntimeError):
    pass


class QdrantKnowledgeStore:
    """Embedded Qdrant collection; ``path=None`` keeps it in memory (evaluation, tests)."""

    def __init__(
        self, path: Path | None, collection_name: str = "english_trainer_knowledge"
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is not installed. Install the project dependencies first."
            ) from exc

        self.collection_name = collection_name
        if path is None:
            self._client = QdrantClient(location=":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))

    def count(self) -> int:
        if not self._client.collection_exists(self.collection_name):
            return 0
        return int(self._client.count(self.collection_name, exact=True).count)

    def replace(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise VectorStoreError("Chunks and vectors must have the same non-zero length.")

        dimensions = len(vectors[0])
        if not dimensions or any(len(vector) != dimensions for vector in vectors):
            raise VectorStoreError("All embedding vectors must have the same dimensions.")

        from qdrant_client.models import Distance, PointStruct, VectorParams

        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector=list(vector),
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upload_points(self.collection_name, points=points, wait=True)

    def search(self, vector: Sequence[float], limit: int = 5) -> list[SearchHit]:
        if not self._client.collection_exists(self.collection_name):
            return []
        result = self._client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=max(1, limit),
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in result.points:
            payload = dict(point.payload or {})
            if payload:
                hits.append(
                    SearchHit(
                        chunk=KnowledgeChunk.from_payload(payload),
                        score=float(point.score),
                    )
                )
        return hits

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
