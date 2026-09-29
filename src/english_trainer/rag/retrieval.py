"""Retrieval strategies over the indexed study material."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Protocol

from english_trainer.rag.embeddings import Embedder
from english_trainer.rag.models import SearchHit

logger = logging.getLogger(__name__)

LEXICAL_BOOST = 0.35

# Kept deliberately small: PT and EN function words that carry no topic.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "as", "da", "de", "do", "e", "em", "explain", "explique",
        "frase", "in", "is", "na", "no", "o", "os", "por", "que", "the", "this",
        "um", "uma", "why",
    }
)  # fmt: skip


class VectorSearch(Protocol):
    def search(self, vector: Sequence[float], limit: int = 5) -> list[SearchHit]: ...


class Retriever(Protocol):
    """Anything that ranks chunks for a question; used by the service and the evaluator."""

    def retrieve(self, query: str, *, limit: int) -> list[SearchHit]: ...


def search_tokens(value: str) -> set[str]:
    normalized = value.lower().replace("doesn't", "does not").replace("don't", "do not")
    return {
        token
        for token in re.findall(r"[a-zà-ÿ]+", normalized)
        if len(token) > 1 and token not in _STOPWORDS
    }


def lexical_overlap_rerank(query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Boost hits that share query terms; the original dense score stays in ``dense_score``."""
    query_tokens = search_tokens(query)
    if not query_tokens:
        return list(hits)

    def boosted(hit: SearchHit) -> SearchHit:
        chunk_tokens = search_tokens(f"{hit.chunk.title} {hit.chunk.topic} {hit.chunk.text}")
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        return hit.with_score(hit.score + LEXICAL_BOOST * overlap)

    return sorted((boosted(hit) for hit in hits), key=lambda hit: hit.score, reverse=True)


class DenseRetriever:
    """Embedding similarity search followed by a lexical-overlap re-ranking."""

    def __init__(self, embedder: Embedder, store: VectorSearch, *, oversample: int = 2) -> None:
        self.embedder = embedder
        self.store = store
        self.oversample = max(1, oversample)

    def retrieve(self, query: str, *, limit: int) -> list[SearchHit]:
        vector = self.embedder.embed([query])[0]
        hits = [
            SearchHit(chunk=hit.chunk, score=hit.score, dense_score=hit.score)
            for hit in self.store.search(vector, limit=limit * self.oversample)
        ]
        return lexical_overlap_rerank(query, hits)[:limit]
