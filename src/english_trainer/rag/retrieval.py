"""Retrieval strategies over the indexed study material."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from english_trainer.rag.embeddings import Embedder
from english_trainer.rag.models import KnowledgeChunk, SearchHit

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


def bm25_tokens(value: str) -> list[str]:
    """Split Portuguese and English text into case and accent insensitive terms."""
    normalized = unicodedata.normalize("NFKD", value.lower())
    plain = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", plain)


class BM25Index:
    """In-memory BM25 over exactly the chunks supplied to the dense index."""

    def __init__(self, chunks: Sequence[KnowledgeChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.terms = [Counter(bm25_tokens(chunk.text)) for chunk in chunks]
        self.avg_length = (
            sum(sum(terms.values()) for terms in self.terms) / len(chunks) if chunks else 0
        )
        self.document_frequency = Counter(term for terms in self.terms for term in terms)

    def search(self, query: str, *, limit: int) -> list[tuple[KnowledgeChunk, float]]:
        if not self.chunks or limit <= 0:
            return []
        query_terms = set(bm25_tokens(query))
        scores: list[tuple[KnowledgeChunk, float]] = []
        for chunk, terms in zip(self.chunks, self.terms, strict=True):
            length = sum(terms.values())
            score = 0.0
            for term in query_terms & terms.keys():
                frequency = terms[term]
                df = self.document_frequency[term]
                idf = math.log1p((len(self.chunks) - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length))
                )
            if score > 0:
                scores.append((chunk, score))
        return sorted(scores, key=lambda item: (-item[1], item[0].id))[:limit]


def reciprocal_rank_fusion(
    dense: Sequence[SearchHit], sparse: Sequence[tuple[KnowledgeChunk, float]], *, k: int = 60
) -> list[SearchHit]:
    """Merge rankings by chunk ID; retain cosine and BM25 scores for diagnostics."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    merged: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense, start=1):
        merged[hit.chunk.id] = SearchHit(hit.chunk, 0.0, dense_score=hit.dense_score)
        scores[hit.chunk.id] = 1 / (k + rank)
    for rank, (chunk, bm25_score) in enumerate(sparse, start=1):
        previous = merged.get(chunk.id)
        merged[chunk.id] = SearchHit(
            chunk,
            0.0,
            dense_score=previous.dense_score if previous else None,
            lexical_score=bm25_score,
        )
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank)
    return sorted(
        (hit.with_score(scores[chunk_id]) for chunk_id, hit in merged.items()),
        key=lambda hit: (-hit.score, hit.chunk.id),
    )


class HybridRetriever:
    """Fuse full-corpus cosine and BM25 rankings using reciprocal ranks."""

    def __init__(self, embedder: Embedder, store: VectorSearch, *, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k
        self.bm25 = BM25Index([])

    def set_chunks(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self.bm25 = BM25Index(chunks)

    def retrieve(self, query: str, *, limit: int) -> list[SearchHit]:
        if limit <= 0 or not self.bm25.chunks:
            return []
        vector = self.embedder.embed([query])[0]
        dense = [
            SearchHit(hit.chunk, hit.score, dense_score=hit.score)
            for hit in self.store.search(vector, limit=len(self.bm25.chunks))
        ]
        sparse = self.bm25.search(query, limit=len(self.bm25.chunks))
        return reciprocal_rank_fusion(dense, sparse, k=self.rrf_k)[:limit]
