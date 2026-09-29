from __future__ import annotations

import pytest

from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.rag.retrieval import (
    BM25Index,
    HybridRetriever,
    bm25_tokens,
    reciprocal_rank_fusion,
)


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id, text, "note.md", "Title", "grammar", "A1", "note")


def test_bm25_tokenizes_portuguese_and_english_and_ranks_repeated_terms() -> None:
    assert bm25_tokens("AÇÃO, café! Don't work: 123.") == [
        "acao",
        "cafe",
        "don",
        "t",
        "work",
        "123",
    ]
    index = BM25Index([_chunk("a", "Café forte café"), _chunk("b", "Cafe fraco")])

    results = index.search("CAFÉ", limit=2)

    assert [chunk.id for chunk, _ in results] == ["a", "b"]
    assert results[0][1] > results[1][1] > 0
    assert index.search("unknown", limit=2) == []


def test_rrf_promotes_item_present_in_both_rankings() -> None:
    a, b, c = (_chunk(letter, letter) for letter in "abc")
    dense = [SearchHit(a, 0.9, dense_score=0.9), SearchHit(b, 0.8, dense_score=0.8)]
    fused = reciprocal_rank_fusion(dense, [(c, 2.0), (b, 1.0)])

    assert [hit.chunk.id for hit in fused] == ["b", "a", "c"]
    assert fused[0].dense_score == 0.8
    assert fused[0].lexical_score == 1.0
    assert fused[2].dense_score is None
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([], [], k=0)


def test_hybrid_retriever_requires_chunks_to_be_prepared() -> None:
    class UnusedEmbedder:
        model = "unused"

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("embedding should not run before preparation")

    class UnusedStore:
        def search(self, vector: list[float], limit: int = 5) -> list[SearchHit]:
            raise AssertionError("search should not run before preparation")

    retriever = HybridRetriever(UnusedEmbedder(), UnusedStore())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=r"set_chunks\(\)"):
        retriever.retrieve("test query", limit=5)
