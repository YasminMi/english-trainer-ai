from __future__ import annotations

from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.rag.relevance import relevant_hits


def _hit(score: float, dense_score: float | None) -> SearchHit:
    chunk = KnowledgeChunk("id", "text", "source", "title", "topic", "A1", "rule")
    return SearchHit(chunk, score, dense_score=dense_score)


def test_gate_uses_raw_cosine_instead_of_boosted_or_fused_score() -> None:
    weak_boosted = _hit(0.75, 0.379)
    strong_fused = _hit(0.03, 0.38)
    missing_cosine = _hit(0.9, None)

    assert relevant_hits([weak_boosted, strong_fused, missing_cosine], min_score=0.38) == [
        strong_fused
    ]


def test_zero_threshold_preserves_hashing_eval_path() -> None:
    assert relevant_hits([_hit(0.0, 0.0)], min_score=0.0)
