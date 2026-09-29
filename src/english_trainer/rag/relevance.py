"""Cheap, deterministic relevance gate over raw cosine scores."""

from __future__ import annotations

from collections.abc import Sequence

from english_trainer.rag.models import SearchHit


def relevant_hits(hits: Sequence[SearchHit], *, min_score: float) -> list[SearchHit]:
    """Keep ranked candidates with enough dense support, ignoring ranking boosts."""
    return [hit for hit in hits if hit.dense_score is not None and hit.dense_score >= min_score]
