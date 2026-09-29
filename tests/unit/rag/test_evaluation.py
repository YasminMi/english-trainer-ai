from __future__ import annotations

from pathlib import Path

import pytest

from english_trainer.config import get_settings
from english_trainer.rag.embeddings import CachingEmbedder, HashingEmbedder
from english_trainer.rag.evaluation import EvalCase, evaluate, load_cases
from english_trainer.rag.models import KnowledgeChunk, SearchHit


def _hit(label: str, score: float = 0.5) -> SearchHit:
    source, _, section = label.partition("#")
    chunk = KnowledgeChunk(
        id=label,
        text=label,
        source=source,
        title=label,
        topic="t",
        level="A1",
        kind="note",
        metadata={"section": section} if section else {},
    )
    return SearchHit(chunk=chunk, score=score, dense_score=score)


class ScriptedRetriever:
    def __init__(self, results: dict[str, list[SearchHit]]) -> None:
        self.results = results

    def retrieve(self, query: str, *, limit: int) -> list[SearchHit]:
        return self.results.get(query, [])[:limit]


def test_metrics_follow_standard_definitions() -> None:
    cases = [
        EvalCase("a", "qa", frozenset({"n.md#A"})),
        EvalCase("b", "qb", frozenset({"n.md#B", "n.md#C"})),
        EvalCase("c", "qc", frozenset({"n.md#D"})),
        EvalCase("off", "qoff", frozenset()),
    ]
    retriever = ScriptedRetriever(
        {
            "qa": [_hit("n.md#A"), _hit("n.md#X")],  # rank 1
            "qb": [_hit("n.md#X"), _hit("n.md#C")],  # rank 2, recall 1/2
            "qc": [_hit("n.md#X")],  # miss
            "qoff": [_hit("n.md#X", score=0.1)],
        }
    )

    report = evaluate(retriever, cases, k=2, gate=lambda hits: hits[0].score >= 0.3)

    assert report.hit_rate(1) == pytest.approx(1 / 3)
    assert report.hit_rate() == pytest.approx(2 / 3)
    assert report.recall() == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.false_accept_rate == 0.0
    assert report.false_reject_rate == 0.0
    assert [miss.case.id for miss in report.misses()] == ["c"]
    assert "hit@2" in report.summary()
    assert "[c] qc" in report.format()


def test_duplicate_labels_count_once_in_the_ranking() -> None:
    case = EvalCase("a", "q", frozenset({"n.md#B"}))
    retriever = ScriptedRetriever({"q": [_hit("n.md#A"), _hit("n.md#A"), _hit("n.md#B")]})

    report = evaluate(retriever, [case], k=3)

    assert report.results[0].first_relevant_rank == 2


def test_load_cases_skips_comments_and_reports_bad_lines(tmp_path: Path) -> None:
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(
        '// comment\n\n{"id": "a", "question": "q", "relevant": ["x"], "tags": ["en"]}\n',
        encoding="utf-8",
    )
    assert load_cases(dataset) == [EvalCase("a", "q", frozenset({"x"}), ("en",))]

    dataset.write_text('{"question": "no id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"gold\.jsonl:1"):
        load_cases(dataset)


def test_gold_set_is_well_formed() -> None:
    cases = load_cases(get_settings().paths.home / "evals" / "rag_golden.jsonl")

    assert len({case.id for case in cases}) == len(cases)
    assert sum(case.answerable for case in cases) >= 30
    assert sum(not case.answerable for case in cases) >= 10


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimensions=64)

    first, second = embedder.embed(["She is happy", "She is happy"])

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert embedder.embed([""])[0] == [0.0] * 64


def test_caching_embedder_only_embeds_new_texts(tmp_path: Path) -> None:
    class CountingEmbedder:
        model = "counting/model:1"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def embed(self, texts):  # type: ignore[no-untyped-def]
            self.seen.extend(texts)
            return [[float(len(text))] for text in texts]

    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, tmp_path)

    assert cached.embed(["a", "bb"]) == [[1.0], [2.0]]
    assert cached.embed(["bb", "ccc", "ccc"]) == [[2.0], [3.0], [3.0]]
    assert inner.seen == ["a", "bb", "ccc"]
    assert (tmp_path / "counting_model_1.json").exists()
