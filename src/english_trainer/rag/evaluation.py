"""Offline retrieval evaluation against a labelled question set.

Each case lists the chunk labels (see :attr:`KnowledgeChunk.label`) that
answer it. Unanswerable cases (``relevant: []``) measure whether the
relevance gate correctly refuses off-topic questions.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from english_trainer.rag.models import SearchHit
from english_trainer.rag.retrieval import Retriever

Gate = Callable[[Sequence[SearchHit]], bool]


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    relevant: frozenset[str]
    tags: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    ranked: tuple[str, ...]
    accepted: bool
    latency_ms: float

    @property
    def first_relevant_rank(self) -> int | None:
        for rank, label in enumerate(self.ranked, start=1):
            if label in self.case.relevant:
                return rank
        return None

    def recall_at(self, k: int) -> float:
        found = self.case.relevant.intersection(self.ranked[:k])
        return len(found) / len(self.case.relevant)

    def hit_at(self, k: int) -> bool:
        rank = self.first_relevant_rank
        return rank is not None and rank <= k


@dataclass
class EvalReport:
    k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def answerable(self) -> list[CaseResult]:
        return [result for result in self.results if result.case.answerable]

    @property
    def unanswerable(self) -> list[CaseResult]:
        return [result for result in self.results if not result.case.answerable]

    def hit_rate(self, k: int | None = None) -> float:
        cases = self.answerable
        return _mean([float(result.hit_at(k or self.k)) for result in cases])

    def recall(self, k: int | None = None) -> float:
        return _mean([result.recall_at(k or self.k) for result in self.answerable])

    @property
    def mrr(self) -> float:
        ranks = [result.first_relevant_rank for result in self.answerable]
        return _mean([1.0 / rank if rank else 0.0 for rank in ranks])

    @property
    def false_accept_rate(self) -> float:
        """Share of off-topic questions that still passed the relevance gate."""
        return _mean([float(result.accepted) for result in self.unanswerable])

    @property
    def false_reject_rate(self) -> float:
        """Share of answerable questions the relevance gate refused."""
        return _mean([float(not result.accepted) for result in self.answerable])

    @property
    def p50_latency_ms(self) -> float:
        return statistics.median(r.latency_ms for r in self.results) if self.results else 0.0

    def misses(self) -> list[CaseResult]:
        return [result for result in self.answerable if not result.hit_at(self.k)]

    def summary(self) -> dict[str, float]:
        return {
            "cases": len(self.results),
            "hit@1": round(self.hit_rate(1), 3),
            f"hit@{self.k}": round(self.hit_rate(), 3),
            f"recall@{self.k}": round(self.recall(), 3),
            "mrr": round(self.mrr, 3),
            "false_accept_rate": round(self.false_accept_rate, 3),
            "false_reject_rate": round(self.false_reject_rate, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
        }

    def cases(self) -> list[dict[str, object]]:
        """Per-case details, for diffing two runs."""
        return [
            {
                "id": result.case.id,
                "rank": result.first_relevant_rank,
                "accepted": result.accepted,
                "top": list(result.ranked[:3]),
            }
            for result in self.results
        ]

    def format(self) -> str:
        lines = ["Retrieval evaluation", "=" * 40]
        lines += [f"{name:<20} {value}" for name, value in self.summary().items()]
        if misses := self.misses():
            lines += ["", f"Misses at k={self.k}:"]
            lines += [
                f"- [{m.case.id}] {m.case.question}\n    got: {', '.join(m.ranked[:3]) or '-'}"
                for m in misses
            ]
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_cases(path: Path) -> list[EvalCase]:
    """Load a JSON Lines dataset; blank lines and ``//`` comments are ignored."""
    cases = []
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
            cases.append(
                EvalCase(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    relevant=frozenset(raw.get("relevant", [])),
                    tags=tuple(raw.get("tags", [])),
                )
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path.name}:{number}: invalid case ({exc})") from exc
    return cases


def evaluate(
    retriever: Retriever,
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    gate: Gate | None = None,
) -> EvalReport:
    """Run every case through ``retriever``; ``gate`` decides if an answer would be given."""
    report = EvalReport(k=k)
    for case in cases:
        started = time.perf_counter()
        hits = retriever.retrieve(case.question, limit=k)
        latency = (time.perf_counter() - started) * 1000
        report.results.append(
            CaseResult(
                case=case,
                ranked=tuple(dict.fromkeys(hit.chunk.label for hit in hits)),
                accepted=gate(hits) if gate else bool(hits),
                latency_ms=latency,
            )
        )
    return report
