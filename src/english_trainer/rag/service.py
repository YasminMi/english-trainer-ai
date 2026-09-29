"""Grounded tutor: retrieve local study material, then answer with citations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from english_trainer.llm.base import ChatModel, GenerationOptions, LLMUnavailableError
from english_trainer.rag.embeddings import EmbeddingError
from english_trainer.rag.index import KnowledgeIndex
from english_trainer.rag.models import KnowledgeChunk, RagSource, SearchHit
from english_trainer.rag.prompts import SYSTEM_PROMPT, build_prompt, learner_context
from english_trainer.rag.retrieval import HybridRetriever, Retriever
from english_trainer.rag.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

ChunkBuilder = Callable[[Mapping[str, Any]], list[KnowledgeChunk]]


class RagError(RuntimeError):
    """The grounded tutor could not produce an answer; the message is user-facing."""


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[RagSource]
    indexed_chunks: int
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RagConfig:
    model: str = "gemma2:2b"
    top_k: int = 2
    min_score: float = 0.20
    max_tokens: int = 160
    timeout: float = 180.0


class RagService:
    def __init__(
        self,
        *,
        index: KnowledgeIndex,
        retriever: Retriever,
        generator: ChatModel,
        chunk_builder: ChunkBuilder,
        config: RagConfig | None = None,
    ) -> None:
        self.index = index
        self.retriever = retriever
        self.generator = generator
        self.chunk_builder = chunk_builder
        self.config = config or RagConfig()

    def prepare(self, content: Mapping[str, Any], *, force: bool = False) -> int:
        chunks = self.chunk_builder(content)
        if not chunks:
            raise RagError("No study content was available for indexing.")
        try:
            count = self.index.sync(chunks, force=force)
            if isinstance(self.retriever, HybridRetriever):
                self.retriever.set_chunks(chunks)
            return count
        except (EmbeddingError, VectorStoreError) as exc:
            raise RagError(str(exc)) from exc

    def retrieve(self, question: str) -> list[SearchHit]:
        """Relevant hits above the configured threshold, best first."""
        try:
            hits = self.retriever.retrieve(question, limit=self.config.top_k * 2)
        except (EmbeddingError, VectorStoreError, IndexError) as exc:
            raise RagError(str(exc)) from exc
        return [hit for hit in hits if self._is_relevant(hit)][: self.config.top_k]

    def ask(
        self,
        question: str,
        content: Mapping[str, Any],
        progress: Mapping[str, Any],
        *,
        force_reindex: bool = False,
    ) -> RagAnswer:
        question = question.strip()
        if not question:
            raise RagError("Write a question before starting the search.")

        timings: dict[str, int] = {}
        with _timer(timings, "index"):
            indexed_chunks = self.prepare(content, force=force_reindex)
        with _timer(timings, "retrieve"):
            hits = self.retrieve(question)
        if not hits:
            logger.info("rag.no_relevant_material", extra={"timings_ms": timings})
            raise RagError("No sufficiently relevant study material was found for this question.")

        prompt = build_prompt(question, hits, learner_context(content, progress))
        with _timer(timings, "generate"):
            answer = self._generate(prompt)

        logger.info(
            "rag.answered",
            extra={
                "hits": [hit.chunk.label for hit in hits],
                "scores": [round(hit.score, 3) for hit in hits],
                "timings_ms": timings,
            },
        )
        sources = [
            RagSource(
                reference=f"S{index}",
                source=hit.chunk.source,
                title=hit.chunk.title,
                topic=hit.chunk.topic,
                score=hit.score,
            )
            for index, hit in enumerate(hits, start=1)
        ]
        return RagAnswer(answer, sources, indexed_chunks, timings)

    def close(self) -> None:
        close = getattr(self.index.store, "close", None)
        if callable(close):
            close()

    def _is_relevant(self, hit: SearchHit) -> bool:
        return hit.dense_score is not None and hit.dense_score >= self.config.min_score

    def _generate(self, prompt: str) -> str:
        try:
            return self.generator.chat(
                [{"role": "user", "content": prompt}],
                system=SYSTEM_PROMPT,
                model=self.config.model,
                options=GenerationOptions(
                    temperature=0.0,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                ),
            )
        except LLMUnavailableError as exc:
            raise RagError(
                "Ollama could not generate an answer from the retrieved material."
            ) from exc


class _timer:
    def __init__(self, sink: dict[str, int], name: str) -> None:
        self.sink = sink
        self.name = name
        self.started = 0.0

    def __enter__(self) -> None:
        self.started = time.perf_counter()

    def __exit__(self, *_exc: object) -> None:
        self.sink[self.name] = round((time.perf_counter() - self.started) * 1000)
