from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from english_trainer.rag.chunking import corpus_fingerprint
from english_trainer.rag.embeddings import EmbeddingError
from english_trainer.rag.models import KnowledgeChunk, RagSource, SearchHit
from english_trainer.rag.vector_store import QdrantKnowledgeStore, VectorStoreError


class RagError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class KnowledgeStore(Protocol):
    def count(self) -> int: ...

    def replace(
        self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def search(self, vector: Sequence[float], limit: int = 5) -> list[SearchHit]: ...


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: list[RagSource]
    indexed_chunks: int


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index_dir: Path,
        generator: Callable[..., str | None],
        chunk_builder: Callable[[Mapping[str, Any]], list[KnowledgeChunk]],
        store: KnowledgeStore | None = None,
        min_score: float = 0.20,
        generation_model: str = "gemma2:2b",
    ) -> None:
        self.embedder = embedder
        self.index_dir = index_dir
        if store is not None:
            self.store = store
        else:
            try:
                self.store = QdrantKnowledgeStore(index_dir / "qdrant")
            except (VectorStoreError, RuntimeError) as exc:
                raise RagError(str(exc)) from exc
        self.generator = generator
        self.chunk_builder = chunk_builder
        self.generation_model = generation_model
        self.min_score = min_score
        self.manifest_path = index_dir / "manifest.json"

    def prepare(self, content: Mapping[str, Any], force: bool = False) -> int:
        chunks = self.chunk_builder(content)
        if not chunks:
            raise RagError("No study content was available for indexing.")

        fingerprint = corpus_fingerprint(chunks, self.embedder.model)
        manifest = self._read_manifest()
        try:
            current_count = self.store.count()
        except (VectorStoreError, RuntimeError) as exc:
            raise RagError(str(exc)) from exc

        if (
            not force
            and manifest.get("fingerprint") == fingerprint
            and manifest.get("embedding_model") == self.embedder.model
            and current_count == len(chunks)
        ):
            return current_count

        try:
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.replace(chunks, vectors)
        except (EmbeddingError, VectorStoreError, RuntimeError) as exc:
            raise RagError(str(exc)) from exc

        self._write_manifest(
            {
                "fingerprint": fingerprint,
                "embedding_model": self.embedder.model,
                "chunk_count": len(chunks),
            }
        )
        return len(chunks)

    def ask(
        self,
        question: str,
        content: Mapping[str, Any],
        progress: Mapping[str, Any],
        *,
        force_reindex: bool = False,
        top_k: int = 2,
    ) -> RagAnswer:
        question = question.strip()
        if not question:
            raise RagError("Write a question before starting the search.")

        indexed_chunks = self.prepare(content, force=force_reindex)
        try:
            query_vector = self.embedder.embed([question])[0]
            candidates = self.store.search(query_vector, limit=max(top_k * 2, top_k))
            candidates = _rerank_hits(question, candidates)
        except (EmbeddingError, VectorStoreError, RuntimeError, IndexError) as exc:
            raise RagError(str(exc)) from exc

        hits = [hit for hit in candidates if hit.score >= self.min_score][:top_k]
        if not hits:
            raise RagError("No sufficiently relevant study material was found for this question.")

        prompt = self._build_prompt(question, hits, content, progress)
        answer = self.generator(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Miu, a personal AI English tutor using retrieval-augmented generation. "
                "Use only the supplied sources. Follow explicit grammar rules exactly, "
                "cite them, and never invent or generalize a rule."
            ),
            model=self.generation_model,
            max_new_tokens=160,
            timeout=180,
            temperature=0,
        )
        if not answer:
            raise RagError("Ollama could not generate an answer from the retrieved material.")

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
        return RagAnswer(text=answer, sources=sources, indexed_chunks=indexed_chunks)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

    def _build_prompt(
        self,
        question: str,
        hits: list[SearchHit],
        content: Mapping[str, Any],
        progress: Mapping[str, Any],
    ) -> str:
        response_language = _response_language(question)
        context = "\n\n".join(
            f"[S{index}] {hit.chunk.title}\n"
            f"Source: {hit.chunk.source}\n"
            f"Topic: {hit.chunk.topic}; Level: {hit.chunk.level}\n"
            f"{hit.chunk.text[:900]}"
            for index, hit in enumerate(hits, start=1)
        )
        learner_context = _learner_context(content, progress)
        return (
            f"Write the entire answer in {response_language}.\n"
            "Use only the retrieved study material below for grammar rules and examples.\n"
            "Cite supporting sources inline as [S1], [S2], and so on.\n"
            "If the material is insufficient, say what is missing instead of inventing a rule.\n"
            "Keep the answer under 100 words and cite references without copying their metadata.\n"
            "When correcting a sentence, give the correction, the exact rule, "
            "and one related example.\n"
            "Do not describe an auxiliary as a conjugated form of the main verb.\n"
            "Finish with one English practice question about the same rule, without its answer.\n\n"
            f"Student context:\n{learner_context}\n\n"
            f"Retrieved study material:\n{context}\n\n"
            f"Student question:\n{question}"
        )

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_manifest(self, data: dict[str, Any]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(self.manifest_path)


def _rerank_hits(question: str, hits: list[SearchHit]) -> list[SearchHit]:
    query_tokens = _search_tokens(question)
    if not query_tokens:
        return hits

    def relevance(hit: SearchHit) -> float:
        chunk_tokens = _search_tokens(f"{hit.chunk.title} {hit.chunk.topic} {hit.chunk.text}")
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        return hit.score + (0.35 * overlap)

    return sorted(hits, key=relevance, reverse=True)


def _search_tokens(value: str) -> set[str]:
    normalized = value.lower()
    normalized = normalized.replace("doesn't", "does not").replace("don't", "do not")
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "da",
        "de",
        "do",
        "e",
        "em",
        "explain",
        "explique",
        "frase",
        "in",
        "is",
        "na",
        "no",
        "o",
        "os",
        "por",
        "que",
        "the",
        "this",
        "um",
        "uma",
        "why",
    }
    return {
        token
        for token in re.findall(r"[a-zà-ÿ]+", normalized)
        if len(token) > 1 and token not in stopwords
    }


def _response_language(question: str) -> str:
    if re.search(r"\b(por que|explique|como|qual|erro|frase|está|estão|não)\b", question.lower()):
        return "Brazilian Portuguese"
    return "English"


def _learner_context(content: Mapping[str, Any], progress: Mapping[str, Any]) -> str:
    labels: dict[str, tuple[str, str]] = {}
    for item in content.get("sentences", []):
        labels[str(item.get("id", ""))] = ("sentence", str(item.get("en", "")))
    for item in content.get("flashcards", []):
        labels[str(item.get("id", ""))] = ("flashcard", str(item.get("en", "")))
    for item in content.get("to_be_exercises", []):
        labels[str(item.get("id", ""))] = ("to be", str(item.get("question", "")))

    difficult: list[tuple[float, int, str]] = []
    for section in ("sentences", "flashcards", "to_be"):
        for item_id, stats in progress.get(section, {}).items():
            seen = int(stats.get("seen", 0))
            correct = int(stats.get("correct", 0))
            errors = max(0, seen - correct)
            if seen and errors:
                kind, label = labels.get(str(item_id), (section, str(item_id)))
                error_rate = errors / seen
                difficult.append((error_rate, errors, f"{kind}: {label} ({errors}/{seen} errors)"))

    if not difficult:
        return "No recurring difficulty has been recorded yet."
    difficult.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return "Recent difficult items:\n" + "\n".join(f"- {item[2]}" for item in difficult[:5])
