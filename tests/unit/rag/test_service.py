from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from english_trainer.llm.base import ChatMessage
from english_trainer.rag import RagConfig, RagError, RagService
from english_trainer.rag.chunking import build_knowledge_chunks
from english_trainer.rag.index import KnowledgeIndex
from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.rag.prompts import learner_context, response_language
from english_trainer.rag.retrieval import DenseRetriever, HybridRetriever, lexical_overlap_rerank
from tests.fakes import FakeChatModel, OfflineChatModel


def _chunk(chunk_id: str, text: str, source: str = "notes/grammar.md") -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        text=text,
        source=source,
        title="Present simple — negative form",
        topic="present_simple",
        level="A1",
        kind="rule",
    )


class FakeEmbedder:
    model = "test-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    def __init__(self, score: float = 0.91) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self.replace_calls = 0
        self.score = score

    def count(self) -> int:
        return len(self.chunks)

    def replace(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks = list(chunks)
        self.replace_calls += 1

    def search(self, vector: Sequence[float], limit: int = 5) -> list[SearchHit]:
        return [SearchHit(chunk=chunk, score=self.score) for chunk in self.chunks[:limit]]


def _service(
    tmp_path: Path,
    chunks: list[KnowledgeChunk],
    *,
    generator: FakeChatModel | OfflineChatModel,
    store: FakeStore | None = None,
    embedder: FakeEmbedder | None = None,
    min_score: float = 0.2,
    hybrid: bool = False,
) -> RagService:
    embedder = embedder or FakeEmbedder()
    store = store or FakeStore()
    return RagService(
        index=KnowledgeIndex(embedder, store, tmp_path / "index"),
        retriever=(
            HybridRetriever(embedder, store)
            if hybrid
            else DenseRetriever(embedder, store, oversample=1)
        ),
        generator=generator,
        chunk_builder=lambda _content: chunks,
        config=RagConfig(min_score=min_score),
    )


def test_chunk_builder_uses_markdown_and_deduplicates_explanations(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "present_simple_a1.md").write_text(
        "# Present Simple A1\n\n"
        "## Sentences\n"
        "- en: She does not work here.\n"
        "  pt: Ela não trabalha aqui.\n",
        encoding="utf-8",
    )
    content = {
        "sentences": [
            {"id": "s1", "en": "I work.", "pt": "Eu trabalho.", "tags": ["present_simple"]},
        ],
        "flashcards": [],
        "to_be_exercises": [
            {
                "question": "She ___ ready.",
                "correct": "is",
                "pt": "Ela está pronta.",
                "explanation": "Use is with she.",
            },
            {
                "question": "He ___ ready.",
                "correct": "is",
                "pt": "Ele está pronto.",
                "explanation": "Use is with she.",
            },
        ],
    }

    chunks = build_knowledge_chunks(content, notes_dir, tmp_path)

    assert len([chunk for chunk in chunks if chunk.kind == "rule"]) == 1
    assert any(chunk.kind == "note" and "She does not work" in chunk.text for chunk in chunks)
    assert any(chunk.kind == "examples" and chunk.topic == "present_simple" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_service_indexes_once_and_builds_a_grounded_prompt(tmp_path: Path) -> None:
    chunks = [_chunk("a" * 64, "Use doesn't with she. After doesn't, use the base verb.")]
    embedder = FakeEmbedder()
    store = FakeStore()
    model = FakeChatModel(replies=["Use doesn't with she [S1].", "Again [S1]."])
    service = _service(tmp_path, chunks, generator=model, store=store, embedder=embedder)

    first = service.ask("Why is 'She don't' wrong?", {}, {})
    second = service.ask("Can you explain again?", {}, {})

    assert first.text.endswith("[S1].")
    assert first.sources[0].source == "notes/grammar.md"
    assert set(first.timings_ms) == {"index", "retrieve", "generate"}
    prompt = model.calls[0]["messages"][0]["content"]
    assert "Use only the retrieved study material" in prompt
    assert "[S1]" in prompt
    assert store.replace_calls == 1
    assert second.indexed_chunks == 1
    assert embedder.calls == 3  # one indexing pass + one query embedding per question


def test_force_reindex_rebuilds_the_store(tmp_path: Path) -> None:
    store = FakeStore()
    service = _service(tmp_path, [_chunk("a" * 64, "rule")], generator=FakeChatModel(), store=store)

    service.prepare({})
    service.prepare({}, force=True)

    assert store.replace_calls == 2


def test_low_scoring_material_is_refused(tmp_path: Path) -> None:
    model = FakeChatModel(replies=["should not be called"])
    service = _service(
        tmp_path,
        [_chunk("a" * 64, "rule")],
        generator=model,
        store=FakeStore(score=0.1),
        min_score=0.3,
    )

    with pytest.raises(RagError, match="relevant"):
        service.ask("What is the capital of France?", {}, {})
    assert model.calls == []


def test_hybrid_service_uses_bm25_and_dense_gate(tmp_path: Path) -> None:
    chunks = [_chunk("a", "Irrelevant material."), _chunk("b", "Explique o passado simples.")]
    model = FakeChatModel(replies=["Use the simple past [S1]."])
    service = _service(tmp_path, chunks, generator=model, hybrid=True)

    answer = service.ask("Explique o passado simples", {}, {})

    assert answer.sources[0].source == "notes/grammar.md"
    assert isinstance(service.retriever, HybridRetriever)
    assert service.retriever.retrieve("Explique o passado simples", limit=1)[0].chunk.id == "b"

    low = _service(
        tmp_path / "low",
        chunks,
        generator=FakeChatModel(),
        store=FakeStore(score=0.1),
        min_score=0.3,
        hybrid=True,
    )
    with pytest.raises(RagError, match="relevant"):
        low.ask("Explique o passado simples", {}, {})


def test_generation_outage_is_reported(tmp_path: Path) -> None:
    service = _service(tmp_path, [_chunk("a" * 64, "rule")], generator=OfflineChatModel())

    with pytest.raises(RagError, match="Ollama"):
        service.ask("Why?", {}, {})


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_questions_are_rejected(tmp_path: Path, question: str) -> None:
    service = _service(tmp_path, [_chunk("a" * 64, "rule")], generator=FakeChatModel())
    with pytest.raises(RagError, match="question"):
        service.ask(question, {}, {})


def test_learner_context_prioritizes_recorded_errors() -> None:
    content = {"sentences": [{"id": "s1", "en": "She works."}]}
    progress = {"sentences": {"s1": {"seen": 4, "correct": 1}}}

    context = learner_context(content, progress)

    assert "She works." in context
    assert "3/4 errors" in context
    assert learner_context({}, {}) == "No recurring difficulty has been recorded yet."


def test_reranking_promotes_lexically_relevant_rule() -> None:
    continuous = _chunk("b" * 64, "She is working now.")
    present_simple = _chunk("c" * 64, "She doesn't like coffee. Do not use: She don't like coffee.")
    hits = [
        SearchHit(chunk=continuous, score=0.44, dense_score=0.44),
        SearchHit(chunk=present_simple, score=0.38, dense_score=0.38),
    ]

    reranked = lexical_overlap_rerank("Explique o erro em She dont like coffee", hits)

    assert reranked[0].chunk is present_simple
    assert reranked[0].dense_score == 0.38


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Por que esta frase está errada?", "Brazilian Portuguese"),
        ("Quando uso am, is e are?", "Brazilian Portuguese"),
        ("Why is this sentence wrong?", "English"),
    ],
)
def test_response_language(question: str, language: str) -> None:
    assert response_language(question) == language


def test_messages_have_expected_shape() -> None:
    message: ChatMessage = {"role": "user", "content": "hi"}
    assert set(message) == {"role", "content"}


def test_rule_variants_are_grouped_under_their_shared_rule(tmp_path: Path) -> None:
    rule = "No presente, usamos: I am / he-she-it is / you-we-they are."
    exercises = [
        {"question": f"{subject} ___ ready.", "correct": verb, "pt": "...",
         "explanation": f"{rule} Aqui, com '{subject}', a forma correta é '{verb}'."}
        for subject, verb in (("He", "is"), ("I", "am"), ("They", "are"))
    ]  # fmt: skip

    chunks = build_knowledge_chunks({"to_be_exercises": exercises}, tmp_path / "notes", tmp_path)

    assert len(chunks) == 1
    assert chunks[0].text.count(rule) == 1
    assert "com 'They'" in chunks[0].text
    assert chunks[0].metadata["example_count"] == 3
