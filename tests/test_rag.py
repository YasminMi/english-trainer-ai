from pathlib import Path

from english_trainer.rag.chunking import build_knowledge_chunks
from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.rag.service import (
    RagService,
    _learner_context,
    _rerank_hits,
    _response_language,
)


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


def test_chunk_builder_uses_markdown_and_deduplicates_explanations(tmp_path):
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
            {"question": "She ___ ready.", "correct": "is", "pt": "Ela está pronta.", "explanation": "Use is with she."},
            {"question": "He ___ ready.", "correct": "is", "pt": "Ele está pronto.", "explanation": "Use is with she."},
        ],
    }

    chunks = build_knowledge_chunks(content, notes_dir=notes_dir)

    assert len([chunk for chunk in chunks if chunk.kind == "rule"]) == 1
    assert any(chunk.kind == "note" and "She does not work" in chunk.text for chunk in chunks)
    assert any(chunk.kind == "examples" and chunk.topic == "present_simple" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


class FakeEmbedder:
    model = "test-embedding"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    def __init__(self):
        self.chunks = []
        self.replace_calls = 0

    def count(self):
        return len(self.chunks)

    def replace(self, chunks, vectors):
        assert len(chunks) == len(vectors)
        self.chunks = list(chunks)
        self.replace_calls += 1

    def search(self, vector, limit=5):
        return [SearchHit(chunk=self.chunks[0], score=0.91)]


def test_rag_service_indexes_once_and_builds_grounded_prompt(tmp_path):
    chunks = [_chunk("a" * 64, "Use doesn't with she. After doesn't, use the base verb.")]
    embedder = FakeEmbedder()
    store = FakeStore()
    captured = {}

    def generate(messages, system_prompt="", **_kwargs):
        captured["prompt"] = messages[0]["content"]
        captured["system"] = system_prompt
        return "Use doesn't with she [S1]."

    service = RagService(
        embedder=embedder,
        store=store,
        index_dir=tmp_path / "index",
        generator=generate,
        chunk_builder=lambda _content: chunks,
        min_score=0.2,
    )

    first = service.ask("Why is 'She don't' wrong?", {}, {})
    second = service.ask("Can you explain again?", {}, {})

    assert first.text.endswith("[S1].")
    assert first.sources[0].source == "notes/grammar.md"
    assert "Use only the retrieved study material" in captured["prompt"]
    assert "[S1]" in captured["prompt"]
    assert store.replace_calls == 1
    assert second.indexed_chunks == 1
    assert embedder.calls == 3


def test_learner_context_prioritizes_recorded_errors():
    content = {
        "sentences": [{"id": "s1", "en": "She works."}],
        "flashcards": [],
        "to_be_exercises": [],
    }
    progress = {"sentences": {"s1": {"seen": 4, "correct": 1}}}

    context = _learner_context(content, progress)

    assert "She works." in context
    assert "3/4 errors" in context


def test_reranking_promotes_lexically_relevant_rule():
    continuous = _chunk("b" * 64, "She is working now.")
    present_simple = _chunk(
        "c" * 64,
        "She doesn't like coffee. Do not use: She don't like coffee.",
    )
    hits = [
        SearchHit(chunk=continuous, score=0.44),
        SearchHit(chunk=present_simple, score=0.38),
    ]

    reranked = _rerank_hits("Explique o erro em She dont like coffee", hits)

    assert reranked[0].chunk is present_simple
    assert _response_language("Por que esta frase está errada?") == "Brazilian Portuguese"