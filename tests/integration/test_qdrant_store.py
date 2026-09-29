from english_trainer.rag.models import KnowledgeChunk
from english_trainer.rag.vector_store import QdrantKnowledgeStore


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        text=text,
        source="notes/test.md",
        title="Test rule",
        topic="test",
        level="A1",
        kind="rule",
    )


def test_qdrant_store_persists_and_searches_vectors(tmp_path):
    store = QdrantKnowledgeStore(tmp_path / "qdrant")
    try:
        store.replace(
            [_chunk("a" * 64, "first"), _chunk("b" * 64, "second")],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        hits = store.search([1.0, 0.0], limit=2)

        assert store.count() == 2
        assert hits[0].chunk.text == "first"
        assert hits[0].score == 1.0
    finally:
        store.close()
