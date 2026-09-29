from __future__ import annotations

import socket
import threading
from collections.abc import Sequence
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from english_trainer.config import OllamaSettings
from english_trainer.llm.base import GenerationOptions, LLMUnavailableError
from english_trainer.llm.ollama import OllamaClient
from english_trainer.rag.embeddings import EmbeddingError, OllamaEmbeddingClient
from english_trainer.rag.index import KnowledgeIndex
from english_trainer.rag.models import KnowledgeChunk, SearchHit
from english_trainer.rag.retrieval import DenseRetriever
from english_trainer.rag.service import RagConfig, RagError, RagService


class _Handler(BaseHTTPRequestHandler):
    status = 200
    body: bytes = b"{}"
    delay = 0.0

    def do_POST(self) -> None:
        if self.delay:
            import time

            time.sleep(self.delay)
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        with suppress(BrokenPipeError):
            self.wfile.write(self.body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def http_provider():
    def start(*, status: int = 200, body: bytes = b"{}", delay: float = 0.0):
        handler = type(
            "ConfiguredHandler",
            (_Handler,),
            {"status": status, "body": body, "delay": delay},
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{server.server_port}", server

    servers: list[ThreadingHTTPServer] = []

    def serve(**kwargs: Any) -> str:
        url, server = start(**kwargs)
        servers.append(server)
        return url

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def _chat(url: str, timeout: float = 0.2) -> str:
    return OllamaClient(
        OllamaSettings(url=url, model="fixture-model", timeout_seconds=timeout)
    ).chat(
        [{"role": "user", "content": "hello"}],
        options=GenerationOptions(timeout=timeout),
    )


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, b'{"error":"internal"}'),
        (404, b'{"error":"model not found"}'),
        (200, b"not-json"),
        (200, b'{"message":{"content":" "}}'),
        (200, b"[]"),
    ],
)
def test_real_ollama_client_translates_http_and_body_failures(
    http_provider, status: int, body: bytes
) -> None:
    url = http_provider(status=status, body=body)

    with pytest.raises(LLMUnavailableError) as error:
        _chat(url)

    assert "fixture-model" in str(error.value) or "response" in str(error.value)
    assert "Traceback" not in str(error.value)


def test_real_ollama_client_translates_connection_refused() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"

    with pytest.raises(LLMUnavailableError, match="Could not reach Ollama"):
        _chat(url)


def test_real_ollama_client_translates_timeout(http_provider) -> None:
    url = http_provider(delay=0.15)

    with pytest.raises(LLMUnavailableError, match="Could not reach Ollama"):
        _chat(url, timeout=0.02)


@pytest.mark.parametrize(
    "body",
    [b"not-json", b"[]", b'{"embeddings":[["not-a-number"]]}', b'{"embeddings":[[]]}'],
)
def test_embedding_client_translates_malformed_or_empty_vectors(http_provider, body: bytes) -> None:
    url = http_provider(body=body)

    with pytest.raises(EmbeddingError):
        OllamaEmbeddingClient("fixture-model", url, timeout=0.2).embed(["text"])


class _Store:
    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self.replace_calls = 0

    def count(self) -> int:
        return len(self.chunks)

    def replace(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None:
        self.replace_calls += 1
        self.chunks = list(chunks)

    def search(self, _vector: Sequence[float], limit: int = 5) -> list[SearchHit]:
        return [SearchHit(chunk=chunk, score=0.9) for chunk in self.chunks[:limit]]


class _GoodEmbedder:
    model = "fixture"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _UnavailableGenerator:
    @property
    def name(self) -> str:
        return "fixture"

    def chat(self, *_args: Any, **_kwargs: Any) -> str:
        raise LLMUnavailableError("provider details")


def _chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        id="a" * 64,
        text="Use does not with she.",
        source="notes.md",
        title="Present simple",
        topic="present_simple",
        level="A1",
        kind="rule",
    )


def test_rag_refuses_failed_embedding_without_partial_index_or_manifest(
    http_provider, tmp_path: Path
) -> None:
    url = http_provider(status=500, body=b'{"error":"embedding failure"}')
    embedder = OllamaEmbeddingClient("fixture-model", url, timeout=0.2)
    store = _Store()
    index_dir = tmp_path / "index"
    service = RagService(
        index=KnowledgeIndex(embedder, store, index_dir),
        retriever=DenseRetriever(embedder, store),
        generator=_UnavailableGenerator(),
        chunk_builder=lambda _content: [_chunk()],
        config=RagConfig(min_score=0),
    )

    with pytest.raises(RagError, match="Could not generate embeddings") as error:
        service.ask("Why?", {}, {})

    assert "Traceback" not in str(error.value)
    assert store.replace_calls == 0
    assert not (index_dir / "manifest.json").exists()


def test_rag_explains_generation_outage_after_successful_retrieval(
    http_provider, tmp_path: Path
) -> None:
    url = http_provider(body=b'{"embeddings":[[1.0,0.0]]}')
    embedder = OllamaEmbeddingClient("fixture-model", url, timeout=0.2)
    store = _Store()
    service = RagService(
        index=KnowledgeIndex(embedder, store, tmp_path / "index"),
        retriever=DenseRetriever(embedder, store),
        generator=_UnavailableGenerator(),
        chunk_builder=lambda _content: [_chunk()],
        config=RagConfig(min_score=0),
    )

    with pytest.raises(RagError, match="could not generate an answer") as error:
        service.ask("Why?", {}, {})

    assert "provider details" not in str(error.value)
    assert store.replace_calls == 1
