import pytest
import requests

from english_trainer.rag.embeddings import EmbeddingError, OllamaEmbeddingClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_ollama_embedding_client_batches_requests(monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse({"embeddings": [[1, 0] for _ in json["input"]]})

    monkeypatch.setattr(requests, "post", post)
    client = OllamaEmbeddingClient(model="embed-test", base_url="http://ollama", batch_size=2)

    vectors = client.embed(["one", "two", "three"])

    assert vectors == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert len(calls) == 2
    assert calls[0][1]["model"] == "embed-test"


def test_ollama_embedding_client_reports_provider_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "post", fail)
    client = OllamaEmbeddingClient(model="embed-test")

    with pytest.raises(EmbeddingError, match="embed-test"):
        client.embed(["question"])
