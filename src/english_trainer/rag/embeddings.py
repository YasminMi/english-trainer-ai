"""Text embedding providers."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import requests

from english_trainer.storage.json_store import read_json_object, write_json_atomic


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic bag-of-words embeddings via the hashing trick.

    It captures no semantics beyond shared words, so it is not used by the app.
    It exists to exercise the dense retrieval path in tests and CI, where
    Ollama is unavailable, with results that are stable across machines.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self.model = f"hashing-{dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-zà-ÿ']+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OllamaEmbeddingClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 120,
        batch_size: int = 32,
    ) -> None:
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                raw_vectors = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(raw_vectors, list) or len(raw_vectors) != len(batch):
                    raise EmbeddingError("Ollama returned an invalid embedding response.")
                batch_vectors: list[list[float]] = []
                for vector in raw_vectors:
                    if not isinstance(vector, list) or not vector:
                        raise EmbeddingError("Ollama returned an empty embedding vector.")
                    batch_vectors.append([float(value) for value in vector])
            except EmbeddingError:
                raise
            except (requests.RequestException, TypeError, ValueError, AttributeError) as exc:
                raise EmbeddingError(
                    f"Could not generate embeddings with Ollama model '{self.model}'."
                ) from exc
            vectors.extend(batch_vectors)
        return vectors


class CachingEmbedder:
    """Persist vectors by ``sha256(model, text)`` so repeated evaluations skip Ollama.

    Only texts missing from the cache are sent to the wrapped embedder.
    """

    def __init__(self, inner: Embedder, directory: Path) -> None:
        self.inner = inner
        self.model = inner.model
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", inner.model)
        self.path = directory / f"{safe_model}.json"
        self.hits = 0
        self.misses = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cache = read_json_object(self.path)
        keys = [hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest() for text in texts]
        pairs = zip(keys, texts, strict=True)
        missing = list(dict.fromkeys(pair for pair in pairs if pair[0] not in cache))
        if missing:
            vectors = self.inner.embed([text for _, text in missing])
            cache.update({key: vector for (key, _), vector in zip(missing, vectors, strict=True)})
            write_json_atomic(self.path, cache)
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [list(map(float, cache[key])) for key in keys]
