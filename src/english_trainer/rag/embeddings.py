from __future__ import annotations

from collections.abc import Sequence

import requests


class EmbeddingError(RuntimeError):
    pass


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
                raw_vectors = response.json().get("embeddings")
            except (requests.RequestException, TypeError, ValueError) as exc:
                raise EmbeddingError(
                    f"Could not generate embeddings with Ollama model '{self.model}'."
                ) from exc

            if not isinstance(raw_vectors, list) or len(raw_vectors) != len(batch):
                raise EmbeddingError("Ollama returned an invalid embedding response.")
            for vector in raw_vectors:
                if not isinstance(vector, list) or not vector:
                    raise EmbeddingError("Ollama returned an empty embedding vector.")
                vectors.append([float(value) for value in vector])
        return vectors
