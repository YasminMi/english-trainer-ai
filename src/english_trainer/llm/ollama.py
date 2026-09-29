"""HTTP client for a local Ollama server."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import requests

from english_trainer.config import OllamaSettings
from english_trainer.llm.base import ChatMessage, GenerationOptions, LLMUnavailableError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin wrapper over the Ollama REST API (``/api/chat``, ``/api/embed``, ``/api/tags``)."""

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = settings.base_url
        self.default_model = settings.model
        self._session = session or requests.Session()

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self, timeout: float = 5.0) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=timeout)
        except requests.RequestException:
            return False
        return response.status_code == 200

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        opts = options or GenerationOptions(timeout=self.settings.timeout_seconds)
        selected = (model or self.default_model).strip()
        payload = {
            "model": selected,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "options": {"temperature": opts.temperature, "num_predict": opts.max_tokens},
            "stream": False,
        }
        data = self._post("/api/chat", payload, timeout=opts.timeout, model=selected)
        message = data.get("message")
        text = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError(f"Ollama model '{selected}' returned an empty answer.")
        return text.strip()

    def embed(self, texts: Sequence[str], *, model: str, timeout: float) -> list[list[float]]:
        data = self._post(
            "/api/embed", {"model": model, "input": list(texts)}, timeout=timeout, model=model
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list):
            raise LLMUnavailableError("Ollama returned an invalid embedding response.")
        return vectors

    def _post(
        self, path: str, payload: dict[str, Any], *, timeout: float, model: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self._session.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning(
                "ollama.request_failed",
                extra={"path": path, "model": model, "error": type(exc).__name__},
            )
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url} with model '{model}'."
            ) from exc
        logger.info(
            "ollama.request",
            extra={
                "path": path,
                "model": model,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        if not isinstance(data, dict):
            raise LLMUnavailableError("Ollama returned a non-object response.")
        return data
