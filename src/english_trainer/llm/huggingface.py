"""Optional Hugging Face Inference API fallback (install the ``huggingface`` extra)."""

from __future__ import annotations

import logging

from english_trainer.config import HuggingFaceSettings
from english_trainer.llm.base import ChatMessage, GenerationOptions, LLMUnavailableError

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Text-generation client; chat messages are flattened into a single prompt.

    Prompts sent here leave the local machine, which is why this provider is
    only used when ``HF_TOKEN`` is explicitly configured.
    """

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self.settings = settings

    @property
    def name(self) -> str:
        return "huggingface"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        if not self.settings.enabled:
            raise LLMUnavailableError("HF_TOKEN is not configured.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise LLMUnavailableError(
                "huggingface_hub is not installed; install the 'huggingface' extra."
            ) from exc

        opts = options or GenerationOptions()
        prompt = "\n\n".join(([system] if system else []) + [m["content"] for m in messages])
        client = InferenceClient(
            model=model or self.settings.model,
            token=self.settings.token.get_secret_value(),
        )
        try:
            text = client.text_generation(prompt, max_new_tokens=opts.max_tokens)
        except Exception as exc:
            logger.warning("huggingface.request_failed", extra={"error": type(exc).__name__})
            raise LLMUnavailableError("Hugging Face inference failed.") from exc
        if not text or not text.strip():
            raise LLMUnavailableError("Hugging Face returned an empty answer.")
        return text.strip()
