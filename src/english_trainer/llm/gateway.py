"""Single entry point for text generation with provider fallback and caching."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from english_trainer.config import Settings
from english_trainer.llm.base import ChatMessage, ChatModel, GenerationOptions, LLMUnavailableError
from english_trainer.llm.cache import ResponseCache, cache_key
from english_trainer.llm.huggingface import HuggingFaceClient
from english_trainer.llm.local_examples import LocalExamplesModel
from english_trainer.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)

_REASONING_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks some local models emit before answering."""
    return _REASONING_BLOCK.sub("", text).strip()


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    provider: str
    cached: bool = False


class TextGenerator:
    """Try each provider in order and return the first usable answer.

    One-shot prompts are cached because they are generated with temperature 0;
    multi-turn chat is never cached.
    """

    def __init__(
        self,
        providers: Sequence[ChatModel],
        *,
        cache: ResponseCache | None = None,
        cache_namespace: str = "",
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required.")
        self.providers = list(providers)
        self.cache = cache
        self.cache_namespace = cache_namespace

    def generate(self, prompt: str, *, max_tokens: int = 160) -> Generation | None:
        """Generate a deterministic answer, or ``None`` when every provider failed."""
        key = cache_key(self.cache_namespace, prompt)
        if self.cache is not None and (cached := self.cache.get(key)) is not None:
            return Generation(text=cached, provider="cache", cached=True)

        options = GenerationOptions(temperature=0.0, max_tokens=max_tokens, timeout=60.0)
        result = self._first_success([{"role": "user", "content": prompt}], None, options)
        if result is not None and self.cache is not None:
            self.cache.put(key, result.text, provider=result.provider)
        return result

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        options: GenerationOptions | None = None,
    ) -> Generation | None:
        return self._first_success(messages, system, options or GenerationOptions())

    def _first_success(
        self,
        messages: list[ChatMessage],
        system: str | None,
        options: GenerationOptions,
    ) -> Generation | None:
        for provider in self.providers:
            try:
                text = strip_reasoning(provider.chat(messages, system=system, options=options))
            except LLMUnavailableError as exc:
                logger.info(
                    "llm.provider_skipped", extra={"provider": provider.name, "reason": str(exc)}
                )
                continue
            if text:
                return Generation(text=text, provider=provider.name)
        logger.warning("llm.all_providers_failed", extra={"providers": len(self.providers)})
        return None


def build_text_generator(
    settings: Settings, *, ollama: OllamaClient | None = None
) -> TextGenerator:
    """Wire the provider chain from configuration: Ollama, then optional fallbacks."""
    providers: list[ChatModel] = [ollama or OllamaClient(settings.ollama)]
    if settings.huggingface.enabled:
        providers.append(HuggingFaceClient(settings.huggingface))
    if settings.ai.local_fallback:
        providers.append(LocalExamplesModel())
    return TextGenerator(
        providers,
        cache=ResponseCache(settings.paths.ai_cache_file),
        cache_namespace=settings.ollama.model,
    )
