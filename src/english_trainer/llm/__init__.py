"""Text generation providers (Ollama, Hugging Face, local examples) behind one gateway."""

from english_trainer.llm.base import (
    ChatMessage,
    ChatModel,
    GenerationOptions,
    LLMError,
    LLMUnavailableError,
)
from english_trainer.llm.gateway import Generation, TextGenerator, build_text_generator
from english_trainer.llm.ollama import OllamaClient

__all__ = [
    "ChatMessage",
    "ChatModel",
    "Generation",
    "GenerationOptions",
    "LLMError",
    "LLMUnavailableError",
    "OllamaClient",
    "TextGenerator",
    "build_text_generator",
]
