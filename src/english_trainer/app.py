"""Composition root: builds every dependency once and hands it to the interfaces.

The CLI and the TUI receive an :class:`Application` instead of importing
modules with hidden global state, which keeps them thin and testable.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from english_trainer.config import Settings, get_settings
from english_trainer.domain.progress import register_ai_use, register_result
from english_trainer.domain.types import Content, PracticeMode, Progress
from english_trainer.llm.base import ChatMessage, GenerationOptions, LLMUnavailableError
from english_trainer.llm.gateway import TextGenerator, build_text_generator
from english_trainer.llm.ollama import OllamaClient
from english_trainer.rag import RagService
from english_trainer.rag.chunking import build_knowledge_chunks
from english_trainer.rag.embeddings import OllamaEmbeddingClient
from english_trainer.services.coach import CoachService
from english_trainer.services.conversation import ConversationSession
from english_trainer.services.dictionary import OnlineDictionary
from english_trainer.services.writing import WritingService
from english_trainer.storage.content import ContentRepository
from english_trainer.storage.progress import ProgressRepository

logger = logging.getLogger(__name__)


@dataclass
class Application:
    settings: Settings
    content_repository: ContentRepository
    progress_repository: ProgressRepository
    content: Content
    progress: Progress
    ollama: OllamaClient
    text_generator: TextGenerator
    chat_generator: TextGenerator
    coach: CoachService
    writing: WritingService
    dictionary: OnlineDictionary
    _rag: RagService | None = field(default=None, repr=False)
    _save_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _rag_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------ progress

    def save_progress(self) -> None:
        # Worker threads in the TUI may save concurrently with the UI thread.
        with self._save_lock:
            self.progress_repository.save(self.progress)

    def record_result(self, mode: PracticeMode, item_id: str, correct: bool) -> None:
        register_result(self.progress, mode, item_id, correct)
        self.save_progress()

    def record_ai_use(self) -> None:
        register_ai_use(self.progress)
        self.save_progress()

    # ------------------------------------------------------------------ AI features

    def new_conversation(self) -> ConversationSession:
        return ConversationSession(self.chat_generator)

    def rag(self) -> RagService:
        """Open the local vector index lazily; it is only needed by the grounded tutor."""
        with self._rag_lock:
            if self._rag is None:
                self._rag = self._build_rag()
            return self._rag

    def close(self) -> None:
        if self._rag is not None:
            self._rag.close()
            self._rag = None

    def _build_rag(self) -> RagService:
        settings = self.settings
        paths = settings.paths

        def generate(
            messages: list[ChatMessage],
            *,
            system_prompt: str = "",
            model: str | None = None,
            max_new_tokens: int = 160,
            timeout: float = 180,
            temperature: float = 0.0,
        ) -> str | None:
            try:
                return self.ollama.chat(
                    messages,
                    system=system_prompt or None,
                    model=model,
                    options=GenerationOptions(
                        temperature=temperature, max_tokens=max_new_tokens, timeout=timeout
                    ),
                )
            except LLMUnavailableError:
                return None

        def chunks(content: dict[str, Any]) -> Any:
            return build_knowledge_chunks(content, paths.notes_dir, paths.home)

        return RagService(
            embedder=OllamaEmbeddingClient(
                model=settings.rag.embed_model, base_url=settings.ollama.base_url
            ),
            index_dir=paths.rag_index_dir,
            generator=generate,
            chunk_builder=chunks,
            min_score=settings.rag.min_score,
            generation_model=settings.rag.model,
        )


def build_application(settings: Settings | None = None) -> Application:
    settings = settings or get_settings()
    paths = settings.paths
    content_repository = ContentRepository(paths)
    progress_repository = ProgressRepository(paths.progress_file)
    ollama = OllamaClient(settings.ollama)
    text_generator = build_text_generator(settings, ollama=ollama)
    logger.info("app.started", extra={"home": str(paths.home), "model": settings.ollama.model})
    return Application(
        settings=settings,
        content_repository=content_repository,
        progress_repository=progress_repository,
        content=content_repository.load(),
        progress=progress_repository.load(),
        ollama=ollama,
        text_generator=text_generator,
        # Conversation needs a real chat model; the offline fallbacks cannot hold one.
        chat_generator=TextGenerator([ollama]),
        coach=CoachService(text_generator),
        writing=WritingService(text_generator),
        dictionary=OnlineDictionary(),
    )
