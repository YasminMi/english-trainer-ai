"""Typed application settings loaded from environment variables.

Each settings group owns an environment prefix, so existing variable names
(``OLLAMA_URL``, ``RAG_MIN_SCORE``...) keep working. Values are validated once
at startup instead of being parsed ad hoc wherever they are used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _default_home() -> Path:
    """Use the repository root when running from a checkout, else the working directory."""
    if (_PACKAGE_ROOT / "data" / "content.json").exists():
        return _PACKAGE_ROOT
    return Path.cwd()


class Paths(BaseModel):
    """Filesystem layout derived from a single home directory."""

    model_config = ConfigDict(frozen=True)

    home: Path

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def notes_dir(self) -> Path:
        return self.home / "notes"

    @property
    def content_file(self) -> Path:
        return self.data_dir / "content.json"

    @property
    def user_content_file(self) -> Path:
        return self.data_dir / "user_content.json"

    @property
    def progress_file(self) -> Path:
        return self.data_dir / "progress.json"

    @property
    def ai_cache_file(self) -> Path:
        return self.data_dir / "ai_cache.json"

    @property
    def conversations_dir(self) -> Path:
        return self.data_dir / "conversations"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def rag_index_dir(self) -> Path:
        return self.data_dir / "rag_index"


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    url: HttpUrl = HttpUrl("http://localhost:11434")
    model: str = "llama3.2"
    timeout_seconds: float = Field(default=120.0, gt=0)

    @property
    def base_url(self) -> str:
        return str(self.url).rstrip("/")


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HF_", extra="ignore")

    token: SecretStr = SecretStr("")
    model: str = "gpt2"

    @property
    def enabled(self) -> bool:
        return bool(self.token.get_secret_value().strip())


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", extra="ignore")

    local_fallback: bool = False


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    model: str = "gemma2:2b"
    embed_model: str = "embeddinggemma"
    min_score: float = Field(default=0.38, ge=0.0, le=1.0)
    top_k: int = Field(default=2, ge=1, le=10)
    retrieval: Literal["dense", "hybrid"] = "dense"


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: LogLevel = "WARNING"
    json_format: bool = False

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class Settings(BaseSettings):
    """Root settings object; build it with :func:`get_settings`."""

    model_config = SettingsConfigDict(env_prefix="ENGLISH_TRAINER_", extra="ignore")

    home: Path = Field(default_factory=_default_home)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    ai: AISettings = Field(default_factory=AISettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @property
    def paths(self) -> Paths:
        return Paths(home=self.home)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once from the environment."""
    return Settings()
