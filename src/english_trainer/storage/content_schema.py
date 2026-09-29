"""Pydantic schema for curated, versioned study content."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SentenceDocument(_StrictModel):
    id: str = Field(min_length=1)
    en: str = Field(min_length=1)
    pt: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class FlashcardDocument(_StrictModel):
    id: str = Field(min_length=1)
    en: str = Field(min_length=1)
    pt: str = Field(min_length=1)
    category: str | None = None


class ExerciseDocument(_StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    correct: str = Field(min_length=1)
    pt: str = Field(min_length=1)
    explanation: str | None = None
    tense: str | None = None
    focus: str | None = None

    @model_validator(mode="after")
    def correct_answer_is_an_option(self) -> ExerciseDocument:
        if self.correct not in self.options:
            raise ValueError("correct must match one of options")
        return self


class ContentDocument(_StrictModel):
    version: Literal[1]
    metadata: dict[str, Any] = Field(default_factory=dict)
    sentences: list[SentenceDocument] = Field(default_factory=list)
    flashcards: list[FlashcardDocument] = Field(default_factory=list)
    to_be_exercises: list[ExerciseDocument] = Field(default_factory=list)
    dictionary: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ids_are_unique(self) -> ContentDocument:
        seen: dict[str, str] = {}
        for section in ("sentences", "flashcards", "to_be_exercises"):
            for index, item in enumerate(getattr(self, section)):
                previous = seen.get(item.id)
                if previous is not None:
                    raise ValueError(
                        f"duplicate id {item.id!r} at {section}.{index}.id "
                        f"(first used at {previous})"
                    )
                seen[item.id] = f"{section}.{index}.id"
        return self
