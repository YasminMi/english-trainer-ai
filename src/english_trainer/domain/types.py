"""Typed shapes of the JSON documents the application reads and writes.

The app stores content and progress as plain JSON objects. ``TypedDict`` keeps
that on-disk format unchanged while giving the type checker, and the schema
validation in :mod:`english_trainer.storage.content`, a single definition.
"""

from __future__ import annotations

from typing import Literal, NotRequired

# Pydantic needs typing_extensions.TypedDict to validate these on Python < 3.12.
from typing_extensions import TypedDict

PracticeMode = Literal["sentences", "flashcards", "to_be"]
PRACTICE_MODES: tuple[PracticeMode, ...] = ("sentences", "flashcards", "to_be")


class Sentence(TypedDict):
    id: str
    en: str
    pt: str
    tags: NotRequired[list[str]]
    source_note: NotRequired[str]


class Flashcard(TypedDict):
    id: str
    en: str
    pt: str
    category: NotRequired[str]
    source_note: NotRequired[str]


class ToBeExercise(TypedDict):
    id: str
    question: str
    options: list[str]
    correct: str
    pt: str
    explanation: NotRequired[str]
    tense: NotRequired[str]
    focus: NotRequired[str]


class Content(TypedDict):
    metadata: dict[str, object]
    sentences: list[Sentence]
    flashcards: list[Flashcard]
    to_be_exercises: list[ToBeExercise]
    dictionary: dict[str, str]


class ItemStats(TypedDict):
    seen: int
    correct: int


class SrsState(TypedDict):
    interval_days: int
    ease: float
    repetitions: int
    due_date: str
    last_reviewed: str | None
    last_result: bool | None


class DayStats(TypedDict):
    attempts: int
    correct: int
    sentences: int
    flashcards: int
    to_be: int


class DailyState(TypedDict):
    goal: int
    history: dict[str, DayStats]
    streak: int
    best_streak: int
    last_study_date: str | None


class Profile(TypedDict, total=False):
    onboarding_completed: bool


class Progress(TypedDict):
    sentences: dict[str, ItemStats]
    flashcards: dict[str, ItemStats]
    to_be: dict[str, ItemStats]
    dictionary_lookups: int
    ai_uses: int
    srs: dict[str, dict[str, SrsState]]
    daily: DailyState
    profile: NotRequired[Profile]
    conversation_sessions: NotRequired[int]
    conversation_turns: NotRequired[int]


def empty_content() -> Content:
    return {
        "metadata": {},
        "sentences": [],
        "flashcards": [],
        "to_be_exercises": [],
        "dictionary": {},
    }
