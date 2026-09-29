"""Miu coach: generate beginner ``to be`` sentences and review translations."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from english_trainer.domain.matching import is_close_answer
from english_trainer.llm.gateway import TextGenerator

Tense = Literal["present", "past"]
MAX_SENTENCES = 12

STYLE = (
    "You are an English tutor for beginners. "
    "Use very simple English. "
    "Explain in simple Portuguese (Brazil). "
    "Be short, practical, and give 1-2 examples. "
    "Never show your reasoning. Do not use tags like <think>. "
    "Output only what is requested."
)

_TARGET_POOLS: tuple[tuple[str, ...], ...] = (
    ("this", "that"),
    ("some", "how"),
    ("they", "we"),
    ("am", "is", "are"),
    ("was", "were"),
)
_TO_BE = re.compile(r"\b(am|is|are|was|were)\b", re.IGNORECASE)
_BE: dict[Tense, dict[str, str]] = {
    "present": {"I": "am", "You": "are", "We": "are", "They": "are", "He": "is", "She": "is"},
    "past": {"I": "was", "You": "were", "We": "were", "They": "were", "He": "was", "She": "was"},
}
_BUILTIN_PT: dict[str, str] = {
    "This is a": "Isto é um/uma",
    "That is a": "Aquilo é um/uma",
    "These are some": "Estas são algumas",
    "How are you?": "Como você está?",
    "They are here.": "Eles/Elas estão aqui.",
    "We are ready.": "Nós estamos prontos/prontas.",
    "I am here.": "Eu estou aqui.",
    "You are here.": "Você está aqui.",
    "He is here.": "Ele está aqui.",
    "She is here.": "Ela está aqui.",
    "I was here.": "Eu estava aqui.",
    "You were here.": "Você estava aqui.",
    "He was here.": "Ele estava aqui.",
    "She was here.": "Ela estava aqui.",
    "We were here.": "Nós estávamos aqui.",
    "They were here.": "Eles/Elas estavam aqui.",
}


@dataclass(frozen=True, slots=True)
class GeneratedSentence:
    en: str
    pt: str
    tip: str = ""


@dataclass(frozen=True, slots=True)
class SentenceBatch:
    sentences: list[GeneratedSentence] = field(default_factory=list)
    provider: str | None = None

    @property
    def used_builtin_examples(self) -> bool:
        """True when no model produced the batch, so nothing was AI-reviewed."""
        return self.provider is None


class Verdict(StrEnum):
    ACCEPTED_LOCALLY = "accepted_locally"
    AI_FEEDBACK = "ai_feedback"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TranslationFeedback:
    verdict: Verdict
    text: str = ""


def generation_prompt(targets: tuple[str, ...], tense: Tense, count: int) -> str:
    return (
        f"{STYLE}\n"
        f"Task: Generate {count} beginner English sentences.\n"
        "Rules:\n"
        "- Use ONLY very basic vocabulary (kid level).\n"
        f"- Focus on: {', '.join(targets)}\n"
        f"- Use verb 'to be' with tense: {tense}\n"
        "- Output format: one sentence per line in this exact format:\n"
        "  EN | PT | DICA\n"
        "  Where EN is a single simple sentence (no extra sentences),\n"
        "  PT is a single simple translation, and DICA is a very short Portuguese tip "
        "(max 8 words).\n"
        "  Do not add anything else. No numbering. No extra text. No reasoning.\n"
        f"- If you are unsure, still output {count} simple sentences.\n"
    )


def review_prompt(en_sentence: str, user_pt: str, expected_pt: str) -> str:
    return (
        f"{STYLE}\n"
        f"English sentence: {en_sentence}\n"
        f"User Portuguese: {user_pt}\n"
        f"Expected Portuguese: {expected_pt}\n"
        "Task: Say if user's translation is acceptable. "
        "If acceptable, say 'ACEITAVEL' and give a short reason. "
        "If not, say 'CORRIGIR' and provide the corrected Portuguese and a short reason.\n"
    )


def parse_sentences(text: str) -> list[GeneratedSentence]:
    """Keep only well-formed ``EN | PT | TIP`` lines that actually use ``to be``."""
    sentences = []
    for raw_line in text.splitlines():
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) < 2 or not _TO_BE.search(parts[0]) or len(parts[0].split()) < 2:
            continue
        en = re.split(r"(?<=[.!?])\s", parts[0])[0].strip()
        sentences.append(
            GeneratedSentence(en=en, pt=parts[1], tip=parts[2] if len(parts) > 2 else "")
        )
    return sentences


class CoachService:
    def __init__(self, generator: TextGenerator, *, rng: random.Random | None = None) -> None:
        self.generator = generator
        self.rng = rng or random.Random()

    def generate_sentences(self, tense: Tense, count: int = 5) -> SentenceBatch:
        count = max(1, min(count, MAX_SENTENCES))
        targets = self.rng.choice(_TARGET_POOLS)
        result = self.generator.generate(generation_prompt(targets, tense, count), max_tokens=220)
        if result is not None and (sentences := parse_sentences(result.text)):
            return SentenceBatch(sentences=sentences[:count], provider=result.provider)
        return SentenceBatch(sentences=self._builtin_sentences(targets, tense, count))

    def check_translation(
        self, en_sentence: str, user_pt: str, expected_pt: str | None
    ) -> TranslationFeedback:
        if expected_pt and is_close_answer(user_pt, expected_pt):
            return TranslationFeedback(Verdict.ACCEPTED_LOCALLY)
        prompt = review_prompt(en_sentence, user_pt, expected_pt or "(not provided)")
        result = self.generator.generate(prompt, max_tokens=220)
        if result is None:
            return TranslationFeedback(Verdict.UNAVAILABLE)
        return TranslationFeedback(Verdict.AI_FEEDBACK, result.text)

    def _builtin_sentences(
        self, targets: tuple[str, ...], tense: Tense, count: int
    ) -> list[GeneratedSentence]:
        nouns = ("book", "ball", "cat", "dog", "car", "house")
        be = _BE[tense]
        english: list[str] = []
        by_target = {
            "this": f"This is a {self.rng.choice(nouns)}.",
            "that": f"That is a {self.rng.choice(nouns)}.",
            "some": f"These are some {self.rng.choice(nouns)}s.",
            "how": "How are you?",
            "they": "They are here.",
            "we": "We are ready.",
        }
        english.extend(by_target[target] for target in targets if target in by_target)
        while len(english) < count:
            subject = self.rng.choice(list(be))
            english.append(f"{subject} {be[subject]} here.")
        return [GeneratedSentence(en=en, pt=_builtin_translation(en)) for en in english[:count]]


def _builtin_translation(en: str) -> str:
    for prefix, translation in _BUILTIN_PT.items():
        if en.startswith(prefix):
            return translation
    return ""
