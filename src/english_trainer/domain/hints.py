"""Feedback hints for translation answers."""

from __future__ import annotations

import re

from english_trainer.domain.matching import normalize_pt_phrase, strip_accents

_PAST_EN = (" was ", " were ", " did ", " did not ", " yesterday", " last ", " ago")
_PRESENT_EN = (
    " am ",
    " is ",
    " are ",
    " do ",
    " does ",
    " do not ",
    " does not ",
    " now",
    " today",
)
_PAST_PT = re.compile(
    r"\b(estava|estavam|foi|foram|fez|fizeram|era|eram|trabalhei|estudou|assistiu|fui)\b"
)
_PRESENT_PT = re.compile(r"\b(esta|estao|estou|faz|fazem|e|sao|trabalho|estudo|assisto)\b")


def detect_tense_hint(en_sentence: str, user_answer: str) -> str | None:
    """Return a hint when the answer's tense contradicts the English sentence."""
    answer = strip_accents(normalize_pt_phrase(user_answer).lower())
    english = f" {en_sentence.lower()} "

    has_past_en = any(marker in english for marker in _PAST_EN)
    has_present_en = any(marker in english for marker in _PRESENT_EN)

    if has_present_en and _PAST_PT.search(answer):
        return "Hint: your answer looks like past tense, but the sentence is in the present."
    if has_past_en and _PRESENT_PT.search(answer):
        return "Hint: your answer looks like present tense, but the sentence is in the past."
    return None
