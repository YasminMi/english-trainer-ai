from __future__ import annotations

from english_trainer.domain.hints import detect_tense_hint


def test_past_answer_for_present_sentence() -> None:
    hint = detect_tense_hint("She does not do homework.", "Ela não fez a lição de casa.")
    assert hint is not None
    assert "present" in hint


def test_present_answer_for_past_sentence() -> None:
    hint = detect_tense_hint("They were not in the kitchen.", "Eles não estão na cozinha.")
    assert hint is not None
    assert "past" in hint


def test_no_hint_when_tenses_agree() -> None:
    assert detect_tense_hint("They were in the kitchen.", "Eles estavam na cozinha.") is None
