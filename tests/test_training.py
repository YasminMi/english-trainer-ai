from english_trainer.training import _detect_tense_hint


def test_detect_tense_hint_for_past_answer_on_present_sentence():
    hint = _detect_tense_hint("She does not do homework.", "Ela não fez a lição de casa.")
    assert hint is not None
    assert "present" in hint


def test_detect_tense_hint_for_present_answer_on_past_sentence():
    hint = _detect_tense_hint("They were not in the kitchen.", "Eles não estão na cozinha.")
    assert hint is not None
    assert "past" in hint
