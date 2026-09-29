import pytest

from english_trainer.progress import (
    _update_item,
    compute_overall,
    daily_goal_status,
    get_srs_weight,
    recommend_practice_mode,
    register_flashcard_result,
    register_sentence_result,
    set_daily_goal,
)


def test_update_item_adds_and_updates():
    section = {}
    _update_item(section, "a", True)
    assert section["a"]["seen"] == 1
    assert section["a"]["correct"] == 1
    _update_item(section, "a", False)
    assert section["a"]["seen"] == 2
    assert section["a"]["correct"] == 1


def test_compute_overall_empty():
    stats = compute_overall({})
    assert stats["total_seen"] == 0
    assert stats["accuracy"] == 0.0


def test_compute_overall_counts():
    progress = {
        "sentences": {"s": {"seen": 2, "correct": 1}},
        "flashcards": {"f": {"seen": 3, "correct": 3}},
        "to_be": {"t": {"seen": 1, "correct": 0}},
        "dictionary_lookups": 5,
    }
    stats = compute_overall(progress)
    assert stats["total_seen"] == 6
    assert stats["total_correct"] == 4
    assert stats["accuracy"] == pytest.approx(4 / 6 * 100)
    assert stats["dictionary_lookups"] == 5


def test_daily_goal_and_progress_fields_present():
    progress = {}
    set_daily_goal(progress, 12)
    register_sentence_result(progress, "s1", True)
    register_flashcard_result(progress, "f1", False)
    stats = compute_overall(progress)

    assert stats["daily_goal"] == 12
    assert stats["today_attempts"] == 2
    assert stats["today_correct"] == 1
    assert stats["streak"] == 1
    assert stats["best_streak"] == 1


def test_srs_weight_boosts_new_and_due_items():
    progress = {}
    assert get_srs_weight(progress, "sentences", "s1") > 1.0

    register_sentence_result(progress, "s1", True)
    weight_after = get_srs_weight(progress, "sentences", "s1")
    assert weight_after < 1.0


def test_recommend_mode_prioritizes_weakest_area():
    progress = {
        "sentences": {"s1": {"seen": 10, "correct": 2}},
        "flashcards": {"f1": {"seen": 10, "correct": 9}},
        "to_be": {"t1": {"seen": 10, "correct": 8}},
    }
    assert recommend_practice_mode(progress) == "sentences"


def test_daily_goal_status_defaults():
    status = daily_goal_status({})
    assert status["goal"] >= 1
    assert status["today_attempts"] == 0
