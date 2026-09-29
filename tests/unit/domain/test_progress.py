from __future__ import annotations

from datetime import date, timedelta

import pytest

from english_trainer.domain.progress import (
    compute_overall,
    count_due_reviews,
    daily_goal_status,
    default_progress,
    normalize_progress,
    recommend_practice_mode,
    register_result,
    set_daily_goal,
    srs_weight,
    update_item_stats,
)
from english_trainer.domain.types import ItemStats

DAY = date(2026, 6, 1)


def test_update_item_stats_counts_attempts_and_hits() -> None:
    section: dict[str, ItemStats] = {}
    update_item_stats(section, "a", correct=True)
    update_item_stats(section, "a", correct=False)
    assert section["a"] == {"seen": 2, "correct": 1}


def test_normalize_progress_fills_missing_sections() -> None:
    progress = normalize_progress({"ai_uses": 3, "daily": {"goal": 5}})

    assert progress["ai_uses"] == 3
    assert progress["sentences"] == {}
    assert progress["daily"]["goal"] == 5
    assert progress["daily"]["streak"] == 0


def test_compute_overall_empty() -> None:
    stats = compute_overall(default_progress(), today=DAY)
    assert stats["total_seen"] == 0
    assert stats["accuracy"] == 0.0


def test_compute_overall_aggregates_every_mode() -> None:
    progress = default_progress()
    progress["sentences"] = {"s": {"seen": 2, "correct": 1}}
    progress["flashcards"] = {"f": {"seen": 3, "correct": 3}}
    progress["to_be"] = {"t": {"seen": 1, "correct": 0}}
    progress["dictionary_lookups"] = 5

    stats = compute_overall(progress, today=DAY)

    assert stats["total_seen"] == 6
    assert stats["total_correct"] == 4
    assert stats["accuracy"] == pytest.approx(4 / 6 * 100)
    assert stats["dictionary_lookups"] == 5


def test_register_result_updates_daily_goal_and_streak() -> None:
    progress = default_progress()
    set_daily_goal(progress, 12)

    register_result(progress, "sentences", "s1", correct=True, today=DAY)
    register_result(progress, "flashcards", "f1", correct=False, today=DAY)
    status = daily_goal_status(progress, today=DAY)

    assert status.goal == 12
    assert status.today_attempts == 2
    assert status.today_correct == 1
    assert status.remaining == 10
    assert status.today_accuracy == pytest.approx(50.0)
    assert status.streak == 1


def test_streak_grows_on_consecutive_days_and_resets_after_a_gap() -> None:
    progress = default_progress()
    for offset in range(3):
        register_result(progress, "to_be", "t", correct=True, today=DAY + timedelta(days=offset))
    assert daily_goal_status(progress, today=DAY + timedelta(days=2)).streak == 3

    register_result(progress, "to_be", "t", correct=True, today=DAY + timedelta(days=10))
    status = daily_goal_status(progress, today=DAY + timedelta(days=10))
    assert status.streak == 1
    assert status.best_streak == 3


def test_srs_intervals_grow_with_correct_answers_and_reset_on_mistakes() -> None:
    progress = default_progress()
    intervals = []
    for _ in range(4):
        register_result(progress, "sentences", "s1", correct=True, today=DAY)
        intervals.append(progress["srs"]["sentences"]["s1"]["interval_days"])
    assert intervals[:2] == [1, 3]
    assert intervals[3] > intervals[2] > intervals[1]

    register_result(progress, "sentences", "s1", correct=False, today=DAY)
    state = progress["srs"]["sentences"]["s1"]
    assert state["interval_days"] == 1
    assert state["repetitions"] == 0
    assert state["last_result"] is False


def test_srs_weight_boosts_new_and_overdue_items() -> None:
    progress = default_progress()
    assert srs_weight(progress, "sentences", "s1", today=DAY) > 1.0

    register_result(progress, "sentences", "s1", correct=True, today=DAY)
    assert srs_weight(progress, "sentences", "s1", today=DAY) < 1.0
    overdue = srs_weight(progress, "sentences", "s1", today=DAY + timedelta(days=5))
    assert overdue > srs_weight(progress, "sentences", "s1", today=DAY + timedelta(days=1))


def test_count_due_reviews() -> None:
    progress = default_progress()
    register_result(progress, "sentences", "s1", correct=True, today=DAY)
    register_result(progress, "to_be", "t1", correct=False, today=DAY)

    assert count_due_reviews(progress, today=DAY)["total"] == 0
    due = count_due_reviews(progress, today=DAY + timedelta(days=1))
    assert due == {"sentences": 1, "flashcards": 0, "to_be": 1, "total": 2}


def test_recommend_mode_prioritizes_weakest_area() -> None:
    progress = default_progress()
    progress["sentences"] = {"s1": {"seen": 10, "correct": 2}}
    progress["flashcards"] = {"f1": {"seen": 10, "correct": 9}}
    progress["to_be"] = {"t1": {"seen": 10, "correct": 8}}

    assert recommend_practice_mode(progress) == "sentences"


def test_daily_goal_is_at_least_one() -> None:
    progress = default_progress()
    assert set_daily_goal(progress, 0) == 1
    assert daily_goal_status(progress, today=DAY).progress_pct == 0.0
