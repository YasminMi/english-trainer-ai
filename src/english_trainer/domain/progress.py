"""Progress rules: per-item stats, daily goal, streaks and spaced repetition (SM-2 variant).

Every function takes an optional ``today`` so behaviour is deterministic in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from english_trainer.domain.types import (
    PRACTICE_MODES,
    DailyState,
    DayStats,
    ItemStats,
    PracticeMode,
    Progress,
    SrsState,
)

DEFAULT_DAILY_GOAL = 20

# Spaced repetition parameters.
INITIAL_EASE = 2.3
MIN_EASE = 1.3
MAX_EASE = 2.7
EASE_BONUS = 0.03
EASE_PENALTY = 0.2
FIRST_INTERVALS = (1, 3)

# Selection weights derived from SRS state.
NEW_ITEM_WEIGHT = 1.35
DUE_BASE_WEIGHT = 1.8
DUE_WEIGHT_PER_DAY = 0.15
MAX_DUE_WEIGHT = 2.8
MIN_NOT_DUE_WEIGHT = 0.35


def default_progress() -> Progress:
    return {
        "sentences": {},
        "flashcards": {},
        "to_be": {},
        "dictionary_lookups": 0,
        "ai_uses": 0,
        "srs": {},
        "daily": _default_daily(),
    }


def _default_daily() -> DailyState:
    return {
        "goal": DEFAULT_DAILY_GOAL,
        "history": {},
        "streak": 0,
        "best_streak": 0,
        "last_study_date": None,
    }


def normalize_progress(raw: dict[str, Any]) -> Progress:
    """Fill missing keys so older or partial progress files stay readable."""
    progress: Any = dict(raw)
    for key, value in default_progress().items():
        progress.setdefault(key, value)
    daily = progress["daily"] if isinstance(progress["daily"], dict) else {}
    for key, value in _default_daily().items():
        daily.setdefault(key, value)
    progress["daily"] = daily
    normalized: Progress = progress
    return normalized


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _ensure_daily(progress: Progress) -> DailyState:
    daily = progress.setdefault("daily", _default_daily())
    for key, value in _default_daily().items():
        daily.setdefault(key, value)  # type: ignore[misc]
    return daily


# --------------------------------------------------------------------------- item results


def update_item_stats(section: dict[str, ItemStats], item_id: str, correct: bool) -> None:
    stats = section.setdefault(item_id, {"seen": 0, "correct": 0})
    stats["seen"] += 1
    if correct:
        stats["correct"] += 1


def _update_streak(progress: Progress, today: date) -> None:
    daily = _ensure_daily(progress)
    last_day = _parse_date(daily.get("last_study_date"))
    if last_day == today:
        return

    if last_day and today == last_day + timedelta(days=1):
        daily["streak"] = int(daily.get("streak", 0)) + 1
    else:
        daily["streak"] = 1
    daily["best_streak"] = max(int(daily.get("best_streak", 0)), daily["streak"])
    daily["last_study_date"] = today.isoformat()


def _register_daily_attempt(
    progress: Progress, mode: PracticeMode, correct: bool, today: date
) -> None:
    history = _ensure_daily(progress)["history"]
    empty: DayStats = {"attempts": 0, "correct": 0, "sentences": 0, "flashcards": 0, "to_be": 0}
    day_stats = history.setdefault(today.isoformat(), empty)
    day_stats["attempts"] = int(day_stats.get("attempts", 0)) + 1
    day_stats["correct"] = int(day_stats.get("correct", 0)) + int(correct)
    day_stats[mode] = int(day_stats.get(mode, 0)) + 1
    _update_streak(progress, today)


def _update_srs(
    progress: Progress, mode: PracticeMode, item_id: str, correct: bool, today: date
) -> None:
    initial: SrsState = {
        "interval_days": 0,
        "ease": INITIAL_EASE,
        "repetitions": 0,
        "due_date": today.isoformat(),
        "last_reviewed": None,
        "last_result": None,
    }
    state = progress.setdefault("srs", {}).setdefault(mode, {}).setdefault(item_id, initial)

    interval = int(state.get("interval_days", 0))
    ease = float(state.get("ease", INITIAL_EASE))
    repetitions = int(state.get("repetitions", 0))

    if correct:
        repetitions += 1
        if repetitions <= len(FIRST_INTERVALS):
            interval = FIRST_INTERVALS[repetitions - 1]
        else:
            interval = max(1, round(interval * ease))
        ease = min(MAX_EASE, ease + EASE_BONUS)
    else:
        repetitions = 0
        interval = 1
        ease = max(MIN_EASE, ease - EASE_PENALTY)

    state["interval_days"] = interval
    state["ease"] = round(ease, 3)
    state["repetitions"] = repetitions
    state["due_date"] = (today + timedelta(days=interval)).isoformat()
    state["last_reviewed"] = today.isoformat()
    state["last_result"] = bool(correct)


def register_result(
    progress: Progress,
    mode: PracticeMode,
    item_id: str,
    correct: bool,
    *,
    today: date | None = None,
) -> None:
    """Record one practice attempt: item stats, daily history, streak and SRS."""
    day = today or date.today()
    update_item_stats(progress.setdefault(mode, {}), item_id, correct)
    _register_daily_attempt(progress, mode, correct, day)
    _update_srs(progress, mode, item_id, correct, day)


def register_dictionary_lookup(progress: Progress) -> None:
    progress["dictionary_lookups"] = int(progress.get("dictionary_lookups", 0)) + 1


def register_ai_use(progress: Progress) -> None:
    progress["ai_uses"] = int(progress.get("ai_uses", 0)) + 1


def register_conversation(progress: Progress, turns: int) -> None:
    progress["conversation_sessions"] = int(progress.get("conversation_sessions", 0)) + 1
    progress["conversation_turns"] = int(progress.get("conversation_turns", 0)) + turns


# --------------------------------------------------------------------------- scheduling


def srs_weight(
    progress: Progress, mode: PracticeMode, item_id: str, *, today: date | None = None
) -> float:
    """Selection multiplier: new and overdue items weigh more than not-yet-due ones."""
    state = progress.get("srs", {}).get(mode, {}).get(item_id)
    if not state:
        return NEW_ITEM_WEIGHT

    due = _parse_date(state.get("due_date"))
    if due is None:
        return 1.0

    days_overdue = ((today or date.today()) - due).days
    if days_overdue >= 0:
        return min(MAX_DUE_WEIGHT, DUE_BASE_WEIGHT + DUE_WEIGHT_PER_DAY * days_overdue)
    return max(MIN_NOT_DUE_WEIGHT, 1.0 + 0.1 * days_overdue)


def count_due_reviews(progress: Progress, *, today: date | None = None) -> dict[str, int]:
    day = today or date.today()
    counts = {"sentences": 0, "flashcards": 0, "to_be": 0, "total": 0}
    srs = progress.get("srs", {})
    for mode in PRACTICE_MODES:
        due = sum(
            1
            for state in srs.get(mode, {}).values()
            if (due_date := _parse_date(state.get("due_date"))) and due_date <= day
        )
        counts[mode] = due
        counts["total"] += due
    return counts


def recommend_practice_mode(progress: Progress) -> PracticeMode:
    """Suggest the mode that mixes the weakest accuracy with the most overdue reviews."""
    due = count_due_reviews(progress)

    def score(mode: PracticeMode) -> float:
        seen, correct = _totals(progress.get(mode, {}))
        weakness = (1.0 - correct / seen) if seen else 0.6
        overdue = min(1.0, due.get(mode, 0) / 20)
        return 0.65 * weakness + 0.35 * overdue

    return max(PRACTICE_MODES, key=score)


# --------------------------------------------------------------------------- goals & summary


def set_daily_goal(progress: Progress, goal: int) -> int:
    value = max(1, int(goal))
    _ensure_daily(progress)["goal"] = value
    return value


@dataclass(frozen=True, slots=True)
class DailyGoalStatus:
    goal: int
    today_attempts: int
    today_correct: int
    streak: int
    best_streak: int

    @property
    def today_accuracy(self) -> float:
        return self.today_correct / self.today_attempts * 100.0 if self.today_attempts else 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.goal - self.today_attempts)

    @property
    def progress_pct(self) -> float:
        return min(100.0, self.today_attempts / self.goal * 100.0)


def daily_goal_status(progress: Progress, *, today: date | None = None) -> DailyGoalStatus:
    daily = _ensure_daily(progress)
    day_stats = daily.get("history", {}).get((today or date.today()).isoformat())
    return DailyGoalStatus(
        goal=max(1, int(daily.get("goal", DEFAULT_DAILY_GOAL))),
        today_attempts=int(day_stats.get("attempts", 0)) if day_stats else 0,
        today_correct=int(day_stats.get("correct", 0)) if day_stats else 0,
        streak=int(daily.get("streak", 0)),
        best_streak=int(daily.get("best_streak", 0)),
    )


def _totals(section: dict[str, ItemStats]) -> tuple[int, int]:
    seen = sum(int(stats.get("seen", 0)) for stats in section.values())
    correct = sum(int(stats.get("correct", 0)) for stats in section.values())
    return seen, correct


def compute_overall(progress: Progress, *, today: date | None = None) -> dict[str, float]:
    """Flat summary used by the progress screens and report exports."""
    per_mode = {mode: _totals(progress.get(mode, {})) for mode in PRACTICE_MODES}
    total_seen = sum(seen for seen, _ in per_mode.values())
    total_correct = sum(correct for _, correct in per_mode.values())
    daily = daily_goal_status(progress, today=today)
    due = count_due_reviews(progress, today=today)

    summary: dict[str, float] = {
        "total_seen": total_seen,
        "total_correct": total_correct,
        "accuracy": (total_correct / total_seen * 100) if total_seen else 0.0,
    }
    for mode, (seen, correct) in per_mode.items():
        summary[f"{mode}_seen"] = seen
        summary[f"{mode}_correct"] = correct
    summary.update(
        {
            "dictionary_lookups": int(progress.get("dictionary_lookups", 0)),
            "ai_uses": int(progress.get("ai_uses", 0)),
            "daily_goal": daily.goal,
            "today_attempts": daily.today_attempts,
            "today_correct": daily.today_correct,
            "today_accuracy": daily.today_accuracy,
            "today_remaining": daily.remaining,
            "goal_progress_pct": daily.progress_pct,
            "streak": daily.streak,
            "best_streak": daily.best_streak,
            "due_reviews_total": due["total"],
            "due_reviews_sentences": due["sentences"],
            "due_reviews_flashcards": due["flashcards"],
            "due_reviews_to_be": due["to_be"],
        }
    )
    return summary
