"""Progress tracking: per-item stats, daily goal, streaks, and spaced repetition."""

from __future__ import annotations

from datetime import date, timedelta

DEFAULT_DAILY_GOAL = 20
PRACTICE_MODES = ("sentences", "flashcards", "to_be")


def _update_item(section: dict[str, dict[str, int]], item_id: str, correct: bool) -> None:
    stats: dict[str, int] = section.setdefault(item_id, {"seen": 0, "correct": 0})
    stats["seen"] += 1
    if correct:
        stats["correct"] += 1


def _today_iso(when: date | None = None) -> str:
    return (when or date.today()).isoformat()


def _safe_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _ensure_daily(progress: dict) -> dict:
    daily = progress.setdefault("daily", {})
    daily.setdefault("goal", DEFAULT_DAILY_GOAL)
    daily.setdefault("history", {})
    daily.setdefault("streak", 0)
    daily.setdefault("best_streak", 0)
    daily.setdefault("last_study_date", None)
    return daily


def _update_streak(progress: dict, when: date | None = None) -> None:
    day = when or date.today()
    day_iso = day.isoformat()
    daily = _ensure_daily(progress)
    last_day = _safe_date(daily.get("last_study_date"))

    if last_day == day:
        return

    if last_day and day == last_day + timedelta(days=1):
        daily["streak"] = int(daily.get("streak", 0)) + 1
    else:
        daily["streak"] = 1

    daily["best_streak"] = max(int(daily.get("best_streak", 0)), int(daily["streak"]))
    daily["last_study_date"] = day_iso


def _register_daily_attempt(
    progress: dict, mode: str, correct: bool, when: date | None = None
) -> None:
    day_iso = _today_iso(when)
    daily = _ensure_daily(progress)
    history = daily.setdefault("history", {})
    day_stats = history.setdefault(
        day_iso,
        {
            "attempts": 0,
            "correct": 0,
            "sentences": 0,
            "flashcards": 0,
            "to_be": 0,
        },
    )
    day_stats["attempts"] = int(day_stats.get("attempts", 0)) + 1
    day_stats["correct"] = int(day_stats.get("correct", 0)) + (1 if correct else 0)
    if mode in PRACTICE_MODES:
        day_stats[mode] = int(day_stats.get(mode, 0)) + 1
    _update_streak(progress, when)


def _ensure_srs(progress: dict, mode: str) -> dict:
    srs = progress.setdefault("srs", {})
    return srs.setdefault(mode, {})


def _update_srs(
    progress: dict, mode: str, item_id: str, correct: bool, when: date | None = None
) -> None:
    day = when or date.today()
    mode_section = _ensure_srs(progress, mode)
    state = mode_section.setdefault(
        item_id,
        {
            "interval_days": 0,
            "ease": 2.3,
            "repetitions": 0,
            "due_date": day.isoformat(),
            "last_reviewed": None,
            "last_result": None,
        },
    )

    interval = int(state.get("interval_days", 0))
    ease = float(state.get("ease", 2.3))
    repetitions = int(state.get("repetitions", 0))

    if correct:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 3
        else:
            interval = max(1, int(round(interval * ease)))
        ease = min(2.7, ease + 0.03)
    else:
        repetitions = 0
        interval = 1
        ease = max(1.3, ease - 0.2)

    due_day = day + timedelta(days=interval)
    state["interval_days"] = interval
    state["ease"] = round(ease, 3)
    state["repetitions"] = repetitions
    state["due_date"] = due_day.isoformat()
    state["last_reviewed"] = day.isoformat()
    state["last_result"] = bool(correct)


def get_srs_weight(progress: dict, mode: str, item_id: str, when: date | None = None) -> float:
    mode_srs = progress.get("srs", {}).get(mode, {})
    state = mode_srs.get(item_id)
    if not state:
        return 1.35

    due = _safe_date(state.get("due_date"))
    if due is None:
        return 1.0

    day = when or date.today()
    days_overdue = (day - due).days
    if days_overdue >= 0:
        return min(2.8, 1.8 + 0.15 * days_overdue)
    return max(0.35, 1.0 + 0.1 * days_overdue)


def count_due_reviews(progress: dict, when: date | None = None) -> dict[str, int]:
    day = when or date.today()
    out = {"sentences": 0, "flashcards": 0, "to_be": 0, "total": 0}
    srs = progress.get("srs", {})

    for mode in PRACTICE_MODES:
        mode_srs = srs.get(mode, {})
        due_count = 0
        for state in mode_srs.values():
            due = _safe_date(state.get("due_date"))
            if due and due <= day:
                due_count += 1
        out[mode] = due_count
        out["total"] += due_count
    return out


def recommend_practice_mode(progress: dict) -> str:
    stats = compute_overall(progress)
    due = count_due_reviews(progress)

    candidates = []
    for mode in PRACTICE_MODES:
        seen = int(stats.get(f"{mode}_seen", 0))
        correct = int(stats.get(f"{mode}_correct", 0))
        acc = (correct / seen) if seen else 0.0
        due_score = min(1.0, due.get(mode, 0) / 20)
        weak_score = (1.0 - acc) if seen else 0.6
        score = 0.65 * weak_score + 0.35 * due_score  # weak items + overdue reviews
        candidates.append((score, mode))

    candidates.sort(reverse=True)
    return candidates[0][1]


def set_daily_goal(progress: dict, goal: int) -> int:
    value = max(1, int(goal))
    daily = _ensure_daily(progress)
    daily["goal"] = value
    return value


def daily_goal_status(progress: dict, when: date | None = None) -> dict[str, float]:
    day_iso = _today_iso(when)
    daily = _ensure_daily(progress)
    history = daily.get("history", {})
    today = history.get(day_iso, {})
    attempts = int(today.get("attempts", 0))
    correct = int(today.get("correct", 0))
    goal = max(1, int(daily.get("goal", DEFAULT_DAILY_GOAL)))
    pct = min(100.0, attempts / goal * 100.0)
    return {
        "goal": goal,
        "today_attempts": attempts,
        "today_correct": correct,
        "today_accuracy": (correct / attempts * 100.0) if attempts else 0.0,
        "remaining": max(0, goal - attempts),
        "goal_progress_pct": pct,
        "streak": int(daily.get("streak", 0)),
        "best_streak": int(daily.get("best_streak", 0)),
    }


def register_sentence_result(progress: dict, sentence_id: str, correct: bool) -> None:
    _update_item(progress.setdefault("sentences", {}), sentence_id, correct)
    _register_daily_attempt(progress, "sentences", correct)
    _update_srs(progress, "sentences", sentence_id, correct)


def register_flashcard_result(progress: dict, flashcard_id: str, correct: bool) -> None:
    _update_item(progress.setdefault("flashcards", {}), flashcard_id, correct)
    _register_daily_attempt(progress, "flashcards", correct)
    _update_srs(progress, "flashcards", flashcard_id, correct)


def register_to_be_result(progress: dict, exercise_id: str, correct: bool) -> None:
    _update_item(progress.setdefault("to_be", {}), exercise_id, correct)
    _register_daily_attempt(progress, "to_be", correct)
    _update_srs(progress, "to_be", exercise_id, correct)


def register_dictionary_lookup(progress: dict) -> None:
    progress["dictionary_lookups"] = int(progress.get("dictionary_lookups", 0)) + 1


def register_ai_use(progress: dict) -> None:
    progress["ai_uses"] = int(progress.get("ai_uses", 0)) + 1


def _compute(section: dict[str, dict[str, int]]) -> tuple[int, int]:
    seen = 0
    correct = 0
    for v in section.values():
        seen += int(v.get("seen", 0))
        correct += int(v.get("correct", 0))
    return seen, correct


def compute_overall(progress: dict) -> dict[str, float]:
    s_seen, s_correct = _compute(progress.get("sentences", {}))
    f_seen, f_correct = _compute(progress.get("flashcards", {}))
    t_seen, t_correct = _compute(progress.get("to_be", {}))

    total_seen: int = s_seen + f_seen + t_seen
    total_correct: int = s_correct + f_correct + t_correct
    acc: float = (total_correct / total_seen * 100) if total_seen else 0.0

    daily = daily_goal_status(progress)
    due = count_due_reviews(progress)

    return {
        "total_seen": total_seen,
        "total_correct": total_correct,
        "accuracy": acc,
        "sentences_seen": s_seen,
        "sentences_correct": s_correct,
        "flashcards_seen": f_seen,
        "flashcards_correct": f_correct,
        "to_be_seen": t_seen,
        "to_be_correct": t_correct,
        "dictionary_lookups": int(progress.get("dictionary_lookups", 0)),
        "ai_uses": int(progress.get("ai_uses", 0)),
        "daily_goal": daily["goal"],
        "today_attempts": daily["today_attempts"],
        "today_correct": daily["today_correct"],
        "today_accuracy": daily["today_accuracy"],
        "today_remaining": daily["remaining"],
        "goal_progress_pct": daily["goal_progress_pct"],
        "streak": daily["streak"],
        "best_streak": daily["best_streak"],
        "due_reviews_total": due["total"],
        "due_reviews_sentences": due["sentences"],
        "due_reviews_flashcards": due["flashcards"],
        "due_reviews_to_be": due["to_be"],
    }


def show_progress_summary(progress: dict) -> None:
    stats: dict[str, float] = compute_overall(progress)
    print("\n=== Progress Summary ===")
    print(f"Total attempts:      {stats['total_seen']}")
    print(f"Total correct:       {stats['total_correct']}")
    print(f"Accuracy:            {stats['accuracy']:.1f}%")
    print()
    print(f"Sentence Training:   {stats['sentences_correct']}/{stats['sentences_seen']}")
    print(f"Flashcards:          {stats['flashcards_correct']}/{stats['flashcards_seen']}")
    print(f"To be Trainer:       {stats['to_be_correct']}/{stats['to_be_seen']}")
    print(f"Dictionary lookups:  {stats['dictionary_lookups']}")
    print(f"Miu interactions:   {stats['ai_uses']}")
    print()
    print(
        f"Daily goal:          {int(stats['today_attempts'])}/{int(stats['daily_goal'])} "
        f"({stats['goal_progress_pct']:.0f}%)"
    )
    print(f"Current streak:      {int(stats['streak'])} day(s)")
    print(f"Best streak:         {int(stats['best_streak'])} day(s)")
    print(
        "Due reviews:         "
        f"{int(stats['due_reviews_total'])} "
        f"(S:{int(stats['due_reviews_sentences'])} "
        f"F:{int(stats['due_reviews_flashcards'])} "
        f"T:{int(stats['due_reviews_to_be'])})"
    )
    conv_sessions = int(progress.get("conversation_sessions", 0))
    conv_turns = int(progress.get("conversation_turns", 0))
    if conv_sessions > 0:
        print(f"Conversations:       {conv_sessions} sessions, {conv_turns} messages sent")
    print("========================")
    input("\nPress ENTER to go back...")
