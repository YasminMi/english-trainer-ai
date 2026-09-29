"""Pick the next study item, favouring unseen, weak and overdue items."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, TypeVar

from english_trainer.domain.progress import srs_weight
from english_trainer.domain.types import PracticeMode, Progress

T = TypeVar("T", bound=Mapping[str, Any])

UNSEEN_WEIGHT = 3.0
MIN_WEIGHT = 0.3


def item_key(item: Mapping[str, Any]) -> str:
    """Stable identifier for progress tracking; older items may lack an ``id``."""
    return str(item.get("id") or item.get("en") or item.get("question", ""))


def accuracy_weight(seen: int, correct: int) -> float:
    """Unseen items get the highest weight; weight decreases as accuracy grows."""
    if seen <= 0:
        return UNSEEN_WEIGHT
    return max(MIN_WEIGHT, 2.5 - (correct / seen) * 2.0)


def weighted_choice(
    items: Sequence[T],
    progress: Progress,
    mode: PracticeMode,
    *,
    today: date | None = None,
    rng: random.Random | None = None,
    key: Callable[[Mapping[str, Any]], str] = item_key,
) -> T:
    """Choose one item using accuracy and spaced-repetition weights.

    Raises:
        ValueError: if ``items`` is empty.
    """
    if not items:
        raise ValueError("Cannot choose from an empty item list.")
    if len(items) == 1:
        return items[0]

    section = progress.get(mode, {})
    weights = []
    for item in items:
        item_id = key(item)
        stats = section.get(item_id, {"seen": 0, "correct": 0})
        weight = accuracy_weight(int(stats.get("seen", 0)), int(stats.get("correct", 0)))
        weights.append(weight * srs_weight(progress, mode, item_id, today=today))

    return (rng or random).choices(list(items), weights=weights, k=1)[0]
