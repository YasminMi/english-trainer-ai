from __future__ import annotations

import random
from collections import Counter
from datetime import date

import pytest

from english_trainer.domain.progress import default_progress
from english_trainer.domain.selection import accuracy_weight, item_key, weighted_choice


def test_accuracy_weight_prefers_unseen_then_weak_items() -> None:
    assert accuracy_weight(0, 0) > accuracy_weight(10, 2) > accuracy_weight(10, 10)
    assert accuracy_weight(10, 10) >= 0.3


def test_item_key_falls_back_for_legacy_items() -> None:
    assert item_key({"id": "s1", "en": "x"}) == "s1"
    assert item_key({"en": "Hello"}) == "Hello"
    assert item_key({"question": "She ___ ready."}) == "She ___ ready."


def test_weighted_choice_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        weighted_choice([], default_progress(), "sentences")


def test_weighted_choice_favours_items_with_more_mistakes() -> None:
    progress = default_progress()
    progress["sentences"] = {
        "known": {"seen": 20, "correct": 20},
        "weak": {"seen": 20, "correct": 2},
    }
    items = [{"id": "known"}, {"id": "weak"}]
    rng = random.Random(7)

    picks = Counter(
        weighted_choice(items, progress, "sentences", today=date(2026, 6, 1), rng=rng)["id"]
        for _ in range(500)
    )

    assert picks["weak"] > picks["known"] * 2
