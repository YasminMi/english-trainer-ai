"""Flashcard mode: word → meaning with SRS-weighted card selection."""

import random
from typing import Any

from .progress import get_srs_weight, register_flashcard_result
from .utils import clear_screen, is_close_answer, pause, read_choice


def _filter(
    cards: list[dict[str, Any]],
    category: str | None,
    source_note: str | None,
) -> list[dict[str, Any]]:
    filtered = list(cards)
    if category is not None:
        filtered = [c for c in filtered if c.get("category") == category]
    if source_note is not None:
        filtered = [c for c in filtered if c.get("source_note") == source_note]
    return filtered


def _weighted_choice(
    cards: list[dict[str, Any]],
    progress_section: dict[str, Any],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick a card, favouring unseen and low-accuracy items."""
    if len(cards) == 1:
        return cards[0]

    weights = []
    for card in cards:
        card_id = card.get("id", card.get("en", ""))
        stats = progress_section.get(str(card_id), {"seen": 0, "correct": 0})
        seen = int(stats.get("seen", 0))
        correct = int(stats.get("correct", 0))
        if seen == 0:
            weight = 3.0
        else:
            accuracy = correct / seen
            weight = max(0.3, 2.5 - accuracy * 2.0)
        if progress is not None:
            weight *= get_srs_weight(progress, "flashcards", str(card_id))
        weights.append(weight)

    return random.choices(cards, weights=weights, k=1)[0]


def run_flashcards_session(
    content: dict[str, Any],
    progress: dict[str, Any],
    category: str | None = None,
    source_note: str | None = None,
) -> None:
    cards = content.get("flashcards", [])
    if not cards:
        print("No flashcards found.")
        pause()
        return

    cards = _filter(cards, category, source_note)
    if not cards:
        print("No flashcards for this category.")
        pause()
        return

    progress_flashcards = progress.setdefault("flashcards", {})
    session_correct = 0
    session_total = 0

    while True:
        clear_screen()
        print("=== Flashcards ===")
        if category:
            print(f"Category: {category}")
        if source_note:
            label = source_note.split("/")[-1].replace(".md", "").replace("_", " ").title()
            print(f"Topic: {label}")
        if session_total > 0:
            pct = int(session_correct / session_total * 100)
            print(f"Session: {session_correct}/{session_total} ({pct}%)")
        print("Type the meaning in Portuguese. Type '0' to go back.\n")

        card: dict[str, Any] = _weighted_choice(cards, progress_flashcards, progress=progress)
        en = card["en"]
        pt = card["pt"]
        card_id = card.get("id", en)

        print(f"Word: {en}")
        user: str = read_choice("Meaning (PT): ")
        if user == "0":
            break

        correct: bool = is_close_answer(user, pt)
        register_flashcard_result(progress, card_id, correct)
        session_total += 1
        if correct:
            session_correct += 1

        if correct:
            print("Correct! ✅")
        else:
            print("Not quite. ❌")
            print(f"Correct: {pt}")

        pause()

    if session_total > 0:
        pct = int(session_correct / session_total * 100)
        print(f"\nSession result: {session_correct}/{session_total} ({pct}%)")
        pause("Press ENTER to go back...")
