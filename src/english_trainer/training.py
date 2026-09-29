"""Sentence training: translate or repeat sentences, with SRS-weighted selection."""

import random
import re
from typing import Any, Dict, List, Optional

from .progress import register_sentence_result, get_srs_weight
from .utils import clear_screen, is_close_answer, normalize_pt_phrase, pause, read_choice, _strip_accents


def _detect_tense_hint(en_sentence: str, user_answer: str) -> Optional[str]:
    user = _strip_accents(normalize_pt_phrase(user_answer).lower())
    english = en_sentence.lower()

    past_en_markers = (" was ", " were ", " did ", " did not ", " yesterday", " last ", " ago")
    present_en_markers = (" am ", " is ", " are ", " do ", " does ", " do not ", " does not ", " now", " today")

    has_past_en = any(marker in f" {english} " for marker in past_en_markers)
    has_present_en = any(marker in f" {english} " for marker in present_en_markers)

    has_past_pt = bool(re.search(r"\b(estava|estavam|foi|foram|fez|fizeram|era|eram|trabalhei|estudou|foi|assistiu|fui)\b", user))
    has_present_pt = bool(re.search(r"\b(esta|estao|estou|faz|fazem|e|sao|trabalho|estudo|assisto)\b", user))

    if has_present_en and has_past_pt:
        return "Hint: your answer looks like past tense, but the sentence is in the present."
    if has_past_en and has_present_pt:
        return "Hint: your answer looks like present tense, but the sentence is in the past."
    return None


def _weighted_choice(
    items: List[Dict[str, Any]],
    progress_section: Dict[str, Any],
    progress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pick an item, giving higher weight to unseen and low-accuracy items."""
    if len(items) == 1:
        return items[0]

    weights = []
    for item in items:
        item_id = item.get("id", item.get("en", ""))
        stats = progress_section.get(str(item_id), {"seen": 0, "correct": 0})
        seen = int(stats.get("seen", 0))
        correct = int(stats.get("correct", 0))
        if seen == 0:
            weight = 3.0
        else:
            accuracy = correct / seen
            weight = max(0.3, 2.5 - accuracy * 2.0)
        if progress is not None:
            weight *= get_srs_weight(progress, "sentences", str(item_id))
        weights.append(weight)

    return random.choices(items, weights=weights, k=1)[0]


def run_sentence_training(
    content: Dict[str, Any],
    progress: Dict[str, Any],
    source_note: Optional[str] = None,
) -> None:
    sentences = content.get("sentences", [])
    if source_note:
        sentences = [item for item in sentences if item.get("source_note") == source_note]
    if not sentences:
        print("No sentences found for this selection.")
        pause()
        return

    progress_sentences = progress.setdefault("sentences", {})
    session_correct = 0
    session_total = 0

    while True:
        clear_screen()
        print("=== Sentence Training ===")
        if source_note:
            label = source_note.split("/")[-1].replace(".md", "").replace("_", " ").title()
            print(f"Topic: {label}")
        if session_total > 0:
            pct = int(session_correct / session_total * 100)
            print(f"Session: {session_correct}/{session_total} ({pct}%)")
        print("Tip: type the meaning in Portuguese OR repeat the sentence in English.")
        print("Type '0' to go back.\n")

        s = _weighted_choice(sentences, progress_sentences, progress=progress)
        en = s["en"]
        pt = s["pt"]
        pt_display = normalize_pt_phrase(pt)
        sid = s.get("id", en)

        print(f"Sentence: {en}")
        user: str = read_choice("Your answer: ")
        if user == "0":
            break

        correct: bool = is_close_answer(user, [pt, pt_display, en])
        register_sentence_result(progress, sid, correct)
        session_total += 1
        if correct:
            session_correct += 1

        if correct:
            print("Correct! ✅")
        else:
            print("Not quite. ❌")
            hint = _detect_tense_hint(en, user)
            if hint:
                print(hint)
            print(f"Expected (PT): {pt_display}")
            print(f"Sentence (EN): {en}")

        pause()

    if session_total > 0:
        pct = int(session_correct / session_total * 100)
        print(f"\nSession result: {session_correct}/{session_total} ({pct}%)")
        pause("Press ENTER to go back...")
