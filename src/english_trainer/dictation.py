"""Dictation: listen to a sentence, type what you heard."""

import random
from typing import Any

from .audio import RATE_NORMAL, RATE_SLOW, audio_available, speak
from .progress import get_srs_weight, register_sentence_result
from .utils import clear_screen, is_close_answer, pause, read_choice


def _weighted_choice(
    sentences: list[dict[str, Any]],
    progress_sentences: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    if len(sentences) == 1:
        return sentences[0]

    weights = []
    for s in sentences:
        sid = str(s.get("id", s.get("en", "")))
        stats = progress_sentences.get(sid, {"seen": 0, "correct": 0})
        seen = int(stats.get("seen", 0))
        correct = int(stats.get("correct", 0))
        weight = 3.0 if seen == 0 else max(0.3, 2.5 - (correct / seen) * 2.0)
        weight *= get_srs_weight(progress, "sentences", sid)
        weights.append(weight)

    return random.choices(sentences, weights=weights, k=1)[0]


def run_dictation(
    content: dict[str, Any],
    progress: dict[str, Any],
    source_note: str | None = None,
) -> None:
    sentences = content.get("sentences", [])
    if source_note:
        sentences = [s for s in sentences if s.get("source_note") == source_note]
    if not sentences:
        print("No sentences available.")
        pause()
        return

    has_audio = audio_available()
    if not has_audio:
        print("Audio not available (edge-tts/pygame not installed).")
        print("Running in reading mode: sentence will be shown instead.")
        pause()

    rate = RATE_NORMAL
    if has_audio:
        speed = read_choice("Speed — 1) Normal  2) Devagar (slow): ").strip()
        if speed == "2":
            rate = RATE_SLOW

    progress_sentences = progress.setdefault("sentences", {})
    session_correct = 0
    session_total = 0

    while True:
        clear_screen()
        print("=== Dictation ===")
        if has_audio:
            print("Listen and type what you hear in English.")
        else:
            print("Read and repeat the sentence in English.")
        if session_total > 0:
            pct = int(session_correct / session_total * 100)
            print(f"Session: {session_correct}/{session_total} ({pct}%)")
        print("Type '0' to go back.\n")

        s = _weighted_choice(sentences, progress_sentences, progress)
        en = s["en"]
        pt = s["pt"]
        sid = str(s.get("id", en))

        if has_audio:
            print("[Playing audio... listen carefully]\n")
            success = speak(en, rate=rate)
            if not success:
                print(f"[Audio failed — showing text: {en}]\n")
        else:
            print(f"Sentence: {en}\n")

        answer = read_choice("Type what you heard (or 'replay' / '0' to quit): ").strip()

        if answer == "0":
            break

        if answer.lower() == "replay":
            if has_audio:
                speak(en, rate=rate)
            else:
                print(f"Sentence: {en}")
            answer = read_choice("Type what you heard: ").strip()
            if answer == "0":
                break

        correct = is_close_answer(answer, en, strict=True)
        register_sentence_result(progress, sid, correct)
        session_total += 1
        if correct:
            session_correct += 1

        if correct:
            print("✅ Correct!")
        else:
            print(f"❌ The sentence was: {en}")
        print(f"Meaning (PT): {pt}")
        pause()

    if session_total > 0:
        pct = int(session_correct / session_total * 100)
        print(f"\nSession result: {session_correct}/{session_total} ({pct}%)")
        pause("Press ENTER to go back...")
