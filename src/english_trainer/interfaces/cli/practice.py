"""Offline practice loops: to be drills, flashcards, sentence training and dictation."""

from __future__ import annotations

from english_trainer.app import Application
from english_trainer.domain.content import available_sources, filter_items, note_label
from english_trainer.domain.hints import detect_tense_hint
from english_trainer.domain.matching import is_close_answer, normalize_pt_phrase
from english_trainer.domain.selection import item_key, weighted_choice
from english_trainer.domain.types import Flashcard, Sentence
from english_trainer.interfaces.cli.io import ask, choose, header, pause, session_line
from english_trainer.services.audio import RATE_NORMAL, RATE_SLOW, audio_available, speak


class _Score:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def add(self, correct: bool) -> None:
        self.total += 1
        self.correct += int(correct)

    def summary(self) -> None:
        if self.total:
            print(f"\nResult — {session_line(self.correct, self.total)}")
            pause("Press ENTER to go back...")


def _pick_note(title: str, items: list[Sentence]) -> str | None:
    """Return the chosen note path, ``""`` for every item, or ``None`` to go back."""
    options = [("All sentences", "")] + [(note_label(n), n) for n in available_sources(items)]
    return choose(title, options)


def run_to_be_trainer(app: Application) -> None:
    exercises = app.content["to_be_exercises"]
    if not exercises:
        pause("No to-be exercises configured.")
        return

    score = _Score()
    while True:
        header("To be Trainer")
        print("Choose the correct form to complete the sentence. Type '0' to go back.\n")
        exercise = weighted_choice(exercises, app.progress, "to_be")
        print(f"Sentence: {exercise['question']}")
        for index, option in enumerate(exercise["options"], start=1):
            print(f"{index}) {option}")

        choice = ask("Your choice (number): ")
        if choice == "0":
            break
        if not choice.isdigit() or not 1 <= int(choice) <= len(exercise["options"]):
            pause("Invalid option.")
            continue

        correct = exercise["options"][int(choice) - 1] == exercise["correct"]
        app.record_result("to_be", item_key(exercise), correct)
        score.add(correct)
        print(
            "Good job! ✅" if correct else f"Try again! ❌\nCorrect answer: {exercise['correct']}"
        )
        print(f"Meaning (PT): {exercise['pt']}")
        if explanation := exercise.get("explanation"):
            print(f"Explanation: {explanation}")
        pause()
    score.summary()


def flashcards_menu(app: Application) -> None:
    cards = app.content["flashcards"]
    categories = [str(c) for c in app.content["metadata"].get("flashcards_categories", []) or []]
    options: list[tuple[str, tuple[str | None, str | None]]] = [("All words", (None, None))]
    options += [(f"{c.replace('_', ' ').capitalize()} only", (c, None)) for c in categories]
    options += [(f"Note: {note_label(n)}", (None, n)) for n in available_sources(cards)]

    while (picked := choose("Flashcards", options)) is not None:
        category, source_note = picked
        run_flashcards(app, filter_items(cards, category=category, source_note=source_note))


def run_flashcards(app: Application, cards: list[Flashcard]) -> None:
    if not cards:
        pause("No flashcards for this selection.")
        return

    score = _Score()
    while True:
        header("Flashcards")
        if score.total:
            print(session_line(score.correct, score.total))
        print("Type the meaning in Portuguese. Type '0' to go back.\n")
        card = weighted_choice(cards, app.progress, "flashcards")
        print(f"Word: {card['en']}")
        answer = ask("Meaning (PT): ")
        if answer == "0":
            break

        correct = is_close_answer(answer, card["pt"])
        app.record_result("flashcards", item_key(card), correct)
        score.add(correct)
        print("Correct! ✅" if correct else f"Not quite. ❌\nCorrect: {card['pt']}")
        pause()
    score.summary()


def sentence_menu(app: Application) -> None:
    sentences = app.content["sentences"]
    while (note := _pick_note("Sentence Training", sentences)) is not None:
        run_sentence_training(app, filter_items(sentences, source_note=note or None))


def run_sentence_training(app: Application, sentences: list[Sentence]) -> None:
    if not sentences:
        pause("No sentences found for this selection.")
        return

    score = _Score()
    while True:
        header("Sentence Training")
        if score.total:
            print(session_line(score.correct, score.total))
        print("Tip: type the meaning in Portuguese OR repeat the sentence in English.")
        print("Type '0' to go back.\n")
        sentence = weighted_choice(sentences, app.progress, "sentences")
        expected_pt = normalize_pt_phrase(sentence["pt"])
        print(f"Sentence: {sentence['en']}")
        answer = ask("Your answer: ")
        if answer == "0":
            break

        correct = is_close_answer(answer, [sentence["pt"], expected_pt, sentence["en"]])
        app.record_result("sentences", item_key(sentence), correct)
        score.add(correct)
        if correct:
            print("Correct! ✅")
        else:
            print("Not quite. ❌")
            if hint := detect_tense_hint(sentence["en"], answer):
                print(hint)
            print(f"Expected (PT): {expected_pt}")
        pause()
    score.summary()


def dictation_menu(app: Application) -> None:
    sentences = app.content["sentences"]
    while (note := _pick_note("Dictation", sentences)) is not None:
        run_dictation(app, filter_items(sentences, source_note=note or None))


def _play_or_show(text: str, *, has_audio: bool, rate: str) -> None:
    if has_audio:
        print("[Playing audio... listen carefully]\n")
        if speak(text, rate=rate):
            return
        print("[Audio failed]")
    print(f"Sentence: {text}\n")


def run_dictation(app: Application, sentences: list[Sentence]) -> None:
    if not sentences:
        pause("No sentences available.")
        return

    has_audio = audio_available()
    rate = RATE_NORMAL
    if not has_audio:
        pause("Audio not available (install the 'audio' extra). Showing sentences instead.")
    elif ask("Speed — 1) Normal  2) Slow: ") == "2":
        rate = RATE_SLOW

    score = _Score()
    while True:
        header("Dictation")
        if score.total:
            print(session_line(score.correct, score.total))
        print("Type what you hear in English. Type '0' to go back.\n")
        sentence = weighted_choice(sentences, app.progress, "sentences")

        _play_or_show(sentence["en"], has_audio=has_audio, rate=rate)
        answer = ask("Type what you heard (or 'replay' / '0' to quit): ")
        if answer.lower() == "replay":
            _play_or_show(sentence["en"], has_audio=has_audio, rate=rate)
            answer = ask("Type what you heard: ")
        if answer == "0":
            break

        correct = is_close_answer(answer, sentence["en"], strict=True)
        app.record_result("sentences", item_key(sentence), correct)
        score.add(correct)
        print("✅ Correct!" if correct else f"❌ The sentence was: {sentence['en']}")
        print(f"Meaning (PT): {sentence['pt']}")
        pause()
    score.summary()
