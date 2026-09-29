"""Non-practice tools: dictionary, content editor, progress, study plan and reports."""

from __future__ import annotations

from typing import Literal

from english_trainer.app import Application
from english_trainer.domain.content import (
    DuplicateEntryError,
    add_dictionary_entry,
    add_flashcard,
    add_sentence,
    find_in_dictionary,
)
from english_trainer.domain.progress import (
    compute_overall,
    recommend_practice_mode,
    register_dictionary_lookup,
    set_daily_goal,
)
from english_trainer.domain.reporting import (
    build_weekly_report,
    export_progress_snapshot,
    format_weekly_report,
)
from english_trainer.interfaces.cli import practice
from english_trainer.interfaces.cli.io import ask, ask_positive_int, choose, header, pause
from english_trainer.services.dictionary import suggest_similar

MODE_LABELS = {
    "to_be": "To be Trainer",
    "flashcards": "Flashcards",
    "sentences": "Sentence Training",
}


def run_dictionary(app: Application) -> None:
    while True:
        header("Mini Dictionary")
        print("Local dictionary + online fallback. Type '0' to go back.\n")
        word = ask("Word: ")
        if word == "0":
            return
        if not word:
            continue
        register_dictionary_lookup(app.progress)

        if meaning := find_in_dictionary(app.content, word):
            pause(f"Meaning (PT): {meaning} ✅")
            continue

        entry = app.dictionary.lookup(word)
        if entry:
            print("Found online ✅")
            for label, value in (
                ("Phonetic", entry.phonetic),
                ("Type", entry.part_of_speech),
                ("Definition (EN)", entry.definition),
                ("Example", entry.example),
            ):
                if value:
                    print(f"{label}: {value}")
        else:
            print("Not found ❌")
            if suggestions := suggest_similar(word, app.content["dictionary"]):
                print("Did you mean:")
                for suggestion in suggestions:
                    print(f" - {suggestion}")
        pause()


def add_content_menu(app: Application) -> None:
    options = [("Sentence", "sentence"), ("Flashcard", "flashcard"), ("Dictionary entry", "word")]
    while (kind := choose("Add new content", options)) is not None:
        user_content = app.content_repository.load_user()
        try:
            if kind == "sentence":
                add_sentence(
                    app.content,
                    user_content,
                    en=ask("English sentence: "),
                    pt=ask("Portuguese meaning: "),
                    tags=ask("Tags (comma-separated, optional): ").split(","),
                )
            elif kind == "flashcard":
                add_flashcard(
                    app.content,
                    user_content,
                    en=ask("English word: "),
                    pt=ask("Portuguese meaning: "),
                    category=ask("Category (optional): ") or None,
                )
            else:
                add_dictionary_entry(
                    app.content,
                    user_content,
                    word=ask("English word: "),
                    meaning=ask("Portuguese meaning: "),
                )
        except (DuplicateEntryError, ValueError) as exc:
            pause(str(exc))
            continue
        app.content_repository.save_user(user_content)
        pause("Content saved locally.")


def show_progress(app: Application) -> None:
    stats = compute_overall(app.progress)
    header("Progress Summary")
    print(f"Total attempts:      {stats['total_seen']:.0f}")
    print(f"Total correct:       {stats['total_correct']:.0f}")
    print(f"Accuracy:            {stats['accuracy']:.1f}%\n")
    print(f"Sentence Training:   {stats['sentences_correct']:.0f}/{stats['sentences_seen']:.0f}")
    print(f"Flashcards:          {stats['flashcards_correct']:.0f}/{stats['flashcards_seen']:.0f}")
    print(f"To be Trainer:       {stats['to_be_correct']:.0f}/{stats['to_be_seen']:.0f}")
    print(f"Dictionary lookups:  {stats['dictionary_lookups']:.0f}")
    print(f"Miu interactions:    {stats['ai_uses']:.0f}\n")
    _print_daily_status(stats)
    sessions = app.progress.get("conversation_sessions", 0)
    if sessions:
        turns = app.progress.get("conversation_turns", 0)
        print(f"Conversations:       {sessions} sessions, {turns} messages sent")
    pause("Press ENTER to go back...")


def _print_daily_status(stats: dict[str, float]) -> None:
    print(
        f"Daily goal:          {stats['today_attempts']:.0f}/{stats['daily_goal']:.0f} "
        f"({stats['goal_progress_pct']:.0f}%)"
    )
    print(f"Current streak:      {stats['streak']:.0f} day(s)")
    print(f"Best streak:         {stats['best_streak']:.0f} day(s)")
    print(
        f"Due reviews:         {stats['due_reviews_total']:.0f} "
        f"(S:{stats['due_reviews_sentences']:.0f} "
        f"F:{stats['due_reviews_flashcards']:.0f} "
        f"T:{stats['due_reviews_to_be']:.0f})"
    )


def daily_plan(app: Application) -> None:
    while True:
        suggested = recommend_practice_mode(app.progress)
        header("Daily Study Plan")
        _print_daily_status(compute_overall(app.progress))
        print(f"Recommended now:     {MODE_LABELS[suggested]}\n")
        print("1) Start recommended practice")
        print("2) Change daily goal")
        print("3) Show weekly report")
        print("0) Back")

        choice = ask("Choose: ")
        if choice == "0":
            return
        if choice == "1":
            start_mode(app, suggested)
        elif choice == "2":
            goal = ask_positive_int("New daily goal (attempts): ")
            if goal is None:
                pause("Please type a whole number greater than zero.")
            else:
                pause(f"Daily goal updated to {set_daily_goal(app.progress, goal)}.")
                app.save_progress()
        elif choice == "3":
            show_weekly_report(app)
        else:
            pause("Invalid option. Press ENTER...")


def start_mode(app: Application, mode: str) -> None:
    if mode == "to_be":
        practice.run_to_be_trainer(app)
    elif mode == "flashcards":
        practice.flashcards_menu(app)
    else:
        practice.sentence_menu(app)


def show_weekly_report(app: Application) -> None:
    header("Weekly Report")
    print(format_weekly_report(build_weekly_report(app.progress)))
    pause()


def reports_menu(app: Application) -> None:
    options: list[tuple[str, Literal["show", "json", "csv"]]] = [
        ("Show weekly report", "show"),
        ("Export snapshot (JSON)", "json"),
        ("Export snapshot (CSV)", "csv"),
    ]
    while (action := choose("Study Report", options)) is not None:
        if action == "show":
            show_weekly_report(app)
        else:
            path = export_progress_snapshot(app.progress, app.settings.paths.reports_dir, action)
            pause(f"Report exported to: {path}")


def onboarding(app: Application) -> None:
    profile = app.progress.setdefault("profile", {})
    if profile.get("onboarding_completed"):
        return
    if compute_overall(app.progress)["total_seen"] == 0:
        header("Welcome to English Trainer AI!")
        print("Let's set up your daily routine. The default goal is 20 attempts.")
        goal = ask_positive_int("Choose your daily goal now (ENTER keeps 20): ")
        if goal is not None:
            set_daily_goal(app.progress, goal)
    profile["onboarding_completed"] = True
    app.save_progress()
