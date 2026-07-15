"""Terminal app entry point and study flow menus."""

from typing import Any, Dict

from .ai_coach import run_ai_coach
from .conversation import run_conversation
from .data_access import (
    load_content,
    load_progress,
    load_user_content,
    save_progress,
    save_user_content,
)
from .dictation import run_dictation
from .flashcards import run_flashcards_session
from .free_writing import run_free_writing
from .mini_dictionary import run_mini_dictionary
from .progress import (
    show_progress_summary,
    compute_overall,
    recommend_practice_mode,
    set_daily_goal,
)
from .rag_cli import run_rag_tutor
from .reporting import build_weekly_report, format_weekly_report, export_progress_snapshot
from .to_be_trainer import run_to_be_trainer
from .training import run_sentence_training
from .utils import clear_screen, read_choice, pause


def run_app() -> None:
    content: Dict[str, Any] = load_content()
    progress: Dict[str, Any] = load_progress()
    _run_onboarding(progress)

    while True:
        clear_screen()
        print("==============================")
        print(" English Trainer - Terminal")
        print("==============================")
        print("1) To be Trainer (am/is/are/was/were)")
        print("2) Flashcards (palavras)")
        print("3) Sentence Training (frases)")
        print("4) Mini Dictionary (local + online)")
        print("5) Miu — AI Coach (gerar frases / corrigir tradução)")
        print("6) Conversation with Miu (conversar em inglês)")
        print("7) Show Progress (progresso)")
        print("8) Add content (sentence/flashcard/dictionary)")
        print("9) Daily Study Plan (goal + recommendation)")
        print("10) Study Report (weekly + export)")
        print("11) Dictation (ouça e escreva)")
        print("12) Free Writing (escreva com correção da IA)")
        print("13) Miu — Grounded Tutor (RAG local + fontes)")
        print("0) Exit")
        print("==============================")

        choice: str = read_choice("Choose: ")

        if choice == "1":
            run_to_be_trainer(content, progress)
        elif choice == "2":
            _flashcards_menu(content, progress)
        elif choice == "3":
            _sentence_training_menu(content, progress)
        elif choice == "4":
            run_mini_dictionary(content, progress)
        elif choice == "5":
            run_ai_coach(content, progress)
        elif choice == "6":
            run_conversation(progress)
        elif choice == "7":
            show_progress_summary(progress)
        elif choice == "8":
            _add_content_menu(content)
        elif choice == "9":
            _daily_study_menu(content, progress)
        elif choice == "10":
            _study_report_menu(progress)
        elif choice == "11":
            _dictation_menu(content, progress)
        elif choice == "12":
            run_free_writing(progress)
        elif choice == "13":
            run_rag_tutor(content, progress)
        elif choice == "0":
            save_progress(progress)
            clear_screen()
            print("Bye! 👋")
            break
        else:
            input("Invalid option. Press ENTER...")

        save_progress(progress)


def _flashcards_menu(content: dict, progress: dict) -> None:
    metadata = content.get("metadata", {})
    categories = metadata.get("flashcards_categories", [])
    notes = _available_sources(content, "flashcards")

    while True:
        clear_screen()
        print("=== Flashcards ===")
        print("1) All words")
        for idx, cat in enumerate(categories, start=2):
            label = cat.replace("_", " ").capitalize()
            print(f"{idx}) {label} only")
        notes_start = len(categories) + 2
        if notes:
            print("--- Markdown Notes ---")
            for idx, note in enumerate(notes, start=notes_start):
                print(f"{idx}) {note}")
        print("0) Back")
        choice: str = read_choice("Choose: ")

        if choice == "0":
            return

        try:
            num = int(choice)
        except ValueError:
            input("Invalid option. Press ENTER...")
            continue

        if num == 1:
            cat = None
            source_note = None
        else:
            cat_idx: int = num - 2
            if 0 <= cat_idx < len(categories):
                cat = categories[cat_idx]
                source_note = None
            elif notes and notes_start <= num < notes_start + len(notes):
                cat = None
                source_note = notes[num - notes_start]
            else:
                input("Invalid option. Press ENTER...")
                continue

        run_flashcards_session(content, progress, category=cat, source_note=source_note)


def _sentence_training_menu(content: dict, progress: dict) -> None:
    notes = _available_sources(content, "sentences")

    while True:
        clear_screen()
        print("=== Sentence Training ===")
        print("1) All sentences")
        for idx, note in enumerate(notes, start=2):
            print(f"{idx}) {note}")
        print("0) Back")

        choice: str = read_choice("Choose: ")
        if choice == "0":
            return

        try:
            num = int(choice)
        except ValueError:
            input("Invalid option. Press ENTER...")
            continue

        if num == 1:
            source_note = None
        else:
            note_idx = num - 2
            if 0 <= note_idx < len(notes):
                source_note = notes[note_idx]
            else:
                input("Invalid option. Press ENTER...")
                continue

        run_sentence_training(content, progress, source_note=source_note)


def _available_sources(content: dict, section: str) -> list[str]:
    items = content.get(section, [])
    sources = {
        item.get("source_note")
        for item in items
        if item.get("source_note")
    }
    return sorted(sources)


def _daily_study_menu(content: dict, progress: dict) -> None:
    mode_labels = {
        "to_be": "To be Trainer",
        "flashcards": "Flashcards",
        "sentences": "Sentence Training",
    }
    while True:
        clear_screen()
        stats = compute_overall(progress)
        suggested = recommend_practice_mode(progress)
        suggested_label = mode_labels.get(suggested, suggested)

        print("=== Daily Study Plan ===")
        print(
            f"Goal today: {int(stats['today_attempts'])}/{int(stats['daily_goal'])} "
            f"({stats['goal_progress_pct']:.0f}%)"
        )
        print(
            f"Streak: {int(stats['streak'])} day(s) | "
            f"Best streak: {int(stats['best_streak'])} day(s)"
        )
        print(
            "Due reviews: "
            f"{int(stats['due_reviews_total'])} "
            f"(S:{int(stats['due_reviews_sentences'])} "
            f"F:{int(stats['due_reviews_flashcards'])} "
            f"T:{int(stats['due_reviews_to_be'])})"
        )
        print(f"Recommended now: {suggested_label}")
        print()
        print("1) Start recommended practice")
        print("2) Start To be Trainer")
        print("3) Start Flashcards")
        print("4) Start Sentence Training")
        print("5) Change daily goal")
        print("6) Show weekly report")
        print("7) Export study snapshot")
        print("0) Back")

        choice = read_choice("Choose: ")
        if choice == "0":
            return
        if choice == "1":
            _run_mode_by_key(suggested, content, progress)
        elif choice == "2":
            run_to_be_trainer(content, progress)
        elif choice == "3":
            _flashcards_menu(content, progress)
        elif choice == "4":
            _sentence_training_menu(content, progress)
        elif choice == "5":
            _configure_daily_goal(progress)
        elif choice == "6":
            _show_weekly_report(progress)
        elif choice == "7":
            _export_study_snapshot(progress)
        else:
            input("Invalid option. Press ENTER...")


def _run_mode_by_key(mode: str, content: dict, progress: dict) -> None:
    if mode == "to_be":
        run_to_be_trainer(content, progress)
    elif mode == "flashcards":
        _flashcards_menu(content, progress)
    else:
        _sentence_training_menu(content, progress)


def _configure_daily_goal(progress: dict) -> None:
    raw = read_choice("New daily goal (attempts): ")
    try:
        goal = int(raw)
    except ValueError:
        print("Invalid number.")
        pause()
        return
    value = set_daily_goal(progress, goal)
    print(f"Daily goal updated to {value}.")
    pause()


def _show_weekly_report(progress: dict) -> None:
    clear_screen()
    report = build_weekly_report(progress)
    print(format_weekly_report(report))
    pause()


def _export_study_snapshot(progress: dict) -> None:
    fmt = read_choice("Format (json/csv): ").strip().lower() or "json"
    if fmt not in {"json", "csv"}:
        print("Invalid format. Use json or csv.")
        pause()
        return
    path = export_progress_snapshot(progress, file_format=fmt)
    print(f"Report exported to: {path}")
    pause()


def _study_report_menu(progress: dict) -> None:
    while True:
        clear_screen()
        print("=== Study Report ===")
        print("1) Show weekly report")
        print("2) Export snapshot (JSON)")
        print("3) Export snapshot (CSV)")
        print("0) Back")

        choice = read_choice("Choose: ")
        if choice == "0":
            return
        if choice == "1":
            _show_weekly_report(progress)
        elif choice == "2":
            path = export_progress_snapshot(progress, file_format="json")
            print(f"Report exported to: {path}")
            pause()
        elif choice == "3":
            path = export_progress_snapshot(progress, file_format="csv")
            print(f"Report exported to: {path}")
            pause()
        else:
            input("Invalid option. Press ENTER...")


def _dictation_menu(content: dict, progress: dict) -> None:
    notes = _available_sources(content, "sentences")

    while True:
        clear_screen()
        print("=== Dictation ===")
        print("1) All sentences")
        for idx, note in enumerate(notes, start=2):
            print(f"{idx}) {note}")
        print("0) Back")

        choice: str = read_choice("Choose: ")
        if choice == "0":
            return

        try:
            num = int(choice)
        except ValueError:
            input("Invalid option. Press ENTER...")
            continue

        if num == 1:
            source_note = None
        else:
            note_idx = num - 2
            if 0 <= note_idx < len(notes):
                source_note = notes[note_idx]
            else:
                input("Invalid option. Press ENTER...")
                continue

        run_dictation(content, progress, source_note=source_note)


def _run_onboarding(progress: dict) -> None:
    profile = progress.setdefault("profile", {})
    if profile.get("onboarding_completed"):
        return

    stats = compute_overall(progress)
    if int(stats.get("total_seen", 0)) > 0:
        profile["onboarding_completed"] = True
        return

    clear_screen()
    print("Welcome to English Trainer AI!")
    print("Let's setup your daily routine.")
    print("Default daily goal is 20 attempts.")
    raw = read_choice("Choose your daily goal now (ENTER keeps 20): ").strip()
    if raw:
        try:
            goal = int(raw)
            set_daily_goal(progress, goal)
        except ValueError:
            print("Invalid number. Keeping default goal.")
            pause()
    profile["onboarding_completed"] = True


def _add_content_menu(content: dict) -> None:
    user_content = load_user_content()

    while True:
        clear_screen()
        print("=== Add new content ===")
        print("1) Sentence")
        print("2) Flashcard")
        print("3) Dictionary entry")
        print("0) Back")
        choice: str = read_choice("Choose: ")

        added = False
        if choice == "0":
            return
        elif choice == "1":
            added = _add_sentence(content, user_content)
        elif choice == "2":
            added = _add_flashcard(content, user_content)
        elif choice == "3":
            added = _add_dictionary_entry(content, user_content)
        else:
            input("Invalid option. Press ENTER...")
            continue

        if added:
            save_user_content(user_content)
            input("Content saved locally. Press ENTER to continue...")


def _add_sentence(content: dict, user_content: dict) -> bool:
    from .utils import _clean_text
    en: str = read_choice("English sentence: ").strip()
    if not en:
        print("No sentence provided.")
        pause()
        return False
    existing = {_clean_text(s.get("en", "")) for s in content.get("sentences", [])}
    if _clean_text(en) in existing:
        print("This sentence already exists.")
        pause()
        return False
    pt: str = read_choice("Portuguese meaning: ").strip()
    tags: str = read_choice("Tags (comma-separated, optional): ").strip()
    entry: dict = {"en": en, "pt": pt}
    if tags:
        entry["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    entry["id"] = f"user_s_{len(user_content.get('sentences', [])) + 1:06d}"
    content.setdefault("sentences", []).append(entry)
    user_content.setdefault("sentences", []).append(entry)
    print("Sentence added.")
    return True


def _add_flashcard(content: dict, user_content: dict) -> bool:
    from .utils import _clean_text
    en: str = read_choice("English word: ").strip()
    if not en:
        print("No word provided.")
        pause()
        return False
    existing = {_clean_text(c.get("en", "")) for c in content.get("flashcards", [])}
    if _clean_text(en) in existing:
        print("This word already exists in flashcards.")
        pause()
        return False
    pt: str = read_choice("Portuguese meaning: ").strip()
    cat: str | None = read_choice("Category (optional): ").strip() or None
    entry: dict = {"en": en, "pt": pt}
    if cat:
        entry["category"] = cat
    entry["id"] = f"user_w_{en.lower().replace(' ', '_')}"
    content.setdefault("flashcards", []).append(entry)
    user_content.setdefault("flashcards", []).append(entry)
    print("Flashcard added.")
    return True


def _add_dictionary_entry(content: dict, user_content: dict) -> bool:
    word: str = read_choice("English word: ").strip()
    if not word:
        print("No word provided.")
        pause()
        return False
    meaning: str = read_choice("Portuguese meaning: ").strip()
    if not meaning:
        print("No meaning provided. Entry skipped.")
        pause()
        return False
    content.setdefault("dictionary", {})[word] = meaning
    user_content.setdefault("dictionary", {})[word] = meaning
    print("Dictionary entry added.")
    return True
