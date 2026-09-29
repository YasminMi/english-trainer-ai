"""Plain terminal interface (``english-trainer --cli``)."""

from __future__ import annotations

from collections.abc import Callable

from english_trainer.app import Application, build_application
from english_trainer.interfaces.cli import assistant, practice, tools
from english_trainer.interfaces.cli.io import ask, clear_screen, header, pause

MenuAction = Callable[[Application], None]

MENU: tuple[tuple[str, MenuAction], ...] = (
    ("To be Trainer (am/is/are/was/were)", practice.run_to_be_trainer),
    ("Flashcards (palavras)", practice.flashcards_menu),
    ("Sentence Training (frases)", practice.sentence_menu),
    ("Mini Dictionary (local + online)", tools.run_dictionary),
    ("Miu — AI Coach (gerar frases / corrigir tradução)", assistant.coach_menu),
    ("Conversation with Miu (conversar em inglês)", assistant.run_conversation),
    ("Show Progress (progresso)", tools.show_progress),
    ("Add content (sentence/flashcard/dictionary)", tools.add_content_menu),
    ("Daily Study Plan (goal + recommendation)", tools.daily_plan),
    ("Study Report (weekly + export)", tools.reports_menu),
    ("Dictation (ouça e escreva)", practice.dictation_menu),
    ("Free Writing (escreva com correção da IA)", assistant.run_free_writing),
    ("Miu — Grounded Tutor (RAG local + fontes)", assistant.run_grounded_tutor),
)


def run(app: Application | None = None) -> None:
    app = app or build_application()
    try:
        tools.onboarding(app)
        while True:
            header("English Trainer - Terminal")
            for index, (label, _) in enumerate(MENU, start=1):
                print(f"{index}) {label}")
            print("0) Exit")

            choice = ask("Choose: ")
            if choice == "0":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(MENU):
                MENU[int(choice) - 1][1](app)
                app.save_progress()
            else:
                pause("Invalid option. Press ENTER...")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        app.save_progress()
        app.close()
    clear_screen()
    print("Bye! 👋")
