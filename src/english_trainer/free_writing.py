"""Free writing: write in English, get AI correction and explanation in Portuguese."""

from typing import Any

from .hf_llm import generate_text
from .progress import register_ai_use
from .utils import clear_screen, pause, read_choice

_TOPICS: list[str] = [
    "your daily routine",
    "your favorite food",
    "your family",
    "your job or studies",
    "a place you like",
    "your weekend",
    "your hobbies",
    "the weather today",
    "your goals for this year",
    "a movie or series you watched recently",
]


def _prompt(text: str, topic: str) -> str:
    return (
        "You are an English teacher for Brazilian beginners. "
        "Respond ONLY in Portuguese (Brazil). Be direct and structured.\n\n"
        f"Topic: {topic}\n"
        f"Student wrote: {text}\n\n"
        "Reply with EXACTLY this structure (4 sections, no extra text):\n\n"
        "AVALIACAO: [one of: OTIMO / ACEITAVEL / PRECISA CORRIGIR]\n\n"
        "ERROS: [list up to 2 errors, or 'Nenhum erro.' if correct. "
        "For each: show the wrong part and the corrected English.]\n\n"
        "VERSAO MELHORADA: [rewrite the student's text in more natural English "
        "— always provide this, even when there are no errors.]\n\n"
        "DICA: [one short practical tip in Portuguese about grammar or vocabulary used here.]\n"
    )


def run_free_writing(progress: dict[str, Any]) -> None:
    while True:
        clear_screen()
        print("=== Free Writing ===")
        print("Write 2-4 sentences in English about a topic.")
        print("The AI will correct your writing and explain mistakes in Portuguese.")
        print("Type '0' to go back.\n")

        print("Choose a topic:")
        for i, t in enumerate(_TOPICS, 1):
            print(f"  {i:2}) {t}")
        print()

        raw = read_choice("Topic number (or write your own): ").strip()
        if raw == "0":
            return

        try:
            idx = int(raw) - 1
            topic = _TOPICS[idx] if 0 <= idx < len(_TOPICS) else raw
        except ValueError:
            topic = raw if raw else "anything"

        clear_screen()
        print(f"Topic: {topic}")
        print("Write 2-4 sentences in English.")
        print("Press ENTER on an empty line when done.\n")

        lines: list[str] = []
        while len(lines) < 4:
            line = read_choice("").strip()
            if not line:
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            continue

        print("\n[Sending to AI...]\n")
        response = generate_text(_prompt(text, topic), max_new_tokens=400)
        register_ai_use(progress)

        print("--- AI Feedback ---")
        if response:
            print(response.strip())
        else:
            print("AI is not available right now (check if Ollama is running).")
        print("-------------------")
        pause()
