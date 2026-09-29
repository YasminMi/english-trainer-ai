"""Free writing: structured AI feedback in Portuguese on a short English text."""

from __future__ import annotations

from english_trainer.llm.gateway import Generation, TextGenerator

TOPICS: tuple[str, ...] = (
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
)
MAX_TEXT_CHARS = 1_200


def feedback_prompt(text: str, topic: str) -> str:
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


def resolve_topic(choice: str) -> str:
    """Accept a 1-based topic number or free text."""
    choice = choice.strip()
    if choice.isdigit() and 1 <= int(choice) <= len(TOPICS):
        return TOPICS[int(choice) - 1]
    return choice or "anything"


class WritingService:
    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator

    def review(self, text: str, topic: str) -> Generation | None:
        text = " ".join(text.split())
        if not text:
            raise ValueError("Write something before asking for feedback.")
        return self.generator.generate(
            feedback_prompt(text[:MAX_TEXT_CHARS], topic), max_tokens=400
        )
