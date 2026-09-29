"""Curated ``to be`` examples used when no model is available (opt-in).

This provider never evaluates learner text: it only answers sentence
generation requests, so it cannot pretend to grade anything.
"""

from __future__ import annotations

import re

from english_trainer.llm.base import ChatMessage, GenerationOptions, LLMUnavailableError

MAX_EXAMPLES = 12

_EXAMPLES: dict[str, list[tuple[str, str, str]]] = {
    "present": [
        ("I am at home.", "Eu estou em casa.", "Use am com I"),
        ("She is tired.", "Ela está cansada.", "Use is com she"),
        ("They are ready.", "Eles estão prontos.", "Use are com they"),
        ("This is a book.", "Isto é um livro.", "Use is com this"),
        ("We are students.", "Nós somos estudantes.", "Use are com we"),
    ],
    "past": [
        ("I was at home.", "Eu estava em casa.", "Use was com I"),
        ("She was tired.", "Ela estava cansada.", "Use was com she"),
        ("They were ready.", "Eles estavam prontos.", "Use were com they"),
        ("This was a book.", "Isto era um livro.", "Use was com this"),
        ("We were students.", "Nós éramos estudantes.", "Use were com we"),
    ],
}


class LocalExamplesModel:
    @property
    def name(self) -> str:
        return "limited-local-examples"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> str:
        prompt = " ".join(message["content"] for message in messages).lower()
        if "generate" not in prompt or "sentence" not in prompt or "to be" not in prompt:
            raise LLMUnavailableError("Local examples only cover 'to be' sentence generation.")

        match = re.search(r"(\d+)\s*sentences", prompt)
        amount = max(1, min(int(match.group(1)) if match else 5, MAX_EXAMPLES))
        examples = _EXAMPLES["past" if "past" in prompt else "present"]
        selected = [examples[index % len(examples)] for index in range(amount)]

        if "en | pt" in prompt or "output format" in prompt:
            return "\n".join(" | ".join(example) for example in selected)
        return "\n".join(example[0] for example in selected)
