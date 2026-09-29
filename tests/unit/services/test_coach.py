from __future__ import annotations

import random

from english_trainer.llm.gateway import TextGenerator
from english_trainer.services.coach import CoachService, Verdict, parse_sentences
from tests.fakes import FakeChatModel, OfflineChatModel


def test_parse_sentences_keeps_only_valid_to_be_lines() -> None:
    text = (
        "Here are your sentences:\n"
        "She is happy. | Ela está feliz. | Use is com she\n"
        "Run fast | Corra rápido | sem to be\n"
        "They were at school. Extra sentence. | Eles estavam na escola.\n"
    )

    sentences = parse_sentences(text)

    assert [(s.en, s.pt, s.tip) for s in sentences] == [
        ("She is happy.", "Ela está feliz.", "Use is com she"),
        ("They were at school.", "Eles estavam na escola.", ""),
    ]


def test_generate_sentences_uses_model_output() -> None:
    model = FakeChatModel(replies=["I am here. | Eu estou aqui. | Use am com I"] * 2)
    coach = CoachService(TextGenerator([model]), rng=random.Random(1))

    batch = coach.generate_sentences("present", count=1)

    assert batch.provider == "fake"
    assert not batch.used_builtin_examples
    assert batch.sentences[0].en == "I am here."


def test_generate_sentences_falls_back_to_builtin_examples() -> None:
    coach = CoachService(TextGenerator([OfflineChatModel()]), rng=random.Random(3))

    batch = coach.generate_sentences("past", count=4)

    assert batch.used_builtin_examples
    assert len(batch.sentences) == 4
    assert all(
        any(v in s.en for v in (" was ", " were ", " is ", " are ")) for s in batch.sentences
    )


def test_check_translation_accepts_locally_without_calling_a_model() -> None:
    model = FakeChatModel(replies=[])
    coach = CoachService(TextGenerator([model]))

    feedback = coach.check_translation("I am tired.", "estou cansado", "Eu estou cansado.")

    assert feedback.verdict is Verdict.ACCEPTED_LOCALLY
    assert model.calls == []


def test_check_translation_reports_unavailable_models() -> None:
    coach = CoachService(TextGenerator([OfflineChatModel()]))

    feedback = coach.check_translation("I am tired.", "eu sou feliz", None)

    assert feedback.verdict is Verdict.UNAVAILABLE
