from __future__ import annotations

import pytest

from english_trainer.domain.matching import (
    clean_text,
    expand_slash_variations,
    is_close_answer,
    normalize_pt_phrase,
    strip_accents,
)


def test_clean_text_removes_punctuation_but_keeps_alternatives() -> None:
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("Cansado/Cansada") == "cansado/cansada"


def test_strip_accents() -> None:
    assert strip_accents("você está à mesa") == "voce esta a mesa"


def test_expand_slash_variations_combines_every_alternative() -> None:
    assert expand_slash_variations("eu sou/estou cansado/cansada") == [
        "eu sou cansado",
        "eu sou cansada",
        "eu estou cansado",
        "eu estou cansada",
    ]


def test_expand_slash_variations_supports_independent_groups() -> None:
    assert set(expand_slash_variations("é/está / são/estão")) == {"é", "está", "são", "estão"}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("happy", "I am happy"),
        ("i am happy", "I am happy"),
        ("estou feliz", "Eu estou feliz"),
        ("pão", "pao"),
        ("pao", "pão"),
        ("cansada", "Eu estou cansado/cansada"),
        ("eu vou pro trabalho", "Eu vou para o trabalho."),
        ("ela vai pra escola", "Ela vai para a escola."),
        ("tem um cachorro", "Há um cachorro."),
        ("estou", "sou/estou (com I)"),
    ],
)
def test_is_close_answer_accepts_equivalent_answers(answer: str, expected: str) -> None:
    assert is_close_answer(answer, expected)


@pytest.mark.parametrize(("answer", "expected"), [("", "anything"), ("hola", "hello")])
def test_is_close_answer_rejects_empty_and_unrelated(answer: str, expected: str) -> None:
    assert not is_close_answer(answer, expected)


def test_is_close_answer_accepts_any_of_several_expected_answers() -> None:
    assert is_close_answer("foo", ["bar", "foo"])


def test_strict_mode_disables_partial_matches() -> None:
    assert is_close_answer("am happy", "I am happy")
    assert not is_close_answer("am happy", "I am happy", strict=True)
    assert is_close_answer("I am happy.", "I am happy", strict=True)


def test_normalize_pt_phrase_fixes_legacy_negatives() -> None:
    assert (
        normalize_pt_phrase("Eles/Elas estavam não na cozinha.")
        == "Eles/Elas não estavam na cozinha."
    )
    assert is_close_answer("eles não estavam na cozinha", "Eles/Elas estavam não na cozinha.")
