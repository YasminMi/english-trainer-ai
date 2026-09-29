from english_trainer.utils import (
    _clean_text,
    _expand_slash_variations,
    is_close_answer,
    normalize_pt_phrase,
)


def test_clean_text_removes_punctuation_and_lowercases():
    assert _clean_text("Hello, World!") == "hello world"
    assert _clean_text("Cansado/Cansada") == "cansado/cansada"


def test_expand_slash_variations_simple():
    variants = _expand_slash_variations("cansado/cansada")
    assert "cansado" in variants
    assert "cansada" in variants


def test_is_close_answer_matches_exact_and_partial():
    assert is_close_answer("happy", "I am happy")
    assert is_close_answer("i am happy", "I am happy")
    assert is_close_answer("estou feliz", "Eu estou feliz")
    assert is_close_answer("pão", "pao")
    assert is_close_answer("pao", "pão")
    assert not is_close_answer("", "anything")
    assert is_close_answer("cansado", "Eu estou cansado/cansada")
    assert is_close_answer("cansada", "Eu estou cansado/cansada")


def test_is_close_answer_multiple_expected():
    assert is_close_answer("foo", ["bar", "foo"])


def test_is_close_answer_rejects_unrelated():
    assert not is_close_answer("hola", "hello")


def test_normalize_pt_phrase_fixes_legacy_to_be_negatives():
    assert (
        normalize_pt_phrase("Eles/Elas estavam não na cozinha.")
        == "Eles/Elas não estavam na cozinha."
    )
    assert is_close_answer("eles não estavam na cozinha", "Eles/Elas estavam não na cozinha.")


def test_is_close_answer_accepts_common_portuguese_contractions():
    assert is_close_answer("eu vou pro trabalho", "Eu vou para o trabalho.")
    assert is_close_answer("ela vai pra escola", "Ela vai para a escola.")
