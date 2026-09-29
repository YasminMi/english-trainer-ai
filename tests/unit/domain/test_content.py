from __future__ import annotations

import pytest

from english_trainer.domain.content import (
    DuplicateEntryError,
    add_dictionary_entry,
    add_flashcard,
    add_sentence,
    available_sources,
    filter_items,
    find_in_dictionary,
    note_label,
)
from english_trainer.domain.types import Content, empty_content


@pytest.fixture
def content() -> Content:
    data = empty_content()
    data["sentences"].append({"id": "base", "en": "Hello.", "pt": "Olá."})
    data["dictionary"]["good morning"] = "bom dia"
    return data


def test_add_sentence_keeps_user_content_separate(content: Content) -> None:
    user = empty_content()

    entry = add_sentence(
        content, user, en=" I study every day. ", pt="Eu estudo.", tags=["routine", " "]
    )

    assert entry["id"] == "user_s_000001"
    assert entry["en"] == "I study every day."
    assert entry["tags"] == ["routine"]
    assert user["sentences"] == [entry]
    assert content["sentences"][-1] is entry


def test_add_sentence_rejects_duplicates_ignoring_case_and_punctuation(content: Content) -> None:
    with pytest.raises(DuplicateEntryError):
        add_sentence(content, empty_content(), en="hello", pt="oi")


def test_add_flashcard_and_dictionary_entry(content: Content) -> None:
    user = empty_content()

    card = add_flashcard(content, user, en="Ice cream", pt="sorvete", category="food")
    add_dictionary_entry(content, user, word="chair", meaning="cadeira")

    assert card["id"] == "user_w_ice_cream"
    assert card["category"] == "food"
    assert user["dictionary"] == {"chair": "cadeira"}
    with pytest.raises(ValueError, match="required"):
        add_dictionary_entry(content, user, word="table", meaning=" ")


def test_find_in_dictionary_normalizes_the_query(content: Content) -> None:
    assert find_in_dictionary(content, "Good  Morning!") == "bom dia"
    assert find_in_dictionary(content, "missing") is None


def test_filter_items_and_sources() -> None:
    items = [
        {"id": "1", "category": "food", "source_note": "notes/a.md"},
        {"id": "2", "category": "food"},
        {"id": "3", "category": "verbs", "source_note": "notes/b.md"},
    ]
    assert [i["id"] for i in filter_items(items, category="food")] == ["1", "2"]
    assert [i["id"] for i in filter_items(items, source_note="notes/b.md")] == ["3"]
    assert available_sources(items) == ["notes/a.md", "notes/b.md"]
    assert note_label("notes/english/grammar/to_be_a1.md") == "To Be A1"
