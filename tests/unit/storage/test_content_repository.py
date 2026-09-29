from __future__ import annotations

import json
from pathlib import Path

import pytest

from english_trainer.config import Paths, get_settings
from english_trainer.storage.content import (
    ContentRepository,
    ContentValidationError,
    validate_content,
)
from english_trainer.storage.json_store import read_json_object

NOTE = """\
# Sample

## Grammar
Use `on` for surfaces.

## Sentences
- en: The book is on the table.
  pt: O livro está sobre a mesa.
  tags: prepositions, place

## Flashcards
- on = sobre

## Dictionary
- chair = cadeira
"""


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    paths = Paths(home=tmp_path)
    paths.data_dir.mkdir()
    (paths.notes_dir / "english").mkdir(parents=True)
    (paths.notes_dir / "english" / "sample.md").write_text(NOTE, encoding="utf-8")
    paths.content_file.write_text(
        json.dumps({"version": 1, "sentences": [{"id": "s1", "en": "Hello.", "pt": "Olá."}]}),
        encoding="utf-8",
    )
    return paths


def test_load_merges_json_user_entries_and_markdown_notes(paths: Paths) -> None:
    repository = ContentRepository(paths)
    user = repository.load_user()
    user["sentences"].append({"id": "user_s_000001", "en": "Good morning.", "pt": "Bom dia."})
    repository.save_user(user)

    content = repository.load()

    assert [s["en"] for s in content["sentences"]] == [
        "Hello.",
        "Good morning.",
        "The book is on the table.",
    ]
    note_sentence = content["sentences"][-1]
    assert note_sentence["id"] == "md_english_sample_s_001"
    assert note_sentence.get("source_note") == "notes/english/sample.md"
    assert note_sentence.get("tags") == ["markdown", "english_sample", "prepositions", "place"]
    assert content["flashcards"][0] == {
        "id": "md_english_sample_f_001",
        "en": "on",
        "pt": "sobre",
        "category": "english_sample",
        "source_note": "notes/english/sample.md",
    }
    assert content["dictionary"] == {"chair": "cadeira"}


def test_user_entries_never_touch_the_versioned_file(paths: Paths) -> None:
    repository = ContentRepository(paths)
    user = repository.load_user()
    user["sentences"].append({"id": "user_s_000001", "en": "Hi.", "pt": "Oi."})
    repository.save_user(user)

    assert len(read_json_object(paths.content_file)["sentences"]) == 1
    assert len(repository.load()["sentences"]) == 3


def test_missing_files_yield_empty_sections(tmp_path: Path) -> None:
    content = ContentRepository(Paths(home=tmp_path)).load()

    assert content["sentences"] == []
    assert content["dictionary"] == {}


def test_validate_content_reports_schema_errors() -> None:
    with pytest.raises(ContentValidationError, match="pt"):
        validate_content({"version": 1, "sentences": [{"id": "s1", "en": "Hello."}]})


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"version": 1, "sentences": [{"id": "s1", "en": "Hi."}]}, "sentences.0.pt"),
        ({"version": 1, "sentences": "bad"}, "sentences"),
        (
            {"version": 1, "sentences": [{"id": "s", "en": "A", "pt": "B"}] * 2},
            "duplicate id",
        ),
        (
            {
                "version": 1,
                "to_be_exercises": [
                    {"id": "x", "question": "q", "options": ["a", "b"], "correct": "c", "pt": "p"}
                ],
            },
            "correct must match",
        ),
        ({"version": 2}, "version"),
    ],
)
def test_broken_versioned_content_has_path_and_field(
    document: dict[str, object], message: str
) -> None:
    with pytest.raises(ContentValidationError, match=message) as error:
        validate_content(document, path="fixture.json")
    assert "fixture.json" in str(error.value)


def test_versioned_content_matches_the_schema() -> None:
    """Guards data/content.json against malformed edits."""
    validate_content(read_json_object(get_settings().paths.content_file))
