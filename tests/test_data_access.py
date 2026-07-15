from pathlib import Path

import pytest

from english_trainer import data_access


def test_save_and_load_progress(tmp_path, monkeypatch):
    progress = {"foo": "bar"}
    tmpdir = tmp_path
    monkeypatch.setattr(data_access, "PROGRESS_FILE", tmpdir / "progress.json")
    monkeypatch.setattr(data_access, "DATA_DIR", tmpdir)

    data_access.save_progress(progress)
    loaded = data_access.load_progress()
    assert loaded.get("foo") == "bar"
    assert "sentences" in loaded
    assert "flashcards" in loaded
    assert not list(tmp_path.glob("*.tmp"))


def test_save_and_load_content(tmp_path, monkeypatch):
    content = {"a": 1}
    tmpdir = tmp_path
    monkeypatch.setattr(data_access, "CONTENT_FILE", tmpdir / "content.json")
    monkeypatch.setattr(data_access, "DATA_DIR", tmpdir)
    monkeypatch.setattr(data_access, "NOTES_DIR", tmpdir / "notes")
    monkeypatch.setattr(data_access, "BASE_DIR", tmpdir)

    data_access.save_content(content)
    loaded = data_access.load_content()
    assert loaded["a"] == 1
    assert loaded["sentences"] == []
    assert loaded["flashcards"] == []


def test_load_content_missing(tmp_path, monkeypatch):
    tmpdir = tmp_path
    monkeypatch.setattr(data_access, "CONTENT_FILE", tmpdir / "content.json")
    monkeypatch.setattr(data_access, "DATA_DIR", tmpdir)
    loaded = data_access.load_content()
    assert "sentences" in loaded
    assert "flashcards" in loaded
    assert "dictionary" in loaded


def test_load_content_merges_markdown_notes(tmp_path, monkeypatch):
    tmpdir = tmp_path
    notes_dir = tmpdir / "notes"
    notes_dir.mkdir()
    note_file = notes_dir / "sample.md"
    note_file.write_text(
        (
            "# Note\n\n"
            "## Sentences\n"
            "- en: The book is on the table.\n"
            "  pt: O livro esta sobre a mesa.\n"
            "  tags: prepositions, place\n\n"
            "## Flashcards\n"
            "- on = sobre\n\n"
            "## Dictionary\n"
            "- chair = cadeira\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(data_access, "CONTENT_FILE", tmpdir / "content.json")
    monkeypatch.setattr(data_access, "DATA_DIR", tmpdir)
    monkeypatch.setattr(data_access, "NOTES_DIR", notes_dir)
    monkeypatch.setattr(data_access, "BASE_DIR", tmpdir)

    loaded = data_access.load_content()

    assert any(item["en"] == "The book is on the table." for item in loaded["sentences"])
    assert any(item["en"] == "on" and item["pt"] == "sobre" for item in loaded["flashcards"])
    assert loaded["dictionary"]["chair"] == "cadeira"


def test_user_content_is_merged_once_and_saved_separately(tmp_path, monkeypatch):
    base = {
        "sentences": [{"id": "base", "en": "Hello.", "pt": "Olá."}],
        "flashcards": [],
        "to_be_exercises": [],
        "dictionary": {},
    }
    user = {
        "sentences": [{"id": "user_s_000001", "en": "Good morning.", "pt": "Bom dia."}],
        "flashcards": [],
        "to_be_exercises": [],
        "dictionary": {},
    }
    monkeypatch.setattr(data_access, "CONTENT_FILE", tmp_path / "content.json")
    monkeypatch.setattr(data_access, "USER_CONTENT_FILE", tmp_path / "user_content.json")
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_access, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(data_access, "BASE_DIR", tmp_path)

    data_access.save_content(base)
    data_access.save_user_content(user)

    assert len(data_access.load_content()["sentences"]) == 2
    assert len(data_access.load_content()["sentences"]) == 2
    assert len(data_access._read_json_object(data_access.CONTENT_FILE)["sentences"]) == 1


def test_invalid_progress_json_uses_defaults(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(data_access, "PROGRESS_FILE", progress_file)

    with pytest.warns(RuntimeWarning, match="Could not read progress.json"):
        loaded = data_access.load_progress()

    assert loaded["sentences"] == {}
    assert loaded["daily"]["goal"] == 20
