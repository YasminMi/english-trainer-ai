"""Loads and saves data from JSON files and Markdown notes."""

import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .utils import _clean_text

BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
NOTES_DIR: Path = BASE_DIR / "notes"
CONTENT_FILE: Path = DATA_DIR / "content.json"
PROGRESS_FILE: Path = DATA_DIR / "progress.json"
AI_CACHE_FILE: Path = DATA_DIR / "ai_cache.json"
USER_CONTENT_FILE: Path = DATA_DIR / "user_content.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for missing or invalid local data."""
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Could not read {path.name}: {exc}", RuntimeWarning, stacklevel=2)
        return {}

    if not isinstance(data, dict):
        warnings.warn(
            f"Ignoring {path.name}: expected a JSON object.", RuntimeWarning, stacklevel=2
        )
        return {}
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to a temporary file and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def load_content() -> dict[str, Any]:
    """Return content merged from JSON and Markdown notes."""
    content = _empty_content()

    loaded = _read_json_object(CONTENT_FILE)
    if loaded:
        content.update(loaded)
        content.setdefault("metadata", {})
        content.setdefault("sentences", [])
        content.setdefault("flashcards", [])
        content.setdefault("to_be_exercises", [])
        content.setdefault("dictionary", {})

    _merge_content(content, load_user_content())
    _merge_content(content, load_markdown_notes())
    return content


def load_user_content() -> dict[str, Any]:
    """Load private user-created study material from the ignored local file."""
    content = _empty_content()
    _merge_content(content, _read_json_object(USER_CONTENT_FILE))
    return content


def load_markdown_notes() -> dict[str, Any]:
    """Parse Markdown notes into content items."""
    parsed = _empty_content()
    parsed["metadata"]["markdown_notes_enabled"] = True
    parsed["metadata"]["markdown_notes_dir"] = (
        str(NOTES_DIR.relative_to(BASE_DIR)) if NOTES_DIR.exists() else "notes"
    )

    if not NOTES_DIR.exists():
        return parsed

    note_paths = sorted(NOTES_DIR.rglob("*.md"))
    metadata_notes = parsed["metadata"].setdefault("markdown_notes", [])
    for note_path in note_paths:
        metadata_notes.append(note_path.relative_to(BASE_DIR).as_posix())
        _merge_content(parsed, _parse_markdown_note(note_path))

    parsed["metadata"]["markdown_note_count"] = len(note_paths)
    return parsed


def _empty_content() -> dict[str, Any]:
    return {
        "metadata": {},
        "sentences": [],
        "flashcards": [],
        "to_be_exercises": [],
        "dictionary": {},
    }


def _merge_content(base: dict[str, Any], extra: dict[str, Any]) -> None:
    base.setdefault("metadata", {}).update(extra.get("metadata", {}))
    base.setdefault("sentences", []).extend(extra.get("sentences", []))
    base.setdefault("flashcards", []).extend(extra.get("flashcards", []))
    base.setdefault("to_be_exercises", []).extend(extra.get("to_be_exercises", []))
    base.setdefault("dictionary", {}).update(extra.get("dictionary", {}))


def _parse_markdown_note(note_path: Path) -> dict[str, Any]:
    content = _empty_content()
    relative_note = note_path.relative_to(BASE_DIR).as_posix()
    note_slug = _slugify(note_path.relative_to(NOTES_DIR).with_suffix("").as_posix())
    text = note_path.read_text(encoding="utf-8")

    for heading, body in _split_markdown_sections(text):
        normalized_heading = _clean_text(heading)
        if normalized_heading == "sentences":
            content["sentences"].extend(_parse_sentence_section(body, note_slug, relative_note))
        elif normalized_heading == "flashcards":
            content["flashcards"].extend(_parse_flashcard_section(body, note_slug, relative_note))
        elif normalized_heading == "dictionary":
            content["dictionary"].update(_parse_dictionary_section(body))

    return content


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
            continue

        if current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def _parse_sentence_section(body: str, note_slug: str, relative_note: str) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    current_en: str | None = None
    current_tags: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("- en:"):
            current_en = line.split(":", 1)[1].strip()
            current_tags = []
            continue

        if line.startswith("tags:") and current_en:
            raw_tags = line.split(":", 1)[1].strip()
            current_tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
            continue

        if line.startswith("pt:") and current_en:
            pt = line.split(":", 1)[1].strip()
            sentence = {
                "id": f"md_{note_slug}_s_{len(sentences) + 1:03d}",
                "en": current_en,
                "pt": pt,
                "tags": ["markdown", note_slug, *current_tags],
                "source_note": relative_note,
            }
            sentences.append(sentence)
            current_en = None
            current_tags = []

    return sentences


def _parse_flashcard_section(body: str, note_slug: str, relative_note: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue

        payload = line[2:].strip()
        if "=" not in payload:
            continue

        en, pt = payload.split("=", 1)
        cards.append(
            {
                "id": f"md_{note_slug}_f_{len(cards) + 1:03d}",
                "en": en.strip(),
                "pt": pt.strip(),
                "category": note_slug,
                "source_note": relative_note,
            }
        )

    return cards


def _parse_dictionary_section(body: str) -> dict[str, str]:
    dictionary: dict[str, str] = {}

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue

        payload = line[2:].strip()
        if "=" not in payload:
            continue

        en, pt = payload.split("=", 1)
        dictionary[en.strip()] = pt.strip()

    return dictionary


def _slugify(value: str) -> str:
    cleaned = _clean_text(value).replace("/", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    return cleaned.strip("_") or "note"


def load_progress() -> dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {
            "sentences": {},
            "flashcards": {},
            "to_be": {},
            "dictionary_lookups": 0,
            "ai_uses": 0,
            "srs": {},
            "daily": {
                "goal": 20,
                "history": {},
                "streak": 0,
                "best_streak": 0,
                "last_study_date": None,
            },
        }

    data = _read_json_object(PROGRESS_FILE)

    data.setdefault("sentences", {})
    data.setdefault("flashcards", {})
    data.setdefault("to_be", {})
    data.setdefault("dictionary_lookups", 0)
    data.setdefault("ai_uses", 0)
    data.setdefault("srs", {})
    daily = data.setdefault("daily", {})
    daily.setdefault("goal", 20)
    daily.setdefault("history", {})
    daily.setdefault("streak", 0)
    daily.setdefault("best_streak", 0)
    daily.setdefault("last_study_date", None)
    return data


def save_progress(progress: dict[str, Any]) -> None:
    _write_json_atomic(PROGRESS_FILE, progress)


def save_content(content: dict[str, Any]) -> None:
    """Write the given content structure back to the JSON file."""
    _write_json_atomic(CONTENT_FILE, content)


def save_user_content(content: dict[str, Any]) -> None:
    """Persist user-created study material outside the versioned content file."""
    _write_json_atomic(USER_CONTENT_FILE, content)


def find_in_dictionary(content: dict[str, Any], word: str) -> str | None:
    dictionary: dict[str, str] = content.get("dictionary", {})
    key: str = _clean_text(word).replace("/", " ")
    key = " ".join(key.split())
    return dictionary.get(key)


def load_ai_cache() -> dict[str, Any]:
    return _read_json_object(AI_CACHE_FILE)


def save_ai_cache(cache: dict[str, Any]) -> None:
    _write_json_atomic(AI_CACHE_FILE, cache)
