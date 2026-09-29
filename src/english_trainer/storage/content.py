"""Load versioned study content, the learner's own entries and Markdown notes."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import TypeAdapter, ValidationError

from english_trainer.config import Paths
from english_trainer.domain.content import merge_content
from english_trainer.domain.types import Content, empty_content
from english_trainer.storage.json_store import read_json_object, write_json_atomic
from english_trainer.storage.markdown_notes import parse_note

logger = logging.getLogger(__name__)

_CONTENT_SCHEMA: TypeAdapter[Content] = TypeAdapter(Content)


class ContentValidationError(ValueError):
    """Raised when a content document does not match the expected schema."""


def validate_content(data: dict[str, Any]) -> Content:
    """Validate a content document against :class:`~english_trainer.domain.types.Content`."""
    document = {**empty_content(), **data}
    try:
        return _CONTENT_SCHEMA.validate_python(document, strict=True)
    except ValidationError as exc:
        raise ContentValidationError(str(exc)) from exc


class ContentRepository:
    """Merges three sources, from most to least curated.

    1. ``data/content.json`` - versioned, curated material.
    2. ``data/user_content.json`` - entries the learner added (git-ignored).
    3. ``notes/**/*.md`` - Markdown study notes.
    """

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    def load(self) -> Content:
        content = empty_content()
        merge_content(content, read_json_object(self.paths.content_file))
        merge_content(content, self.load_user())
        merge_content(content, self.load_notes())
        return content

    def load_user(self) -> Content:
        content = empty_content()
        merge_content(content, read_json_object(self.paths.user_content_file))
        return content

    def load_notes(self) -> Content:
        content = empty_content()
        notes_dir = self.paths.notes_dir
        if not notes_dir.exists():
            return content

        note_paths = sorted(notes_dir.rglob("*.md"))
        for path in note_paths:
            merge_content(content, parse_note(path, notes_dir=notes_dir, home=self.paths.home))
        content["metadata"]["markdown_notes"] = [
            path.relative_to(self.paths.home).as_posix() for path in note_paths
        ]
        logger.debug("content.notes_loaded", extra={"notes": len(note_paths)})
        return content

    def save_user(self, content: Content) -> None:
        """Persist learner-created entries outside the versioned content file."""
        write_json_atomic(self.paths.user_content_file, content)
