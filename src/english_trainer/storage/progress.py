"""Persistence for the learner's progress file."""

from __future__ import annotations

from pathlib import Path

from english_trainer.domain.progress import default_progress, normalize_progress
from english_trainer.domain.types import Progress
from english_trainer.storage.json_store import read_json_object, write_json_atomic


class ProgressRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Progress:
        raw = read_json_object(self.path)
        return normalize_progress(raw) if raw else default_progress()

    def save(self, progress: Progress) -> None:
        write_json_atomic(self.path, progress)
