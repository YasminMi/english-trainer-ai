from __future__ import annotations

from pathlib import Path

from english_trainer.domain.progress import register_result
from english_trainer.storage.progress import ProgressRepository


def test_new_learner_gets_default_progress(tmp_path: Path) -> None:
    progress = ProgressRepository(tmp_path / "progress.json").load()

    assert progress["sentences"] == {}
    assert progress["daily"]["goal"] == 20


def test_progress_round_trip(tmp_path: Path) -> None:
    repository = ProgressRepository(tmp_path / "progress.json")
    progress = repository.load()
    register_result(progress, "flashcards", "f1", correct=True)

    repository.save(progress)

    assert repository.load()["flashcards"] == {"f1": {"seen": 1, "correct": 1}}


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text("{invalid", encoding="utf-8")

    assert ProgressRepository(path).load()["daily"]["goal"] == 20
