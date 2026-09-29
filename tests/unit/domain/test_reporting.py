from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from english_trainer.domain.progress import default_progress
from english_trainer.domain.reporting import (
    build_weekly_report,
    export_progress_snapshot,
    format_weekly_report,
)
from english_trainer.domain.types import Progress


@pytest.fixture
def progress() -> Progress:
    data = default_progress()
    data["daily"]["history"] = {
        "2026-05-30": {"attempts": 5, "correct": 4, "sentences": 2, "flashcards": 2, "to_be": 1},
        "2026-05-31": {"attempts": 3, "correct": 1, "sentences": 1, "flashcards": 1, "to_be": 1},
        "2026-06-01": {"attempts": 2, "correct": 2, "sentences": 2, "flashcards": 0, "to_be": 0},
        "2026-05-01": {"attempts": 99, "correct": 99, "sentences": 99, "flashcards": 0, "to_be": 0},
    }
    return data


def test_weekly_report_covers_the_last_seven_days_only(progress: Progress) -> None:
    report = build_weekly_report(progress, today=date(2026, 6, 1))

    assert len(report.days) == 7
    assert report.period_start == "2026-05-26"
    assert report.attempts == 10
    assert report.correct == 7
    assert report.sentences == 5
    assert report.accuracy == pytest.approx(70.0)


def test_format_weekly_report_lists_every_day(progress: Progress) -> None:
    text = format_weekly_report(build_weekly_report(progress, today=date(2026, 6, 1)))

    assert "Accuracy: 70.0%" in text
    assert text.count(" attempts | ") == 7


def test_export_snapshot_json_and_csv(tmp_path: Path, progress: Progress) -> None:
    now = datetime(2026, 6, 1, 9, 30)

    json_path = export_progress_snapshot(progress, tmp_path, "json", now=now)
    csv_path = export_progress_snapshot(progress, tmp_path, "csv", now=now)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["weekly"]["attempts"] == 10
    assert json_path.name == "study_report_20260601_093000.json"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["section", "key", "value"]
    assert ["weekly", "attempts", "10"] in rows


def test_export_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="json"):
        export_progress_snapshot(default_progress(), tmp_path, "xml")  # type: ignore[arg-type]
