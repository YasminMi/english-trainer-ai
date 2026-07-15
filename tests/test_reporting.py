from pathlib import Path

from english_trainer import reporting


def test_build_weekly_report_aggregates_daily_history():
    progress = {
        "daily": {
            "history": {
                "2026-05-30": {"attempts": 5, "correct": 4, "sentences": 2, "flashcards": 2, "to_be": 1},
                "2026-05-31": {"attempts": 3, "correct": 1, "sentences": 1, "flashcards": 1, "to_be": 1},
                "2026-06-01": {"attempts": 2, "correct": 2, "sentences": 2, "flashcards": 0, "to_be": 0},
            }
        }
    }
    report = reporting.build_weekly_report(progress, when=reporting.date(2026, 6, 1))
    assert report["attempts"] == 10
    assert report["correct"] == 7
    assert report["sentences"] == 5
    assert len(report["days"]) == 7


def test_export_progress_snapshot_writes_json_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "DATA_DIR", Path(tmp_path))
    progress = {"sentences": {}, "flashcards": {}, "to_be": {}}

    json_path = reporting.export_progress_snapshot(progress, file_format="json")
    csv_path = reporting.export_progress_snapshot(progress, file_format="csv")

    assert json_path.exists()
    assert csv_path.exists()
    assert json_path.suffix == ".json"
    assert csv_path.suffix == ".csv"
