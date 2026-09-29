"""Weekly study reports and exportable progress snapshots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from english_trainer.domain.progress import compute_overall
from english_trainer.domain.types import Progress

ExportFormat = Literal["json", "csv"]
REPORT_DAYS = 7


def _as_int(value: object) -> int:
    try:
        return int(value or 0)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class DayReport:
    date: str
    attempts: int
    correct: int
    sentences: int
    flashcards: int
    to_be: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts * 100.0 if self.attempts else 0.0


@dataclass(frozen=True, slots=True)
class WeeklyReport:
    period_start: str
    period_end: str
    days: list[DayReport] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return sum(day.attempts for day in self.days)

    @property
    def correct(self) -> int:
        return sum(day.correct for day in self.days)

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts * 100.0 if self.attempts else 0.0

    @property
    def sentences(self) -> int:
        return sum(day.sentences for day in self.days)

    @property
    def flashcards(self) -> int:
        return sum(day.flashcards for day in self.days)

    @property
    def to_be(self) -> int:
        return sum(day.to_be for day in self.days)

    def to_dict(self) -> dict[str, object]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "attempts": self.attempts,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "sentences": self.sentences,
            "flashcards": self.flashcards,
            "to_be": self.to_be,
            "days": [asdict(day) | {"accuracy": day.accuracy} for day in self.days],
        }


def build_weekly_report(progress: Progress, *, today: date | None = None) -> WeeklyReport:
    end = today or date.today()
    start = end - timedelta(days=REPORT_DAYS - 1)
    history = progress.get("daily", {}).get("history", {})

    days = []
    for offset in range(REPORT_DAYS):
        key = (start + timedelta(days=offset)).isoformat()
        stats: dict[str, object] = dict(history.get(key, {}))
        days.append(
            DayReport(
                date=key,
                attempts=_as_int(stats.get("attempts")),
                correct=_as_int(stats.get("correct")),
                sentences=_as_int(stats.get("sentences")),
                flashcards=_as_int(stats.get("flashcards")),
                to_be=_as_int(stats.get("to_be")),
            )
        )
    return WeeklyReport(period_start=start.isoformat(), period_end=end.isoformat(), days=days)


def format_weekly_report(report: WeeklyReport) -> str:
    lines = [
        "=== Weekly Study Report ===",
        f"Period: {report.period_start} to {report.period_end}",
        f"Attempts: {report.attempts}",
        f"Correct: {report.correct}",
        f"Accuracy: {report.accuracy:.1f}%",
        f"By mode: sentences={report.sentences} flashcards={report.flashcards} "
        f"to_be={report.to_be}",
        "",
        "Daily breakdown:",
    ]
    lines.extend(
        f"{day.date} | {day.attempts} attempts | {day.accuracy:.0f}% | "
        f"S:{day.sentences} F:{day.flashcards} T:{day.to_be}"
        for day in report.days
    )
    return "\n".join(lines)


def export_progress_snapshot(
    progress: Progress,
    out_dir: Path,
    file_format: ExportFormat = "json",
    *,
    now: datetime | None = None,
) -> Path:
    """Write a summary plus weekly report to ``out_dir`` and return the file path."""
    if file_format not in ("json", "csv"):
        raise ValueError("file_format must be 'json' or 'csv'")

    moment = now or datetime.now()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"study_report_{moment:%Y%m%d_%H%M%S}.{file_format}"
    summary = compute_overall(progress, today=moment.date())
    weekly = build_weekly_report(progress, today=moment.date())

    if file_format == "json":
        payload = {
            "generated_at": moment.isoformat(timespec="seconds"),
            "summary": summary,
            "weekly": weekly.to_dict(),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        writer.writerows(["summary", key, value] for key, value in summary.items())
        writer.writerow(["weekly", "period_start", weekly.period_start])
        writer.writerow(["weekly", "period_end", weekly.period_end])
        writer.writerow(["weekly", "attempts", weekly.attempts])
        writer.writerow(["weekly", "correct", weekly.correct])
        writer.writerow(["weekly", "accuracy", f"{weekly.accuracy:.2f}"])
        writer.writerow([])
        writer.writerow(
            ["date", "attempts", "correct", "accuracy", "sentences", "flashcards", "to_be"]
        )
        writer.writerows(
            [
                day.date,
                day.attempts,
                day.correct,
                f"{day.accuracy:.2f}",
                day.sentences,
                day.flashcards,
                day.to_be,
            ]
            for day in weekly.days
        )
    return out
