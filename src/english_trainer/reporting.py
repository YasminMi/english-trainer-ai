"""Weekly study reports and progress snapshots."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .data_access import DATA_DIR
from .progress import compute_overall


def _reports_dir() -> Path:
    out = DATA_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_weekly_report(progress: dict[str, Any], when: date | None = None) -> dict[str, Any]:
    day = when or date.today()
    start = day - timedelta(days=6)
    history = progress.get("daily", {}).get("history", {})

    days = []
    total_attempts = 0
    total_correct = 0
    total_sentences = 0
    total_flashcards = 0
    total_to_be = 0

    for offset in range(7):
        current = start + timedelta(days=offset)
        key = current.isoformat()
        stats = history.get(key, {})
        attempts = _safe_int(stats.get("attempts"))
        correct = _safe_int(stats.get("correct"))
        sentences = _safe_int(stats.get("sentences"))
        flashcards = _safe_int(stats.get("flashcards"))
        to_be = _safe_int(stats.get("to_be"))

        total_attempts += attempts
        total_correct += correct
        total_sentences += sentences
        total_flashcards += flashcards
        total_to_be += to_be

        days.append(
            {
                "date": key,
                "attempts": attempts,
                "correct": correct,
                "accuracy": (correct / attempts * 100.0) if attempts else 0.0,
                "sentences": sentences,
                "flashcards": flashcards,
                "to_be": to_be,
            }
        )

    return {
        "period_start": start.isoformat(),
        "period_end": day.isoformat(),
        "attempts": total_attempts,
        "correct": total_correct,
        "accuracy": (total_correct / total_attempts * 100.0) if total_attempts else 0.0,
        "sentences": total_sentences,
        "flashcards": total_flashcards,
        "to_be": total_to_be,
        "days": days,
    }


def format_weekly_report(report: dict[str, Any]) -> str:
    lines = [
        "=== Weekly Study Report ===",
        f"Period: {report['period_start']} to {report['period_end']}",
        f"Attempts: {report['attempts']}",
        f"Correct: {report['correct']}",
        f"Accuracy: {report['accuracy']:.1f}%",
        f"By mode: sentences={report['sentences']} flashcards={report['flashcards']} to_be={report['to_be']}",
        "",
        "Daily breakdown:",
    ]

    for day in report.get("days", []):
        lines.append(
            f"{day['date']} | {day['attempts']} attempts | {day['accuracy']:.0f}% | "
            f"S:{day['sentences']} F:{day['flashcards']} T:{day['to_be']}"
        )

    return "\n".join(lines)


def export_progress_snapshot(progress: dict[str, Any], file_format: str = "json") -> Path:
    fmt = (file_format or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        raise ValueError("file_format must be 'json' or 'csv'")

    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = _reports_dir()

    summary = compute_overall(progress)
    weekly = build_weekly_report(progress, when=now.date())

    if fmt == "json":
        out = out_dir / f"study_report_{stamp}.json"
        payload = {
            "generated_at": now.isoformat(timespec="seconds"),
            "summary": summary,
            "weekly": weekly,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    out = out_dir / f"study_report_{stamp}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for k, v in summary.items():
            w.writerow(["summary", k, v])
        w.writerow(["weekly", "period_start", weekly["period_start"]])
        w.writerow(["weekly", "period_end", weekly["period_end"]])
        w.writerow(["weekly", "attempts", weekly["attempts"]])
        w.writerow(["weekly", "correct", weekly["correct"]])
        w.writerow(["weekly", "accuracy", f"{weekly['accuracy']:.2f}"])
        w.writerow([])
        w.writerow(["date", "attempts", "correct", "accuracy", "sentences", "flashcards", "to_be"])
        for day in weekly["days"]:
            w.writerow(
                [
                    day["date"],
                    day["attempts"],
                    day["correct"],
                    f"{day['accuracy']:.2f}",
                    day["sentences"],
                    day["flashcards"],
                    day["to_be"],
                ]
            )
    return out
