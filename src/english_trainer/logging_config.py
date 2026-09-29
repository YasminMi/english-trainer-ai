"""Logging setup shared by the CLI and the TUI.

Logs go to ``stderr`` (or a file for the TUI, which owns the terminal). The JSON
formatter emits one object per line so events such as guardrail decisions and
retrieval timings can be filtered with standard tools.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from english_trainer.config import LoggingSettings

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, including ``extra={...}`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        )
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class KeyValueFormatter(logging.Formatter):
    """Human-readable formatter that appends ``extra`` fields as ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        if not extras:
            return base
        pairs = " ".join(f"{key}={value!r}" for key, value in sorted(extras.items()))
        return f"{base} | {pairs}"


def configure_logging(settings: LoggingSettings, *, log_file: Path | None = None) -> None:
    """Configure the ``english_trainer`` logger hierarchy.

    Only the package logger is configured so third-party libraries keep their
    own defaults and the host application stays in control of the root logger.
    """
    handler: logging.Handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    if settings.json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            KeyValueFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )

    logger = logging.getLogger("english_trainer")
    for previous in logger.handlers[:]:
        logger.removeHandler(previous)
        previous.close()
    logger.addHandler(handler)
    logger.setLevel(settings.level)
    logger.propagate = False
