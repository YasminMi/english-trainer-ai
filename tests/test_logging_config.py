from __future__ import annotations

import json
import logging
from pathlib import Path

from english_trainer.config import LoggingSettings
from english_trainer.logging_config import configure_logging


def test_json_logs_include_extra_fields(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    configure_logging(LoggingSettings(level="INFO", json_format=True), log_file=log_file)

    logging.getLogger("english_trainer.test").info("rag.retrieved", extra={"hits": 3})
    for handler in logging.getLogger("english_trainer").handlers:
        handler.flush()

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["event"] == "rag.retrieved"
    assert record["hits"] == 3
    assert record["level"] == "INFO"


def test_level_filters_lower_severity(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    configure_logging(LoggingSettings(level="warning"), log_file=log_file)

    logging.getLogger("english_trainer.test").info("hidden")
    logging.getLogger("english_trainer.test").warning("shown", extra={"reason": "x"})
    for handler in logging.getLogger("english_trainer").handlers:
        handler.flush()

    text = log_file.read_text(encoding="utf-8")
    assert "hidden" not in text
    assert "shown" in text
    assert "reason='x'" in text
