"""Crash-safe JSON persistence for single-user local data."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning ``{}`` when the file is missing or unusable.

    Corrupt local files are logged instead of crashing the app, so a bad
    progress file never locks the learner out of their study material.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("storage.read_failed", extra={"path": path.name, "error": str(exc)})
        return {}
    if not isinstance(data, dict):
        logger.warning("storage.not_an_object", extra={"path": path.name})
        return {}
    return data


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Write to a temporary sibling file, fsync it, then atomically replace ``path``."""
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
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
