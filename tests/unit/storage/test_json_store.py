from __future__ import annotations

import logging
from pathlib import Path

import pytest

from english_trainer.storage.json_store import read_json_object, write_json_atomic


def test_round_trip_preserves_unicode_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.json"

    write_json_atomic(target, {"pt": "você está à mesa"})

    assert read_json_object(target) == {"pt": "você está à mesa"}
    assert not list(target.parent.glob("*.tmp"))


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert read_json_object(tmp_path / "missing.json") == {}


@pytest.mark.parametrize("payload", ["{invalid", "[1, 2, 3]"])
def test_unusable_file_is_logged_and_read_as_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: str
) -> None:
    target = tmp_path / "progress.json"
    target.write_text(payload, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="english_trainer"):
        assert read_json_object(target) == {}
    assert caplog.records
