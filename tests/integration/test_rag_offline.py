"""End-to-end retrieval over the real study material with a deterministic embedder."""

from __future__ import annotations

import pytest

from english_trainer.__main__ import main
from english_trainer.interfaces.evaluation import DEFAULT_DATASET


def test_eval_rag_command_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval-rag", "--embedder", "hashing", "--json"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"embedder": "hashing-512"' in output
    assert '"hit@5"' in output


def test_quality_gate_fails_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval-rag", "--embedder", "hashing", "--min-hit-rate", "1.01"]) == 1
    assert "Quality gate failed" in capsys.readouterr().err
    assert DEFAULT_DATASET.name == "rag_golden.jsonl"
