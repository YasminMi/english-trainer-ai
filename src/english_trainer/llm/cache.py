"""Persistent cache for deterministic generations (same model + prompt)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from english_trainer.storage.json_store import read_json_object, write_json_atomic


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, key: str) -> str | None:
        entry = read_json_object(self.path).get(key)
        text = entry.get("text") if isinstance(entry, dict) else None
        return text if isinstance(text, str) else None

    def put(self, key: str, text: str, *, provider: str) -> None:
        data = read_json_object(self.path)
        data[key] = {"text": text, "model": provider}
        write_json_atomic(self.path, data)
