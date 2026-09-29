"""Online dictionary fallback (dictionaryapi.dev) and local suggestions."""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    word: str
    phonetic: str | None = None
    part_of_speech: str | None = None
    definition: str | None = None
    example: str | None = None


def suggest_similar(word: str, known_words: Iterable[str], *, limit: int = 3) -> list[str]:
    return difflib.get_close_matches(word.lower(), list(known_words), n=limit, cutoff=0.6)


class OnlineDictionary:
    """Returns the first meaning for a word; network problems yield ``None``."""

    def __init__(self, *, session: requests.Session | None = None, timeout: float = 10.0) -> None:
        self._session = session or requests.Session()
        self.timeout = timeout

    def lookup(self, word: str) -> DictionaryEntry | None:
        word = word.strip().lower()
        if not word:
            return None
        try:
            response = self._session.get(API_URL.format(word=quote(word)), timeout=self.timeout)
        except requests.RequestException as exc:
            logger.info("dictionary.request_failed", extra={"error": type(exc).__name__})
            return None
        if response.status_code != 200:
            return None
        try:
            data = response.json()
            entry = data[0]
            meaning = entry["meanings"][0]
            definition = meaning["definitions"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            return None
        return DictionaryEntry(
            word=word,
            phonetic=entry.get("phonetic"),
            part_of_speech=meaning.get("partOfSpeech"),
            definition=definition.get("definition"),
            example=definition.get("example"),
        )
