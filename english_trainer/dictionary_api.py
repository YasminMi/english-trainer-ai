"""Free Dictionary API client (dictionaryapi.dev) — online fallback for local lookups."""

from typing import Any, Dict, Optional
import requests

API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"


def fetch_dictionary_entry(word: str) -> Optional[Dict[str, Any]]:
    try:
        r: requests.Response = requests.get(API_URL.format(word=word), timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        if not isinstance(data, list) or not data:
            return None

        entry = data[0]
        phonetic = entry.get("phonetic")

        meanings = entry.get("meanings", [])
        if not meanings:
            return None
        m0 = meanings[0]
        pos = m0.get("partOfSpeech")

        defs = m0.get("definitions", [])
        if not defs:
            return None
        d0 = defs[0]
        definition = d0.get("definition")
        example = d0.get("example")

        return {
            "word": word,
            "phonetic": phonetic,
            "part_of_speech": pos,
            "definition": definition,
            "example": example,
        }
    except requests.RequestException:
        return None
