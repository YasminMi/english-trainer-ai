"""Dictionary mode: local lookup with online fallback and fuzzy suggestions."""

from typing import Dict, Any
import difflib

from .data_access import find_in_dictionary
from .dictionary_api import fetch_dictionary_entry
from .progress import register_dictionary_lookup
from .utils import pause, read_choice, clear_screen


def run_mini_dictionary(content: Dict[str, Any], progress: Dict[str, Any]) -> None:
    dictionary = content.get("dictionary", {})
    if not dictionary:
        print("No dictionary data.")
        pause()
        return

    while True:
        clear_screen()
        print("=== Mode 4: Mini Dictionary ===")
        print("Local dictionary + online fallback.")
        print("Type a word in English (example: this, that, were).")
        print("Type '0' to go back.\n")

        word: str = read_choice("Word: ")
        if word == "0":
            break

        register_dictionary_lookup(progress)

        meaning: str | None = find_in_dictionary(content, word)
        if meaning:
            print(f"Meaning (PT): {meaning} ✅")
            pause()
            continue

        entry: Dict[str, Any] | None = fetch_dictionary_entry(word.lower())
        if entry:
            print("Found online ✅")
            if entry.get("phonetic"):
                print(f"Phonetic: {entry['phonetic']}")
            if entry.get("part_of_speech"):
                print(f"Type: {entry['part_of_speech']}")
            if entry.get("definition"):
                print(f"Definition (EN): {entry['definition']}")
            if entry.get("example"):
                print(f"Example: {entry['example']}")
        else:
            print("Not found ❌")
            close_matches = difflib.get_close_matches(word.lower(), dictionary.keys(), n=3, cutoff=0.6)
            if close_matches:
                print("Did you mean:")
                for m in close_matches:
                    print(f" - {m}")

        pause()
