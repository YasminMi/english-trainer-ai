"""Shared utilities: text normalization, answer matching, and CLI helpers."""

import os
import platform
import re
import unicodedata
from typing import Iterable


def clear_screen() -> None:
    system: str = platform.system().lower()
    if "windows" in system:
        os.system("cls")
    else:
        os.system("clear")


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s/]", "", text)
    return " ".join(text.split())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_pt_phrase(text: str) -> str:
    """Fix legacy PT-BR negation word order (e.g. 'estou não' → 'não estou')."""
    normalized = text
    replacements = [
        (r"^(Eu) estou não (.+)$", r"\1 não estou \2"),
        (r"^(Você) está não (.+)$", r"\1 não está \2"),
        (r"^(Ele) está não (.+)$", r"\1 não está \2"),
        (r"^(Ela) está não (.+)$", r"\1 não está \2"),
        (r"^(Isso) está não (.+)$", r"\1 não está \2"),
        (r"^(Nós) estamos não (.+)$", r"\1 não estamos \2"),
        (r"^(Eles/Elas) estão não (.+)$", r"\1 não estão \2"),
        (r"^(Eu) estava não (.+)$", r"\1 não estava \2"),
        (r"^(Você) estava não (.+)$", r"\1 não estava \2"),
        (r"^(Ele) estava não (.+)$", r"\1 não estava \2"),
        (r"^(Ela) estava não (.+)$", r"\1 não estava \2"),
        (r"^(Isso) estava não (.+)$", r"\1 não estava \2"),
        (r"^(Nós) estávamos não (.+)$", r"\1 não estávamos \2"),
        (r"^(Eles/Elas) estavam não (.+)$", r"\1 não estavam \2"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _normalize_pt_synonyms(text: str) -> str:
    # existential há/existe → tem  ("há um cachorro" = "tem um cachorro")
    text = re.sub(r"\bha\b", "tem", text)          # "há" after accent strip → "ha"
    text = re.sub(r"\bexistem\b", "tem", text)
    text = re.sub(r"\bexiste\b", "tem", text)
    # optional leading subject pronoun (PT-BR commonly drops it)
    text = re.sub(r"^(eu|voce|ele|ela|isso|nos|eles|elas)\s+", "", text)
    # "a gente" → "nos"
    text = re.sub(r"\ba gente\b", "nos", text)
    return text


def _normalize_matching_text(text: str) -> str:
    normalized = _clean_text(text)
    normalized = normalize_pt_phrase(normalized)
    normalized = _strip_accents(normalized)          # é→e, ã→a, ê→e …
    normalized = re.sub(r"\bpra\b", "para", normalized)
    normalized = re.sub(r"\bpro\b", "para o",  normalized)
    normalized = re.sub(r"\bpras\b", "para as", normalized)
    normalized = re.sub(r"\bpros\b", "para os", normalized)
    normalized = re.sub(r"\bpara\s+(a|o|as|os)\b", "para", normalized)
    normalized = re.sub(r"\bem\s+(a|o|as|os)\b", "em", normalized)
    normalized = _normalize_pt_synonyms(normalized)
    return " ".join(normalized.split())


def _strip_parentheticals(text: str) -> str:
    """Remove contextual notes in parentheses, e.g. 'sou/estou (com I)' → 'sou/estou'."""
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def _expand_slash_variations(text: str) -> list[str]:
    text = _clean_text(text)
    if "/" not in text:
        return [text]

    # " / " with surrounding spaces separates independent alternative groups,
    # e.g. "é/está / são/estão" means any of é, está, são, estão are valid.
    phrases = re.split(r"\s+/\s+", text)

    all_results: list[str] = []
    for phrase in phrases:
        if "/" not in phrase:
            all_results.append(phrase)
            continue
        tokens = phrase.split()
        results: list[str] = [""]
        for token in tokens:
            if "/" in token:
                options = token.split("/")
                new_results = []
                for base in results:
                    for opt in options:
                        new_results.append((base + " " + opt).strip())
                results = new_results
            else:
                results = [(base + " " + token).strip() for base in results]
        all_results.extend(results)

    return list(dict.fromkeys(all_results))


def is_close_answer(user_answer: str, expected: str | list[str], strict: bool = False) -> bool:
    """Match user answer against expected.

    strict=True disables partial-phrase matching (required for dictation).
    """
    ua: str = _clean_text(user_answer)
    if not ua:
        return False

    if isinstance(expected, str):
        candidates: Iterable[str] = [expected]
    else:
        candidates = expected

    expanded_expected: list[str] = []
    for exp in candidates:
        clean_exp = _strip_parentheticals(exp)
        expanded_expected.extend(_expand_slash_variations(clean_exp))
        normalized_exp = normalize_pt_phrase(clean_exp)
        if normalized_exp != clean_exp:
            expanded_expected.extend(_expand_slash_variations(normalized_exp))

    ua_simple: str = " ".join(ua.replace("/", " ").split())
    ua_normalized: str = _normalize_matching_text(ua_simple)
    ua_tokens = ua_normalized.split()

    def _is_phrase_subset(short_text: str, long_text: str) -> bool:
        short_tokens = short_text.split()
        long_tokens = long_text.split()
        if not short_tokens or len(short_tokens) > len(long_tokens):
            return False
        width = len(short_tokens)
        for i in range(0, len(long_tokens) - width + 1):
            if long_tokens[i : i + width] == short_tokens:
                return True
        return False

    for ex in expanded_expected:
        ex_normalized = _normalize_matching_text(ex)
        if ua_simple == ex or ua_normalized == ex_normalized:
            return True
        if not strict and (len(ua_normalized) >= 4 or len(ua_tokens) >= 2) and _is_phrase_subset(ua_normalized, ex_normalized):
            return True
    return False


def read_choice(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def pause(msg: str = "Press ENTER to continue...") -> None:
    input(f"\n{msg}")
