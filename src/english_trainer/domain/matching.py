"""Answer matching tolerant to accents, punctuation and common PT-BR variations."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_LEGACY_NEGATIONS: tuple[tuple[str, str], ...] = tuple(
    (rf"^({subject}) {verb} não (.+)$", rf"\1 não {verb} \2")
    for subject, verb in (
        ("Eu", "estou"),
        ("Você", "está"),
        ("Ele", "está"),
        ("Ela", "está"),
        ("Isso", "está"),
        ("Nós", "estamos"),
        ("Eles/Elas", "estão"),
        ("Eu", "estava"),
        ("Você", "estava"),
        ("Ele", "estava"),
        ("Ela", "estava"),
        ("Isso", "estava"),
        ("Nós", "estávamos"),
        ("Eles/Elas", "estavam"),
    )
)

_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bpra\b", "para"),
    (r"\bpro\b", "para o"),
    (r"\bpras\b", "para as"),
    (r"\bpros\b", "para os"),
    (r"\bpara\s+(a|o|as|os)\b", "para"),
    (r"\bem\s+(a|o|as|os)\b", "em"),
)

_SYNONYMS: tuple[tuple[str, str], ...] = (
    # Existential "há/existe" and "tem" are interchangeable in PT-BR ("há" loses its accent first).
    (r"\bha\b", "tem"),
    (r"\bexistem\b", "tem"),
    (r"\bexiste\b", "tem"),
    # PT-BR commonly drops the subject pronoun.
    (r"^(eu|voce|ele|ela|isso|nos|eles|elas)\s+", ""),
    (r"\ba gente\b", "nos"),
)


def clean_text(text: str) -> str:
    """Lowercase, drop punctuation (keeping ``/`` alternatives) and collapse whitespace."""
    text = re.sub(r"[^\w\s/]", "", text.lower())
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_pt_phrase(text: str) -> str:
    """Fix legacy PT-BR negation word order (``estou não`` -> ``não estou``)."""
    for pattern, replacement in _LEGACY_NEGATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def _normalize_for_matching(text: str) -> str:
    normalized = strip_accents(normalize_pt_phrase(clean_text(text)))
    for pattern, replacement in (*_CONTRACTIONS, *_SYNONYMS):
        normalized = re.sub(pattern, replacement, normalized)
    return " ".join(normalized.split())


def _strip_parentheticals(text: str) -> str:
    """Remove contextual notes such as ``sou/estou (com I)``."""
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def expand_slash_variations(text: str) -> list[str]:
    """Expand ``cansado/cansada`` style alternatives into every concrete phrase.

    `` / `` (with spaces) separates independent alternatives, so
    ``é/está / são/estão`` accepts any of the four verbs.
    """
    text = clean_text(text)
    if "/" not in text:
        return [text]

    results: list[str] = []
    for phrase in re.split(r"\s+/\s+", text):
        variants = [""]
        for token in phrase.split():
            options = token.split("/")
            variants = [f"{base} {option}".strip() for base in variants for option in options]
        results.extend(variants)
    return list(dict.fromkeys(results))


def _contains_phrase(short_text: str, long_text: str) -> bool:
    short_tokens = short_text.split()
    long_tokens = long_text.split()
    width = len(short_tokens)
    if not short_tokens or width > len(long_tokens):
        return False
    return any(
        long_tokens[start : start + width] == short_tokens
        for start in range(len(long_tokens) - width + 1)
    )


def is_close_answer(
    user_answer: str, expected: str | Iterable[str], *, strict: bool = False
) -> bool:
    """Return whether ``user_answer`` matches any expected answer.

    ``strict`` disables partial-phrase matching, which dictation requires.
    """
    answer = clean_text(user_answer)
    if not answer:
        return False

    candidates = [expected] if isinstance(expected, str) else list(expected)
    accepted: list[str] = []
    for candidate in candidates:
        base = _strip_parentheticals(candidate)
        accepted.extend(expand_slash_variations(base))
        fixed = normalize_pt_phrase(base)
        if fixed != base:
            accepted.extend(expand_slash_variations(fixed))

    answer_plain = " ".join(answer.replace("/", " ").split())
    answer_normalized = _normalize_for_matching(answer_plain)
    long_enough = len(answer_normalized) >= 4 or len(answer_normalized.split()) >= 2

    for option in accepted:
        option_normalized = _normalize_for_matching(option)
        if answer_plain == option or answer_normalized == option_normalized:
            return True
        if not strict and long_enough and _contains_phrase(answer_normalized, option_normalized):
            return True
    return False
