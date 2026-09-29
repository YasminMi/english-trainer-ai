"""Terminal input/output helpers shared by every CLI flow."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def clear_screen() -> None:
    # ANSI clear + cursor home works in modern Windows terminals and every POSIX one,
    # and avoids spawning a shell (the previous `os.system("cls")`).
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    elif os.environ.get("TERM") == "dumb":
        print()


def ask(prompt: str) -> str:
    """Read one stripped line.

    ``EOFError`` (Ctrl+D, or piped input running out) propagates so the main
    loop can exit cleanly instead of re-prompting forever.
    """
    return input(prompt).strip()


def pause(message: str = "Press ENTER to continue...") -> None:
    print(f"\n{message}")
    ask("")


def header(title: str) -> None:
    clear_screen()
    print(f"=== {title} ===")


def ask_positive_int(prompt: str) -> int | None:
    """Return a positive integer, or ``None`` for empty or invalid input."""
    raw = ask(prompt)
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def choose(title: str, options: Sequence[tuple[str, T]], *, back_label: str = "Back") -> T | None:
    """Render a numbered menu and return the chosen value, or ``None`` for back.

    Invalid input re-renders the menu instead of silently picking something.
    """
    while True:
        header(title)
        for index, (label, _) in enumerate(options, start=1):
            print(f"{index}) {label}")
        print(f"0) {back_label}")
        choice = ask("Choose: ")
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        pause("Invalid option. Press ENTER...")


def session_line(correct: int, total: int) -> str:
    pct = int(correct / total * 100) if total else 0
    return f"Session: {correct}/{total} ({pct}%)"
