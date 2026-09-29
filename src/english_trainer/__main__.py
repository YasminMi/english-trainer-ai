"""Command-line entry point: ``english-trainer`` or ``python -m english_trainer``."""

from __future__ import annotations

import argparse
import io
import sys


def _configure_stdio() -> None:
    # Windows terminals may not default to UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="english-trainer",
        description="Offline-first English trainer with spaced repetition and a local RAG tutor.",
    )
    parser.add_argument("--cli", action="store_true", help="use the plain terminal interface")
    args = parser.parse_args(argv)

    _configure_stdio()

    from dotenv import load_dotenv

    load_dotenv()

    if args.cli:
        from english_trainer.cli import run_app

        run_app()
        return 0

    try:
        from english_trainer.tui_app import run_tui
    except ModuleNotFoundError as exc:
        if exc.name != "textual":
            raise
        print("Textual is not installed. Run: pip install textual", file=sys.stderr)
        print("Or use the terminal interface: english-trainer --cli", file=sys.stderr)
        return 1

    run_tui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
