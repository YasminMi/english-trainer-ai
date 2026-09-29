"""Command-line entry point: ``english-trainer`` or ``python -m english_trainer``.

``english-trainer``            Textual TUI (default)
``english-trainer cli``        plain terminal interface (``--cli`` still works)
``english-trainer eval-rag``   retrieval quality report on the gold question set
"""

from __future__ import annotations

import argparse
import io
import sys


def _configure_stdio() -> None:
    # Windows terminals may not default to UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    from english_trainer import __version__
    from english_trainer.interfaces import evaluation

    parser = argparse.ArgumentParser(
        prog="english-trainer",
        description="Offline-first English trainer with spaced repetition and a local RAG tutor.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser("tui", help="start the Textual interface (default)")
    commands.add_parser("cli", help="start the plain terminal interface")
    commands.add_parser("validate-content", help="validate the shipped versioned study content")
    evaluation.add_arguments(
        commands.add_parser("eval-rag", help="evaluate retrieval on the gold question set")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = "cli" if args.cli else (args.command or "tui")

    _configure_stdio()

    from dotenv import load_dotenv

    load_dotenv()

    from english_trainer.config import get_settings
    from english_trainer.logging_config import configure_logging

    settings = get_settings()
    # The TUI owns the terminal, so its logs go to a file instead of stderr.
    log_file = (
        settings.paths.data_dir / "logs" / "english-trainer.log" if command == "tui" else None
    )
    configure_logging(settings.logging, log_file=log_file)

    if command == "eval-rag":
        from english_trainer.interfaces import evaluation

        return evaluation.run(args, settings)

    if command == "validate-content":
        from english_trainer.storage.content import ContentRepository, ContentValidationError

        try:
            ContentRepository(settings.paths).validate_versioned()
        except (ContentValidationError, ValueError) as exc:
            print(f"Content validation failed:\n{exc}", file=sys.stderr)
            return 1
        print(f"Valid content: {settings.paths.content_file}")
        return 0

    if command == "cli":
        from english_trainer.interfaces import cli

        cli.run()
        return 0

    try:
        from english_trainer.interfaces.tui.app import run
    except ModuleNotFoundError as exc:
        if exc.name != "textual":
            raise
        print("Textual is not installed. Run: pip install textual", file=sys.stderr)
        print("Or use the terminal interface: english-trainer cli", file=sys.stderr)
        return 1

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
