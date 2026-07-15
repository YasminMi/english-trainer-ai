import sys
import io

# Windows terminals may not default to UTF-8.
if sys.platform == "win32" and isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    if "--cli" in sys.argv:
        from english_trainer.cli import run_app
        run_app()
    else:
        try:
            from english_trainer.tui_app import run_tui
        except ModuleNotFoundError as exc:
            if exc.name != "textual":
                raise
            print("Textual não instalado. Execute: pip install textual")
            print("Ou use o modo terminal: python main.py --cli")
        else:
            run_tui()
