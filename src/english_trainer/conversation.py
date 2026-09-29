"""
Mode 6: Free Conversation with AI.

- Chat freely in English with an AI conversation partner
- The AI uses simple language (A1/A2) and corrects grammar gently
- At the end of each session, the AI reviews all grammar mistakes
- Optionally saved to data/conversations/ for review
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hf_llm import generate_conversation_reply, OLLAMA_URL
from .utils import clear_screen, read_choice, pause

BASE_DIR = Path(__file__).resolve().parents[2]
CONV_DIR = BASE_DIR / "data" / "conversations"

SYSTEM_PROMPT = """\
You are Miu, a friendly personal AI English tutor and conversation partner for a Brazilian beginner (A1/A2 level).

Rules:
- Always respond in English using very simple words and short sentences.
- If the user makes a grammar mistake, gently show the correction on its own line like this:
  Correction: "I am tired" not "I are tired"
- After correcting, continue the conversation naturally — do not dwell on the error.
- Ask one follow-up question to keep the conversation going.
- Be encouraging and patient.
- Keep each reply short: 2-4 sentences maximum.
- Do not use bullet points or headers. Write naturally.
- Never use parentheses. Do not add asides, translations, or extra explanations in parentheses or brackets.
"""

REVIEW_PROMPT = """\
You are an English teacher reviewing a Brazilian beginner's messages.
Find all grammar mistakes and list them clearly.

Output format - one mistake per line:
  [X] what they wrote -> [OK] correct version
      Dica: short explanation in Portuguese (max 10 words)

Rules:
- Only list real mistakes: wrong verb form, missing word, wrong word order, spelling, missing preposition.
- Do not list things that were correct.
- If there were no mistakes, write only: "Otimo! Nenhum erro encontrado."
- Do not add any text before or after the list.
- Be encouraging.
"""


def _check_ollama() -> bool:
    import requests
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _get_error_review(history: List[Dict[str, str]]) -> Optional[str]:
    """Ask the AI to review all user messages and list grammar mistakes."""
    user_messages = [m for m in history if m["role"] == "user"]
    if len(user_messages) < 2:
        return None

    user_text = "\n".join(
        f"Message {i+1}: {m['content']}"
        for i, m in enumerate(user_messages)
    )
    review_messages = [
        {"role": "user", "content": f"Here are all my messages from our conversation:\n\n{user_text}\n\nPlease review my grammar."}
    ]
    return generate_conversation_reply(review_messages, system_prompt=REVIEW_PROMPT)


def _save_conversation(history: List[Dict[str, str]], review: Optional[str]) -> Path:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = CONV_DIR / f"conversation_{timestamp}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Conversation — {timestamp}\n")
        f.write("=" * 40 + "\n\n")
        for msg in history:
            role = "You" if msg["role"] == "user" else "Miu"
            f.write(f"{role}: {msg['content']}\n\n")
        if review:
            f.write("=" * 40 + "\n")
            f.write("Grammar Review\n")
            f.write("=" * 40 + "\n\n")
            f.write(review + "\n")
    return path


def run_conversation(progress: Dict[str, Any]) -> None:
    clear_screen()
    print("=== Conversation with Miu ===")
    print("Chat freely in English. Miu will correct your mistakes gently.")
    print("Type '0' or 'bye' to end the conversation.\n")

    if not _check_ollama():
        print("Ollama is not running.")
        print()
        print("To start Ollama:")
        print("  1. Install Ollama: https://ollama.com")
        print("  2. Open a terminal and run:  ollama serve")
        print("  3. Download a model:         ollama pull llama3.2")
        print()
        print("After that, restart this app and try again.")
        pause()
        return

    history: List[Dict[str, str]] = []

    starter = generate_conversation_reply(
        [{"role": "user", "content": "Hello! Let's start a conversation to practice my English."}],
        system_prompt=SYSTEM_PROMPT,
    )
    if not starter:
        print("Could not connect to the AI. Is a model loaded in Ollama?")
        print("Tip: run  ollama pull llama3.2  in a terminal.")
        pause()
        return

    print(f"Miu: {starter}\n")
    history.append({"role": "user", "content": "Hello! Let's start a conversation to practice my English."})
    history.append({"role": "assistant", "content": starter})

    while True:
        user_input = read_choice("You: ").strip()

        if user_input == "0" or user_input.lower() in ("exit", "quit", "bye"):
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        print("Miu is thinking...", end="\r")
        reply = generate_conversation_reply(history, system_prompt=SYSTEM_PROMPT)

        if reply is None:
            print("Miu: (no response — check if Ollama is still running)\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"Miu: {reply}\n")

    user_turns = [m for m in history if m["role"] == "user"][1:]
    turns = len(user_turns)

    if turns == 0:
        print("\nNo messages sent.")
        return

    print()
    print(f"Conversation ended — {turns} message{'s' if turns != 1 else ''} sent.")

    print("\nAnalyzing your grammar mistakes...\n")
    review = _get_error_review(history)

    if review:
        print("=" * 40)
        print(" Grammar Review")
        print("=" * 40)
        print(review.strip())
        print("=" * 40)
    else:
        print("(Could not generate grammar review.)")

    print()
    save = read_choice("Save this conversation + review? (y/n): ").strip().lower()
    if save == "y":
        path = _save_conversation(history, review)
        rel = path.relative_to(BASE_DIR)
        print(f"Saved to: {rel}")

    progress["conversation_sessions"] = int(progress.get("conversation_sessions", 0)) + 1
    progress["conversation_turns"] = int(progress.get("conversation_turns", 0)) + turns

    pause()

