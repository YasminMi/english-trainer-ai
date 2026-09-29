"""Miu coach: generates level-appropriate sentences and explains mistakes in simple Portuguese."""

from typing import Dict, Any, List
import re
import random

from .hf_llm import generate_text
from .progress import register_ai_use
from .utils import clear_screen, pause, read_choice, _clean_text, is_close_answer


SYSTEM_STYLE = (
    "You are an English tutor for beginners. "
    "Use very simple English. "
    "Explain in simple Portuguese (Brazil). "
    "Be short, practical, and give 1-2 examples. "
    "Never show your reasoning. Do not use tags like <think>. "
    "Output only what is requested."
)


def _prompt_generate_sentences(targets: List[str], tense: str, n: int) -> str:
    targets_str: str = ", ".join(targets)
    return (
        f"{SYSTEM_STYLE}\n"
        f"Task: Generate {n} beginner English sentences.\n"
        f"Rules:\n"
        f"- Use ONLY very basic vocabulary (kid level).\n"
        f"- Focus on: {targets_str}\n"
        f"- Use verb 'to be' with tense: {tense}\n"
        f"- Output format: one sentence per line in this exact format:\n"
        f"  EN | PT | DICA\n"
        f"  Where EN is a single simple sentence (no extra sentences),\n"
        f"  PT is a single simple translation, and DICA is a very short Portuguese tip (max 8 words).\n"
        f"  Do not add anything else. No numbering. No extra text. No reasoning.\n"
        f"- If you are unsure, still output {n} simple sentences.\n"
    )


def _prompt_explain_mistake(en_sentence: str, user_pt: str, expected_pt: str) -> str:
    return (
        f"{SYSTEM_STYLE}\n"
        f"English sentence: {en_sentence}\n"
        f"User Portuguese: {user_pt}\n"
        f"Expected Portuguese: {expected_pt}\n"
        f"Task: Say if user's translation is acceptable. "
        f"If acceptable, say 'ACEITAVEL' and give a short reason. "
        f"If not, say 'CORRIGIR' and provide the corrected Portuguese and a short reason.\n"
    )


def _pick_targets() -> List[str]:
    pools: List[List[str]] = [
        ["this", "that"],
        ["some", "how"],
        ["they", "we"],
        ["am", "is", "are"],
        ["was", "were"],
    ]
    return random.choice(pools)


def _fallback_sentences(targets: List[str], tense: str, n: int) -> List[str]:
    nouns: List[str] = ["book", "ball", "cat", "dog", "car", "house"]
    subjects: List[str] = ["I", "You", "We", "They", "He", "She", "It"]
    be_present: Dict[str, str] = {"I": "am", "You": "are", "We": "are", "They": "are", "He": "is", "She": "is", "It": "is"}
    be_past: Dict[str, str] = {"I": "was", "You": "were", "We": "were", "They": "were", "He": "was", "She": "was", "It": "was"}
    be: Dict[str, str] = be_present if tense == "present" else be_past

    out: List[str] = []
    tset: List[str] = [t.lower() for t in targets]

    if "this" in tset:
        out.append("This is a " + random.choice(nouns) + ".")
    if "that" in tset:
        out.append("That is a " + random.choice(nouns) + ".")
    if "some" in tset:
        out.append("These are some " + random.choice(nouns) + "s.")
    if "how" in tset:
        out.append("How are you?")
    if "they" in tset:
        out.append("They are here.")
    if "we" in tset:
        out.append("We are ready.")

    i = 0
    while len(out) < n and i < 20:
        subj: str = random.choice(subjects)
        out.append(f"{subj} {be[subj]} here.")
        i += 1

    return out[:n]


def _fallback_pt(en: str) -> str:
    replacements: Dict[str, str] = {
        "This is a": "Isto é um/uma",
        "That is a": "Aquilo é um/uma",
        "These are some": "Estas são algumas",
        "How are you?": "Como você está?",
        "They are here.": "Eles/Elas estão aqui.",
        "We are ready.": "Nós estamos prontos/prontas.",
        "I am here.": "Eu estou aqui.",
        "You are here.": "Você está aqui.",
        "He is here.": "Ele está aqui.",
        "She is here.": "Ela está aqui.",
        "It is here.": "Isso está aqui.",
    }
    for k, v in replacements.items():
        if en.startswith(k):
            return v
    return "Tradução simples."


def _strip_think_tags(lines: List[str]) -> List[str]:
    result = []
    skip = False
    for ln in lines:
        if "<think>" in ln.lower():
            skip = True
            continue
        if skip:
            if "</think>" in ln.lower():
                skip = False
            continue
        if ln.startswith("<") and ln.endswith(">"):
            continue
        result.append(ln)
    return result


def _is_valid_sentence(s: str) -> bool:
    words = re.findall(r"[A-Za-z]+", s)
    return len(words) >= 2 and bool(re.search(r"\b(am|is|are|was|were)\b", s.lower()))


def run_ai_coach(content: Dict[str, Any], progress: Dict[str, Any]) -> None:
    while True:
        clear_screen()
        print("=== Miu — AI Coach ===")
        print("1) Generate new sentences (gerar frases)")
        print("2) Check my translation (corrigir minha tradução)")
        print("0) Back")
        choice: str = read_choice("Choose: ")

        if choice == "0":
            return
        if choice == "1":
            _ai_generate_sentences(progress)
        elif choice == "2":
            _ai_check_translation(content, progress)
        else:
            print("Invalid option.")
            pause()


def _ai_generate_sentences(progress: Dict[str, Any]) -> None:
    clear_screen()
    print("AI will generate sentences for beginners.")
    tense: str = read_choice("Choose tense (present/past): ").lower().strip()
    if tense not in ("present", "past"):
        tense = "present"

    targets: List[str] = _pick_targets()
    n_str: str = read_choice("How many sentences? (default 5): ").strip()
    try:
        n: int = max(1, min(int(n_str) if n_str else 5, 12))
    except ValueError:
        n = 5

    def _parse(text: str | None) -> List[str]:
        if not text:
            return []
        raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [ln for ln in _strip_think_tags(raw) if _is_valid_sentence(ln)]

    lines = _parse(generate_text(_prompt_generate_sentences(targets, tense, n), max_new_tokens=220))
    register_ai_use(progress)

    if not lines:
        retry = (
            f"Generate exactly {n} very simple English sentences using verb 'to be' in {tense} tense. "
            f"Use only basic words. Output one sentence per line. No extra text."
        )
        lines = _parse(generate_text(retry, max_new_tokens=160))
        register_ai_use(progress)

    used_builtin_fallback = False
    if not lines:
        lines = _fallback_sentences(targets, tense, n)
        used_builtin_fallback = True

    print("\n--- Generated sentences ---")
    if used_builtin_fallback:
        print("Limited built-in examples: no AI model evaluated this output.\n")
    didactic = []
    for ln in lines:
        parts: List[str] = [p.strip() for p in ln.split("|")]
        if len(parts) == 3 and all(parts):
            en, pt, tip = parts
            en = re.split(r"[.!?]", en)[0].strip()
            pt = re.split(r"[.!?]", pt)[0].strip()
            if en and pt and tip:
                didactic.append([en, pt, tip])
        else:
            break

    if didactic:
        for en, pt, tip in didactic[:n]:
            print(f"{en}")
            print(f"PT: {pt}")
            print(f"Dica: {tip}")
            print("")
    else:
        for ln in lines[:n]:
            print(ln)
            print(f"PT: {_fallback_pt(ln)}")
            print("")

    print("---------------------------")
    print("\nTip: copy good sentences into data/content.json to train offline.")
    pause()


def _ai_check_translation(content: Dict[str, Any], progress: Dict[str, Any]) -> None:
    clear_screen()
    print("Type an English sentence you are training (or choose from your list).")
    en: str = read_choice("EN sentence: ")
    if not en:
        return
    user_pt: str = read_choice("Your PT: ")

    expected_pt: str = ""
    for s in content.get("sentences", []):
        if _clean_text(s.get("en", "")) == _clean_text(en):
            expected_pt = s.get("pt", "")
            break

    if not expected_pt:
        expected_pt = read_choice("Expected PT (optional, press ENTER to skip): ").strip() or "(not provided)"

    if expected_pt != "(not provided)" and is_close_answer(user_pt, expected_pt):
        print("✅ Acceptable (local check).")
        pause()
        return

    text: str | None = generate_text(
        _prompt_explain_mistake(en_sentence=en, user_pt=user_pt, expected_pt=expected_pt),
        max_new_tokens=220,
    )
    register_ai_use(progress)

    if not text:
        print("AI is not available right now (try again later).")
        print("Tip: set HF_TOKEN in .env or env var to improve reliability.")
        pause()
        return

    print("\n--- AI feedback ---")
    print(text.strip())
    print("-------------------")
    pause()

