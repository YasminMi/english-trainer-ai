"""To be trainer: multiple-choice drills for am / is / are / was / were."""

import random
from typing import Dict, Any, List

from .utils import pause, clear_screen, read_choice
from .progress import register_to_be_result, get_srs_weight


def _weighted_choice(exercises: List[Dict[str, Any]], progress: Dict[str, Any]) -> Dict[str, Any]:
    if len(exercises) == 1:
        return exercises[0]

    progress_section = progress.get("to_be", {})
    weights = []
    for ex in exercises:
        ex_id = str(ex.get("id", ex.get("question", "")))
        stats = progress_section.get(ex_id, {"seen": 0, "correct": 0})
        seen = int(stats.get("seen", 0))
        correct = int(stats.get("correct", 0))
        if seen == 0:
            weight = 3.0
        else:
            accuracy = correct / seen
            weight = max(0.3, 2.5 - accuracy * 2.0)
        weight *= get_srs_weight(progress, "to_be", ex_id)
        weights.append(weight)
    return random.choices(exercises, weights=weights, k=1)[0]


def run_to_be_trainer(content: Dict[str, Any], progress: Dict[str, Any]) -> None:
    exercises: List[Dict[str, Any]] = content.get("to_be_exercises", [])
    if not exercises:
        print("No to-be exercises configured.")
        pause()
        return

    while True:
        clear_screen()
        print("=== Mode 3: To be Trainer ===")
        print("Choose the correct form to complete the sentence.")
        print("Type '0' to go back.\n")

        ex: Dict[str, Any] = _weighted_choice(exercises, progress)

        q = ex["question"]
        options = ex["options"]
        correct_answer = ex["correct"]
        pt = ex["pt"]
        explanation: Any | None = ex.get("explanation")  
        ex_id = ex.get("id", q)

        print(f"Sentence: {q}")
        for i, opt in enumerate(options, start=1):
            print(f"{i}) {opt}")

        choice: str = read_choice("Your choice (number): ")

        if choice == "0":
            break

        try:
            idx: int = int(choice) - 1
        except ValueError:
            print("Invalid option.")
            pause()
            continue
        if not 0 <= idx < len(options):
            print("Invalid option.")
            pause()
            continue
        chosen = options[idx]

        correct = chosen == correct_answer
        register_to_be_result(progress, ex_id, correct)

        if correct:
            print("Good job! ✅")
        else:
            print("Try again! ❌")
            print(f"Correct answer: {correct_answer}")

        print(f"Meaning (PT): {pt}")

        if explanation:
            print(f"Explanation: {explanation}")  

        pause()
