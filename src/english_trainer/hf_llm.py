"""Text generation through Ollama, Hugging Face or limited local examples."""

from __future__ import annotations

import hashlib
import os
import re

import requests

from .data_access import load_ai_cache, save_ai_cache

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip() or "http://localhost:11434"
HF_MODEL = os.getenv("HF_MODEL", "gpt2").strip() or "gpt2"


def _hf_token() -> str:
    return os.getenv("HF_TOKEN", "").strip()


def _use_local_fallback() -> bool:
    return os.getenv("AI_LOCAL_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_key(model: str, prompt: str) -> str:
    value = f"{model}\n{prompt}".encode()
    return hashlib.sha256(value).hexdigest()


def generate_conversation_reply(
    messages: list[dict[str, str]],
    system_prompt: str = "",
    model: str | None = None,
    max_new_tokens: int = 300,
    timeout: float = 120,
    temperature: float = 0.7,
) -> str | None:
    """Return an Ollama chat reply, or ``None`` when the provider is unavailable."""
    selected_model = (model or DEFAULT_MODEL).strip()
    full_messages: list[dict[str, str]] = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": selected_model,
        "messages": full_messages,
        "options": {"temperature": temperature, "num_predict": max_new_tokens},
        "stream": False,
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip() or None
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def generate_text(prompt: str, model: str | None = None, max_new_tokens: int = 160) -> str | None:
    """Generate text with Ollama, then Hugging Face, then the opt-in limited fallback."""
    selected_model = (model or DEFAULT_MODEL).strip()
    key = _cache_key(selected_model, prompt)
    cache = load_ai_cache()
    cached = cache.get(key)
    if isinstance(cached, dict) and isinstance(cached.get("text"), str):
        return cached["text"]

    text = _generate_with_ollama(prompt, selected_model, max_new_tokens)
    source = selected_model

    if text is None and _hf_token():
        text = _generate_with_hugging_face(prompt, max_new_tokens)
        source = HF_MODEL

    if text is None and _use_local_fallback():
        text = _limited_local_examples(prompt)
        source = "limited-local-examples"

    if text:
        cache[key] = {"text": text, "model": source}
        save_ai_cache(cache)
    return text


def _generate_with_ollama(prompt: str, model: str, max_new_tokens: int) -> str | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": max_new_tokens},
        "stream": False,
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip() or None
    except (requests.RequestException, TypeError, ValueError, KeyError):
        return None


def _generate_with_hugging_face(prompt: str, max_new_tokens: int) -> str | None:
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=HF_MODEL, token=_hf_token())
        text = client.text_generation(prompt, max_new_tokens=max_new_tokens)
        return text.strip() if text else None
    except Exception:
        # Provider-specific clients expose different exception types across versions.
        return None


def _limited_local_examples(prompt: str) -> str | None:
    """Return curated ``to be`` examples; this function never evaluates user text."""
    normalized = prompt.lower()
    if "generate" not in normalized or "sentence" not in normalized or "to be" not in normalized:
        return None

    match = re.search(r"(\d+)\s*sentences", normalized)
    amount = max(1, min(int(match.group(1)) if match else 5, 12))
    tense = "past" if "past" in normalized else "present"
    examples = {
        "present": [
            ("I am at home.", "Eu estou em casa.", "Use am com I"),
            ("She is tired.", "Ela está cansada.", "Use is com she"),
            ("They are ready.", "Eles estão prontos.", "Use are com they"),
            ("This is a book.", "Isto é um livro.", "Use is com this"),
            ("We are students.", "Nós somos estudantes.", "Use are com we"),
        ],
        "past": [
            ("I was at home.", "Eu estava em casa.", "Use was com I"),
            ("She was tired.", "Ela estava cansada.", "Use was com she"),
            ("They were ready.", "Eles estavam prontos.", "Use were com they"),
            ("This was a book.", "Isto era um livro.", "Use was com this"),
            ("We were students.", "Nós éramos estudantes.", "Use were com we"),
        ],
    }[tense]

    selected = [examples[index % len(examples)] for index in range(amount)]
    if "en | pt" in normalized or "output format" in normalized:
        return "\n".join(" | ".join(example) for example in selected)
    return "\n".join(example[0] for example in selected)
