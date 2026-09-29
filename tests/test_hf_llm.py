import requests

from english_trainer import hf_llm


class DummyResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def test_generate_text_ollama_unreachable(monkeypatch, tmp_path):
    def fake_post(*args, **kwargs):
        raise requests.RequestException("fail")

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(hf_llm, "load_ai_cache", lambda: {})
    monkeypatch.setattr(hf_llm, "save_ai_cache", lambda new: None)
    monkeypatch.setenv("HF_TOKEN", "")
    text = hf_llm.generate_text("hello")
    assert text is None


def test_generate_text_cache(monkeypatch):
    cache = {}
    monkeypatch.setattr(hf_llm, "load_ai_cache", lambda: cache)
    monkeypatch.setattr(hf_llm, "save_ai_cache", lambda new: cache.update(new))

    cache["key"] = {"text": "cached", "model": "m"}
    monkeypatch.setattr(hf_llm, "_cache_key", lambda model, prompt: "key")

    text = hf_llm.generate_text("anything")
    assert text == "cached"


def test_local_fallback_does_not_evaluate_free_writing(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr(hf_llm, "load_ai_cache", lambda: {})
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("AI_LOCAL_FALLBACK", "1")

    assert hf_llm.generate_text("Evaluate this student's free writing") is None


def test_local_fallback_returns_curated_sentence_examples(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr(hf_llm, "load_ai_cache", lambda: {})
    monkeypatch.setattr(hf_llm, "save_ai_cache", lambda _cache: None)
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("AI_LOCAL_FALLBACK", "1")

    text = hf_llm.generate_text(
        "Generate 2 sentences with verb to be in past tense. Output format: EN | PT | DICA"
    )

    assert text is not None
    assert text.splitlines() == [
        "I was at home. | Eu estava em casa. | Use was com I",
        "She was tired. | Ela estava cansada. | Use was com she",
    ]
