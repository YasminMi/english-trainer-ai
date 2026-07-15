from english_trainer import cli


def test_add_sentence_keeps_user_content_separate(monkeypatch):
    answers = iter(["I study every day.", "Eu estudo todos os dias.", "routine"])
    monkeypatch.setattr(cli, "read_choice", lambda _prompt: next(answers))
    content = {"sentences": []}
    user_content = {"sentences": []}

    added = cli._add_sentence(content, user_content)

    assert added is True
    assert content["sentences"] == user_content["sentences"]
    assert user_content["sentences"][0]["id"].startswith("user_s_")


def test_add_sentence_rejects_duplicate(monkeypatch):
    answers = iter(["Hello."])
    monkeypatch.setattr(cli, "read_choice", lambda _prompt: next(answers))
    monkeypatch.setattr(cli, "pause", lambda *_args, **_kwargs: None)
    content = {"sentences": [{"id": "base", "en": "Hello.", "pt": "Olá."}]}
    user_content = {"sentences": []}

    assert cli._add_sentence(content, user_content) is False
    assert user_content["sentences"] == []
