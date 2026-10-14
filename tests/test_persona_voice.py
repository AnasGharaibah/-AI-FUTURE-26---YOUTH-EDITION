"""Group C: one persona, one shared Pipeline, no Arabic read by a non-Arabic voice."""
from Agent import context, rag, tools, voice
from DB.pipeline import Pipeline


def test_one_persona_everywhere():
    texts = [context.SYSTEM_PROMPT, context.SYSTEM_PROMPT_AR] + [t.description for t in tools.TOOLS]
    assert not any("company" in t.lower() or "employee" in t.lower() for t in texts)
    assert "Verity" in context.SYSTEM_PROMPT and "Verity" in context.SYSTEM_PROMPT_AR
    assert "government" in tools.search_documents.description


def test_rag_and_tools_share_one_pipeline():
    assert rag.db() is tools._db() is Pipeline.shared()


def test_arabic_not_spoken_without_arabic_voice(monkeypatch):
    monkeypatch.setattr(voice, "installed_voices", lambda: {"en": ["Microsoft Zira"]})
    wav, engine, spoken = voice.synthesize("رسوم الجواز 25 ديناراً [1].")
    assert wav is None and engine == "no-arabic-voice" and spoken
    assert voice.check_voices() is False


def test_unknown_voice_list_still_tries(monkeypatch):
    monkeypatch.setattr(voice, "installed_voices", lambda: None)
    assert voice.has_voice("ar") and voice.check_voices()


def test_mac_prefers_majed_else_any_arabic_voice(monkeypatch):
    monkeypatch.setattr(voice, "installed_voices", lambda: {"ar": ["Maged", "Majed"], "en": ["Alex"]})
    assert voice._say_voice("ar") == "Majed"
    assert voice._say_voice("en") == "Alex"                             # Samantha missing -> any English voice
    monkeypatch.setattr(voice, "installed_voices", lambda: {"ar": ["Tarik"]})
    assert voice._say_voice("ar") == "Tarik"
