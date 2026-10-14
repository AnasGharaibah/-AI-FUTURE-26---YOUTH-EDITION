"""Group D: real token streaming, normalized refusal matching, temp wav cleanup, CONFIG read at call time."""
import tempfile
from pathlib import Path

import pytest

from Agent import brain, llm, rag, voice
from test_rag import FakePipe


@pytest.fixture
def pipe(monkeypatch):
    p = FakePipe()
    monkeypatch.setattr(rag, "db", lambda: p)
    monkeypatch.setattr(llm, "rewrite_to_msa", lambda q, model=None: q)
    rag.reset()
    return p


def stream_reply(monkeypatch, *pieces):
    monkeypatch.setattr(llm, "chat_stream", lambda msgs, model=None, max_tokens=400, temperature=None: iter(pieces))


# ---------------------------------------------------------------- #13 streaming
def test_answer_stream_yields_tokens_then_checked_answer(pipe, monkeypatch):
    stream_reply(monkeypatch, "رسوم الجواز ", "25 ", "ديناراً [1].")
    events = list(rag.answer_stream("كم رسوم الجواز؟", use_history=False))
    kinds = [k for k, _ in events]
    assert kinds[0] == "retrieved" and kinds.count("token") == 3 and kinds[-1] == "done"
    assert events[-1][1]["answer"] == "رسوم الجواز 25 ديناراً [1]."


def test_stream_replaces_draft_when_the_guard_changes_it(pipe, monkeypatch):
    stream_reply(monkeypatch, "الرسوم 30 ديناراً [1].")                 # 30 is not in the sources
    monkeypatch.setattr(rag, "_generate", lambda *a, **k: "الرسوم 25 ديناراً [1].")
    monkeypatch.setattr(brain, "MODE", "rag")
    out = list(brain.stream("كم رسوم الجواز؟", "t-stream"))
    assert ("text", "الرسوم 30 ديناراً [1].") in out
    assert ("replace", "الرسوم 25 ديناراً [1].") in out
    assert out[-1][0] == "sources"


def test_non_streaming_answer_unchanged(pipe, monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda *a, **k: "الرسوم 25 ديناراً [1].")
    assert rag.answer("كم رسوم الجواز؟", use_history=False)["answer"] == "الرسوم 25 ديناراً [1]."


# ---------------------------------------------------------------- #10 refusal variants
@pytest.mark.parametrize("text", ["لم أجد هذه المعلومة في الوثائق المتوفرة.", "لم اجد هذه المعلومه في الوثائق المتاحة",
                                  "I couldn't find this information in the documents.",
                                  "I Could Not Find This Information in the available documents"])
def test_refusal_variants(text):
    assert rag.is_refusal(text)


def test_strip_refusal_keeps_the_answer():
    text = "مدة الإنجاز 5 أيام عمل [1].\n- المستعجل 48 ساعة [1].\n\nلم اجد هذه المعلومه في الوثائق المتاحة."
    out = rag.strip_refusal(text)
    assert not rag.is_refusal(out) and "5 أيام عمل [1]." in out and "48 ساعة" in out


def test_stray_refusal_variant_removed_from_cited_answer(pipe, monkeypatch):
    monkeypatch.setattr(rag, "_generate",
                        lambda *a, **k: "الرسوم 25 ديناراً [1]. لم أجد هذه المعلومة في الوثائق المتاحة.")
    r = rag.answer("كم رسوم الجواز؟", use_history=False)
    assert r["answer"] == "الرسوم 25 ديناراً [1]." and r["sources"] and "stripped_refusal" in r["flags"]


# ---------------------------------------------------------------- #16 temp wav cleanup
def test_speak_deletes_its_temp_wav(monkeypatch):
    winsound = pytest.importorskip("winsound")
    played = []
    monkeypatch.setattr(winsound, "PlaySound", lambda path, flags: played.append(Path(path).exists()))
    monkeypatch.setattr(voice, "synthesize", lambda text, language=None: (b"RIFF....", "windows", text))
    before = set(Path(tempfile.gettempdir()).glob("yoth-tts-*.wav"))
    assert voice.speak("hello", block=True) == "windows"
    assert played == [True]                                             # it existed while playing ...
    assert set(Path(tempfile.gettempdir()).glob("yoth-tts-*.wav")) == before   # ... and is gone after


# ---------------------------------------------------------------- #18 CONFIG read at call time
def test_listen_reads_language_from_config_at_call_time(monkeypatch):
    seen = []
    monkeypatch.setattr(voice, "record", lambda seconds=None: b"")
    monkeypatch.setattr(voice, "transcribe", lambda audio, language: seen.append(language) or {"text": ""})
    monkeypatch.setitem(voice.CONFIG, "stt_language", "en")
    voice.listen()
    monkeypatch.setitem(voice.CONFIG, "stt_language", None)
    voice.listen()
    voice.listen(language="ar")
    assert seen == ["en", None, "ar"]
