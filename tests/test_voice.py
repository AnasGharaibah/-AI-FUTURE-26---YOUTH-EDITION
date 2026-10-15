import numpy as np

from Agent import voice
from Agent.voice import clean_text, digits_to_words, number_to_arabic_words as w


def test_numbers():
    assert w(0) == "صفر"
    assert w(5) == "خمسة"
    assert w(12) == "اثنا عشر"
    assert w(25) == "خمسة وعشرون"
    assert w(100) == "مئة"
    assert w(150) == "مئة وخمسون"
    assert w(200) == "مئتان"
    assert w(999) == "تسعمئة وتسعة وتسعون"
    assert w(1000) == "ألف"
    assert w(2000) == "ألفان"
    assert w(3000) == "ثلاثة آلاف"
    assert w(12000) == "اثنا عشر ألف"
    assert w(1250) == "ألف ومئتان وخمسون"
    assert w(999999).startswith("تسعمئة وتسعة وتسعون ألف")
    assert w(1_000_000) == "1 0 0 0 0 0 0"        # out of range -> digits


def test_digits_to_words():
    assert digits_to_words("الرسوم 25 دينارا") == "الرسوم خمسة وعشرون دينارا"
    assert digits_to_words("2.5") == "اثنان فاصلة خمسة"
    assert digits_to_words("0791234567") == " ".join(["صفر", "سبعة", "تسعة", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة"])


def test_clean_text_arabic_speakable():
    out = clean_text("الرسوم 25 دينارا [1][2]. **ملاحظة** يرجى المراجعة. ثالثا. رابعا.")
    assert "[" not in out and "*" not in out and "25" not in out and "خمسة وعشرون" in out
    assert len([x for x in out.split(".") if x.strip()]) <= 3


def test_stt_drops_whisper_silence_hallucinations(monkeypatch):
    from types import SimpleNamespace as NS
    seg = lambda text, nsp=0.1, lp=-0.3: NS(text=text, no_speech_prob=nsp, avg_logprob=lp)   # noqa: E731
    segs = [seg(" كم رسوم تجديد جواز السفر؟"), seg(" ترجمة نانسي قنقر"), seg(" اشتركوا في القناة"),
            seg(" شيء ما", nsp=0.9, lp=-1.5), seg(voice.CONFIG["stt_prompt_ar"][:40]), seg(" كم الرسوم؟")]
    fake = NS(transcribe=lambda *a, **k: (iter(segs), NS(language="ar")))
    monkeypatch.setattr(voice, "_whisper", lambda: fake)
    out = voice._stt(np.zeros(16000, np.float32), "ar")["text"]
    assert out == "كم رسوم تجديد جواز السفر؟ كم الرسوم؟"
