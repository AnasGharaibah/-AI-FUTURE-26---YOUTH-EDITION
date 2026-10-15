"""NVIDIA setup: reranker on with >= 8 GB VRAM, model candidates by VRAM, Whisper device / compute type."""
from Agent import llm, voice
from DB import pipeline as P


def test_reranker_auto_by_vram(monkeypatch):
    pipe = P.Pipeline.__new__(P.Pipeline)                       # no Chroma needed for this check
    pipe.cfg, pipe._auto_rerank = dict(P.DEFAULTS), None
    monkeypatch.setattr(P, "gpu_vram_gb", lambda: 8.59)          # RTX 4060 Laptop: 8188 MiB
    assert pipe.reranker_enabled()
    pipe._auto_rerank = None
    monkeypatch.setattr(P, "gpu_vram_gb", lambda: 6.4)
    assert not pipe.reranker_enabled()
    pipe.cfg["use_reranker"] = True                             # explicit setting wins
    assert pipe.reranker_enabled()


def test_model_candidates_by_vram():
    assert llm.candidates(8.59) == ("qwen2.5:7b", "qwen3:8b")
    assert llm.candidates(12.9) == ("qwen2.5:7b", "qwen3:8b", "qwen3:14b", "qwen2.5:14b")


def test_whisper_cuda_int8_float16_and_cpu_int8(monkeypatch):
    made = []

    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            made.append((name, device, compute_type))
            if device == "cuda" and fail:
                raise RuntimeError("CUDA failed")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(voice, "cuda_device", lambda: "cuda")
    monkeypatch.setattr(voice, "_cuda_dlls", lambda: None)
    for fail, expected in ((False, [("large-v3-turbo", "cuda", "int8_float16")]),
                           (True, [("large-v3-turbo", "cuda", "int8_float16"), ("large-v3-turbo", "cpu", "int8")])):
        made.clear()
        monkeypatch.setattr(voice, "_whisper_model", None)
        voice._whisper()
        voice._whisper()                                        # second call: no second load
        assert made == expected
    monkeypatch.setattr(voice, "_whisper_model", None)


def test_whisper_loads_once_across_threads(monkeypatch):
    import threading
    import time
    loads = []
    monkeypatch.setattr(voice, "_whisper_model", None)
    monkeypatch.setattr(voice, "_load_whisper", lambda: loads.append(1) or time.sleep(0.2) or "model")
    threads = [threading.Thread(target=voice._whisper) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert loads == [1] and voice._whisper() == "model"
    monkeypatch.setattr(voice, "_whisper_model", None)


def test_transcribe_reads_language_at_call_time_and_keeps_raw(monkeypatch):
    import numpy as np
    seen = []
    monkeypatch.setattr(voice, "_stt", lambda data, language: seen.append(language) or
                        {"text": "  كم رسوم ﺍﻟﺠﻮﺍﺯ ٢٥  ", "language": "ar"})
    audio = (np.sin(np.arange(16000) / 5) * 0.2).astype(np.float32)
    monkeypatch.setitem(voice.CONFIG, "stt_language", None)
    r = voice.transcribe(audio)
    assert seen == [None] and r["raw"] == "كم رسوم ﺍﻟﺠﻮﺍﺯ ٢٥" and r["text"] == "كم رسوم الجواز 25"


def test_speech_chunks_and_engine_fallback(monkeypatch):
    chunks = voice.speech_chunks("الرسوم 25 دينار [1]. وتستغرق المعاملة خمسة أيام عمل في مراكز الخدمة، ويمكن "
                                 "متابعتها إلكترونياً عبر البوابة الرسمية للخدمات الحكومية [2]. See https://x.y")
    assert chunks[0] == ("الرسوم خمسة وعشرون دينار.", "ar")
    assert all("[" not in c and "http" not in c for c, _ in chunks) and len(chunks) == 3     # long sentence cut at ،

    def down(text, lang):
        raise ConnectionRefusedError("worker stopped")
    monkeypatch.setattr(voice, "_omnivoice", down)
    monkeypatch.setattr(voice, "_system_voice", lambda text, lang: (b"RIFF", "windows"))
    assert voice._synth("مرحبا", "ar") == (b"RIFF", "windows")                              # no exception
    monkeypatch.setitem(voice.CONFIG, "tts_engine", "system")
    monkeypatch.setattr(voice, "_omnivoice", lambda t, l: (_ for _ in ()).throw(AssertionError("not called")))
    assert voice._synth("hello", "en") == (b"RIFF", "windows")
