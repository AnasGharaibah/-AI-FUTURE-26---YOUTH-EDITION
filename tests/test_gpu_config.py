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


def test_whisper_cuda_float16_and_cpu_int8(monkeypatch):
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
    for fail, expected in ((False, [("large-v3-turbo", "cuda", "float16")]),
                           (True, [("large-v3-turbo", "cuda", "float16"), ("large-v3-turbo", "cpu", "int8")])):
        made.clear()
        voice._whisper.cache_clear()
        voice._whisper()
        assert made == expected
    voice._whisper.cache_clear()
