"""OmniVoice TTS worker (https://github.com/k2-fsa/OmniVoice). Own env: OmniVoice pins torch 2.8.0, the main
project uses a newer torch. Keeps the model loaded; the app (Agent/voice.py) calls it over HTTP on 127.0.0.1.

    cd voice_env
    uv run python tts_server.py                 # http://127.0.0.1:8765   (YOTH_TTS_DEVICE=cpu to keep it off the GPU)

    POST /tts  {"text": "...", "language": "ar" | "en"}  ->  audio/wav (24 kHz)
    GET  /health

Voice cloning: voices/ar_ref.wav + voices/ar_ref.txt (8-10 s of clean speech and its exact transcript), and
voices/en_ref.wav + en_ref.txt for English. Without them OmniVoice uses its own default voice.
"""
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "voices"
PORT = int(os.getenv("YOTH_TTS_PORT", "8765"))
MODEL = os.getenv("YOTH_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.getenv("YOTH_TTS_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 24000
LANGS = {"ar": "arb", "en": "en"}           # OmniVoice ids: Standard Arabic = "arb"

log = logging.getLogger("uvicorn.error")
_model = None
_prompts: dict[str, object] = {}
_lock = threading.Lock()                    # one generation at a time (one GPU, one model)


def model():
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
        _model = OmniVoice.from_pretrained(MODEL, device_map=DEVICE, dtype=dtype)
        log.info("OmniVoice loaded on %s", DEVICE)
    return _model


def voice_prompt(lang: str):
    """Cached voice-clone prompt from voices/<lang>_ref.wav (+ .txt); None when there is no reference recording."""
    if lang not in _prompts:
        wav, txt = VOICES / f"{lang}_ref.wav", VOICES / f"{lang}_ref.txt"
        if wav.exists() and txt.exists():
            _prompts[lang] = model().create_voice_clone_prompt(
                ref_audio=str(wav), ref_text=txt.read_text(encoding="utf-8").strip())
            log.info("voice clone prompt ready for %s (%s)", lang, wav.name)
        else:
            _prompts[lang] = None
            log.info("no voices/%s_ref.wav + .txt - OmniVoice's default voice for %s", lang, lang)
    return _prompts[lang]


@asynccontextmanager
async def lifespan(_app):
    model()                                 # load at startup, not on the first request
    for lang in LANGS:
        voice_prompt(lang)
    yield


app = FastAPI(title="Yoth TTS worker", lifespan=lifespan)


class TTSIn(BaseModel):
    text: str
    language: str = "ar"


@app.get("/health")
def health():
    vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.startswith("cuda") else 0
    return {"status": "ok", "model": MODEL, "device": DEVICE, "loaded": _model is not None,
            "cloned_voices": [k for k, v in _prompts.items() if v is not None], "vram_gb": round(vram, 2)}


@app.post("/tts")
def tts(body: TTSIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    lang = body.language if body.language in LANGS else "ar"
    kw = {"text": text}
    prompt = voice_prompt(lang)
    if prompt is not None:
        kw["voice_clone_prompt"] = prompt
    with _lock:
        try:
            audio = model().generate(language=LANGS[lang], **kw)
        except TypeError:                   # an OmniVoice version without the language argument
            audio = model().generate(**kw)
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio[0], dtype=np.float32), SAMPLE_RATE, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
