"""OmniVoice TTS worker. Runs in its own env (torch 2.8.0) and keeps the model loaded.

    cd voice_env && uv run python tts_server.py            # http://127.0.0.1:8765

    POST /tts  {"text": "...", "language": "arb"|"en"}  ->  audio/wav bytes
    GET  /health
"""
import io
import os

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

PORT = int(os.getenv("YOTH_TTS_PORT", "8765"))
MODEL = os.getenv("YOTH_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.getenv("YOTH_TTS_DEVICE", "mps")
SAMPLE_RATE = 24000

app = FastAPI(title="Yoth TTS worker")
_model = None


def model():
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        _model = OmniVoice.from_pretrained(MODEL, device_map=DEVICE, dtype=torch.float16)
    return _model


class TTSIn(BaseModel):
    text: str
    language: str = "en"


@app.on_event("startup")
def warm():
    model()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "device": DEVICE, "loaded": _model is not None}


@app.post("/tts")
def tts(body: TTSIn):
    if not body.text.strip():
        raise HTTPException(400, "empty text")
    audio = model().generate(text=body.text, language=body.language)
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio[0], dtype=np.float32), SAMPLE_RATE, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
