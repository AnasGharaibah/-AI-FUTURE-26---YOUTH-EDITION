"""REST API.   uv run uvicorn api:app --reload      docs at http://127.0.0.1:8000/docs"""
import json
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from Agent.brain import MODEL, ask, stream

app = FastAPI(title="Yoth AI")


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None          # reuse it to continue a conversation


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
def chat(body: ChatIn):
    tid = body.thread_id or uuid.uuid4().hex
    return {"thread_id": tid, "answer": ask(body.message, tid)}


@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    tid = body.thread_id or uuid.uuid4().hex

    def events():
        for kind, val in stream(body.message, tid):
            yield f"data: {json.dumps({'type': kind, 'data': val, 'thread_id': tid})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")



# ------------------------------------------------------------------ voice
class TTSIn(BaseModel):
    text: str
    language: str | None = None           # "ar" | "en" | None = detect


@app.post("/stt")
def stt(audio: UploadFile = File(...), language: str | None = Form("ar")):
    """Audio upload (wav/mp3/m4a) -> {"text", "language", "seconds"}. language=auto for mixed speech."""
    from Agent.voice import transcribe
    return transcribe(audio.file.read(), None if language in (None, "", "auto") else language)


@app.post("/tts")
def tts(body: TTSIn):
    """Text -> wav bytes. Header X-TTS-Engine says which engine produced it."""
    from Agent.voice import synthesize
    wav, engine, _ = synthesize(body.text, body.language)
    return Response(wav or b"", media_type="audio/wav", headers={"X-TTS-Engine": engine})
