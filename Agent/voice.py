"""Voice in/out.   STT: mlx-whisper (local, Apple Silicon).   TTS: OmniVoice worker, fallback macOS `say`.

    from Agent.voice import listen, transcribe, speak
    listen(seconds=None, language="ar")      # mic -> {"text", "language", "seconds"}  (stops on silence)
    transcribe("q.wav", language=None)       # file / numpy array
    speak("مرحبا", block=True)               # -> engine used: "omnivoice" | "say"

OmniVoice needs torch 2.8.0, so it lives in voice_env/ and is called over HTTP:
    cd voice_env && uv run python tts_server.py
"""
import io
import logging
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from DB.arabic import normalize_display

log = logging.getLogger("yoth.voice")
ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # --- STT
    "stt_model": "mlx-community/whisper-large-v3-turbo",      # light option: mlx-community/whisper-small-mlx
    "stt_language": "ar",                                      # "ar" | "en" | None (auto, mixed speech)
    "sample_rate": 16000,
    "silence_rms": 0.01,                                       # RMS below this counts as silence
    "silence_seconds": 1.2,                                    # stop after this much silence
    "max_record_seconds": 30,
    "min_speech_seconds": 0.3,
    # --- TTS
    "tts_engine": "auto",                                      # "auto" (omnivoice -> say) | "omnivoice" | "say"
    "tts_url": "http://127.0.0.1:8765",
    "tts_timeout": 60,
    "max_spoken_chars": 300,
    "max_spoken_sentences": 3,
    "say_voices": {"ar": "Majed", "en": "Samantha"},
    "omnivoice_langs": {"ar": "arb", "en": "en"},
    "arabic_ratio": 0.30,
}


# =============================================================== STT
def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def record(seconds: float | None = None):
    """Record from the mic. seconds=N -> fixed length. seconds=None -> push-to-talk style:
    starts now, stops after `silence_seconds` of silence once speech was heard (or on Ctrl+C / max length).
    Returns float32 mono 16 kHz."""
    import sounddevice as sd
    sr = CONFIG["sample_rate"]
    if seconds:
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0]

    block = int(sr * 0.1)
    chunks, heard, quiet = [], False, 0.0
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block) as stream:
            while len(chunks) * 0.1 < CONFIG["max_record_seconds"]:
                data, _ = stream.read(block)
                data = data[:, 0].copy()
                chunks.append(data)
                if _rms(data) >= CONFIG["silence_rms"]:
                    heard, quiet = True, 0.0
                elif heard:
                    quiet += 0.1
                    if quiet >= CONFIG["silence_seconds"]:
                        break
    except KeyboardInterrupt:           # Ctrl+C while talking = "I'm done"; before any speech = quit
        if not heard:
            raise
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _load_audio(audio) -> np.ndarray:
    """path | bytes | ndarray -> float32 mono 16 kHz."""
    sr = CONFIG["sample_rate"]
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32).reshape(-1)
    import soundfile as sf
    try:
        data, file_sr = sf.read(io.BytesIO(audio) if isinstance(audio, bytes) else str(audio),
                                dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if file_sr != sr:
            data = np.interp(np.linspace(0, len(data), int(len(data) * sr / file_sr), endpoint=False),
                             np.arange(len(data)), data).astype(np.float32)
        return data
    except Exception:                    # mp3 / m4a / anything soundfile can't read -> ffmpeg
        from mlx_whisper.audio import load_audio
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".bin") as f:
                f.write(audio)
                f.flush()
                return load_audio(f.name)
        return load_audio(str(audio))


def transcribe(audio, language: str | None = "ar") -> dict:
    """audio: file path (.wav/.mp3/.m4a), raw bytes or numpy array. language: "ar" | "en" | None (auto).
    Returns {"text", "language", "seconds"}; empty text (never an exception) when nothing was said."""
    t0 = time.time()
    try:
        data = _load_audio(audio)
    except Exception as e:
        log.warning("could not read audio: %s", e)
        return {"text": "", "language": language or "", "seconds": round(time.time() - t0, 2)}
    if len(data) < CONFIG["min_speech_seconds"] * CONFIG["sample_rate"] or _rms(data) < 1e-4:
        return {"text": "", "language": language or "", "seconds": round(time.time() - t0, 2)}
    import mlx_whisper
    res = mlx_whisper.transcribe(data, path_or_hf_repo=CONFIG["stt_model"], language=language,
                                 task="transcribe", condition_on_previous_text=False, verbose=None)
    return {"text": normalize_display(res.get("text") or ""), "language": res.get("language") or language or "",
            "seconds": round(time.time() - t0, 2)}


def listen(seconds: float | None = None, language: str | None = CONFIG["stt_language"]) -> dict:
    """Record from the mic, then transcribe."""
    t0 = time.time()
    out = transcribe(record(seconds), language)
    out["seconds"] = round(time.time() - t0, 2)
    return out


# =============================================================== TTS
_CITE = re.compile(r"\s*(\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\])+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENT = re.compile(r"(?<=[.!?؟۔。])\s+|\n+")


def clean_text(text: str, max_chars: int | None = None, max_sentences: int | None = None) -> str:
    """Make an answer speakable: drop citations [1][2], URLs, markdown symbols; keep the first sentences."""
    max_chars = max_chars or CONFIG["max_spoken_chars"]
    max_sentences = max_sentences or CONFIG["max_spoken_sentences"]
    text = _URL.sub("", text)
    text = _CITE.sub("", text)
    text = re.sub(r"\(\s*[\w.\-]+\.(?:pdf|docx|pptx|xlsx|html?|md|txt)[^)]*\)", "", text, flags=re.I)  # (file.pdf, p.1)
    text = re.sub(r"[*#|`_~>]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    sents = [s for s in _SENT.split(text) if s.strip()]
    if detect_language(text) == "ar":
        sents = [digits_to_words(x) for x in sents]
    out = ""
    for s in sents[:max_sentences]:
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = f"{out} {s}".strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0]
    return out


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum("؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ" for c in letters)
    return "ar" if arabic / len(letters) > CONFIG["arabic_ratio"] else "en"


# ----------------------------------------------------------- numbers -> Arabic words (for natural TTS)
_ONES = ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
         "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_TENS = {2: "عشرون", 3: "ثلاثون", 4: "أربعون", 5: "خمسون", 6: "ستون", 7: "سبعون", 8: "ثمانون", 9: "تسعون"}
_HUNDREDS = {1: "مئة", 2: "مئتان", 3: "ثلاثمئة", 4: "أربعمئة", 5: "خمسمئة", 6: "ستمئة", 7: "سبعمئة",
             8: "ثمانمئة", 9: "تسعمئة"}


def _below_1000(n: int) -> str:
    parts = []
    h, r = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if r:
        if r < 20:
            parts.append(_ONES[r])
        else:
            t, o = divmod(r, 10)
            parts.append(f"{_ONES[o]} و{_TENS[t]}" if o else _TENS[t])
    return " و".join(parts)


def number_to_arabic_words(n: int) -> str:
    """0..999999 -> Arabic words (25 -> خمسة وعشرون). Outside the range -> digits separated by spaces."""
    if not 0 <= n <= 999999:
        return " ".join(str(n))
    if n == 0:
        return _ONES[0]
    th, rest = divmod(n, 1000)
    out = []
    if th:
        if th == 1:
            out.append("ألف")
        elif th == 2:
            out.append("ألفان")
        elif th <= 10:
            out.append(f"{_below_1000(th)} آلاف")
        else:
            out.append(f"{_below_1000(th)} ألف")
    if rest:
        out.append(_below_1000(rest))
    return " و".join(out)


_NUM_RE = re.compile(r"(?<![\w.])(\d+)(?:[.,](\d+))?(?![\w])")


def digits_to_words(text: str) -> str:
    """Replace numbers in `text` by Arabic words. 7+ digit numbers / leading zeros (phones, IDs) are spoken digit by digit;
    decimals read as 'x فاصلة y'."""
    def sub(m):
        whole, frac = m.group(1), m.group(2)
        if len(whole) > 6 or (len(whole) > 1 and whole.startswith("0")):
            return " ".join(_ONES[int(d)] for d in whole)
        out = number_to_arabic_words(int(whole))
        if frac:
            out += " فاصلة " + " ".join(_ONES[int(d)] for d in frac)
        return out
    return _NUM_RE.sub(sub, text)


_play_lock = threading.Lock()


def _play_wav(path: str, block: bool):
    def run():
        with _play_lock:
            subprocess.run(["afplay", path], check=False)
    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


def _omnivoice(text: str, lang: str) -> bytes:
    import urllib.request, json
    req = urllib.request.Request(CONFIG["tts_url"] + "/tts",
                                 data=json.dumps({"text": text, "language": CONFIG["omnivoice_langs"][lang]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=CONFIG["tts_timeout"]) as r:
        return r.read()


def synthesize(text: str, language: str | None = None, clean: bool = True):
    """text -> (wav_bytes | None, engine, spoken_text). `say` output is converted to wav."""
    spoken = clean_text(text) if clean else text.strip()
    if not spoken:
        return None, "none", ""
    lang = language or detect_language(spoken)
    engine = CONFIG["tts_engine"]
    if engine in ("auto", "omnivoice"):
        try:
            return _omnivoice(spoken, lang), "omnivoice", spoken
        except Exception as e:
            log.warning("OmniVoice unavailable (%s) - falling back to say", e)
            if engine == "omnivoice":
                raise
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = f"{d}/o.aiff", f"{d}/o.wav"
        subprocess.run(["say", "-v", CONFIG["say_voices"][lang], "-o", aiff, spoken], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff, wav], check=True)
        return Path(wav).read_bytes(), "say", spoken


def speak(text: str, language: str | None = None, block: bool = True, save_to: str | None = None) -> str:
    """Clean the text, synthesize it, play it (and optionally save a .wav). Returns the engine used.
    Never raises: if every engine fails it returns "none"."""
    try:
        wav, engine, spoken = synthesize(text, language)
    except Exception as e:
        log.error("TTS failed: %s", e)
        return "none"
    if wav is None:
        return "none"
    log.info("TTS engine: %s | %r", engine, spoken)
    print(f"   [tts: {engine}]")
    path = save_to or tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    Path(path).write_bytes(wav)
    _play_wav(path, block)
    return engine


# =============================================================== loop
def voice_loop(answer_fn, language: str | None = CONFIG["stt_language"]):
    """listen -> answer_fn(text) -> print answer -> speak it. Ctrl+C at the prompt exits.
    answer_fn(text) -> str (full answer, with sources)."""
    print("Voice mode - speak after the prompt, pause to send. Ctrl+C to quit.\n")
    while True:
        try:
            print("🎤 listening...", flush=True)
            heard = listen(None, language)
        except KeyboardInterrupt:
            break
        if not heard["text"]:
            continue
        print(f"you> {heard['text']}   (stt {heard['seconds']}s)")
        t0 = time.time()
        try:
            answer = answer_fn(heard["text"])
        except KeyboardInterrupt:
            break
        print(f"ai> {answer}   (answer {time.time() - t0:.1f}s)")
        t1 = time.time()
        speak(answer)
        print(f"   (tts {time.time() - t1:.1f}s)\n")
