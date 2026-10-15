"""Voice in/out.   STT: faster-whisper large-v3-turbo on CUDA (int8_float16; CPU int8 fallback).
TTS: 1) OmniVoice worker (voice_env/tts_server.py, CUDA, voice cloning from voices/*_ref.wav)
     2) fallback, always available: Windows OneCore voices (Hoda / Naayf for Arabic) / macOS `say`.
Arabic is never read by an English voice (check_voices() warns at startup).

    from Agent.voice import listen, transcribe, speak
    listen(seconds=None, language="ar")      # mic -> {"text", "raw", "language", "seconds"}  (stops on silence)
    transcribe("q.wav", language=None)       # file / bytes / numpy array; None = auto-detect (mixed ar/en)
    speak("مرحبا", block=True)               # -> engines used: "omnivoice" | "windows" | "omnivoice+windows" | ...
"""
import io
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from DB.arabic import normalize_display

log = logging.getLogger("yoth.voice")
ROOT = Path(__file__).resolve().parent.parent
IS_MAC = sys.platform == "darwin"

CONFIG = {
    # --- STT
    "whisper_model": "large-v3-turbo",                         # faster-whisper; light option: "small"
    "whisper_device": "auto",                                  # "auto" (cuda if available) | "cpu" | "cuda"
    "whisper_compute_cuda": "int8_float16",                    # ~0.8 GB VRAM
    "whisper_compute_cpu": "int8",
    "beam_size": 5,
    "stt_language": "ar",                                      # "ar" | "en" | None (auto, mixed speech)
    # Arabic speech: the service words, so Whisper writes "كم رسوم ... جواز السفر" (not "كما ... السفري")
    "stt_prompt_ar": "أسئلة عن جواز السفر وبطاقة الهوية ودفتر العائلة: كم الرسوم؟ ما الوثائق المطلوبة؟ "
                     "تجديد، إصدار لأول مرة، بدل فاقد، بدل تالف.",
    "sample_rate": 16000,
    "silence_rms": 0.01,                                       # RMS below this counts as silence
    "silence_seconds": 1.2,                                    # stop after this much silence
    "max_record_seconds": 15,
    "min_speech_seconds": 0.3,
    # --- TTS
    "tts_engine": "auto",                                      # "auto" (OmniVoice -> system voice) | "omnivoice" | "system"
    "tts_url": os.getenv("YOTH_TTS_URL", "http://127.0.0.1:8765"),   # voice_env/tts_server.py
    "tts_timeout": 30,
    "max_spoken_chars": 300,
    "max_spoken_sentences": 3,
    "chunk_chars": 80,                                         # sentences longer than this are also cut at "،"
    "say_voices": {"ar": "Majed", "en": "Samantha"},
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
    except Exception:                    # webm / mp3 / m4a / anything soundfile can't read
        return _decode_av(io.BytesIO(audio) if isinstance(audio, bytes) else str(audio))


def _decode_av(src) -> np.ndarray:
    """Any audio/video container -> float32 mono 16 kHz with PyAV (no ffmpeg install needed)."""
    import av
    resampler = av.AudioResampler(format="flt", layout="mono", rate=CONFIG["sample_rate"])
    out = []
    with av.open(src) as f:
        for frame in f.decode(audio=0):
            out += [r.to_ndarray().reshape(-1) for r in resampler.resample(frame)]
    out += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)


def cuda_device() -> str:
    """"cuda" when torch sees an NVIDIA GPU, else "cpu"."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _cuda_dlls():
    """Windows: faster-whisper (CTranslate2) loads cuBLAS / cuDNN lazily from PATH; the CUDA build of torch ships
    them in torch/lib, so put that folder on the DLL search path and PATH (else: "cublas64_12.dll is not found")."""
    if sys.platform != "win32":
        return
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir() and str(lib) not in os.environ.get("PATH", ""):
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        log.warning("could not add torch's CUDA libraries to the DLL path: %s", e)


_whisper_model = None
_whisper_lock = threading.Lock()


def _whisper():
    """The faster-whisper model, loaded once (thread-safe: two first requests never load it twice)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                _whisper_model = _load_whisper()
    return _whisper_model


def _load_whisper():
    """faster-whisper large-v3-turbo. GPU: int8_float16; CPU fallback: int8. Downloaded on first use (~1.6 GB)."""
    device = CONFIG["whisper_device"]
    if device == "auto":
        device = cuda_device()
    if device == "cuda":
        _cuda_dlls()
    from faster_whisper import WhisperModel
    compute = CONFIG["whisper_compute_cuda"] if device == "cuda" else CONFIG["whisper_compute_cpu"]
    try:
        return WhisperModel(CONFIG["whisper_model"], device=device, compute_type=compute)
    except Exception as e:
        if device != "cuda":
            raise
        log.warning("Whisper on CUDA failed (%s) - using the CPU (%s)", e, CONFIG["whisper_compute_cpu"])
        return WhisperModel(CONFIG["whisper_model"], device="cpu", compute_type=CONFIG["whisper_compute_cpu"])


# What Whisper "hears" in silence / breathing: subtitle credits from its training data, mostly Arabic ones.
_HALLUCINATION = re.compile(
    r"ترجمة\s+نانسي|نانسي\s+قنقر|اشتركوا?\s+(?:في\s+)?(?:القناة|قناتي)|لا\s+تنسوا?\s+(?:الاشتراك|الإعجاب)|"
    r"شكرا\s+(?:لكم\s+)?(?:على|علي)\s+المشاهدة|شكرًا\s+(?:لكم\s+)?على\s+المشاهدة|المترجم\s*:|تفريغ\s+نصي|"
    r"thanks?\s+(?:you\s+)?for\s+watching|please\s+subscribe|subtitles?\s+by|amara\.org", re.I)


def _real_speech(seg) -> bool:
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:      # Whisper itself thinks it is silence
        return False
    return not _HALLUCINATION.search(seg.text)


def _stt(data: np.ndarray, language: str | None) -> dict:
    prompt = CONFIG["stt_prompt_ar"] if language == "ar" else None
    segments, info = _whisper().transcribe(data, language=language, task="transcribe", beam_size=CONFIG["beam_size"],
                                           vad_filter=True, condition_on_previous_text=False, initial_prompt=prompt)
    kept = [s.text.strip() for s in segments if _real_speech(s)]
    if prompt:          # in near-silence Whisper can repeat its prompt; a short real question ("كم الرسوم") is kept
        kept = [t for t in kept if len(t) < 25 or t.strip(" .،؟?") not in prompt]
    return {"text": " ".join(kept), "language": info.language}


FROM_CONFIG = "config"          # default language marker: read CONFIG["stt_language"] when called, not at import


def _language(language):
    return CONFIG["stt_language"] if language == FROM_CONFIG else language


def transcribe(audio, language: str | None = FROM_CONFIG) -> dict:
    """audio: file path (.wav/.mp3/.m4a/.webm), raw bytes or numpy array.
    language: "ar" | "en" | None (auto-detect, for mixed Arabic/English); default CONFIG["stt_language"].
    Returns {"text" (normalize_display), "raw" (as Whisper wrote it), "language", "seconds"};
    empty text (never an exception) when nothing was said."""
    language = _language(language)
    t0 = time.time()
    try:
        data = _load_audio(audio)
    except Exception as e:
        log.warning("could not read audio: %s", e)
        return {"text": "", "raw": "", "language": language or "", "seconds": round(time.time() - t0, 2)}
    if len(data) < CONFIG["min_speech_seconds"] * CONFIG["sample_rate"] or _rms(data) < 1e-4:
        return {"text": "", "raw": "", "language": language or "", "seconds": round(time.time() - t0, 2)}
    res = _stt(data, language)
    raw = (res.get("text") or "").strip()
    return {"text": normalize_display(raw), "raw": raw, "language": res.get("language") or language or "",
            "seconds": round(time.time() - t0, 2)}


def listen(seconds: float | None = None, language: str | None = FROM_CONFIG) -> dict:
    """Record from the mic, then transcribe. language: "ar" | "en" | None (auto); default CONFIG["stt_language"]."""
    language = _language(language)
    t0 = time.time()
    out = transcribe(record(seconds), language)
    out["seconds"] = round(time.time() - t0, 2)
    return out


# =============================================================== TTS
_CITE = re.compile(r"\s*(\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\])+")
_URL = re.compile(r"(?:\b(?:see|visit|at)\s+|(?:راجع|انظر|زر|زوروا)\s+)?(?:https?://\S+|www\.\S+)", re.I)
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


def _play_wav(path: str, block: bool, delete_after: bool = False):
    def run():
        try:
            with _play_lock:
                if sys.platform == "win32":
                    import winsound
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                else:
                    subprocess.run(["afplay", path], check=False)
        finally:
            if delete_after:
                Path(path).unlink(missing_ok=True)
    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


# Windows: the WinRT speech API (Windows.Media.SpeechSynthesis). Unlike the old System.Speech it can use the voices
# added in Settings > Time & language > Speech (Arabic: Hoda, Naayf, ...).
_WINDOWS_TTS_PS = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($op, [Type]$type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $null = $t.Wait(-1); $t.Result }
if ($env:YOTH_TTS_LIST) {
    [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | ForEach-Object { $_.Language + '|' + $_.DisplayName }
    return
}
$s = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -like "$($env:YOTH_TTS_LANG)*" } | Select-Object -First 1
if ($v) { $s.Voice = $v }
$text = [IO.File]::ReadAllText($env:YOTH_TTS_IN, [Text.Encoding]::UTF8)
$stream = Await ($s.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$size = [uint32]$stream.Size
$null = Await ($reader.LoadAsync($size)) ([uint32])
$bytes = New-Object byte[] $size
$reader.ReadBytes($bytes)
[IO.File]::WriteAllBytes($env:YOTH_TTS_OUT, $bytes)
"""


def _powershell(script: str, **env) -> str:
    """Run a PowerShell script from a temp .ps1 file (passed inline with -Command, the WinRT awaits hang)."""
    with tempfile.TemporaryDirectory() as d:
        ps1 = Path(d) / "script.ps1"
        ps1.write_text(script, encoding="utf-8-sig")              # BOM: PowerShell 5.1 reads the file as UTF-8
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-File", str(ps1)], env={**os.environ, **env}, check=True, capture_output=True,
                              stdin=subprocess.DEVNULL, timeout=60, encoding="utf-8", errors="replace").stdout


@lru_cache(maxsize=1)
def installed_voices() -> dict[str, list[str]] | None:
    """System TTS voices by language: {"ar": ["Microsoft Naayf", ...], "en": [...]}; None when they can't be listed."""
    voices: dict[str, list[str]] = {}
    try:
        if sys.platform == "win32":
            for line in _powershell(_WINDOWS_TTS_PS, YOTH_TTS_LIST="1").splitlines():
                lang, _, name = line.strip().partition("|")
                if name:
                    voices.setdefault(lang[:2].lower(), []).append(name)
        elif IS_MAC:
            out = subprocess.run(["say", "-v", "?"], capture_output=True, encoding="utf-8", errors="replace", timeout=10).stdout
            for line in out.splitlines():                  # "Majed               ar_001    # ..."
                m = re.match(r"(.+?)\s+([a-z]{2})[_-]\w+\s+#", line)
                if m:
                    voices.setdefault(m.group(2), []).append(m.group(1).strip())
    except Exception as e:
        log.warning("could not list TTS voices: %s", e)
        return None
    return voices


def has_voice(lang: str) -> bool:
    """True when a voice for `lang` is installed - or when the voices can't be listed (then just try)."""
    voices = installed_voices()
    return voices is None or bool(voices.get(lang))


def check_voices() -> bool:
    """Print a clear warning when no Arabic TTS voice is installed (Arabic answers will then not be spoken).
    Returns True when an Arabic voice exists (or the voices can't be listed)."""
    if has_voice("ar"):
        return True
    how = ("Settings > Time & language > Speech > Add voices > Arabic" if sys.platform == "win32"
           else "System Settings > Accessibility > Spoken Content > System voice > Manage Voices > Arabic (Majed)")
    print(f"WARNING: no Arabic text-to-speech voice is installed - Arabic answers will be shown but not spoken.\n"
          f"         Add one in {how}, then restart.")
    return False


def _say_voice(lang: str) -> str | None:
    """macOS voice for `lang`: the configured one (Majed / Samantha) when installed, else any voice of that language."""
    preferred, voices = CONFIG["say_voices"].get(lang), installed_voices()
    if voices is None:                                     # unknown: trust the configured voice
        return preferred
    names = voices.get(lang, [])
    return preferred if preferred in names else (names[0] if names else None)


# ---------------------------------------------------------------- engines
def _omnivoice(text: str, lang: str) -> bytes:
    """Engine 1: the OmniVoice worker (voice_env/tts_server.py, own env, keeps the model loaded on the GPU)."""
    import json
    import urllib.request
    req = urllib.request.Request(CONFIG["tts_url"] + "/tts", data=json.dumps({"text": text, "language": lang}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=CONFIG["tts_timeout"]) as r:
        return r.read()


def _system_voice(text: str, lang: str) -> tuple[bytes | None, str]:
    """Engine 2 (always available): Windows OneCore voices (Hoda / Naayf) or macOS `say`."""
    if lang == "ar" and not has_voice("ar"):
        log.warning("no Arabic system voice installed - not speaking: %r", text[:60])
        return None, "no-arabic-voice"
    with tempfile.TemporaryDirectory() as d:
        if sys.platform == "win32":
            src, wav = Path(d) / "in.txt", Path(d) / "out.wav"
            src.write_text(text, encoding="utf-8")
            _powershell(_WINDOWS_TTS_PS, YOTH_TTS_LANG=lang, YOTH_TTS_IN=str(src), YOTH_TTS_OUT=str(wav))
            return wav.read_bytes(), "windows"
        aiff, wav = Path(d) / "o.aiff", Path(d) / "o.wav"
        voice = _say_voice(lang)
        subprocess.run(["say", *(["-v", voice] if voice else []), "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff), str(wav)], check=True)
        return wav.read_bytes(), "say"


def _synth(text: str, lang: str) -> tuple[bytes | None, str]:
    """One piece of text -> (wav, engine). OmniVoice first (unless tts_engine="system"); any error -> system voice."""
    if CONFIG["tts_engine"] in ("auto", "omnivoice"):
        try:
            return _omnivoice(text, lang), "omnivoice"
        except Exception as e:
            log.warning("OmniVoice unavailable (%s) - falling back to the system voice", type(e).__name__)
            if CONFIG["tts_engine"] == "omnivoice":
                raise
    return _system_voice(text, lang)


def speech_chunks(text: str, language: str | None = None, clean: bool = True) -> list[tuple[str, str]]:
    """Cleaned text -> [(piece, lang)] to synthesize one after another: sentences (. ؟ ! ?), long ones also cut at
    the Arabic / Latin comma, so the first audio is ready quickly."""
    spoken = clean_text(text) if clean else text.strip()
    pieces = []
    for sent in [x.strip() for x in _SENT.split(spoken) if x.strip()]:
        parts = re.split(r"(?<=[،,])\s+", sent) if len(sent) > CONFIG["chunk_chars"] else [sent]
        pieces += [x.strip() for x in parts if x.strip()]
    return [(x, language or detect_language(x)) for x in pieces]


def synthesize(text: str, language: str | None = None, clean: bool = True):
    """text -> (wav_bytes | None, engine, spoken_text) in one piece (the web API). Arabic is never read by an
    English voice: with no OmniVoice and no Arabic system voice -> (None, "no-arabic-voice", spoken)."""
    spoken = clean_text(text) if clean else text.strip()
    if not spoken:
        return None, "none", ""
    wav, engine = _synth(spoken, language or detect_language(spoken))
    log.info("TTS engine: %s | %r", engine, spoken[:60])
    return wav, engine, spoken


last_speak = {}                  # stats of the last speak(): engines, time_to_first_audio, pieces


def speak(text: str, language: str | None = None, block: bool = True, save_to: str | None = None) -> str:
    """Clean the text and play it piece by piece: piece 1 starts playing while the next pieces are synthesized
    (lower time-to-first-audio). Temp files are deleted after playback. Returns the engine(s) used, e.g.
    "omnivoice" or "omnivoice+windows" after a fallback. Never raises: if every engine fails it returns "none".
    save_to: also write the first piece's wav there (debugging)."""
    import queue
    t0 = time.time()
    chunks = speech_chunks(text, language)
    if not chunks:
        return "none"
    q, engines, stats = queue.Queue(), [], {"pieces": len(chunks), "time_to_first_audio": None}

    def produce():
        for piece, lang in chunks:
            try:
                wav, engine = _synth(piece, lang)
            except Exception as e:
                log.error("TTS failed for %r: %s", piece[:40], e)
                wav, engine = None, "none"
            engines.append(engine)
            log.info("TTS engine: %s | %r", engine, piece[:60])
            q.put(wav)
        q.put(StopIteration)

    def consume():
        first = True
        while (wav := q.get()) is not StopIteration:
            if not wav:
                continue
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="yoth-tts-")
            os.close(fd)
            Path(path).write_bytes(wav)
            if first and save_to:
                Path(save_to).write_bytes(wav)
            if first:
                stats["time_to_first_audio"] = round(time.time() - t0, 2)
                first = False
            _play_wav(path, True, delete_after=True)
        stats["total"] = round(time.time() - t0, 2)

    threading.Thread(target=produce, daemon=True).start()
    if block:
        consume()
    else:
        threading.Thread(target=consume, daemon=True).start()
    used = "+".join(dict.fromkeys(e for e in engines if e)) or "none"
    stats["engines"] = engines
    last_speak.clear()
    last_speak.update(stats)
    if "no-arabic-voice" in engines:
        print("   [tts: skipped Arabic - no Arabic voice installed and OmniVoice not running]")
    print(f"   [tts: {used}]")
    return used


# =============================================================== loop
def voice_loop(answer_fn, language: str | None = FROM_CONFIG):
    """listen -> answer_fn(text) -> print answer -> speak it. Ctrl+C at the prompt exits.
    answer_fn(text) -> str (full answer, with sources)."""
    check_voices()
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
