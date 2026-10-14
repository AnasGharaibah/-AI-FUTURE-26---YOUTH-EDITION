# Yoth AI

## Setup
```bash
uv sync                                   # Python env (Windows without uv on PATH: py -m uv sync)
ollama pull bge-m3                        # embeddings
ollama pull qwen3:4b-instruct             # chat model (YOTH_MODEL=... to switch)
```

## Web app
```bash
uv run uvicorn api:app                    # then open http://127.0.0.1:8000
```
`api.py` serves the page in `UI/` and the API it uses:

- **Chat** – `POST /api/chat` runs `Agent/rag.py` and streams the answer with numbered sources; click a citation to
  open the document at the cited passage. Searches all documents, one folder, or one document.
- **Library** – the files in `data/raw/`. Upload (`POST /api/upload`) saves into `data/raw/` and re-indexes; delete
  removes the file and re-indexes. Folders and stars are kept in `data/library.json`.
- **Voice** – the browser records the mic; `POST /api/stt` transcribes it, `POST /api/tts` reads answers aloud.

Chats are saved in the browser (local storage). The first PDF upload downloads the Marker models once (slow).

## Voice

Speak Arabic or English, hear the answer. 100% local.

**STT** – macOS: `mlx-whisper`; Windows/Linux: [OpenAI Whisper](https://github.com/openai/whisper) (`turbo`,
downloaded on first use, ~1.6 GB; audio is decoded with PyAV, so no ffmpeg install is needed).
**TTS** – macOS `say` (Majed for Arabic); Windows speech voices (Hoda / Naayf for Arabic, via the WinRT speech API).
Arabic is never read by an English voice: without an Arabic voice, Arabic answers are shown but not spoken, and a
warning is printed at startup. Add one in **Settings → Time & language → Speech** (Windows) or **System Settings →
Accessibility → Spoken Content → Manage Voices** (macOS).

```bash
uv run python -m Agent.main --voice       # terminal voice chat (--lang en | auto for mixed speech)
```
Speak after "listening…"; the question is sent after ~1.2 s of silence. Ctrl+C quits.

```python
from Agent.voice import listen, transcribe, speak
transcribe("q.wav", language=None)       # {"text", "language", "seconds"}
speak("The limit is 150 JOD [1].")       # -> "say" | "windows" | "no-arabic-voice"
```
All settings (models, language, silence threshold, spoken length) are in `CONFIG` at the top of `Agent/voice.py`.

### Troubleshooting
- *No sound / empty transcript*: allow the mic (macOS: **System Settings → Privacy & Security → Microphone**;
  browser: the mic prompt); raise `silence_rms` if background noise never "ends" a recording, lower it if speech is cut off.
- *Slow answers / swap*: 8 GB is tight. Check `ollama ps`; use a smaller LLM (`YOTH_MODEL=...`) or a smaller
  speech model (`stt_model` / `whisper_model` in `CONFIG`), and close other apps.

## Arabic government-service mode

Documents in `data/raw/` (PDF, DOCX, HTML, TXT, ...) are parsed, Arabic-normalized, chunked along the
structure of government documents (المادة / الفصل / أولاً / "الوثائق المطلوبة:" / "الرسوم:" ...), embedded with
`bge-m3` and searched with vector + BM25 (Arabic-aware tokens), fused by RRF.

```bash
uv run python -m DB.main ingest                    # parse + index data/raw   (add --force to rebuild)
uv run python -m DB.main search "شو الأوراق المطلوبة لتجديد الجواز؟" [--rerank] [--no-rewrite]
uv run python -m Agent.main                        # chat   (YOTH_MODEL=gemma3:4b to switch model)
uv run python -m Agent.main --voice                # voice
uv run python -m Agent.eval --tag mytest           # evaluate on questions_ar.csv   (--table compares saved runs)
uv run pytest                                      # unit tests
```

Answer path (`Agent/rag.py`): follow-up -> standalone question -> MSA rewrite (dialect -> Fusha) -> hybrid search ->
numbered sources -> LLM with the Arabic system prompt -> guard (missing `[n]` citation = one retry; numbers that are
not in the sources are logged). `YOTH_MODE=agent` switches back to the old tool-calling agent.

`samples_ar/` has four *fictional* sample documents; `questions_ar.csv` the evaluation questions.

### Known limits
- No OCR: scanned PDFs are reported as `empty` / `partial` in `data/parsed/_manifest.json`.
- PDFs: pdfium (Marker) swaps the letters of Arabic ligatures (`املنتهي`); `DB/parse.py` repairs this against a second
  extractor (pypdf) when the swapped word is a unique anagram of a word pypdf saw. RTL **table cells** can still come out
  with scrambled word order (numbers stay correct) - ingest warns when many words look garbled. DOCX/HTML/TXT are clean.
- Reversed Arabic text (a bad PDF text layer) is detected and warned about, not fixed.
