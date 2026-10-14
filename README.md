# Yoth AI

## Voice mode

Speak Arabic or English, hear the answer. 100% local.

**STT** – `mlx-whisper` (`whisper-large-v3-turbo`, downloaded on first use, ~1.6 GB).
**TTS** – OmniVoice in an isolated env (`voice_env/`, needs torch 2.8.0; the main project uses a newer torch).
If the OmniVoice worker is not running, speech automatically falls back to macOS `say` (Majed / Samantha).

### Setup
```bash
uv sync                                   # main env (mlx-whisper, sounddevice, soundfile)
cd voice_env && uv sync && cd ..          # OmniVoice env (optional - without it `say` is used)
```
macOS: allow the terminal/VS Code to use the mic – **System Settings → Privacy & Security → Microphone**.

### Run
```bash
(cd voice_env && uv run python tts_server.py)        # optional, terminal 1 - keeps OmniVoice loaded
uv run python -m Agent.main --voice                  # terminal 2 (--lang en | auto for mixed speech)
uv run uvicorn api:app                               # API: POST /stt (audio upload), POST /tts (text -> wav)
```
Speak after "listening…"; the question is sent after ~1.2 s of silence. Ctrl+C quits.

```python
from Agent.voice import listen, transcribe, speak
transcribe("q.wav", language=None)       # {"text", "language", "seconds"}
speak("The limit is 150 JOD [1].")       # -> "omnivoice" | "say"
```
All settings (models, language, silence threshold, spoken length, engine) are in `CONFIG` at the top of `Agent/voice.py`.

### Troubleshooting
- *No sound / empty transcript*: check the mic permission above; raise `silence_rms` if background noise never "ends" a recording, lower it if speech is cut off.
- *Slow answers / swap*: 8 GB is tight. Check `ollama ps`; use a smaller LLM (`YOTH_MODEL=...`) or `stt_model = mlx-community/whisper-small-mlx`, and close other apps.
- *OmniVoice not used*: `curl 127.0.0.1:8765/health`; the worker needs ~1 min to load. Force an engine with `CONFIG["tts_engine"] = "say"`.

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
