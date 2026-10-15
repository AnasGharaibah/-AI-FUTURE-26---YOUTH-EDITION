# Yoth AI – Government Services Chatbot

Answers citizens' questions about Jordan's **passport**, **family book** and **smart ID card** services (first issue,
renewal, lost, damaged), only from the 12 official service documents, with the organisers' model
`command-r7b-arabic:7b` running locally in Ollama. Built for the AI Future 26 hackathon (42 Irbid · Youth Edition).

## Quick start (evaluation run)

```bash
ollama pull command-r7b-arabic:7b
python main.py questions.json          # writes answers.json next to questions.json
```

- `main.py` uses **only the Python standard library**: no venv and no `pip install`. It talks to Ollama at
  `http://127.0.0.1:11434`.
- The 12 service files are read from `data/raw/`. It falls back to `data/`, or to the folder in `YOTH_DATA_DIR`.
- Messages are answered in order. Every chat is a new user, and earlier chats are forgotten.
- `answers.json` keeps the input structure exactly and adds `answer` and `contexts` to every message. It is saved
  after every message, so a crash still leaves a valid partial file.
- A message that fails gets an error answer, and the run continues.
- Arabic console output is forced to UTF-8, so a Windows console does not crash.

**Before the run (offline check):**
1. Run `ollama list` and check that `command-r7b-arabic:7b` is there.
2. Set the Ollama server settings below and restart Ollama.
3. Turn Wi-Fi off and run `python main.py eval/practice/questions.json` once.
4. Run `ollama ps` and check that it shows `100% GPU`.

### Talk to it

```bash
python chat.py          # type a question; 'new' = new chat, 'exit' = quit
```

**What it prints for each question:**
- the answer and its contexts;
- the route, and what the bot understood;
- flags (retries, typo fixes, memory);
- the time taken and the prompt tokens.

**Memory within a chat:**
- The bot remembers what you said earlier about the same document. For example, "أنا بالأردن ومش عسكري وجوازي
  مخلص ورقه" given in reply to its question is used for the next "كم المبلغ؟".
- If the fee depends on something you didn't say, it lists the cases and asks **one** question about what is
  missing.

**Logs:** everything is saved in `chat_logs/`:
- `chat_log.md` is readable, and better in VS Code when the terminal shows Arabic in the wrong order.
- `chat_log.json` holds all chats in the `answers_example.json` format: `chats → messages → message_id, question,
  answer, contexts`.

### Ollama settings

| Where | Setting |
|---|---|
| Server (Windows: user environment variables, then restart Ollama) | `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_IGPU_ENABLE=1` (needed for laptops with an integrated GPU; without it Ollama runs on the CPU) |
| Every request (`Agent/ollama_http.py`) | `num_ctx 8192`, `temperature 0`, `top_p 1`, `num_predict 450`, `repeat_penalty 1.05`, `keep_alive 30m`, no `think` parameter |

`num_ctx` goes up to 12288, and at most 16384, only when a prompt is estimated above ~7k tokens. Every call logs
Ollama's `prompt_eval_count` and warns above 90% of `num_ctx`, because Ollama cuts long prompts silently.

## How it works

```
message ─► router ──┬─ out of scope ─► fixed reply, contexts []
                    ├─ unclear      ─► asks only what is missing, contexts []
                    └─ services     ─► numbered snippets ─► LLM answer with [n] ─► guards ─► answer + contexts
```

### 1. Data (`Agent/services.py`)
All 12 files have the same nine fields:

| Field | Snippet label |
|---|---|
| الفئة المستفيدة | الفئة المستفيدة |
| أماكن تقديم الخدمة | أماكن تقديم الخدمة |
| شروط تقديم الخدمة | الشروط |
| الوثائق المطلوبة للحصول على الخدمة | الوثائق المطلوبة |
| إجراءات تقديم الخدمة | الإجراءات |
| الزمن المستغرق لتقديم الخدمة | الزمن المستغرق |
| شركاء الخدمة | شركاء الخدمة |
| ملاحظات | ملاحظات |
| الرسوم | الرسوم |

Each file is split on these labels.

Cleaning removes the website menu lines, empty and exactly repeated lines, tabs and stray symbols (`]`, `‘`). The
facts stay word for word, typos included.

Each (service, field) becomes one snippet. Fields longer than 600 characters are cut at line breaks.

### 2. Finding the right service (`Agent/router.py`)
The four passport files look almost the same, so picking the wrong file gives a perfect-looking wrong answer. Routing
is therefore rule-based first:

- **Lexicon:** document words are matched together with Jordanian dialect:
  - Documents: جواز / باسبور, هوية / بطاقة ذكية, دفتر عيلة / رب الأسرة.
  - Situations: تجديد / خلص / انتهى, ضاع / انسرق, تالف / انكسر / اهترى, أول مرة / أطلع.
- **Pairing:** each situation word is paired with the nearest document word in the same clause. So "البطاقة الذكية
  لأول مرة، وتجديد جواز السفر" routes to two services, and a long story about several people routes each problem
  separately.
- **No situation:** a message that names only a document (for example "كم رسوم الجواز؟") gets all four of its services.
  The model then lists the cases and asks one question.
- **LLM router:** used only when the lexicon is unsure. That means no document word, a word from a service that is not
  in the data (رخصة, تصديق, شهادة ميلاد…), conflicting situations, or a follow-up in a conversation. Ollama enforces a
  JSON schema: `in_scope`, `services` (enum of the 12 ids), `needs_clarification`, `standalone_question`. The router
  sees the last 2 Q/A of the chat, so "وكم بتكلف؟" is resolved from the previous turn.
- **Fixed replies (no LLM answer):**
  - Out of scope: "عذراً، لا تتوفر لدي معلومات عن هذه الخدمة…".
  - Unclear: asks only for the part that is missing (the document, the situation, or both).

### 3. Answering (`Agent/competition.py`)
- **Sources:** only the fields the question needs are sent (fees, documents, conditions, place, time, procedure…). The
  notes (ملاحظات) are always sent, because extra fees, deadlines and exceptions are usually there. Snippets look like
  this:
  ```
  [1] (جواز سفر بدل تالف | ملاحظات)
  إذا ثبت إن التلف متعمد تحول المعاملة إلى لجنة التحقيق ...
  [2] (جواز سفر بدل تالف | الرسوم)
  100 دينار
  ```
- **Answer:** the model writes plain Arabic (English for English questions) and puts `[n]` after every fact. It uses the
  organisers' style of strict system prompt: sources only, copy amounts exactly, apply the user's situation, read the
  notes, correct wrong assumptions, answer every part.
- **Situation facts computed in code:** a 7B model compares numbers badly. In testing it called a 17-year-old "under
  16". So the code works out these facts and adds them below the question:
  - age compared with 16 and 18;
  - days left before expiry compared with 180;
  - days since marriage compared with 90 (family book late fee);
  - expired, or pages full ("مخلص ورقه" means full pages, not expired);
  - which loss it is (first, second, third);
  - military or not;
  - inside or outside Jordan.

  The facts are taken from all of the user's messages in the chat, and the newest one wins.
- **Conversation:** in a multi-message chat about the same document, the answer prompt also sees the user's last
  messages and the question the bot just asked.
- **Typos:** ة inside a word is read as و (جةاز → جواز).
- **One question back:** if the model asks several questions, only the first is kept. A fee answer that lists the
  cases gets one question about the missing facts.

### 4. Not making things up
- **Grounded contexts:** `contexts` holds exactly the snippets the answer cited, in citation order, as
  `"<service> - <field>: <text>"`. The `[n]` markers are then removed from the answer text (`YOTH_KEEP_CITATIONS=1`
  keeps them).
- **No citation:** the answer is regenerated once with "يجب ذكر رقم المصدر بعد كل معلومة". If it still has none,
  `contexts` becomes the snippets whose numbers and words overlap the answer. Refusals and clarifying questions get
  `[]`.
- **Number guard:** every number in the answer must appear in the cited snippets or in the user's message. Numbers are
  normalised first: Arabic-Indic digits, separators, and number words such as خمسة وعشرون = 25, دينارين = 2 and
  السادسة عشر = 16. If a number fails, the answer is regenerated once with a warning. If it still fails, the sentence
  with that number is dropped.
- **Not in the documents:** the model must reply with "لا تتوفر لدي هذه المعلومة في الوثائق المتاحة.", and `contexts`
  is `[]`.
- **Clean output:** greetings and "هل ترغب بمعرفة المزيد؟" offers are removed, so the judges read only the answer.

## Testing and evaluation

```bash
python main.py eval/practice/questions.json               # practice file -> eval/practice/answers.json
python -m Agent.comp_eval --file eval/smoke.json --show   # smoke: one question per type, prints answer + contexts
python -m Agent.comp_eval --tag ar                        # our eval set, Arabic answer instructions
python -m Agent.comp_eval --tag en --lang en              # A/B: English answer instructions
python -m Agent.comp_eval --table                         # compare saved runs (results/comp_*.json)
uv run pytest tests/test_competition.py                   # offline unit tests (no LLM needed)
```

- `eval/comp_eval.json` has 42 messages:
  - all 12 services and every question type in the brief (simple fact, fee for a situation, documents, hidden in
    notes, comparison, dialect, wrong assumption, unclear, not in the data, long story, English);
  - 3 multi-turn chats.
- Each message carries the expected route and the words the answer must contain.
- The report gives:
  - route accuracy and answer accuracy;
  - % of answers with at least one citation;
  - % of answers whose numbers are all grounded;
  - seconds per message (average and p90);
  - the largest prompt in tokens compared with `num_ctx`.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `YOTH_MODEL` | `command-r7b-arabic:7b` | Ollama chat model |
| `YOTH_PROMPT_LANG` | `ar` | language of the answer prompt's instructions (`ar` / `en`) |
| `YOTH_KEEP_CITATIONS` | `0` | `1` keeps `[n]` in the answer text |
| `YOTH_DATA_DIR` | `data/raw` | folder with the 12 service files |
| `YOTH_DEBUG` | `0` | `1` also writes `answers.debug.json` (route, flags, prompt tokens per call) |
| `YOTH_LLM_TIMEOUT` | `300` | seconds per Ollama request (one retry on connection errors) |

## Project layout

```
main.py                 competition entry point (stdlib only)
chat.py                 interactive terminal chat, logs to chat_logs/
Agent/competition.py    answer step, guards, Chat (one per conversation)
Agent/router.py         lexicon router + JSON-schema LLM router
Agent/services.py       parsing / cleaning of the 12 files, snippets, contexts
Agent/ollama_http.py    Ollama /api/chat client: fixed options, num_ctx choice, token logging
Agent/comp_eval.py      eval runner (metrics, A/B, smoke)
DB/arabic.py            Arabic normalisation, number words, number guard
data/raw/               the 12 service documents
eval/                   comp_eval.json, smoke.json, practice/ (questions.json + answers_example.json)
tests/                  unit tests (test_competition.py for this pipeline)
```

---

## Web app, voice and general document search

The repo also has an interactive app built on a general RAG pipeline (vector + BM25 search over any documents in
`data/raw/`). The competition run does not use it.

```bash
uv sync                                   # Python env (Windows without uv on PATH: py -m uv sync)
ollama pull bge-m3                        # embeddings for the search index
uv run python -m DB.main ingest           # parse + index data/raw   (--force to rebuild)
uv run uvicorn api:app                    # web app on http://127.0.0.1:8000
uv run python -m Agent.main               # terminal chat   (--voice for voice)
```

- **Web app:** `api.py` serves `UI/`.
  - Chat with numbered sources: click a citation to open the passage.
  - A document library with upload, folders and stars.
  - Voice: speech-to-text with Whisper, speech output with the system's Arabic voice.
- **Voice:**
  - STT: macOS `mlx-whisper`; Windows/Linux `faster-whisper`.
  - TTS: macOS `say`; Windows speech voices (Hoda / Naayf).
  - Arabic answers are never read by an English voice. Add an Arabic voice in Settings → Time & language → Speech
    (Windows) or Accessibility → Spoken Content (macOS).
  - Settings are in `CONFIG` in `Agent/voice.py`.
- **Search / agent:**
  - `Agent/rag.py` (follow-up → MSA rewrite → hybrid search → cited answer).
  - `Agent/agent.py` (tool-calling agent).
  - Eval: `uv run python -m Agent.eval --modes rag,agent`. Note that `questions_v1.csv` / `questions_v2.csv` were
    written for the old sample documents in `samples_ar/`.

## Known limits
- **Speed:** about 8–28 s per message on an AMD Radeon 860M iGPU (prompt ~160 tokens/s, generation ~10 tokens/s).
  Comparisons and long stories are the slowest.
- **7B reasoning:** the model can still pick the wrong rule in complex fee cases. The computed situation facts cover
  age, days left, days since marriage and "abroad", but not every condition.
- **Lexicon coverage:** dialect words that are not in the lexicon go to the LLM router, which is slower and less
  predictable.
- **Web-app search:** no OCR for scanned PDFs, and RTL table cells in PDFs can come out with scrambled word order.
