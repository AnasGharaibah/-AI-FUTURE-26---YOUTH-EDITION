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
- If the fee depends on something you didn't say, it lists the cases. It only asks back when it can't tell which
  document you mean.

**Logs:** everything is saved in `chat_logs/`:
- `chat_log.md` is readable, and better in VS Code when the terminal shows Arabic in the wrong order.
- `chat_log.json` holds all chats in the `answers_example.json` format: `chats → messages → message_id, question,
  answer, contexts`.

### Add a new service page

A new page in the same format as the 12 files (the 9 fields, from the same website) becomes a service without any
code change:

```bash
python -m Agent.add_document page.txt --doc birth_certificate --words "شهادة ولادة"   # add
python -m Agent.add_document page.txt --dry-run                                     # only check it
python -m Agent.add_document --list                                                 # added services
python -m Agent.add_document --remove birth_certificate_first_issue                 # take it out again
```

What the command does:
1. Checks the page's fields.
2. Reads the service name and the situation from the title (لأول مرة / تجديد / بدل فاقد / بدل تالف). Use
   `--situation` if the title doesn't say.
3. Uses the existing document if the title names one. A new document type needs an English id (`--doc`), and its
   keywords are its name with and without "ال" plus any `--words`.
4. Copies the page to `data/raw/<doc>_<situation>.txt` and records the service in `data/services.json`.
5. Checks that a question about it is routed to it.

After that, `main.py`, `chat.py` and the web app all pick it up: router keywords, the router's list of allowed ids,
the prompts and replies that name the documents, and the clarifying question. Two things to know:
- A situation without a page (for example renewing a document that has only a first-issue page) is declined.
- Fee rules exist only for the 12 original services, so for an added service all of its fee lines are sent. The 12
  originals can't be replaced or removed.

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
                    ├─ no document  ─► asks which document, contexts []
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
- **No situation:** a message that names only a document (for example "كم رسوم الجواز؟") gets all four of its services,
  and the answer lists the cases.
- **Conversation state:** each chat keeps a (document, situation) state. A new message replaces only the slots it
  names:
  - "وللهوية؟" keeps the situation;
  - "وإذا كان تالف؟" keeps the document;
  - "للبطاقة الذكية" after "ضاع مني" becomes a lost smart ID card;
  - "أول / تاني / ثالث مرة" after a loss is a loss count, not a first issue.

  Follow-ups need no LLM call.
- **Asking back:** only when no document type is detected (in the message or in the chat state).
- **LLM router:** only decides in scope or out of scope, when no document is known or a word from a service that is
  not in the data appears (رخصة, تصديق, شهادة ميلاد…). Ollama enforces a JSON schema.
- **Fixed replies (no LLM answer):**
  - Out of scope: "عذراً، لا تتوفر لدي معلومات عن هذه الخدمة…".
  - No document: asks which document, and also the situation if that is missing too.

### 3. Answering (`Agent/competition.py`, `Agent/fees.py`, `Agent/situation.py`)
- **Sources:** only the fields the question needs are sent (fees, documents, conditions, place, time, procedure…). Fees
  and notes are one snippet per line, and **every note line is always sent**:
  ```
  [3] (تجديد جواز السفر | الرسوم)
  50 دينار لمن هم فوق سن السادسة عشر بيوم وجواز السفر منتهي او متبقي به اقل من (180) يوم
  [4] (تجديد جواز السفر | ملاحظات)
  يضاف 25 دينار في حال قدم في مكتب الخدمة المستعجلة
  ```
- **Situation facts computed in code** (`situation.py`): a 7B model compares numbers badly. In testing it called a
  17-year-old "under 16". The code reads these facts from the user's words, and the newest message in the chat wins:
  - age;
  - days left before expiry;
  - expired, still valid, or pages full ("مخلص ورقه");
  - military or not;
  - inside or outside Jordan (an embassy or a foreign country counts as outside);
  - which loss it is;
  - time since the previous loss or issue;
  - days since marriage;
  - the express office.
- **Fee rules** (`fees.py`): for each service, the code **selects** the fee line(s) that match these facts. Examples:
  - expired passport, age 16 or over: the "فوق سن السادسة عشر" line;
  - second loss of an ID card within 3 years: the fee line and the "استيفاء قيمة التعهد السابق" line.

  It also selects the note lines that answer the question (the two weeks for a lost card, intentional damage, the head
  of family after a death, the express office). Only the selected fee lines are sent, and a hint points the model at
  them. Lines are chosen by a phrase from the data, never by an amount, and the code never writes answer text. When a
  deciding fact is missing, all fee lines are sent and the answer lists the cases. Every rule is tested with two
  phrasings that are not in the test questions (`tests/test_fees.py`).
- **Long stories:** a message about several people or problems is split. Each service is answered on its own, with
  the clause about it plus what was asked, and only that clause's facts (the son's age is not the brother's). The
  parts are joined as "• <service>: …".
- **Conversation:** in a multi-message chat about the same document, the answer prompt also sees the user's last
  messages.
- **Typos:** ة inside a word is read as و (جةاز → جواز).

### 4. Not making things up
- **Grounded contexts:** `contexts` holds exactly the snippets the answer cited, in citation order, as
  `"<service> - <field>: <text>"`. The `[n]` markers are then removed from the answer text (`YOTH_KEEP_CITATIONS=1`
  keeps them).
- **No citation:** the answer is regenerated once with "يجب ذكر رقم المصدر بعد كل معلومة". If it still has none,
  `contexts` becomes the snippets whose numbers and words overlap the answer.
- **Asked back or nothing found:** if the model asks a question back although the document is known, it is asked once
  to answer from the sources. If it still cites nothing, the bot declines ("لا تتوفر لدي هذه المعلومة في الوثائق
  المتاحة.", `contexts: []`). Closing questions without a citation are removed from answers.
- **Number guard:** every number in the answer must appear in the cited snippets or in the user's message, or be a
  sum of the amounts in the selected lines (50 + 25 = 75 when the user asks for the total). Numbers are
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
python eval/score.py answers.json test_expected.json --show-fails   # score a run against a rubric file
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
Agent/services.py       parsing / cleaning of the service files, snippets, contexts, registry of added services
Agent/situation.py      the user's facts (age, expiry, abroad, loss count ...) computed from the messages
Agent/fees.py           which fee / note lines apply to those facts
Agent/add_document.py   python -m Agent.add_document: add a new 9-field service page
Agent/ollama_http.py    Ollama /api/chat client: fixed options, num_ctx choice, token logging
Agent/comp_eval.py      eval runner (metrics, A/B, smoke)
DB/arabic.py            Arabic normalisation, number words, number guard
data/raw/               the 12 service documents
eval/                   comp_eval.json, smoke.json, practice/ (questions.json + answers_example.json)
tests/                  unit tests (test_competition.py for this pipeline)
```

---

## Web app, voice and general document search

```bash
uv sync                                   # Python env (Windows without uv on PATH: py -m uv sync)
uv run uvicorn api:app                    # web app on http://127.0.0.1:8000
```

The web chat uses **the same competition pipeline** as `main.py` (same model, router, guards, memory). The answer
arrives once it is checked (not word by word). Each `[n]` is clickable and opens the service file with the cited
field highlighted. The browser keeps the conversation, so follow-ups work. No ingest is needed: the viewer shows
the `.txt` service files as they are.

`YOTH_MODE=agent` or `YOTH_MODE=rag` switches the chat back to the older general pipelines. These do vector + BM25
search over any documents in `data/raw/`, and need `ollama pull bge-m3` and
`uv run python -m DB.main ingest`.

```bash
uv run python -m Agent.main               # terminal chat of the older pipelines   (--voice for voice)
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
