# Yoth AI – Government Services Chatbot

A chatbot that answers citizens' questions about Jordan's **passport**, **family book** and **smart ID card** services
(first issue, renewal, lost, damaged). Every answer comes **only from the 12 official service documents** and is written
by the organisers' model, `command-r7b-arabic:7b`, which runs locally in Ollama. Built for the **AI Future 26 hackathon
(42 Irbid · Youth Edition)**.

| | |
|---|---|
| Entry point | `python main.py questions.json` → `answers.json` |
| Model | `command-r7b-arabic:7b` (Cohere Command R7B Arabic, 5.1 GB, 16k context), via Ollama |
| Knowledge | `data/raw/` – 12 service pages (3 documents × 4 situations), 9 fields each |
| Runs | fully offline; `main.py` needs only the Python standard library |
| Also | terminal chat (`chat.py`), web app (`api.py` + `UI/`), command to add new service pages |

## Contents
1. [Quick start (evaluation run)](#1-quick-start-evaluation-run)
2. [Setup in detail](#2-setup-in-detail)
3. [Ways to use it](#3-ways-to-use-it)
4. [How it works](#4-how-it-works)
5. [How it avoids making things up](#5-how-it-avoids-making-things-up)
6. [Why there is no vector database](#6-why-there-is-no-vector-database)
7. [Speed](#7-speed)
8. [Testing and evaluation](#8-testing-and-evaluation)
9. [Configuration](#9-configuration)
10. [Project layout](#10-project-layout)
11. [Web app, API and the older pipelines](#11-web-app-api-and-the-older-pipelines)
12. [Rules of the challenge – how they are met](#12-rules-of-the-challenge--how-they-are-met)
13. [Known limits](#13-known-limits)

---

## 1. Quick start (evaluation run)

```bash
ollama pull command-r7b-arabic:7b      # once, while online
python main.py questions.json          # answers.json is written next to questions.json
```

What `main.py` does:
- It reads `questions.json`: `{"chats": [{"chat_id", "messages": [{"message_id", "question"}]}]}`.
- It answers the messages of each chat **in order**. A later message can depend on an earlier one, and every chat is
  a new user, so earlier chats are forgotten.
- It writes `answers.json` with **exactly the same structure**. Every message gets two added keys and nothing else is
  changed:
  ```json
  {"message_id": "chat-01-1",
   "question": "كم يستغرق تجديد دفتر العائلة؟",
   "answer": "يستغرق تجديد دفتر العائلة 45 دقيقة.",
   "contexts": ["تجديد دفتر العائلة - الزمن المستغرق: 45 دقيقة"]}
  ```
  - `answer` is the text the user reads.
  - `contexts` lists the exact pieces of the documents the answer is based on, or `[]` when the bot declines or asks
    back.
- `answers.json` is **saved after every message**, so an interruption still leaves a valid partial file. A message
  that fails gets an error answer and the run continues. At the end, every message is guaranteed to have both keys.
- Every file is read and written as UTF-8, and console output is forced to UTF-8, so Arabic does not crash a Windows
  console.
- It prints one line per message (time and the start of the answer) and a total at the end.

### Checklist before the evaluation (internet off)
1. `ollama list` shows `command-r7b-arabic:7b`.
2. The Ollama server settings in [2.2](#22-ollama-server-settings) are set, and Ollama was restarted afterwards.
3. Turn Wi-Fi off and run `python main.py eval/practice/questions.json`. It must finish, and
   `eval/practice/answers.json` must have the same structure as `eval/practice/answers_example.json`.
4. While the model is loaded, `ollama ps` shows `100% GPU`.
5. On Windows the command can also be `py main.py questions.json`. No virtual environment is needed.

---

## 2. Setup in detail

### 2.1 What is needed
| For | Needs |
|---|---|
| `main.py`, `chat.py`, `python -m Agent.add_document`, `python eval/score.py` | Python ≥ 3.10 (stdlib only) + Ollama + the model |
| Web app, unit tests, the older search pipelines | `uv sync` (the project venv, see `pyproject.toml`) |

### 2.2 Ollama server settings
These are read when the Ollama server starts. On Windows, set them as **user environment variables** and then
restart Ollama: quit it from the tray icon and start it again.

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION','1','User')
[Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE','q8_0','User')
[Environment]::SetEnvironmentVariable('OLLAMA_IGPU_ENABLE','1','User')   # laptops with an integrated GPU
```

| Variable | Why |
|---|---|
| `OLLAMA_FLASH_ATTENTION=1` | Faster prompt processing and a smaller memory footprint. |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | Halves the memory of the context cache, with no visible quality loss. |
| `OLLAMA_IGPU_ENABLE=1` | Without it, Ollama ignores integrated GPUs (AMD Radeon 860M here) and runs on the CPU. With it, `ollama ps` shows `100% GPU`. |

To verify, look in `%LOCALAPPDATA%\Ollama\server.log`. It should show `OLLAMA_FLASH_ATTENTION:true`,
`OLLAMA_KV_CACHE_TYPE:q8_0`, and an `inference compute` line naming the GPU.

### 2.3 Settings of every model call (`Agent/ollama_http.py`)
| Option | Value | Why |
|---|---|---|
| `num_ctx` | 8192 | Ollama's default context is small and cuts long prompts silently. The largest prompts here are about 3.5k tokens. It goes up to 12288, and at most 16384 (the model's limit), only when a prompt is estimated above ~7k tokens. |
| `temperature` / `top_p` | 0 / 1 | Deterministic: the same question gives the same answer. |
| `num_predict` | 450 | Enough for a document list. Prevents run-on answers. |
| `repeat_penalty` | 1.05 | Light. Lists repeat words legitimately. |
| `keep_alive` | 30m | The model stays loaded between messages. |
| `think` | not sent | Command R7B is not a reasoning model. |

Every call logs Ollama's `prompt_eval_count` and warns when a prompt uses more than 90% of `num_ctx`.

### 2.4 Python environment (only for the web app and tests)
```bash
uv sync                    # Windows without uv on PATH: py -m uv sync
```

---

## 3. Ways to use it

### 3.1 Competition run: `main.py`
See [section 1](#1-quick-start-evaluation-run). `YOTH_DEBUG=1 python main.py questions.json` also writes
`answers.debug.json`, with the route, flags, time and prompt tokens of every message and model call.

### 3.2 Talk to it in the terminal: `chat.py`
```bash
python chat.py             # type a question; 'new' = new chat, 'exit' = quit
```
For every question it prints:
- the answer;
- its contexts;
- the route: which service(s), and whether they came from the keyword rules (`lexicon`), the conversation (`state`)
  or the model router (`llm`);
- what the bot understood;
- flags: retries, typo fixes, memory, dropped numbers, …;
- the time and the prompt size of each model call.

Questions in one chat form a conversation until you type `new`. Everything is saved in `chat_logs/`:
- `chat_log.md` is readable. Open it in VS Code if the terminal shows Arabic in the wrong order.
- `chat_log.json` holds all chats in the `answers_example.json` format (`chats → messages → message_id, question,
  answer, contexts`).

### 3.3 Web app
```bash
uv run uvicorn api:app     # http://127.0.0.1:8000
```
The chat uses the same pipeline as `main.py`. See [section 11](#11-web-app-api-and-the-older-pipelines).

### 3.4 Add a new service page: `python -m Agent.add_document`
A new page in the **same 9-field format** (copied from the same government website) becomes a service with no code
change:

```bash
python -m Agent.add_document page.txt --doc birth_certificate --words "شهادة ولادة"   # add
python -m Agent.add_document page.txt --dry-run                                     # only check it
python -m Agent.add_document --list                                                 # services added so far
python -m Agent.add_document --remove birth_certificate_first_issue                 # take one out again
```

What it does:
1. **Checks the page.** It must have the field labels and at least the fees (الرسوم). Missing fields are reported.
2. **Reads the title.** The service name is the page title. The situation comes from it: لأول مرة / إصدار →
   first issue, تجديد → renewal, بدل فاقد → lost, بدل تالف → damaged. Use `--situation` if the title doesn't say.
3. **Finds the document.** If the title names a document the bot already knows, that one is used. A **new document
   type** needs an English id (`--doc birth_certificate`, used in file names). Its keywords are its name with and
   without "ال" (شهادة ميلاد / شهادة الميلاد) plus every `--words`.
4. **Saves it.** The page is copied to `data/raw/<doc>_<situation>.txt`, and the service and document keywords are
   recorded in `data/services.json`.
5. **Checks it in a fresh process.** A question like "كم رسوم <name>؟" must be routed to the new service. It prints
   `OK`, or tells you to add `--words`.

From the next run, `main.py`, `chat.py` and the web app all use the new service:
- the router's keywords;
- the router model's list of allowed ids;
- the answer prompt and the out-of-scope reply, which name the documents;
- the clarifying question ("هل تقصد … أم شهادة ميلاد؟").

Rules:
- A situation that has no page, such as renewing a document that only has a first-issue page, is **declined**.
- The 12 original services can't be replaced or removed.
- Fee rules ([4.5](#45-fee-and-note-line-selection-agentfeespy)) exist only for the 12 originals, so for an added
  service all of its fee lines are sent.

---

## 4. How it works

```
question
  │  fix typos (ة inside a word → و)
  ▼
ROUTER (Agent/router.py) ─────────────── no document known ──► "هل تقصد جواز السفر أم ...؟"          contexts []
  │  keywords + chat state                out of scope ─────► "عذراً، لا تتوفر لدي معلومات ..."       contexts []
  │  (model only to decide in/out of scope)
  ▼  service ids, e.g. ["passport_renewal"]
FACTS (Agent/situation.py)        age 40, expired, express office ...
  ▼
SOURCES (services.py + fees.py)   only the fields the question needs, only the fee lines that apply, all notes
  ▼
MODEL (command-r7b-arabic)        plain Arabic answer, [n] after every fact
  ▼
GUARDS (competition.py)           citation retry · ask-back retry → decline · number guard · cleanup
  ▼
answer (without [n]) + contexts (the cited lines)
```

Each step below is followed through one example question:

> **عمري 40 وجوازي منتهي، وبدي أقدم بمكتب الخدمة المستعجلة. قديش المجموع؟**

### 4.1 Data and cleaning (`Agent/services.py`)
All 12 files are pages from the same website, with the same nine fields after a few menu lines. Each file is split on
these labels:

| Label in the file | Field name in sources and contexts |
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

Cleaning removes:
- the website menu lines (الرئيسية / الخدمات / دليل الخدمات / شارك and the title);
- empty and exactly repeated lines;
- tabs and stray symbols (`]`, `‘`).

The facts stay word for word, including the typos in the source. Text is Unicode-normalised (NFKC, no tatweel, ASCII
digits). Everything is parsed once into memory, so there is no index to build and nothing to download.

Each service has a fixed name used in the sources and contexts:

| | First issue | Renewal | Lost | Damaged |
|---|---|---|---|---|
| Passport | جواز سفر لأول مرة | تجديد جواز السفر | جواز سفر بدل فاقد | جواز سفر بدل تالف |
| Family book | دفتر عائلة لأول مرة | تجديد دفتر العائلة | دفتر عائلة بدل فاقد | دفتر عائلة بدل تالف |
| Smart ID | بطاقة ذكية لأول مرة | تجديد البطاقة الذكية | بطاقة ذكية بدل فاقد | بطاقة ذكية بدل تالف |

### 4.2 Finding the right service (`Agent/router.py`)
The four passport files share most of their text, so a wrong file gives a perfect-looking wrong answer. Routing is
therefore **rule-based and deterministic**.

**Keywords (lexicon).** Matching is done on normalised Arabic (no hamza variants, ة→ه, ى→ي), so spelling variants
match too.

| Kind | Words (examples, incl. Jordanian dialect) |
|---|---|
| Passport | جواز، باسبور، passport |
| Family book | دفتر، دفتر عيلة، رب الأسرة، القيد المدني، family book |
| Smart ID | هوية، بطاقة ذكية، البطاقة الشخصية، بطاقتي، id card |
| Lost | فاقد، ضاع، ضيعت، انسرق، مفقود، lost |
| Damaged | تالف، خربان، انكسر، تمزق، اهترى، damaged |
| Renewal | تجديد، أجدد؛ weaker: منتهي، خلص، انتهى، ساري، expired |
| First issue | أول مرة؛ weaker: إصدار، أطلع، الحصول، فتح دفتر |

Some phrases contain a document word but are not about that document, such as "بطاقة الجسور" or "الهوية المقدسية".
These are ignored. "لأول مرة" next to a loss word counts losses, not a first issue.

**Pairing per clause.** The message is split into clauses (at ، . ؟ and words like وكمان / أما), and each situation
word is paired with the **nearest** document word. Examples:
- "قديش صورة بحتاج للبطاقة الذكية لأول مرة، وقديش لتجديد جواز السفر؟" → `id_card_first_issue` + `passport_renewal`.
- "ابني ضاع جوازه، وبطاقة زوجتي انكسرت" → `passport_lost` + `id_card_damaged`.

The router also remembers which clause each service came from. Long stories need this
([4.7](#47-long-stories)).

**A document but no situation.** "كم رسوم الجواز؟" routes to all four passport services, and the answer lists the
cases.

**Conversation state.** Each chat keeps a (document, situation) state. A new message replaces **only the slots it
names**, and no model call is needed:

| Before | Message | Route |
|---|---|---|
| تجديد جواز السفر | وإذا كان تالف؟ | passport_damaged |
| دفتر عائلة بدل تالف | وللهوية؟ | id_card_damaged |
| "ضاع مني" (no document yet) | للبطاقة الذكية | id_card_lost |
| جواز سفر بدل فاقد | وإذا صارت تاني مرة؟ | passport_lost (second loss) |
| جواز سفر بدل فاقد | وكم بتكلف إذا أول مرة؟ | passport_lost (first loss) |
| تجديد دفتر العائلة | ووين بقدر أقدم؟ | family_book_renewal |

**When the model router is used.** It is used only when no document is known (in the message or the chat), or when a
word from a service outside the data appears (رخصة، تصديق، شهادة ميلاد، تذكرة، فيزا…). It is called with an
Ollama-enforced JSON schema (`in_scope`, `services` as an enum of the service ids, `needs_clarification`,
`standalone_question`), and **only decides in scope or out of scope**. The bot never guesses a document.

**Fixed replies (no answer call, `contexts: []`):**
- Out of scope: "عذراً، لا تتوفر لدي معلومات عن هذه الخدمة. أستطيع المساعدة في خدمات جواز السفر ودفتر العائلة
  والبطاقة الذكية (إصدار أول مرة، تجديد، بدل فاقد، بدل تالف)."
- No document known: "هل تقصد جواز السفر أم دفتر العائلة أم البطاقة الذكية؟". If the situation is also unknown, it
  adds "وهل هي إصدار أول مرة أم تجديد أم بدل فاقد أم بدل تالف؟".
- A known document with a situation that has no page (only possible for added services) → the out-of-scope reply.

*Example:* "جوازي منتهي … بدي أقدم" gives passport + renewal → `passport_renewal`, from the lexicon in under a
millisecond.

### 4.3 The user's situation (`Agent/situation.py`)
A 7B model compares numbers badly. In testing it called a 17-year-old "under 16". So the code reads these facts from
the user's words:

| Fact | Read from (examples) |
|---|---|
| age | عمري 17، ابني عمره 10، بنتي 15 سنة، adult / بالغ |
| days left before expiry | باقي عليه 3 شهور، ضايل عليه شهرين |
| expired / still valid | منتهي، خلص، فايت عليه شهر / ساري، مش منتهي |
| pages full | مخلص ورقه، امتلأت صفحاته (not "expired") |
| military | عسكري، بالجيش، الدرك / مش عسكري |
| inside / outside Jordan | بالأردن / برا، بالخارج، من السفارة، a country name (ألمانيا، السعودية …) |
| which loss | أول / ثاني / تاني / ثالث مرة (only in a chat about a lost document) |
| time since the previous loss or issue | قبل سنتين، صارلي سنة، خلال سنتين |
| days since marriage | تزوجت من 4 شهور، كتبنا الكتاب من 6 شهور |
| express office | مستعجل، الخدمة المستعجلة |

These facts are used in two ways:
- **Hints** below the question in the prompt, for example "العمر المذكور 40 سنة، أي 18 سنة أو أكثر."
- **Choosing the fee lines** ([4.5](#45-fee-and-note-line-selection-agentfeespy)).

In a conversation, the facts come from all of the user's messages about the same document, and the newest statement
wins. In a long story, each service gets the facts of its own clause.

*Example:* age 40, expired, express office.

### 4.4 Choosing the fields (`Agent/competition.py`)
Only the fields the question is about are sent. This keeps prompts small (≈1–2.5k tokens) and keeps the model on
topic:

| Field | Selected by words like |
|---|---|
| الرسوم (+ الشروط) | رسوم، كم، قديش، بتكلف، مبلغ، سعر، غرامة، كفالة، تعهد، أغلى، أرخص |
| الوثائق المطلوبة | أوراق، وثائق، المطلوب، بلزمني، شو بدها توخذ معها، صور |
| الشروط (+ الرسوم) | شروط، مين بقدر، يحق، موافقة، ولي، قاصر، عمر |
| أماكن تقديم الخدمة | وين، مكان، سفارة، برا، خارج |
| الزمن المستغرق | كم بياخد، وقت، مدة، متى، أسرع، نفس اليوم |
| الإجراءات | خطوات، كيف، إجراءات، بصمة، استلام |
| شركاء الخدمة | جهات، وزارة، مخابرات، تجنيد |
| الفئة المستفيدة | مستفيد، أجنبي، مقدسي |
| **ملاحظات** | **always**: extra fees, deadlines and exceptions are usually here |

No match means all fields are sent. In a follow-up ("وإذا كان تالف؟") the topic words of the earlier questions count
too. Multi-service prompts above ~9,000 characters drop the procedure, partner and beneficiary fields.

Fees and notes become **one snippet per line**, so a single rule can be chosen and cited. Other fields are one
snippet each, cut at line breaks into pieces of at most 600 characters.

### 4.5 Fee and note line selection (`Agent/fees.py`)
Fees "depend on the person". For each service, a rule picks the fee line(s) that apply to the facts from 4.3.
**Only those fee lines are sent** (plus every note line), and a hint points the model at them ("المصادر التي تنطبق
على حالة المستخدم في «تجديد جواز السفر»: [3]، [7]"). Lines are named by a phrase of the data (or "the first line"),
**never by an amount**, and the code never writes answer text. When a deciding fact is unknown, every fee line is
sent and the answer lists the cases.

| Service | Rule (in order) |
|---|---|
| جواز سفر لأول مرة | military → "لأفراد القوات" · under 16 → "دون سن 16" · abroad → "خارج البلاد" · otherwise (inside, or age known) → "داخل البلاد" |
| تجديد جواز السفر | abroad → "خارج البلاد" · military → "لأفراد القوات" · under 16 → "دون السادسة عشر" · pages full and not expired → "الممتلئ الصفحات" · expired or < 180 days left → "فوق سن السادسة عشر … منتهي" · valid / ≥ 180 days → "غير المنتهي" · express office → note "المستعجلة" |
| جواز سفر بدل فاقد | first loss → "للمرة الاولى" · later loss → "لاي مرة" (incl. the previous guarantee) |
| جواز سفر بدل تالف | one fee line · intentional damage asked → note "متعمد" |
| دفتر عائلة لأول مرة | issue fee · + the late fine unless married < 90 days ago |
| تجديد دفتر العائلة | expired → "منتهي المدة" line · valid → "غير منتهي المدة" line |
| دفتر عائلة بدل فاقد | 1st loss → fee + pledge (note "لأول مرة") · 2nd → fee + pledge + previous pledge if < 3 years (note "للمرة الثانية") · 3rd → fee + previous pledge (note "للمرة الثالثة": the guarantee) |
| دفتر عائلة بدل تالف | fee · + the previous pledge unless the last book is ≥ 3 years old |
| بطاقة ذكية لأول مرة | fee · + the late fine unless under 18 |
| بطاقة ذكية بدل فاقد | 1st → fee · 2nd → fee + previous pledge if < 3 years (note "للمرة الثانية") · 3rd → fee + previous pledge (notes "للمرة الثالثة": committee + guarantee) · asking when it can be collected → note "أسبوعين" |
| family book (any) | death / several wives asked → notes "أكثر من زوجة"، "وفاة الأب والأم" · deadlines asked → notes "خلال ثلاثة أشهر"، "خلال 30 يوم" |
| تجديد البطاقة الذكية، بطاقة ذكية بدل تالف | one fee line, nothing to choose |

Every rule is tested with **two phrasings that are not in the test questions**, and a test checks that they are not
(`tests/test_fees.py`, 63 cases).

*Example:* expired, age 40 → only the "50 دينار لمن هم فوق سن السادسة عشر …" fee line is sent. The express office
selects the note "يضاف 25 دينار في حال قدم في مكتب الخدمة المستعجلة". The user asked for the total ("المجموع"), so
the prompt also says "إذا طلب المستخدم المجموع فاجمع المبالغ التي تنطبق عليه".

### 4.6 The prompt and the answer
The system prompt is strict:
- answer only from the numbered sources, never from prior knowledge;
- answer only the question asked, in simple MSA (English for English questions);
- `[n]` after every fact;
- copy amounts exactly;
- apply the user's situation;
- read the notes;
- correct wrong assumptions;
- answer every part;
- if the answer isn't in the sources, say "لا تتوفر لدي هذه المعلومة في الوثائق المتاحة.";
- don't paste whole documents.

The user message:
```
المصادر:
[1] (تجديد جواز السفر | الشروط)
...
[3] (تجديد جواز السفر | الرسوم)
50 دينار لمن هم فوق سن السادسة عشر بيوم وجواز السفر منتهي او متبقي به اقل من (180) يوم
[4] (تجديد جواز السفر | ملاحظات)
يضاف 25 دينار في حال قدم في مكتب الخدمة المستعجلة

سؤال المستخدم (كما كتبه): عمري 40 وجوازي منتهي، وبدي أقدم بمكتب الخدمة المستعجلة. قديش المجموع؟
السؤال الموضح: ...
حالة المستخدم (محسوبة من كلامه): العمر المذكور 40 سنة، أي 18 سنة أو أكثر. الوثيقة منتهية الصلاحية. يريد التقديم في
مكتب الخدمة المستعجلة. المصادر التي تنطبق على حالة المستخدم في «تجديد جواز السفر»: [3]، [4]. اذكر كل ما فيها.
```
In a conversation, "ما قاله المستخدم سابقاً في هذه المحادثة" (the user's last messages) is added before the
question.

The answer comes back as plain text with citations, for example "رسوم التجديد 50 دينار [3]، ويضاف 25 دينار في مكتب
الخدمة المستعجلة [4]، فالمجموع 75 دينار." Answers are plain text with citations rather than JSON, because that is what
Command R is trained for (grounded generation).

The instructions are in Arabic. An English version (`YOTH_PROMPT_LANG=en`) can be compared on the eval set.

### 4.7 Long stories
A long message (≥ 100 characters) about two or more services from **different clauses** is split into one
sub-question per service:
- the clause(s) about that service;
- plus what the user asks ("قديش رح ندفع على كل وحدة؟");
- with only that clause's facts.

The son's age is not the brother's, and the brother's "Germany" doesn't make the son's passport an abroad fee. The
parts are answered one after another and joined as "• <service>: …". The contexts are the union of the parts'
contexts. A comparison in one clause ("أيهما أغلى: جواز بدل تالف ولا بدل فاقد؟") is not split: it is answered from
both services together.

### 4.8 The output
- `contexts` = the cited snippets, in citation order, without duplicates, as `"<service> - <field>: <text>"` (for
  example "تجديد جواز السفر - الرسوم: 50 دينار لمن هم فوق سن …"). The judges recognise the source from this text.
- `answer` = the text with the `[n]` markers removed (`YOTH_KEEP_CITATIONS=1` keeps them). Greetings ("مرحباً بك!")
  and offers ("هل ترغب بمعرفة المزيد؟") are removed.

---

## 5. How it avoids making things up

| Guard | What happens |
|---|---|
| Only the right service's lines | The router picks the service deterministically. The fee rules send only the lines that apply. The model never sees the look-alike services' fees. |
| Strict prompt | Sources only, copy amounts exactly, fixed sentence when the answer is missing. |
| Citation required | No `[n]` → regenerated once with "يجب ذكر رقم المصدر بعد كل معلومة". Still none → `contexts` = the snippets whose numbers and words overlap the answer. |
| No asking back when the document is known | The model asks a question instead of answering → regenerated once with "answer directly from the sources, or say it is not there". Still nothing cited → **declined** ("لا تتوفر لدي هذه المعلومة في الوثائق المتاحة.", `contexts: []`). Closing questions without a citation are removed from answers. |
| Number guard | Every number in the answer must be in the **cited** lines, the user's own messages, or be a sum of the amounts in the selected lines (50 + 25 = 75). Normalised: Arabic-Indic digits, separators, and number words (خمسة وعشرون = 25، دينارين = 2، السادسة عشر = 16), and amounts glued to words in the data ("250دينار"). Fails → regenerated once with the offending numbers named → still fails → **the sentence is dropped** (with any dangling "المبلغ هو:" line). |
| Stray "not found" | A cited answer that also says "not found" has that sentence removed. An answer that only says "not found" gets `contexts: []`. |
| Out of scope | Services that sound related but aren't in the data (رخصة القيادة، تصديق الشهادة، شهادة الميلاد…) are declined without an answer call. |
| Typos | ة inside a word is read as و (جةاز → جواز، ةرقه → ورقه) before routing. |

---

## 6. Why there is no vector database

The data is 12 small pages (≈3–4 KB each) with an identical structure. Similarity search over chunks is the usual
approach, but here it would hurt:
- The four passport pages are nearly identical text. A search often returns the right field from the **wrong** page,
  which is exactly the trap the brief warns about. Keywords + the 9-field structure pick the exact page and field.
- Rules are deterministic (same question, same route) and take well under a millisecond.
- There are no embeddings to compute and no index to build, so nothing can be missing when the evaluation laptop is
  offline.

The repo still contains a general vector + BM25 pipeline for arbitrary documents. The competition run doesn't use it
([section 11](#11-web-app-api-and-the-older-pipelines)).

---

## 7. Speed

Measured on the development laptop: AMD Radeon 860M iGPU, 31 GB RAM, `100% GPU`.

| Step | Time |
|---|---|
| Routing, facts, fee rules, building the sources | ≈ 0.25 ms per question |
| Model router (only when no document is known / out-of-scope word) | ≈ 8–12 s |
| Writing the answer | ≈ 8–30 s, typically ≈ 15 s: reading the prompt at ≈ 160 tokens/s, writing at ≈ 10 tokens/s |
| A retry (missing citation, asked back, unsupported number) | + one more answer call |
| Long story with N services | N answer calls |
| First call after Ollama starts | + ≈ 10–15 s to load the model |

Average on the 80-question test file: **≈ 20 s per message** (1,586 s for 80 messages, previous version). Nearly all
of it is the model writing, and finding the information is effectively instant.

---

## 8. Testing and evaluation

```bash
uv run pytest                                             # all unit tests (no model needed)
python main.py eval/practice/questions.json               # practice file -> eval/practice/answers.json
python -m Agent.comp_eval --file eval/smoke.json --show   # smoke test: one question per type, answers + contexts
python -m Agent.comp_eval --tag ar                        # our eval set, Arabic answer instructions
python -m Agent.comp_eval --tag en --lang en              # A/B: English answer instructions
python -m Agent.comp_eval --table                         # compare saved runs (results/comp_*.json)
python eval/score.py answers.json test_expected.json --show-fails   # score a run against a rubric file
```

### 8.1 Unit tests (`tests/`)
- **`test_competition.py`** covers:
  - parsing the 12 files and the snippet format;
  - routing without the model (13 phrasings), and conversation slots (6 cases);
  - citations, number words, glued amounts;
  - situation facts;
  - asking back → retry → decline, and removing closing questions;
  - "only the matching fee lines are sent";
  - the long-story split;
  - chat memory.
- **`test_fees.py`** covers every fee rule with two phrasings that are not in `test_questions.json` (63 cases).
- **`test_add_document.py`** runs the add / route / remove cycle on a temporary copy of the data.
- The suite currently runs 195 tests. 194 pass; the one that fails (`test_gpu_config` voice test) needs
  `faster_whisper` installed via `uv sync` and is unrelated to the chatbot.

### 8.2 Eval sets
| File | Content |
|---|---|
| `eval/practice/` | the organisers' practice `questions.json` + `answers_example.json` |
| `eval/smoke.json` | 11 chats, one per question type in the brief |
| `eval/comp_eval.json` | 45 messages (41 chats, 3 of them multi-turn): all 12 services and every question type, with the expected route and the words the answer must contain |
| `test_questions.json` + `test_expected.json` | 80 messages with a rubric per message (behavior, services, must / must-not include, allowed numbers, bonus) |

### 8.3 The scorer (`eval/score.py`)
It scores the three criteria of the brief for every message:
- **Correct answer.**
  - "answer": every required fact is present (digits or number words) and no forbidden phrase is.
  - "decline": refused, with `contexts: []`.
  - "clarify": asks back.
- **Right source.** The contexts come from the expected service(s), and from no other.
- **Grounded.** Every number is in the contexts, the question, or the allowed sums.

Overall = 60% correct + 20% source + 20% grounded. This weighting is ours: the brief only says correctness "matters
most". Bonus (conversation) messages are reported separately. `--show-fails` lists every lost point with the reason.

### 8.4 Results so far
| Run | Correct | Source | Grounded | Overall | s / message |
|---|---|---|---|---|---|
| Practice file (6 messages) | 6 / 6, structure identical to `answers_example.json` | | | | ≈ 13 |
| 80 test messages, main (66) – version before the fee rules | 72.7% | 95.5% | 98.5% | 82.4% | ≈ 20 |
| 80 test messages, bonus conversations (14) – same version | 57.1% | 57.1% | 100% | 65.7% | |
| Our eval set (45 messages), first version | ≈ 84% (route accuracy 100%) | | 96.9% numbers grounded, 100% of answers cited | | ≈ 18 |

The fee rules, conversation slots, long-story split and decline rule (section 4) target the errors that run showed:
- the wrong fee line for an expired passport;
- missing pledge / guarantee lines;
- mixed-up facts in long stories;
- follow-ups that lost the situation;
- asking back instead of answering.

Re-score with `python main.py test_questions.json` and then `python eval/score.py answers.json test_expected.json`.

---

## 9. Configuration

| Variable | Default | Effect |
|---|---|---|
| `YOTH_MODEL` | `command-r7b-arabic:7b` | Ollama chat model |
| `OLLAMA_HOST_URL` | `http://127.0.0.1:11434` | where Ollama listens |
| `YOTH_LLM_TIMEOUT` | `300` | seconds per model call (one retry on connection errors) |
| `YOTH_PROMPT_LANG` | `ar` | language of the answer prompt's instructions: `ar` or `en` |
| `YOTH_KEEP_CITATIONS` | `0` | `1` keeps `[n]` in the answer text |
| `YOTH_DATA_DIR` | `data/raw` (then `data/`) | folder with the service files |
| `YOTH_REGISTRY` | `data/services.json` | services added with `Agent.add_document` |
| `YOTH_DEBUG` | `0` | `1`: `main.py` also writes `answers.debug.json` |
| `YOTH_MODE` | `competition` | web chat: `competition`, or the older `agent` / `rag` pipelines |

---

## 10. Project layout

```
main.py                   competition entry point: questions.json -> answers.json (stdlib only)
chat.py                   terminal chat; logs to chat_logs/chat_log.md + chat_log.json
api.py, UI/               web app (FastAPI + one-page UI)

Agent/competition.py      the answer step: Chat (one per conversation), sources, prompt, guards, long stories
Agent/router.py           keyword router, conversation slots, JSON-schema model router, fixed replies
Agent/situation.py        the user's facts (age, expiry, abroad, loss count, ...) from the messages
Agent/fees.py             which fee / note lines apply to those facts
Agent/services.py         parsing / cleaning of the service pages, snippets, contexts, registry of added services
Agent/add_document.py     python -m Agent.add_document: add / list / remove a 9-field service page
Agent/ollama_http.py      Ollama /api/chat client: fixed options, num_ctx choice, token logging, GPU check
Agent/comp_eval.py        eval runner (metrics, A/B of the prompt language, smoke test)
DB/arabic.py              Arabic normalisation, number words, number guard

data/raw/                 the 12 service pages (+ pages added with add_document)
data/services.json        added services (created by add_document)
eval/                     comp_eval.json, smoke.json, score.py, practice/ (questions + answers_example)
tests/                    unit tests: test_competition.py, test_fees.py, test_add_document.py, ...

Agent/rag.py, agent.py, brain.py, llm.py, DB/*   the older general search pipelines (web app YOTH_MODE=agent|rag)
```

---

## 11. Web app, API and the older pipelines

```bash
uv sync
uv run uvicorn api:app             # http://127.0.0.1:8000
```

**Chat.** It uses the competition pipeline: the same model, routing, fee rules, guards and memory as `main.py`.
- The answer appears once it has passed the guards (not word by word).
- Every `[n]` is clickable and opens the service file with the cited field highlighted.
- The browser sends the conversation with each question, so follow-ups work.
- Greetings get a short reply from the same model.

**Library.** It shows the files in `data/raw/` and opens `.txt` pages directly, with no ingest step. Uploads (+ Add
documents) are saved to `data/raw/` and indexed for the older search pipelines. To make an uploaded service page part
of the chatbot, register it with `python -m Agent.add_document`.

**Voice.**
- Speech-to-text: macOS `mlx-whisper`; Windows/Linux `faster-whisper`.
- Speech output: macOS `say`; Windows speech voices (Hoda / Naayf).
- Arabic is never read by an English voice. Add an Arabic voice in Windows Settings → Time & language → Speech.

**API.**

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | `{question, history}` → newline-delimited JSON events: `trace`, `sources`, `final`, `verification`, `done` |
| `GET /api/health` | status, chat mode and model |
| `GET /api/library`, `POST /api/upload`, `PATCH`/`DELETE /api/docs/{id}`, `POST`/`DELETE /api/folders` | document library |
| `GET /api/doc/{id}/page/{n}`, `GET /api/doc/{id}/file` | document viewer |
| `POST /api/stt`, `POST /api/tts` | voice |
| `POST /api/export/docx` | export answers to Word |
| `GET /api/overview` | service overview (older pipelines) |

**The older pipelines** (`YOTH_MODE=agent` or `rag`):
- They do hybrid vector (`bge-m3`) + BM25 search over any uploaded documents (PDF, Word, HTML, …).
- They need `ollama pull bge-m3` and `uv run python -m DB.main ingest`.
- Terminal chat: `uv run python -m Agent.main`.
- Eval: `uv run python -m Agent.eval --modes rag,agent`. Its `questions_v1.csv` / `questions_v2.csv` were written for
  the old sample documents in `samples_ar/`.

---

## 12. Rules of the challenge – how they are met

| Rule | How |
|---|---|
| Use only the provided files | The only knowledge is `data/raw/` (12 pages). The prompt forbids prior knowledge. The number guard drops numbers that aren't in the cited lines. |
| Every answer generated by the provided model at run time | Each answer is written by `command-r7b-arabic:7b` while `main.py` runs. Code only selects which lines of the data the model sees and checks the result. There are no pre-written answers, no other AI models and no online services. The fixed replies are only for declining or asking which document. |
| Clean the data without changing facts | Cleaning removes menus, duplicate lines and stray symbols. Facts and wording are kept. |
| `python main.py questions.json` with the internet off | Standard library only, Ollama on localhost, no downloads at run time. |
| Exact output format, every message answered | Same structure plus `answer` + `contexts`, saved after every message, errors caught per message. |

---

## 13. Known limits
- **Speed.** About 20 s per message on this iGPU. Long stories with several services and retries take longer.
- **7B reasoning.** The fee rules cover the conditions in the 12 pages, but the model can still word a case wrongly or
  leave out part of a long answer. The number guard catches invented numbers, not missing ones.
- **Lexicon coverage.** Dialect words that aren't in the lexicon can leave a question without a recognised document.
  The bot then asks which document rather than guessing.
- **Added services.** They get the generic pipeline: all of their fee lines, and no fee rules.
- **Older pipelines.** They have no OCR for scanned PDFs, and RTL table cells in PDFs can come out with scrambled word
  order.
