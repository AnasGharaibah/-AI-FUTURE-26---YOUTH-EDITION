"""Step 5 - per-document summaries + the Service Map (run at the end of ingest; cached by file hash).

For every document the LLM writes, grounded ONLY in the document text:
    {"file", "service", "summary", "topics", "sections", "cases", "key_facts": [{"fact", "source_page"}],
     "example_questions"}
Short documents go to the model whole; long ones are map-reduced (notes per group of sections, then merged).
Code-side checks: every number in summary / key_facts / cases must occur in the document (else that sentence /
fact / case is dropped), topics must occur in the document, sections come from the indexed parent sections.

    from DB.summaries import build
    build(pipeline)                       # data/summaries.json + data/service_map.md
    uv run python -m DB.main summaries [--force]
"""
import hashlib
import json
import re
from pathlib import Path

from .arabic import check_numbers, drop_number_sentences, is_arabic, normalize_display, normalize_search, tokens
from .chunks import ntok
from .parse import PAGE_RE

ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_PATH = ROOT / "data/summaries.json"
MAP_PATH = ROOT / "data/service_map.md"
VERSION = 4                         # bump when the prompt / checks change -> everything is regenerated
SINGLE_PASS_TOKENS = 2500             # documents up to this size are summarized in one call
SECTION_BATCH_TOKENS = 1500           # map step: sections grouped up to this size per call
MAP_TOKENS = 700                      # hard limit for the Service Map
MAX_FACTS, MAX_EXAMPLES = 5, 5

# canonical topic -> words that show the topic is really in the document (compared after normalize_search)
TOPICS = {                            # specific words only: "دينار" or "خلال" alone would match salary tables etc.
    "الرسوم": ("رسوم", "fee"),
    "الوثائق المطلوبة": ("الوثائق المطلوبه", "المستندات المطلوبه"),
    "الخطوات": ("الخطوات", "خطوات التقديم"),
    "مدة الإنجاز": ("مده الانجاز", "ايام عمل", "يوم عمل"),
    "الشروط": ("الشروط", "يشترط", "العمر الادني"),
    "مكان التقديم": ("مكان التقديم", "مراكز الخدمه", "مراكز الاحوال", "البوابه الالكترونيه", "مركز الخدمه"),
    "الغرامات": ("غرامه", "يعاقب"),
    "المواد القانونية": ("الماده (",),
    "التظلم والاعتراض": ("التظلم",),
    "صلاحية الوثيقة": ("صالحه لمده", "صلاحيه"),
}
CASE_WORD_SHARE = 0.7                 # a case is kept when this share of its words occurs in the document

_SYSTEM = ("أنت محرر دقيق يكتب بطاقة تعريف لوثيقة رسمية. اعتمد فقط على نص الوثيقة المرفق، ولا تضف أي معلومة من خارجه. "
           "انسخ الأرقام والرسوم والمدد حرفياً كما وردت. اكتب بالعربية. أعد كائن JSON صالحاً فقط، بدون أي شرح.")

_SCHEMA = """أعد JSON بهذه الحقول بالضبط:
{
  "service": "اسم الخدمة أو موضوع الوثيقة باختصار",
  "summary": "جملتان أو ثلاث: ما هذه الخدمة أو الوثيقة ولمن هي",
  "topics": ["فقط المواضيع الموجودة فعلاً في الوثيقة، من هذه القائمة: %s"],
  "cases": ["الحالات التي تغيّر الإجابة، مثل: جواز 5 سنوات / 10 سنوات، إصدار جديد / تجديد"],
  "key_facts": [{"fact": "حقيقة مهمة منسوخة حرفياً من الوثيقة", "source_page": رقم الصفحة أو null}],
  "example_questions": ["3 إلى 5 أسئلة طبيعية قد يسألها المواطن، واحد منها على الأقل باللهجة العامية"]
}
key_facts: خمس حقائق كحد أقصى.""" % "، ".join(TOPICS)

# Ollama structured output: the reply is always valid JSON with these fields (small models break JSON otherwise)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "service": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "cases": {"type": "array", "items": {"type": "string"}},
        "key_facts": {"type": "array", "items": {"type": "object", "properties": {
            "fact": {"type": "string"}, "source_page": {"type": ["integer", "null"]}}, "required": ["fact"]}},
        "example_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["service", "summary", "topics", "cases", "key_facts", "example_questions"],
}


# ---------------------------------------------------------------- document text
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_text(md: str) -> str:
    """Parsed markdown with its page markers turned into readable "[صفحة N]" lines (pages are 1-based)."""
    return PAGE_RE.sub(lambda m: f"\n[صفحة {int(m.group(1)) + 1}]\n", md).strip()


def page_count(md: str) -> int:
    return len(PAGE_RE.findall(md)) or 1


def _sections(parents: list[dict], source: str) -> list[str]:
    """Unique section titles of a document, in order, from the indexed parent sections."""
    out = []
    for p in parents:
        if p.get("source") != source or not p.get("section"):
            continue
        title = p["section"].split(" > ")[-1].strip()
        if title and title not in out:
            out.append(title)
    return out[:20]


# ---------------------------------------------------------------- LLM calls
def _default_chat():
    from Agent.llm import chat                      # lazy: DB works without the Agent package (same as the rewriter)
    return chat


def parse_json(text: str):
    """The summary object in an LLM reply (code fences / text around it are ignored); None if there is none.
    Only an object with "service" or "summary" counts - never a nested key_facts item of a broken reply."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and ("service" in obj or "summary" in obj):
                        return obj
                except json.JSONDecodeError:
                    pass
                break
        start = text.find("{", start + 1)
    return None


def _ask_json(chat, prompt: str, max_tokens: int = 1500):
    msgs = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    obj = parse_json(chat(msgs, max_tokens=max_tokens, temperature=0, format=JSON_SCHEMA))
    if obj is None:                                  # one retry with a blunt reminder
        msgs.append({"role": "user", "content": "أعد كائن JSON صالحاً فقط."})
        obj = parse_json(chat(msgs, max_tokens=max_tokens, temperature=0, format=JSON_SCHEMA))
    if obj is None:
        print("  (no valid JSON from the model - only the heading and sections are used)")
    return obj or {}


def _map_notes(chat, text: str, parents: list[dict], source: str) -> str:
    """Long documents: short factual notes per group of sections (map), to be merged into one summary (reduce)."""
    groups, cur = [], ""
    bodies = [p["text"] for p in parents if p.get("source") == source] or [text]
    for body in bodies:
        if cur and ntok(cur + body) > SECTION_BATCH_TOKENS:
            groups.append(cur)
            cur = ""
        cur += body + "\n\n"
    if cur.strip():
        groups.append(cur)
    notes = []
    for g in groups:
        msgs = [{"role": "system", "content": _SYSTEM.replace("أعد كائن JSON صالحاً فقط، بدون أي شرح.", "")},
                {"role": "user", "content": "اكتب ملاحظات قصيرة عن هذا الجزء من الوثيقة: موضوعه، والحقائق المهمة "
                 "(منسوخة حرفياً مع رقم الصفحة إن وُجد)، والحالات التي تغيّر الإجابة.\n\n" + g}]
        notes.append(chat(msgs, max_tokens=400, temperature=0))
    return "\n\n".join(notes)


# ---------------------------------------------------------------- validation
def _first_heading(md: str, source: str) -> str:
    m = re.search(r"(?m)^#+\s*\**(.+?)\**\s*$", md)
    return normalize_display(m.group(1)) if m else source


def _present(topic: str, doc_norm: str) -> bool:
    words = TOPICS.get(topic, ()) + (normalize_search(topic),)
    return any(normalize_search(w) in doc_norm for w in words)


def validate(raw: dict, *, source: str, md: str, sections: list[str]) -> dict:
    """Keep only what the document supports. Returns the stored summary dict (+ "dropped": what was removed)."""
    doc = document_text(md)
    doc_norm = normalize_search(doc)
    pages = page_count(md)
    dropped = []

    def text(v) -> str:
        return normalize_display(str(v or "")).strip()

    summary = text(raw.get("summary"))
    bad = check_numbers(summary, doc)
    if bad:
        dropped.append(f"summary numbers {bad}")
        summary = drop_number_sentences(summary, bad)

    facts = []
    for f in raw.get("key_facts") or []:
        fact = text(f.get("fact") if isinstance(f, dict) else f)
        if not fact:
            continue
        bad = check_numbers(fact, doc)
        if bad:
            dropped.append(f"fact {fact[:40]!r}: numbers {bad}")
            continue
        page = f.get("source_page") if isinstance(f, dict) else None
        try:
            page = int(page) if page is not None and 1 <= int(page) <= pages else None
        except (TypeError, ValueError):
            page = None
        facts.append({"fact": fact, "source_page": page})

    cases, arabic_doc = [], is_arabic(doc[:3000])
    for c in raw.get("cases") or []:
        c = text(c)
        if not c:
            continue
        words = tokens(c)
        sides = {normalize_search(s).strip() for s in re.split(r"[/؛;|]| أو ", c) if s.strip()}
        # stem match ("جديد" counts in "جديدة"); an Arabic case of an English document can only be checked by numbers
        grounded = (sum(w in doc_norm for w in words) / len(words) >= CASE_WORD_SHARE if arabic_doc
                    else bool(re.search(r"\d", c))) if words else False
        if check_numbers(c, doc):
            dropped.append(f"case {c[:40]!r}: numbers {check_numbers(c, doc)}")
        elif not grounded:
            dropped.append(f"case {c[:40]!r}: words not in the document")
        elif "/" in c and len(sides) < 2:                         # "5 سنوات / 5 سنوات" is not a choice
            dropped.append(f"case {c[:40]!r}: both sides are the same")
        elif c not in cases:
            cases.append(c)

    topics = []
    for t in raw.get("topics") or []:
        t = text(t)
        if t and t not in topics and _present(t, doc_norm):
            topics.append(t)
    if not topics:                                   # e.g. an English document: its own section titles instead
        topics = sections[:4]

    examples = [text(q) for q in raw.get("example_questions") or [] if text(q)][:MAX_EXAMPLES]
    return {"file": source, "service": text(raw.get("service")) or _first_heading(md, source),
            "summary": summary, "topics": topics, "sections": sections, "cases": cases[:6],
            "key_facts": facts[:MAX_FACTS], "example_questions": examples, "dropped": dropped}


_DIALECT = re.compile(r"(^|\s)(شو|ايش|إيش|قديش|اديش|أديش|وين|بدي|بدّي|بقدر|بكم|ليش|هسا|هلأ|كيف بقدر|لازم)(\s|؟|$)")


def _ensure_dialect(chat, questions: list[str]) -> list[str]:
    """At least one example question in colloquial Arabic; small models write only MSA, so rewrite one."""
    if not questions or any(_DIALECT.search(q) for q in questions):
        return questions
    shots = [("ما الوثائق المطلوبة لتجديد جواز السفر؟", "شو الأوراق اللي بدي اياها عشان اجدد الجواز؟"),
             ("كم رسوم إصدار بطاقة الهوية؟", "قديش رسوم الهوية؟"),
             ("أين أقدم طلب رخصة القيادة؟", "وين بقدر اقدم على رخصة السواقة؟"),
             ("هل يمكن تجديد الجواز إلكترونياً؟", "بقدر اجدد الجواز اونلاين؟")]
    msgs = [{"role": "system", "content": "أعد صياغة السؤال باللهجة العامية الأردنية في سطر واحد، بدون أي شرح. "
                                          "استخدم كلمات مثل: شو، قديش، وين، بدي، بقدر."}]
    for msa, dialect in shots:
        msgs += [{"role": "user", "content": msa}, {"role": "assistant", "content": dialect}]
    msgs.append({"role": "user", "content": questions[-1]})
    q = normalize_display(chat(msgs, max_tokens=60, temperature=0).splitlines()[0] if questions else "").strip(" \"'«»")
    return questions[:-1] + [q] if q and _DIALECT.search(q) else questions


def summarize_document(chat, source: str, md: str, parents: list[dict]) -> dict:
    sections = _sections(parents, source)
    doc = document_text(md)
    if ntok(doc) <= SINGLE_PASS_TOKENS:
        raw = _ask_json(chat, f"{_SCHEMA}\n\nالوثيقة ({source}):\n{doc}")
    else:
        notes = _map_notes(chat, doc, parents, source)
        raw = _ask_json(chat, f"{_SCHEMA}\n\nملاحظات عن أجزاء الوثيقة ({source}):\n{notes}")
    s = validate(raw, source=source, md=md, sections=sections)
    if is_arabic(doc[:3000]):
        s["example_questions"] = _ensure_dialect(chat, s["example_questions"])
    return s


# ---------------------------------------------------------------- Service Map
def _shorten(text: str, words: int | None) -> str:
    first = re.split(r"(?<=[.!؟?])\s+", text.strip())[0] if text else ""
    if words is None:
        return first
    w = first.split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


# (summary words, max topics, max cases) - tried in order until the map fits MAP_TOKENS; files are never dropped
_LEVELS = [(None, 6, 3), (18, 4, 2), (10, 3, 1), (5, 3, 1), (0, 2, 1), (0, 0, 0)]


def service_map(summaries: list[dict], limit: int = MAP_TOKENS) -> str:
    text = ""
    for words, n_topics, n_cases in _LEVELS:
        lines = [f"# خريطة الخدمات ({len(summaries)} وثائق)"]
        for s in summaries:
            line = f"- **{s['service']}** (`{s['file']}`)"
            one = _shorten(s.get("summary", ""), words) if words != 0 else ""
            if one:
                line += f": {one}"
            if n_topics and s.get("topics"):
                line += " | المواضيع: " + "، ".join(s["topics"][:n_topics])
            if n_cases and s.get("cases"):
                line += " | الحالات: " + "؛ ".join(s["cases"][:n_cases])
            lines.append(line)
        text = "\n".join(lines)
        if ntok(text) <= limit:
            break
    return text


# ---------------------------------------------------------------- build
def load() -> dict:
    """{file name: summary dict} from data/summaries.json (empty when not generated yet)."""
    if not SUMMARIES_PATH.exists():
        return {}
    return {k: v["summary"] for k, v in json.loads(SUMMARIES_PATH.read_text(encoding="utf-8")).items()}


def build(pipeline, force: bool = False, chat=None, model: str | None = None) -> dict:
    """Summaries for every parsed document (regenerated only when the raw file changed, or with force), then the
    Service Map. Returns {file: summary}."""
    chat = chat or _default_chat()
    if model is None:
        try:
            from Agent.llm import MODEL as model
        except Exception:
            model = ""
    cache = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8")) if SUMMARIES_PATH.exists() else {}
    parents_path = Path(pipeline.store.parents_path)
    parents = list(json.loads(parents_path.read_text(encoding="utf-8")).values()) if parents_path.exists() else []
    raw_dir, parsed_dir = Path(pipeline.cfg["raw_dir"]), Path(pipeline.cfg["parsed_dir"])
    out, changed = {}, 0
    for md_path in sorted(parsed_dir.glob("*.md")):
        source = md_path.name[:-3]
        raw = raw_dir / source
        h = file_hash(raw) if raw.exists() else hashlib.sha256(md_path.read_bytes()).hexdigest()
        hit = cache.get(source)
        if not force and hit and hit.get("hash") == h and hit.get("version") == VERSION:
            out[source] = hit
            continue
        md = md_path.read_text(encoding="utf-8")
        if len(PAGE_RE.sub("", md).strip()) < 50:
            print(f"summaries: skip {source} (no text)")
            continue
        print(f"summaries: {source} ...", flush=True)
        s = summarize_document(chat, source, md, parents)
        if s["dropped"]:
            print(f"  dropped (not in the document): {s['dropped']}")
        out[source] = {"hash": h, "version": VERSION, "model": model, "summary": s}
        changed += 1
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    smap = service_map([v["summary"] for v in out.values()])
    MAP_PATH.write_text(smap + "\n", encoding="utf-8")
    print(f"summaries: {changed} generated, {len(out) - changed} cached; service map {ntok(smap)} tokens "
          f"-> {MAP_PATH}")
    return {k: v["summary"] for k, v in out.items()}
