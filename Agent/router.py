"""Which of the 12 services is a message about?

1. Lexicon router (no LLM): document words (جواز / باسبور / هوية / دفتر عيلة ...) and situation words
   (تجديد / ضاع / تالف / أول مرة ...). Each situation word is paired with the nearest document word, so
   "البطاقة الذكية لأول مرة، وتجديد جواز السفر" gives two services. A document without any situation means all
   four of its situations (the answer step then lists the cases).
2. LLM router (JSON schema enforced by Ollama) only when the lexicon is unsure: no document word, conflicting
   situations, a word from an uncovered service, or a follow-up message in a conversation.

    route("قديش بتكلف الهوية اذا ضاعت؟")  -> Route(services=["id_card_lost"], source="lexicon", ...)
"""
import json
import logging
import re
from dataclasses import dataclass, field

from DB.arabic import _DIACRITICS, _SEARCH_MAP, normalize_display, normalize_search

from . import ollama_http
from .services import BUILTIN_SERVICES, DOC_EXTRA_WORDS, DOCS, NAMES, SERVICE_IDS, SITUATIONS, docs_or

log = logging.getLogger("yoth.router")


# ---------------------------------------------------------------- lexicon
def _rx(*words: str) -> re.Pattern:
    """Regex over normalize_search text; patterns get the same letter folding (no lower(): \\S would become \\s)."""
    return re.compile("|".join(_DIACRITICS.sub("", w).translate(_SEARCH_MAP) for w in words))


_PRE = r"(?:\b|ال|بال|وال|لل|و|ب|ل)"            # clitics before a word
DOC_WORDS = {
    "passport": _rx("جواز", "باسبور", "بسبور", "passport"),
    "family_book": _rx("دفتر", "رب الأسرة", "رب الاسرة", "رب للأسرة", "القيد المدني", "قيد مدني", "السجل المدني",
                       "سجل مدني", "family book"),
    "id_card": _rx("هوية", "هويتي", "هويته", "هويتها", "هويات", "بطاقة ذكية", "البطاقة الذكية", "بطاقة الأحوال",
                   "البطاقة الشخصية", "بطاقة شخصية", "بطاقتي", "بطاقته", "بطاقتها", _PRE + r"بطاقة\b",
                   "id card", "identity card", "smart card", "national id"),
}
# document words inside other things: "بطاقة الجسور", "الهوية الفلسطينية", "الهويات المقدسية"
NOT_DOC = _rx(r"بطاق\S* (إحصاءات |احصاءات )?(ال)?جسور", r"هوي\S* (ال)?(فلسطيني|مقدسي|إسرائيلي)\S*")
# documents added with `python -m Agent.add_document` (data/services.json)
EXTRA_DOC_WORDS = {d: _rx(*(re.escape(w) for w in words)) for d, words in DOC_EXTRA_WORDS.items()}
DOC_WORDS.update(EXTRA_DOC_WORDS)

SIT_STRONG = {
    "lost": _rx("فاقد", "فقدان", "فقد", "ضاع", "ضايع", "ضيعت", "ضيع", "انسرق", "سرق", "مسروق", "مفقود",
                "lost", "stolen"),
    "damaged": _rx("تالف", _PRE + r"تلف(?!ون|زيون)", "خربان", "خربت", _PRE + "خرب", "انكسر", "مكسور", "تكسر",
                   "تمزق", "ممزق", "مقطع", "انقطع", "مشقوق", "انشق", "انحرق", "محروق", "اهترى", "مهتري", "تبهدل",
                   "damaged", "torn", "broken"),
    "renewal": _rx("تجديد", "جدد", "أجدد", "يجدد", "بجدد", "نجدد", "تجدد", "renew"),
    "first_issue": _rx("أول مرة", "اول مرة", "اول مره", "first time", "ما عمري طلعت", "ما طلعت", "ما عندي",
                       "ما معي"),
}
SIT_WEAK = {      # only used when no strong situation word is present; renewal wins over first_issue
    "renewal": _rx("منتهي", "انتهى", "انتهت", "انتهاء", "خلص", "خلصت", "ساري", "صلاحية", "امتلأ", "امتلأت",
                   "ممتلئ", "مليان", "expired"),
    "first_issue": _rx("إصدار", "اصدار", "استخراج", "أطلع", "اطلع", "بطلع", "يطلع", "تطلع", "نطلع", "طلع", "أحصل",
                       "احصل", "الحصول", r"فتح\S* (ال)?دفتر", r"فتح\S* (ال)?سجل", r"فتح\S* (ال)?قيد", "نطلع دفتر", "طلعنا دفتر", "issue", "get a"),
}
COMPARE = _rx("أغلى", "اغلى", "أرخص", "ارخص", "الفرق", r"\bفرق\b", "مقارنة", "أيهما", "ايهما", r"\bأيها\b",
              r"\bولا\b", r"\bوال\b", r"\bأم\b", r"\bأو\b", "compare", "difference", "cheaper", "expensive",
              r"\bor\b", r"\bvs\b")
# services that sound related but are not in the data -> let the LLM router decide (usually in_scope=false)
OUT_OF_SCOPE = _rx("رخصة", "سواقة", "سياقة", "قيادة", "مركبة", "سيارة", "شهادة ميلاد", "شهادة الميلاد",
                   "شهادة جامعية", "الشهادة الجامعية", "تصديق", "فيزا", "تأشيرة", "عدم محكومية", "تذكرة", "طيران",
                   "جواز دبلوماسي", "جواز خاص", "جواز مؤقت", "جواز سفر مؤقت", "جواز السفر المؤقت", "وثيقة سفر",
                   "تجنيس", "ضريبة", "تصريح عمل", "licen", "visa")
CLAUSE_SPLIT = re.compile(r"[،,.؟?!؛;:\n]|\s(?:وكمان|كمان|وكذلك|وأيضا|وايضا|وأيضاً|بالإضافة|بالاضافة|أما|اما|and also)\s")
# "أول / تاني / ثالث مرة": a loss count when the chat is about a lost document
COUNT_WORDS = _rx(r"(المرة|مرة|للمرة) (الأولى|الاولى|الثانية|ثانية|التانية|تانية|الثالثة|ثالثة|التالتة|تالتة)",
                  r"(أول|اول|ثاني|تاني|ثالث|تالت) مرة", "first time", "second time", "third time")

SIT_ASK = "إصدار {doc} لأول مرة أم تجديده أم بدل فاقد أم بدل تالف"


def doc_of(service_id: str) -> str:
    return next(d for d in DOCS if service_id.startswith(d + "_"))


def sit_of(service_id: str) -> str:
    return service_id[len(doc_of(service_id)) + 1:]


@dataclass
class Route:
    services: list[str] = field(default_factory=list)
    in_scope: bool = True
    needs_clarification: bool = False
    standalone: str = ""
    source: str = "lexicon"            # lexicon | state | llm
    docs: list[str] = field(default_factory=list)          # documents named in the message (lexicon)
    sits: list[str] = field(default_factory=list)          # situations named in the message (lexicon)
    unsure: str = ""                   # why the lexicon was unsure ("" = sure)
    clauses: dict[str, list[str]] = field(default_factory=dict)   # service -> the clauses (display text) it came from


def _mentions(rx_map: dict, text: str) -> list[tuple[int, str]]:
    return sorted((m.start(), key) for key, rx in rx_map.items() for m in rx.finditer(text))


def _clause(text: str) -> tuple[dict[str, list[str]], list[str], bool]:
    """One normalized clause -> ({document: [situations nearest to it]}, situations, conflicting situations).
    Every situation word is paired with the nearest document word, so "ضاع جوازه ... وبطاقة زوجتي انكسرت" gives
    passport: [lost], id_card: [damaged]."""
    clean = NOT_DOC.sub(lambda m: " " * len(m.group()), text)
    docs = _mentions(DOC_WORDS, clean)
    sits = _mentions(SIT_STRONG, text)
    if {k for _, k in sits} - {"first_issue"}:
        sits = [x for x in sits if x[1] != "first_issue"]     # "ضاع لأول مرة" counts losses, it is not a 1st issue
    if not sits:
        weak = _mentions(SIT_WEAK, text)
        kind = next((k for k in SIT_WEAK if any(w == k for _, w in weak)), None)
        sits = [x for x in weak if x[1] == kind]
    kinds = list(dict.fromkeys(k for _, k in sits))
    by_doc = {d: [] for _, d in docs}
    for pos, sit in sits:
        if docs:
            d = min(docs, key=lambda x: abs(x[0] - pos))[1]
            if sit not in by_doc[d]:
                by_doc[d].append(sit)
    return by_doc, kinds, len(kinds) > 1


def lexicon(message: str) -> Route:
    """Rule-based route. route.unsure != "" -> the lexicon alone is not enough."""
    text = normalize_search(message)
    r = Route(standalone=message)
    known = text
    for rx in EXTRA_DOC_WORDS.values():       # an added document ("شهادة الميلاد") is no longer out of scope
        known = rx.sub(" ", known)
    if OUT_OF_SCOPE.search(known):
        r.unsure = "out_of_scope_word"
    pairs, open_docs, pending, last_sits, last_doc, conflict = [], [], [], [], None, False
    for clause in filter(str.strip, CLAUSE_SPLIT.split(normalize_display(message))):
        by_doc, sits, clash = _clause(normalize_search(clause))
        r.docs += [d for d in by_doc if d not in r.docs]
        r.sits += [s for s in sits if s not in r.sits]
        conflict |= clash
        if len(by_doc) > 1 and all(any(p[0] == d for p in pairs) for d in by_doc):
            continue                          # "كم رسوم جواز ابني وكم رسوم بطاقة زوجتي؟" restates earlier clauses
        if by_doc:
            for d, ds in by_doc.items():
                if ds:
                    pairs += [(d, s, [clause]) for s in ds]
                else:                         # "جوازي ضاع، والهوية كمان" -> the clause's / previous situations
                    open_docs.append((d, sits or [p for p, _ in pending] or last_sits,
                                      [clause] + [c for _, c in pending]))
            pending, last_doc = [], (list(by_doc)[-1], clause)
        elif sits:
            if last_doc:                      # "جوازي باقي عليه 3 شهور، كم رسوم التجديد؟" -> the last document
                pairs += [(last_doc[0], s, [last_doc[1], clause]) for s in sits]
            else:
                pending = [(s, clause) for s in sits]   # the document comes in a later clause
        last_sits = sits or last_sits
    for d, sits, clauses in open_docs:
        if any(p[0] == d for p in pairs):
            continue                          # the document got its situation elsewhere in the message
        pairs += [(d, s, clauses) for s in (sits or SITUATIONS)]   # only the document named -> all four
    for d, s, clauses in pairs:
        sid = f"{d}_{s}"
        if sid not in SERVICE_IDS:            # an added document may not have all four situations
            continue
        if sid not in r.services:
            r.services.append(sid)
        r.clauses.setdefault(sid, [])
        r.clauses[sid] += [c for c in clauses if c not in r.clauses[sid]]
    if not r.docs:
        r.unsure = r.unsure or "no_document"
    elif conflict and not COMPARE.search(text) and len(r.docs) == 1:
        r.unsure = r.unsure or "conflicting_situations"
    return r


# ---------------------------------------------------------------- LLM router
ROUTER_SYSTEM = """أنت مصنّف أسئلة لخدمات دائرة الأحوال المدنية والجوازات في الأردن. الخدمات المتاحة فقط:
جواز السفر، دفتر العائلة، البطاقة الذكية — ولكل منها: إصدار أول مرة، تجديد، بدل فاقد، بدل تالف.
معرّفات الخدمات: passport_first_issue, passport_renewal, passport_lost, passport_damaged,
family_book_first_issue, family_book_renewal, family_book_lost, family_book_damaged,
id_card_first_issue, id_card_renewal, id_card_lost, id_card_damaged.
افهم اللهجة الأردنية: باسبور=جواز سفر، هوية=بطاقة ذكية، دفتر عيلة=دفتر عائلة، ضاع/انسرق=فاقد،
خربان/انكسر/تمزق=تالف، خلص/انتهى/أجدد=تجديد، أطلع لأول مرة=إصدار أول مرة.
- اختر كل الخدمات التي يسأل عنها المستخدم (قد تكون أكثر من خدمة عند المقارنة أو القصة الطويلة).
- إذا كان السؤال عن خدمة غير هذه الخدمات فاجعل in_scope=false.
- إذا لم يحدد المستخدم الوثيقة ولا يمكن معرفتها من سياق المحادثة فاجعل needs_clarification=true.
- اكتب standalone_question: سؤال كامل واضح بالفصحى يتضمن الوثيقة والحالة من سياق المحادثة."""
_ADDED = [sid for sid in SERVICE_IDS if sid not in BUILTIN_SERVICES]
if _ADDED:                                    # services added with `python -m Agent.add_document`
    ROUTER_SYSTEM += ("\n- خدمات أخرى متاحة أيضاً: " + "، ".join(f"{NAMES[s]} ({s})" for s in _ADDED)
                      + ". هذه الخدمات ضمن النطاق.")

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "in_scope": {"type": "boolean"},
        "services": {"type": "array", "items": {"type": "string", "enum": list(SERVICE_IDS)}},
        "needs_clarification": {"type": "boolean"},
        "standalone_question": {"type": "string"},
    },
    "required": ["in_scope", "services", "needs_clarification", "standalone_question"],
}


def _history_text(history: list[tuple[str, str]]) -> str:
    if not history:
        return "لا يوجد"
    return "\n".join(f"المستخدم: {q}\nالمساعد: {a[:400]}" for q, a in history[-2:])


def llm_route(message: str, history: list[tuple[str, str]], model: str | None = None) -> dict:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"سياق المحادثة السابق:\n{_history_text(history)}\n\nسؤال المستخدم الحالي:\n{message}"}]
    out = ollama_http.chat(msgs, format=ROUTER_SCHEMA, kind="router", model=model, num_predict=200)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        log.warning("router returned invalid JSON: %r", out[:200])
        return {}
    data["services"] = [s for s in dict.fromkeys(data.get("services") or []) if s in SERVICE_IDS]
    return data


def route(message: str, history: list[tuple[str, str]] | None = None,
          state: tuple[list[str], list[str]] | None = None, model: str | None = None) -> Route:
    """Lexicon first. In a conversation, `state` = (documents, situations) of the chat so far: the message only
    replaces the slots it names ("وللهوية؟" keeps the situation, "وإذا تالف؟" keeps the document, "تاني مرة" after a
    loss keeps "lost"). The LLM router only decides in / out of scope when no document is known or a word from an
    uncovered service appears; a missing document is always asked back, never guessed."""
    history = history or []
    st_docs, st_sits = state or ([], [])
    r = lexicon(message)
    text = normalize_search(message)
    if history and (st_docs or st_sits) and r.unsure != "out_of_scope_word":
        sits = r.sits
        if "lost" in st_sits and COUNT_WORDS.search(text) and set(sits) <= {"first_issue"}:
            sits = []                         # "وإذا أول مرة؟" after a lost passport = the first loss
        docs = r.docs or st_docs
        sits = sits or st_sits
        if docs and sits and (not r.docs or not r.sits or sits != r.sits):
            r.services = [f"{d}_{s}" for d in docs for s in sits if f"{d}_{s}" in SERVICE_IDS]
            r.source, r.unsure = "state", ""
            r.docs, r.sits = docs, sits
            r.in_scope = bool(r.services)     # e.g. renewal of an added document that has no renewal page
            return r
        if docs and not sits:                 # a document but still no situation: all four, as without a chat
            r.unsure = ""
    if not r.unsure:
        if r.docs and not r.services:         # a known document, but no page for that situation
            r.in_scope = False
        return r
    data = llm_route(message, history, model)
    log.info("llm router (%s): %s", r.unsure, data)
    r.source = "llm"
    r.in_scope = bool(data.get("in_scope", True)) if data else r.unsure != "out_of_scope_word"
    if not r.in_scope:
        r.services = []
    elif not r.docs:                          # no document type detected -> ask (never guess one)
        r.services, r.needs_clarification = [], True
    else:                                     # an uncovered-service word next to a known document, still in scope
        r.source = "lexicon"
    return r


def clarification(r: Route) -> str:
    """Ask only for the part that is missing."""
    if len(r.docs) == 1 and not r.sits:
        return f"هل تقصد {SIT_ASK.format(doc=DOCS[r.docs[0]])}؟"
    if r.sits and not r.docs:
        return f"هل تقصد {docs_or()}؟"
    return f"هل تقصد {docs_or()}؟ وهل هي إصدار أول مرة أم تجديد أم بدل فاقد أم بدل تالف؟"
