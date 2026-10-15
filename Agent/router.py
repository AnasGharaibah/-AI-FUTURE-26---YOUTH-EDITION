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

from DB.arabic import _DIACRITICS, _SEARCH_MAP, normalize_search

from . import ollama_http
from .services import DOCS, SERVICE_IDS, SITUATIONS

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
                       "احصل", "الحصول", "فتح دفتر", "فتحت دفتر", "افتح دفتر", "فتح سجل", "فتح قيد", "issue", "get a"),
}
COMPARE = _rx("أغلى", "اغلى", "أرخص", "ارخص", "الفرق", r"\bفرق\b", "مقارنة", "أيهما", "ايهما", r"\bأيها\b",
              r"\bولا\b", r"\bوال\b", r"\bأم\b", r"\bأو\b", "compare", "difference", "cheaper", "expensive",
              r"\bor\b", r"\bvs\b")
# services that sound related but are not in the data -> let the LLM router decide (usually in_scope=false)
OUT_OF_SCOPE = _rx("رخصة", "سواقة", "سياقة", "قيادة", "مركبة", "سيارة", "شهادة ميلاد", "شهادة الميلاد",
                   "شهادة جامعية", "الشهادة الجامعية", "تصديق", "فيزا", "تأشيرة", "عدم محكومية", "تذكرة", "طيران",
                   "جواز دبلوماسي", "جواز خاص", "جواز مؤقت", "جواز سفر مؤقت", "جواز السفر المؤقت", "وثيقة سفر",
                   "تجنيس", "ضريبة", "تصريح عمل", "licen", "visa")
CLAUSE_SPLIT = re.compile(r"[،,.؟?!؛;:\n]|\s(?:وكمان|كمان|وكذلك|وايضا|بالاضافه|اما|and also)\s")

SIT_ASK = "إصدار {doc} لأول مرة أم تجديده أم بدل فاقد أم بدل تالف"


def doc_of(service_id: str) -> str:
    return next(d for d in DOCS if service_id.startswith(d + "_"))


@dataclass
class Route:
    services: list[str] = field(default_factory=list)
    in_scope: bool = True
    needs_clarification: bool = False
    standalone: str = ""
    source: str = "lexicon"            # lexicon | llm | lexicon-fallback
    docs: list[str] = field(default_factory=list)          # documents named in the message (lexicon)
    sits: list[str] = field(default_factory=list)          # situations named in the message (lexicon)
    unsure: str = ""                   # why the lexicon was unsure ("" = sure)


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
    """Rule-based route. route.unsure != "" -> ask the LLM router."""
    text = normalize_search(message)
    r = Route(standalone=message)
    if OUT_OF_SCOPE.search(text):
        r.unsure = "out_of_scope_word"
    pairs, open_docs, pending, last_sits, last_doc, conflict = [], [], [], [], None, False
    for clause in filter(str.strip, CLAUSE_SPLIT.split(text)):
        by_doc, sits, clash = _clause(clause)
        r.docs += [d for d in by_doc if d not in r.docs]
        r.sits += [s for s in sits if s not in r.sits]
        conflict |= clash
        if len(by_doc) > 1 and all(any(p[0] == d for p in pairs) for d in by_doc):
            continue                          # "كم رسوم جواز ابني وكم رسوم بطاقة زوجتي؟" restates earlier clauses
        if by_doc:
            for d, ds in by_doc.items():
                if ds:
                    pairs += [(d, s) for s in ds]
                else:                         # "جوازي ضاع، والهوية كمان" -> the clause's / previous situations
                    open_docs.append((d, sits or pending or last_sits))
            pending, last_doc = [], list(by_doc)[-1]
        elif sits:
            if last_doc:                      # "جوازي باقي عليه 3 شهور، كم رسوم التجديد؟" -> the last document
                pairs += [(last_doc, s) for s in sits]
            else:
                pending = sits                # the document comes in a later clause
        last_sits = sits or last_sits
    for d, sits in open_docs:
        if any(p[0] == d for p in pairs):
            continue                          # the document got its situation elsewhere in the message
        pairs += [(d, s) for s in (sits or SITUATIONS)]   # only the document named -> all four situations
    r.services = list(dict.fromkeys(f"{d}_{s}" for d, s in pairs))
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


def route(message: str, history: list[tuple[str, str]] | None = None, prev_services: list[str] | None = None,
          model: str | None = None) -> Route:
    """Lexicon first; the LLM router when the lexicon is unsure or the message continues a conversation."""
    history = history or []
    r = lexicon(message)
    if history and not r.docs:
        r.unsure = r.unsure or "follow_up"
    if not r.unsure:
        return r
    # what the lexicon would do on its own (used when the LLM's answer is unusable)
    fallback = list(r.services)
    if not fallback and prev_services and history:
        prev_docs = list(dict.fromkeys(doc_of(s) for s in prev_services))
        fallback = [f"{d}_{s}" for d in prev_docs for s in r.sits] if r.sits else list(prev_services)
    data = llm_route(message, history, model)
    log.info("llm router (%s): %s", r.unsure, data)
    if not data:
        r.source, r.services = "lexicon-fallback", fallback
        r.needs_clarification = not fallback and r.unsure != "out_of_scope_word"
        r.in_scope = bool(fallback) or r.needs_clarification
        return r
    r.source = "llm"
    r.standalone = (data.get("standalone_question") or "").strip() or message
    r.in_scope = bool(data.get("in_scope", True))
    r.needs_clarification = bool(data.get("needs_clarification")) and r.in_scope
    r.services = data["services"] if r.in_scope and not r.needs_clarification else []
    if r.in_scope and not r.needs_clarification and not r.services:
        if fallback:
            r.services, r.source = fallback, "lexicon-fallback"
        else:
            r.needs_clarification = True
    if r.needs_clarification and fallback and history:      # a follow-up we can resolve from the conversation
        r.services, r.needs_clarification, r.source = fallback, False, "lexicon-fallback"
    if not history and not r.docs and len({doc_of(s) for s in r.services}) > 1:
        r.services, r.needs_clarification = [], True       # "بدي أجدد" -> the LLM guessed every document: ask
    if not r.in_scope and r.docs and r.sits and r.unsure != "out_of_scope_word":
        r.in_scope, r.services, r.source = True, fallback, "lexicon-fallback"   # the lexicon clearly saw a service
    return r


def clarification(r: Route) -> str:
    """Ask only for the part that is missing."""
    if len(r.docs) == 1 and not r.sits:
        return f"هل تقصد {SIT_ASK.format(doc=DOCS[r.docs[0]])}؟"
    if r.sits and not r.docs:
        return "هل تقصد جواز السفر أم دفتر العائلة أم البطاقة الذكية؟"
    return ("هل تقصد جواز السفر أم دفتر العائلة أم البطاقة الذكية؟ "
            "وهل هي إصدار أول مرة أم تجديد أم بدل فاقد أم بدل تالف؟")
