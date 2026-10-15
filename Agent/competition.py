"""Competition pipeline (AI Future 26 brief): one chat = one Chat(); every message -> {"answer", "contexts"}.

    route (lexicon, LLM router only when unsure) -> fixed reply (out of scope / unclear)
      or numbered snippets of the routed services -> command-r7b-arabic answers in plain Arabic with [n] citations
      -> guards: no citation = one retry, then overlap fallback; numbers not in the cited snippets = one retry,
         then the sentence is dropped -> contexts = the cited snippets, [n] stripped from the answer

    from Agent.competition import Chat
    chat = Chat()
    chat.ask("شو الأوراق المطلوبة لتجديد دفتر العائلة؟")   # {"answer", "contexts", "debug"}
    chat.ask("وبقدر أقدّم الطلب من السفارة؟")              # follow-up: the router sees the last 2 Q/A
"""
import logging
import os
import re
import time

from DB.arabic import check_numbers, drop_number_sentences, is_arabic, normalize_display, normalize_search, tokens

from . import ollama_http, router
from .services import FIELDS, context_text, snippets, sources_block

log = logging.getLogger("yoth.competition")

KEEP_CITATIONS = os.getenv("YOTH_KEEP_CITATIONS", "0") == "1"     # keep "[n]" in the answer text
PROMPT_LANG = os.getenv("YOTH_PROMPT_LANG", "ar")                  # instructions of the ANSWER prompt: ar | en
HISTORY_TURNS = 2

OUT_OF_SCOPE = ("عذراً، لا تتوفر لدي معلومات عن هذه الخدمة. أستطيع المساعدة في خدمات جواز السفر ودفتر العائلة "
                "والبطاقة الذكية (إصدار أول مرة، تجديد، بدل فاقد، بدل تالف).")
NOT_FOUND = "لا تتوفر لدي هذه المعلومة في الوثائق المتاحة."
NOT_FOUND_EN = "This information is not available in the documents I have."

ANSWER_SYSTEM = {
    "ar": """أنت مساعد رسمي يجيب عن أسئلة المواطنين حول خدمات جواز السفر ودفتر العائلة والبطاقة الذكية في الأردن.
اعتمد فقط على المصادر المرقّمة المرفقة، ولا تستخدم معلوماتك السابقة أبداً حتى لو كنت تعرف الجواب.
القواعد:
1. أجب عن السؤال المطروح فقط، باختصار ووضوح، بالعربية الفصحى البسيطة (وبالإنجليزية إذا كان السؤال بالإنجليزية).
2. اذكر رقم المصدر بعد كل معلومة هكذا [2].
3. انسخ المبالغ والمدد والأعداد كما وردت في المصادر حرفياً، ولا تحسب ولا تقدّر.
4. حدّد حالة المستخدم من كلامه (العمر، هل الوثيقة منتهية، داخل الأردن أو خارجه، عدد مرات الفقدان...) وطبّق
   القاعدة المناسبة لها فقط. إذا كانت القاعدة تعتمد على معلومة لم يذكرها فاذكر الحالات باختصار ثم اسأله سؤالاً واحداً.
5. اقرأ الملاحظات دائماً: الرسوم الإضافية والمهل والاستثناءات غالباً موجودة فيها.
6. إذا كان في سؤال المستخدم افتراض خاطئ فصحّحه بوضوح من المصادر.
7. إذا تضمّن السؤال عدة أجزاء فأجب عن كل جزء.
8. إذا لم تجد الجواب في المصادر فقل حرفياً: "لا تتوفر لدي هذه المعلومة في الوثائق المتاحة."
9. لا تنسخ المستند كاملاً، ولا تضف معلومات لم يُسأل عنها.""",
    "en": """You are an official assistant answering citizens' questions about Jordan's passport, family book and smart ID card services.
Rely ONLY on the numbered sources provided. Never use prior knowledge, even if you know the answer.
Rules:
1. Answer only the question asked, briefly and clearly, in simple Modern Standard Arabic (in English if the question is in English).
2. Cite the source number after every fact, like this: [2].
3. Copy amounts, durations and counts exactly as written in the sources. Do not calculate or estimate.
4. Work out the user's situation from their words (age, whether the document has expired, inside or outside Jordan, how many times it was lost...) and apply only the matching rule. If the rule depends on information they did not give, list the cases briefly and then ask them one question.
5. Always read the notes (ملاحظات): extra fees, deadlines and exceptions are often there.
6. If the question contains a wrong assumption, correct it clearly from the sources.
7. If the question has several parts, answer every part.
8. If the answer is not in the sources, say exactly: "لا تتوفر لدي هذه المعلومة في الوثائق المتاحة."
9. Do not copy the whole document and do not add information that was not asked for.""",
}
CITE_REMINDER = "\n\nيجب ذكر رقم المصدر بعد كل معلومة."
NUMBER_REMINDER = ("\n\nتنبيه: الأرقام التالية غير موجودة في المصادر التي ذكرتها: {bad}. استخدم فقط الأرقام الموجودة "
                   "حرفياً في المصادر، ولا تحسب ولا تقدّر، واذكر رقم المصدر بعد كل معلومة.")

# which fields a question needs (normalize_search patterns); notes are always sent (rule 5)
INTENTS = {
    "الرسوم": r"رسوم|رسم|تكلف|كلفه|تكاليف|سعر|اسعار|بدفع|ادفع|يدفع|دفع|مصاري|فلوس|مصروف|دينار|غرامه|كفاله|تعهد|"
              r"اغلي|ارخص|مجان|قديش|شو بعمل|شو اعمل|ماذا افعل|fee|cost|price|pay|fine|cheap|expensive|what do i do",
    "الوثائق المطلوبة": r"وثائق|اوراق|ورق|مستند|مطلوب|بلزم|يلزم|لازم|احضر|اجيب|بجيب|نجيب|صور|صوره|ثبوت|"
                         r"شو بعمل|شو اعمل|بعمل ايش|ماذا افعل|document|paper|bring|photo|what do i do",
    "الشروط": r"شروط|شرط|مين بقدر|من يحق|يحق|بحق|بقدر|يقدر|بتقدر|موافق|حضور|شخصي|ولي|قاصر|زوج|عمر|عمري|سن|سنه|"
              r"سنين|توكيل|تفويض|condition|eligib|allowed|can i|who can|consent",
    "أماكن تقديم الخدمة": r"وين|اين|مكان|اماكن|سفار|قنصل|برا|خارج|مكتب|مركز|where|embassy|abroad|office",
    "الزمن المستغرق": r"وقت|مده|يستغرق|بياخد|بتاخد|بيطول|دقيق|دقايق|ساعه|متي|اديش|how long|time|minutes",
    "الإجراءات": r"خطوات|اجراء|كيف|مراحل|بصم|رساله|دور|طباع|استلم|تسليم|شو بعمل|شو اعمل|بعمل|steps|procedure|how do",
    "شركاء الخدمة": r"شركاء|جهات|جهه|وزاره|مخابرات|قاضي|تجنيد|تعبئه|امني|partner|agency",
    "الفئة المستفيدة": r"مستفيد|اجنبي|غير اردني|مقدسي|اردني|foreign|citizen",
}
OPTIONAL_WHEN_LARGE = ("الإجراءات", "شركاء الخدمة", "الفئة المستفيدة")
MAX_SOURCE_CHARS = 9000            # ~4.7k tokens: drop optional fields of multi-service prompts above this

_CITE = re.compile(r"\[\s*(\d+(?:\s*[,،\-–]\s*\d+)*)\s*\]")
_SENTENCE = re.compile(r"(?<=[.!?؟])\s+")


# ---------------------------------------------------------------- snippets
def wanted_fields(question: str) -> list[str] | None:
    """Fields the question needs (+ notes), or None = all fields."""
    q = normalize_search(question)
    hit = [f for f, rx in INTENTS.items() if re.search(rx, q)]
    if not hit:
        return None
    if "الرسوم" in hit or "الشروط" in hit:      # fees and conditions depend on each other (age, abroad, consent)
        hit += ["الرسوم", "الشروط"]
    return [f for f in FIELDS if f in hit or f == "ملاحظات"]


def build_snippets(service_ids: list[str], question: str) -> list[dict]:
    fields = wanted_fields(question)
    snips = snippets(service_ids, fields)
    if sum(len(s["text"]) for s in snips) > MAX_SOURCE_CHARS:
        keep = [f for f in (fields or FIELDS) if f not in OPTIONAL_WHEN_LARGE] or None
        snips = snippets(service_ids, keep)
    return snips


# ---------------------------------------------------------------- citations
def cited_numbers(text: str, n: int) -> list[int]:
    """[2], [1, 3], [2-4], [1][2] -> source numbers in citation order (valid ones only, deduplicated)."""
    out = []
    for m in _CITE.finditer(text):
        for part in re.split(r"\s*[,،]\s*", m.group(1)):
            ends = [int(x) for x in re.split(r"\s*[\-–]\s*", part) if x]
            rng = range(ends[0], ends[1] + 1) if len(ends) == 2 and 0 <= ends[1] - ends[0] < 20 else ends[:1]
            out += [i for i in rng if 1 <= i <= n and i not in out]
    return out


def strip_citations(text: str) -> str:
    text = _CITE.sub("", text)
    text = re.sub(r"[ \t]+([.,،؛:؟?!)])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def is_not_found(text: str) -> bool:
    t = normalize_search(text)
    return any(normalize_search(k) in t for k in (NOT_FOUND, "لا تتوفر لدي هذه المعلومه", "لم اجد هذه المعلومه",
                                                  NOT_FOUND_EN, "not available in the documents"))


def is_clarifying(text: str) -> bool:
    """A short answer that only asks the user something back."""
    return len(text) < 260 and text.rstrip().endswith(("؟", "?")) and not re.search(r"\d", text)


_OFFER = re.compile(normalize_search(r"^\s*(هل (ترغب|تريد|تود|تودين|ترغبين|لديك|لديكم|عندك|تحتاج|يمكنني|بإمكاني)"
                                     r"|لا تتردد|إذا كان لديك|إذا كانت لديك|إذا احتجت|يسعدني|أتمنى)"))


_GREETING = re.compile(r"^\s*(مرحب[اًا]*|[أا]هل[اًا]*|السلام عليكم|شكر[اًا]* لسؤالك)[^.!؟\n]{0,40}[.!،,]\s*")


def strip_offers(text: str) -> str:
    """Drop the opening "مرحباً بك!" and the closing "هل ترغب بمعرفة المزيد؟" / "لا تتردد في السؤال" lines."""
    m = _GREETING.match(text)
    if m and m.end() < len(text):
        text = text[m.end():].lstrip()
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or _OFFER.search(normalize_search(_CITE.sub("", lines[-1])))):
        lines.pop()
    if lines:                                           # an offer at the end of the last line
        sents = _SENTENCE.split(lines[-1])
        while len(sents) > 1 and _OFFER.search(normalize_search(sents[-1])):
            sents.pop()
        lines[-1] = " ".join(sents)
    return "\n".join(lines).strip() or text


def strip_not_found(text: str) -> str:
    kept = [s for s in _SENTENCE.split(text) if not is_not_found(s)]
    return " ".join(kept).strip()


def overlap_fallback(answer: str, snips: list[dict], limit: int = 4) -> list[int]:
    """No citation at all: the snippets whose numbers / content words occur in the answer (1-based, best first)."""
    a_nums = set(re.findall(r"\d+", normalize_display(answer)))
    a_toks = set(tokens(answer))
    scored = []
    for i, s in enumerate(snips, 1):
        s_toks = set(tokens(s["text"]))
        nums = a_nums & set(re.findall(r"\d+", s["text"]))
        shared = a_toks & s_toks
        score = 3 * len(nums) + len(shared) / max(4, len(s_toks)) * 10
        if nums or len(shared) >= 3:
            scored.append((score, i))
    return [i for _, i in sorted(scored, reverse=True)[:limit]]


# ---------------------------------------------------------------- user situation (computed, not left to the 7B model)
_AGE = re.compile(r"(?:عمر\w*|age)\s*(?:هو\s*|is\s*)?(\d{1,2})|(\d{1,2})\s*(?:سنه|سنين|سنوات|عام|years?)\b")
_LEFT = re.compile(r"(?:باقي|بقي|متبقي|ضايل|ظايل|left)\D{0,15}?(\d{1,3}|سنتين|سنه|شهرين|شهر)\s*(يوم|ايام|شهر|شهور|اشهر|سنه|سنين|سنوات)?")
_ABROAD = re.compile(r"\bبرا\b|بالخارج|في الخارج|خارج (الاردن|البلاد|المملكه)|مغترب|مقيم (ب|في)\s*(?!الاردن)\w+|abroad")
_MARRIED = re.compile(r"(?:تزوج\w*|زواج\w*|عقد\w*|كتب\w* الكتاب)\D{0,12}?\bمن\s*(\d{1,3}|سنتين|سنه|شهرين|شهر)\s*"
                      r"(يوم|ايام|شهر|شهور|اشهر|سنه|سنين|سنوات)?")
_DAYS = {"يوم": 1, "ايام": 1, "شهر": 30, "شهور": 30, "اشهر": 30, "سنه": 365, "سنين": 365, "سنوات": 365}


def situation_hints(question: str) -> list[str]:
    """Facts a 7B model gets wrong when it compares numbers itself: "عمري 17" -> "16 or older, under 18",
    "باقي عليه 3 شهور" -> "about 90 days: less than 180". Added below the question, never shown to the user."""
    q, hints = normalize_search(question), []
    m = _AGE.search(q)
    if m:
        age = int(m.group(1) or m.group(2))
        if 0 < age < 120:
            band = "أقل من 16 سنة" if age < 16 else ("16 سنة أو أكثر وأقل من 18" if age < 18 else "18 سنة أو أكثر")
            hints.append(f"العمر المذكور {age} سنة، أي {band}.")
    m = _LEFT.search(q)
    if m:
        num, unit = m.group(1), m.group(2) or ""
        if num.isdigit():
            days = int(num) * _DAYS.get(unit, 30)
        else:
            days = {"سنه": 365, "سنتين": 730, "شهر": 30, "شهرين": 60}[num]
        hints.append(f"المدة المتبقية على انتهاء الوثيقة حوالي {days} يوماً، أي "
                     f"{'أقل من 180 يوماً' if days < 180 else 'أكثر من 180 يوماً'}.")
    m = _MARRIED.search(q)
    if m:
        days = int(m.group(1)) * _DAYS.get(m.group(2), 30) if m.group(1).isdigit() else \
            {"سنه": 365, "سنتين": 730, "شهر": 30, "شهرين": 60}[m.group(1)]
        hints.append(f"مضى على عقد الزواج حوالي {days} يوماً، أي "
                     f"{'أقل من 90 يوماً' if days < 90 else 'أكثر من 90 يوماً'}.")
    if _ABROAD.search(q):
        hints.append("المستخدم خارج الأردن.")
    return hints


# ---------------------------------------------------------------- LLM
def _messages(question: str, standalone: str, snips: list[dict], reminder: str = "") -> list[dict]:
    user = f"المصادر:\n{sources_block(snips)}\n\nسؤال المستخدم (كما كتبه): {question}\nالسؤال الموضح: {standalone}"
    hints = situation_hints(question)
    if hints:
        user += "\nحالة المستخدم (محسوبة من كلامه): " + " ".join(hints)
    return [{"role": "system", "content": ANSWER_SYSTEM[PROMPT_LANG]}, {"role": "user", "content": user + reminder}]


def _generate(question, standalone, snips, model, reminder="") -> str:
    text = ollama_http.chat(_messages(question, standalone, snips, reminder), kind="answer", model=model)
    text = re.sub(r"\*\*|__|^#+\s*", "", normalize_display(text), flags=re.M)      # no markdown emphasis/headers
    return strip_offers(text.strip())


# ---------------------------------------------------------------- one chat
class Chat:
    """One user conversation. Messages must be asked in order; a new Chat forgets everything."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.history: list[tuple[str, str]] = []           # (question, answer)
        self.prev_services: list[str] = []

    def ask(self, question: str) -> dict:
        t0 = time.time()
        q = normalize_display(question)
        r = router.route(q, self.history[-HISTORY_TURNS:], self.prev_services, self.model)
        dbg = {"route": r.services, "route_source": r.source, "unsure": r.unsure, "standalone": r.standalone,
               "flags": [], "cited": False}
        if not r.in_scope:
            ans, contexts = OUT_OF_SCOPE, []
            dbg["flags"].append("out_of_scope")
        elif r.needs_clarification or not r.services:
            ans, contexts = router.clarification(r), []
            dbg["flags"].append("clarify")
        else:
            ans, contexts = self._answer(q, r.standalone or q, r.services, dbg)
            self.prev_services = r.services
        self.history.append((q, ans))
        dbg["seconds"] = round(time.time() - t0, 2)
        return {"answer": ans, "contexts": contexts, "debug": dbg}

    def _answer(self, q: str, standalone: str, services: list[str], dbg: dict) -> tuple[str, list[str]]:
        snips = build_snippets(services, q if standalone == q else f"{q} {standalone}")
        dbg["snippets"] = len(snips)
        text = _generate(q, standalone, snips, self.model)
        cites = cited_numbers(text, len(snips))
        if not cites and not (is_not_found(text) or is_clarifying(text)):
            dbg["flags"].append("retry_no_citation")
            text = _generate(q, standalone, snips, self.model, CITE_REMINDER)
            cites = cited_numbers(text, len(snips))
        # citation markers that point nowhere are dropped
        text = _CITE.sub(lambda m: m.group(0) if cited_numbers(m.group(0), len(snips)) else "", text)
        if cites and is_not_found(text):                  # a cited answer with a stray "not found" sentence
            text = strip_not_found(text) or text
            dbg["flags"].append("stripped_not_found")
        if is_not_found(text) and not cites:
            return (NOT_FOUND if is_arabic(q) else NOT_FOUND_EN), []
        if not cites and is_clarifying(text):
            return text, []
        dbg["cited"] = bool(cites)
        if not cites:
            cites = overlap_fallback(text, snips)
            dbg["flags"].append("overlap_fallback")

        # numbers must be in the cited snippets (or the user's own message)
        user_facts = q + "\n" + " ".join(situation_hints(q))

        def evidence(ids):
            return "\n".join(snips[i - 1]["text"] for i in ids) + "\n" + user_facts
        bad = check_numbers(text, evidence(cites))
        if bad:
            dbg["flags"].append("numbers_retry:" + ",".join(bad))
            retry = _generate(q, standalone, snips, self.model, NUMBER_REMINDER.format(bad="، ".join(bad)))
            r_cites = cited_numbers(retry, len(snips))
            if r_cites and not is_not_found(retry):
                still = check_numbers(retry, evidence(r_cites))
                if len(still) < len(bad):
                    text, cites, bad = retry, r_cites, still
            if bad:
                kept = drop_number_sentences(text, bad)
                dbg["flags"].append("numbers_dropped:" + ",".join(bad))
                text = kept or text
                cites = cited_numbers(text, len(snips)) or cites
        contexts = list(dict.fromkeys(context_text(snips[i - 1]) for i in cites))
        answer = text if KEEP_CITATIONS else strip_citations(text)
        return answer, contexts
