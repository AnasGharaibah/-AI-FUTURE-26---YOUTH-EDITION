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
    "الرسوم": r"رسوم|رسم|تكلف|كلفه|تكاليف|سعر|اسعار|مبلغ|بكم|ثمن|حقه|بدفع|ادفع|يدفع|دفع|مصاري|فلوس|مصروف|دينار|غرامه|كفاله|تعهد|"
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


_OFFER = re.compile(normalize_search(r"^\s*(هل (ترغب|تريد|تود|تودين|ترغبين|لديك|لديكم|عندك|تحتاج|يمكنني|بإمكاني|هناك أي|هناك اي|من أسئلة)"
                                     r"|لا تتردد|إذا كان لديك|إذا كانت لديك|إذا احتجت|يسعدني|أتمنى)"))


_GREETING = re.compile(r"^\s*(مرحب[اًا]*|[أا]هل[اًا]*|السلام عليكم|شكر[اًا]* لسؤالك|يسعدني|يسرني)[^.!؟\n]{0,80}[.!،,]\s*")


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
_LOSS = re.compile(r"ضاع|ضايع|ضيع|فقد|فاقد|سرق|lost|stolen")
_ADULT = re.compile(r"بالغ|adult|grown")
_EXPIRED = re.compile(r"منتهي|انتهي|انتهت|خلص|خلصت|expired")
_NOT_EXPIRED = re.compile(r"(غير|مش|مو|ما|لم|ليس)\s*(منتهي|انتهي|انتهت|خلص)|ساري|باقي|بقي|متبقي|not expired|valid")
_NTH_TIME = [      # highest first: "ضاع للمرة الثانية بعد سنتين من أول مرة" is the second time
    ("الثالثة", re.compile(r"(المره|مره|للمره) (الثالثه|ثالثه)|ثالث مره|third time|3rd time")),
    ("الثانية", re.compile(r"(المره|مره|للمره) (الثانيه|ثانيه)|ثاني مره|مره كمان|مره ثانيه|second time|2nd time")),
    ("الأولى", re.compile(r"(المره|للمره) الاولي|اول مره|first time")),
]
_FULL_PAGES = re.compile(r"خلص\w*\s+(ال)?(ورق|صفح)\w*|(ورق|صفح)\w*\s+(خلص|مخلص|ممتلي|امتلا|مليان)\w*|"
                         r"(ممتلي|امتلا|مليان)\w*\s+(ال)?صفح\w*|full pages|pages are full|out of pages")
_INSIDE = re.compile(r"بالاردن|في الاردن|داخل الاردن|جوا البلد|داخل البلد|in jordan|inside jordan")
_NOT_MILITARY = re.compile(r"(مش|مو|ما|لست|ليس|ليست|غير)\s*(انا\s*)?(عسكري|بالجيش|جيش)|not (in the )?(military|army)")
_MILITARY = re.compile(r"عسكري|بالجيش|القوات المسلحه|امن عام|الدرك|دفاع مدني|حفظ السلام|military|army")
_DAYS = {"يوم": 1, "ايام": 1, "شهر": 30, "شهور": 30, "اشهر": 30, "سنه": 365, "سنين": 365, "سنوات": 365}


def _hints(question: str) -> list[tuple[str, str]]:
    """[(kind, hint)] for one message - see situation_hints."""
    q, hints = normalize_search(question), []
    m = _AGE.search(q)
    if m:
        age = int(m.group(1) or m.group(2))
        if 0 < age < 120:
            band = "أقل من 16 سنة" if age < 16 else ("16 سنة أو أكثر وأقل من 18" if age < 18 else "18 سنة أو أكثر")
            hints.append(("age", f"العمر المذكور {age} سنة، أي {band}."))
    elif _ADULT.search(q):
        hints.append(("age", "صاحب الطلب بالغ، أي 18 سنة أو أكثر."))
    m = _LEFT.search(q)
    if m:
        num, unit = m.group(1), m.group(2) or ""
        days = int(num) * _DAYS.get(unit, 30) if num.isdigit() else {"سنه": 365, "سنتين": 730, "شهر": 30, "شهرين": 60}[num]
        hints.append(("left", f"المدة المتبقية على انتهاء الوثيقة حوالي {days} يوماً، أي "
                              f"{'أقل من 180 يوماً' if days < 180 else 'أكثر من 180 يوماً'}."))
    m = _MARRIED.search(q)
    if m:
        days = int(m.group(1)) * _DAYS.get(m.group(2), 30) if m.group(1).isdigit() else \
            {"سنه": 365, "سنتين": 730, "شهر": 30, "شهرين": 60}[m.group(1)]
        hints.append(("married", f"مضى على عقد الزواج حوالي {days} يوماً، أي "
                                 f"{'أقل من 90 يوماً' if days < 90 else 'أكثر من 90 يوماً'}."))
    if _FULL_PAGES.search(q):
        hints.append(("pages", "صفحات جواز السفر ممتلئة."))
    if _EXPIRED.search(_FULL_PAGES.sub(" ", q)) and not _NOT_EXPIRED.search(q):   # "مخلص ورقه" = full, not expired
        hints.append(("expired", "الوثيقة منتهية الصلاحية."))
    if _NOT_MILITARY.search(q):
        hints.append(("military", "المستخدم ليس من العسكريين أو الأجهزة الأمنية."))
    elif _MILITARY.search(q):
        hints.append(("military", "المستخدم من العسكريين أو الأجهزة الأمنية أو الدفاع المدني."))
    count = next((n for n, rx in _NTH_TIME if rx.search(q)), None) if _LOSS.search(q) else None
    if count:
        hints.append(("loss", f"هذه المرة {count} لفقدان الوثيقة، أي ليست المرة الأولى."
                              if count != "الأولى" else "هذه المرة الأولى لفقدان الوثيقة."))
    if _ABROAD.search(q):
        hints.append(("place", "المستخدم خارج الأردن."))
    elif _INSIDE.search(q):
        hints.append(("place", "المستخدم داخل الأردن."))
    return hints


def situation_hints(question: str) -> list[str]:
    """Facts a 7B model gets wrong when it compares numbers itself: "عمري 17" -> "16 or older, under 18",
    "باقي عليه 3 شهور" -> "about 90 days: less than 180", "مخلص ورقه" -> pages full (not expired).
    Added below the question, never shown to the user."""
    return [h for _, h in _hints(question)]


def chat_hint_map(messages: list[str]) -> dict[str, str]:
    """{kind: hint} over the user's messages in this chat, newest first: one hint per kind, the latest wins
    ("عمري 15" earlier, "عمري 17" now -> 17)."""
    out = {}
    for msg in messages:
        for kind, h in _hints(msg):
            out.setdefault(kind, h)
    return out


def chat_hints(messages: list[str]) -> list[str]:
    return list(chat_hint_map(messages).values())


def fix_typos(text: str) -> str:
    """ة is only written at the end of a word: "جةاز" / "ةانا" / "ةرقه" are و typed on the ة key."""
    return re.sub(r"ة(?=[ء-ي])", "و", text)


def one_question(text: str) -> str:
    """The model sometimes asks a list of questions back; keep only the first one, at the end."""
    lines = text.split("\n")
    asks = [i for i, line in enumerate(lines) if line.strip().endswith(("؟", "?"))]
    if len(asks) < 2:
        return text
    first = re.sub(r"^\s*(\d+\s*[.)\-–]|[-*•])\s*", "", lines[asks[0]])
    rest = [line for i, line in enumerate(lines) if i not in asks]
    rest = [re.sub(r"(الأسئلة|الاسئلة) التالية", "السؤال التالي", line) for line in rest]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(rest).rstrip() + "\n" + first).strip()


# what a fee rule depends on: (pattern in the fee/notes text, hint kinds that already answer it, question part)
_FEE_FACTORS = (
    (r"16|السادسه عشر|18|الثامنه عشر", {"age"}, "كم عمرك"),
    (r"180|منتهي", {"expired", "left", "pages"}, "وهل الوثيقة منتهية أو باقٍ على انتهائها أقل من 180 يوماً"),
    (r"خارج البلاد|الخارج", {"place"}, "وهل أنت داخل الأردن أم خارجه"),
    (r"المره|لاي مره", {"loss"}, "وهل هذه أول مرة تفقد فيها الوثيقة"),
)


def ask_missing(text: str, snips: list[dict], hints_kinds: set[str]) -> str:
    """An answer that lists several amounts without asking anything gets ONE question about the facts the fee rules
    depend on that the user has not given yet ("كم لازم أدفع؟" -> the cases + "كم عمرك، وهل الوثيقة منتهية...؟")."""
    if text.rstrip().endswith(("؟", "?")) or len(set(re.findall(r"(\d+)\s*(?:دينار|دنانير)", text))) < 3:
        return text
    fees = normalize_search(" ".join(s["text"] for s in snips if s["field"] in ("الرسوم", "ملاحظات")))
    parts = [ask for rx, known, ask in _FEE_FACTORS if re.search(rx, fees) and not known & hints_kinds]
    if not parts:
        return text
    question = "، ".join(parts).removeprefix("و")
    return f"{text.rstrip()}\n\nحتى أحدد المبلغ الذي ينطبق عليك: {question}؟"


def drop_dangling(text: str) -> str:
    """After a sentence was dropped: no last line that only announces something ("المبلغ المطلوب هو:")."""
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or lines[-1].rstrip().endswith((":", "："))):
        lines.pop()
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- LLM
def _messages(question: str, standalone: str, snips: list[dict], reminder: str = "", convo: str = "",
              hints: list[str] | None = None) -> list[dict]:
    user = f"المصادر:\n{sources_block(snips)}\n\n{convo}سؤال المستخدم (كما كتبه): {question}\nالسؤال الموضح: {standalone}"
    hints = situation_hints(question) if hints is None else hints
    if hints:
        user += "\nحالة المستخدم (محسوبة من كلامه): " + " ".join(hints)
    return [{"role": "system", "content": ANSWER_SYSTEM[PROMPT_LANG]}, {"role": "user", "content": user + reminder}]


def _generate(question, standalone, snips, model, reminder="", convo="", hints=None) -> str:
    text = ollama_http.chat(_messages(question, standalone, snips, reminder, convo, hints), kind="answer", model=model)
    text = re.sub(r"\*\*|__|^#+\s*", "", normalize_display(text), flags=re.M)      # no markdown emphasis/headers
    return one_question(strip_offers(text.strip()))


# ---------------------------------------------------------------- one chat
class Chat:
    """One user conversation. Messages must be asked in order; a new Chat forgets everything."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.history: list[tuple[str, str]] = []           # (question, answer)
        self.prev_services: list[str] = []

    def ask(self, question: str) -> dict:
        t0 = time.time()
        q = fix_typos(normalize_display(question))
        r = router.route(q, self.history[-HISTORY_TURNS:], self.prev_services, self.model)
        dbg = {"route": r.services, "route_source": r.source, "unsure": r.unsure, "standalone": r.standalone,
               "flags": [], "cited": False}
        if q != normalize_display(question):
            dbg["flags"].append("typos_fixed")
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

    def _earlier(self, services: list[str]) -> list[tuple[str, str]]:
        """Earlier turns of this chat about the same document(s) - the user's situation lives there
        ("أنا بالأردن ومش عسكري" answered to the bot's question one turn before "كم المبلغ؟")."""
        docs = {router.doc_of(s) for s in services}
        if not self.history or not docs & {router.doc_of(s) for s in self.prev_services}:
            return []
        return self.history[-3:]

    def _answer(self, q: str, standalone: str, services: list[str], dbg: dict) -> tuple[str, list[str]]:
        earlier = self._earlier(services)
        user_msgs = [q] + [eq for eq, _ in reversed(earlier)]           # newest first
        topic = q                         # "وإذا كان تالف؟" / "كم المبلغ؟" -> the fields of the earlier questions too
        if wanted_fields(q) is None or earlier:
            topic = " ".join(user_msgs)
        snips = build_snippets(services, topic if standalone == q else f"{topic} {standalone}")
        dbg["snippets"] = len(snips)
        hint_map = chat_hint_map(user_msgs)
        hints = list(hint_map.values())
        convo = ""
        if earlier:
            convo = "ما قاله المستخدم سابقاً في هذه المحادثة:\n" + "\n".join(f"- {eq}" for eq, _ in earlier) + "\n"
            last_answer = earlier[-1][1]
            if last_answer.rstrip().endswith(("؟", "?")):            # the user is answering the bot's question
                convo += f"آخر ما سأله المساعد للمستخدم: {last_answer[-300:]}\n"
            convo += "\n"
            dbg["flags"].append(f"memory:{len(earlier)}")

        def generate(reminder=""):
            return _generate(q, standalone, snips, self.model, reminder, convo, hints)

        text = generate()
        cites = cited_numbers(text, len(snips))
        if not cites and not (is_not_found(text) or is_clarifying(text)):
            dbg["flags"].append("retry_no_citation")
            text = generate(CITE_REMINDER)
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

        # numbers must be in the cited snippets (or what the user said in this chat)
        user_facts = "\n".join(user_msgs + hints)

        def evidence(ids):
            return "\n".join(snips[i - 1]["text"] for i in ids) + "\n" + user_facts
        bad = check_numbers(text, evidence(cites))
        if bad:
            dbg["flags"].append("numbers_retry:" + ",".join(bad))
            retry = generate(NUMBER_REMINDER.format(bad="، ".join(bad)))
            r_cites = cited_numbers(retry, len(snips))
            if r_cites and not is_not_found(retry):
                still = check_numbers(retry, evidence(r_cites))
                if len(still) < len(bad):
                    text, cites, bad = retry, r_cites, still
            if bad:
                kept = drop_dangling(drop_number_sentences(text, bad))
                dbg["flags"].append("numbers_dropped:" + ",".join(bad))
                text = kept or text
                cites = cited_numbers(text, len(snips)) or cites
        contexts = list(dict.fromkeys(context_text(snips[i - 1]) for i in cites))
        text = ask_missing(text, [snips[i - 1] for i in cites], set(hint_map))
        answer = text if KEEP_CITATIONS else strip_citations(text)
        return answer, contexts
