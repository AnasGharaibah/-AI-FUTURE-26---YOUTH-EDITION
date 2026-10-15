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

from . import fees, ollama_http, router, situation
from .services import BUILTIN_DOCS, DOCS, FIELDS, NAMES, context_text, docs_and, snippets, sources_block

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
_BUILTIN_DOCS_AND = "جواز السفر ودفتر العائلة والبطاقة الذكية"
if docs_and() != _BUILTIN_DOCS_AND:            # services added with `python -m Agent.add_document`
    OUT_OF_SCOPE = OUT_OF_SCOPE.replace("جواز السفر ودفتر العائلة والبطاقة الذكية", docs_and())
    ANSWER_SYSTEM["ar"] = ANSWER_SYSTEM["ar"].replace(_BUILTIN_DOCS_AND, docs_and())
    ANSWER_SYSTEM["en"] = ANSWER_SYSTEM["en"].replace(
        "smart ID card services", "smart ID card services, and also: "
        + "، ".join(n for d, n in DOCS.items() if d not in BUILTIN_DOCS))
CITE_REMINDER = "\n\nيجب ذكر رقم المصدر بعد كل معلومة."
NUMBER_REMINDER = ("\n\nتنبيه: الأرقام التالية غير موجودة في المصادر التي ذكرتها: {bad}. استخدم فقط الأرقام الموجودة "
                   "حرفياً في المصادر، ولا تحسب ولا تقدّر، واذكر رقم المصدر بعد كل معلومة.")

# which fields a question needs (normalize_search patterns); notes are always sent (rule 5)
INTENTS = {
    "الرسوم": r"رسوم|رسم|تكلف|كلفه|تكاليف|سعر|اسعار|مبلغ|بكم|ثمن|حقه|بدفع|ادفع|يدفع|دفع|مصاري|فلوس|مصروف|دينار|غرامه|كفاله|تعهد|"
              r"اغلي|ارخص|مجان|قديش|شو بعمل|شو اعمل|ماذا افعل|fee|cost|price|pay|fine|cheap|expensive|what do i do",
    "الوثائق المطلوبة": r"وثائق|اوراق|ورق|مستند|مطلوب|بلزم|يلزم|لازم|احضر|اجيب|بجيب|نجيب|صور|صوره|ثبوت|تاخذ|توخذ|"
                         r"ياخذ|اخذ معي|معها|معه|ماخد|"
                         r"شو بعمل|شو اعمل|بعمل ايش|ماذا افعل|document|paper|bring|photo|what do i do",
    "الشروط": r"شروط|شرط|مين بقدر|من يحق|يحق|بحق|بقدر|يقدر|بتقدر|موافق|حضور|شخصي|ولي|قاصر|زوج|عمر|عمري|سن|سنه|"
              r"سنين|توكيل|تفويض|condition|eligib|allowed|can i|who can|consent",
    "أماكن تقديم الخدمة": r"وين|اين|مكان|اماكن|سفار|قنصل|برا|خارج|مكتب|مركز|where|embassy|abroad|office",
    "الزمن المستغرق": r"وقت|مده|يستغرق|بياخد|بتاخد|بيطول|دقيق|دقايق|ساعه|متي|اديش|اسرع|ابطا|سرعه|بسرعه|"
                       r"نفس اليوم|how long|time|minutes|faster|quicker",
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


def renumber_citations(text: str, cites: list[int]) -> str:
    """[7] [2, 9] -> [1] [2][3] following the order of `cites`, so source n of a UI answer is cites[n - 1]."""
    new = {old: i for i, old in enumerate(cites, 1)}
    return _CITE.sub(lambda m: "".join(f"[{new[i]}]" for i in cited_numbers(m.group(0), 10 ** 6) if i in new), text)


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


# ---------------------------------------------------------------- user situation (Agent/situation.py)
def situation_hints(question: str) -> list[str]:
    """Facts a 7B model gets wrong when it compares numbers itself ("عمري 17" -> "16 or older, under 18")."""
    return [h for _, h in situation.hints(situation.facts(question))]


def chat_facts(messages: list[str], loss_context: bool = False) -> situation.Facts:
    """Facts over the user's messages (newest first, the newest statement wins). In a chat about a lost document
    "تاني مرة" counts losses even without a loss word in that message."""
    loss_context = loss_context or any(situation._LOSS.search(normalize_search(m)) for m in messages)
    return situation.merge([situation.facts(m, loss_context) for m in messages])


def chat_hint_map(messages: list[str]) -> dict[str, str]:
    return dict(situation.hints(chat_facts(messages)))


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


def strip_questions(text: str) -> str:
    """The document is known, so no question back: drop closing questions that cite nothing
    ("... 50 دينار [3]. هل جوازك منتهي؟" -> "... 50 دينار [3].")."""
    def asks(s: str) -> bool:
        return s.strip().endswith(("؟", "?")) and not _CITE.search(s)
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or asks(lines[-1])):
        lines.pop()
    if lines:
        sents = _SENTENCE.split(lines[-1])
        while len(sents) > 1 and asks(sents[-1]):
            sents.pop()
        lines[-1] = " ".join(sents)
        lines[-1] = re.sub(r"[،,:]\s*$", ".", lines[-1].rstrip())
    return "\n".join(lines).strip() or text


def drop_dangling(text: str) -> str:
    """After a sentence was dropped: no last line that only announces something ("المبلغ المطلوب هو:")."""
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or lines[-1].rstrip().endswith((":", "："))):
        lines.pop()
    return "\n".join(lines).strip()


# ---------------------------------------------------------------- fee / note line selection (Agent/fees.py)
_TOTAL = re.compile(r"مجموع|كلها|كله|اجمالي|بالمجمل|سوا|total|altogether|in total")


def select_snippets(services: list[str], topic: str, question: str,
                    facts_by_sid: dict[str, situation.Facts]) -> tuple[list[dict], dict[str, fees.Pick]]:
    """The snippets of the routed services, with only the fee lines that apply to the user's facts (all of them when
    a deciding fact is unknown). Every note line stays."""
    snips = build_snippets(services, topic)
    picks = {sid: fees.pick(sid, facts_by_sid[sid], question) for sid in services}
    out = []
    for sid in services:
        mine = [s for s in snips if s["sid"] == sid]
        p = picks[sid]
        if p.fee is not None:
            keep = [s for s in mine if s["field"] == "الرسوم" and any(fees.matches(k, s["text"], s["line"]) for k in p.fee)]
            if keep:                                          # a rule that matches nothing never empties the fees
                mine = [s for s in mine if s["field"] != "الرسوم" or s in keep]
        out += mine
    return out, picks


def selection_hints(snips: list[dict], picks: dict[str, fees.Pick]) -> list[str]:
    """"The sources that apply to the user in «service»: [3], [5]" - points at lines, never states an amount."""
    out = []
    for sid, p in picks.items():
        refs = []
        if p.fee is not None:
            refs += [i for i, s in enumerate(snips, 1) if s["sid"] == sid and s["field"] == "الرسوم"]
        refs += [i for i, s in enumerate(snips, 1) if s["sid"] == sid and s["field"] == "ملاحظات"
                 and any(fees.matches(k, s["text"], s["line"]) for k in p.notes)]
        if refs:
            out.append(f"المصادر التي تنطبق على حالة المستخدم في «{NAMES[sid]}»: "
                       + "، ".join(f"[{i}]" for i in dict.fromkeys(refs)) + ". اذكر كل ما فيها.")
    return out


def allowed_sums(snips: list[dict], picks: dict[str, fees.Pick]) -> list[str]:
    """Totals the model may write (50 + 25 = 75): sums of the amounts in the selected fee / note lines, read from
    the data. Only for the number guard - never shown or sent to the model."""
    vals = []
    for sid, p in picks.items():
        lines = [s["text"] for s in snips if s["sid"] == sid and (
            (s["field"] == "الرسوم" and p.fee is not None) or
            (s["field"] == "ملاحظات" and any(fees.matches(k, s["text"], s["line"]) for k in p.notes)))]
        vals += fees.amounts(lines)
    sums = {sum(vals)} | {a + b for i, a in enumerate(vals) for b in vals[i + 1:]}
    return [str(x) for x in sums if len(vals) >= 2]


# ---------------------------------------------------------------- LLM
ANSWER_REMINDER = ("\n\nأجب عن السؤال مباشرة من المصادر مع رقم المصدر بعد كل معلومة، ولا تسأل المستخدم أي سؤال. "
                   f"إذا لم يكن الجواب في المصادر فقل حرفياً: \"{NOT_FOUND}\"")
TOTAL_NOTE = "\nإذا طلب المستخدم المجموع فاجمع المبالغ التي تنطبق عليه."


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


_ASKS = re.compile(r"\bكم\b|قديش|\bشو\b|ماذا|\bما\b|\bهل\b|متى|متي|\bوين\b|\bكيف\b|\bايش\b|بدي اعرف|اريد ان اعرف|"
                   r"المطلوب|الرسوم|how|what|which|when|where")


# ---------------------------------------------------------------- one chat
class Chat:
    """One user conversation. Messages must be asked in order; a new Chat forgets everything."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.history: list[tuple[str, str]] = []           # (question, answer)
        self.prev_services: list[str] = []
        self.state: tuple[list[str], list[str]] = ([], [])  # (documents, situations) of the conversation so far
        self.last_cited: str | None = None                  # last answer with [n] renumbered over last_sources
        self.last_sources: list[dict] = []

    def _set_state(self, services: list[str], docs: list[str], sits: list[str]):
        if services:
            self.state = (list(dict.fromkeys(router.doc_of(s) for s in services)),
                          list(dict.fromkeys(router.sit_of(s) for s in services)))
        else:                                               # "ضاع مني، شو أعمل؟" -> remember "lost" for "للهوية"
            self.state = (docs or self.state[0], sits or self.state[1])

    def restore(self, pairs: list[tuple[str, str]]):
        """Continue a conversation kept elsewhere (the web UI sends its history with every request)."""
        for q, a in pairs:
            q = fix_typos(normalize_display(q))
            self.history.append((q, strip_citations(a)))
            lx = router.lexicon(q)
            docs, sits = lx.docs or self.state[0], lx.sits or self.state[1]
            services = [f"{d}_{s}" for d in docs for s in sits if f"{d}_{s}" in NAMES]
            self._set_state(services, docs, sits)
            if services:
                self.prev_services = services

    def ask(self, question: str) -> dict:
        """-> {"answer", "contexts", "debug", "cited_answer" (with [n] for the UI), "sources" (the cited snippets,
        source n = sources[n - 1])}"""
        self.last_cited, self.last_sources = None, []
        t0 = time.time()
        q = fix_typos(normalize_display(question))
        r = router.route(q, self.history[-HISTORY_TURNS:], self.state, self.model)
        standalone = r.standalone or q
        if r.source == "state":                             # the follow-up with the service it continues
            standalone = f"{q} ({'، '.join(NAMES[s] for s in r.services)})"
        dbg = {"route": r.services, "route_source": r.source, "unsure": r.unsure, "standalone": standalone,
               "flags": [], "cited": False}
        if q != normalize_display(question):
            dbg["flags"].append("typos_fixed")
        if not r.in_scope:
            ans, contexts = OUT_OF_SCOPE, []
            dbg["flags"].append("out_of_scope")
        elif r.needs_clarification or not r.services:
            ans, contexts = router.clarification(r), []
            dbg["flags"].append("clarify")
            self._set_state([], r.docs, r.sits)
        else:
            if self._is_story(q, r):
                ans, contexts = self._answer_parts(q, r, dbg)
            else:
                facts_by_sid = None
                if len(r.services) > 1 and all(sid in r.clauses for sid in r.services) and r.source == "lexicon":
                    facts_by_sid = {sid: chat_facts([" ".join(r.clauses[sid])], sid.endswith("_lost"))
                                    for sid in r.services}
                ans, contexts = self._answer(q, standalone, r.services, dbg, facts_by_sid)
            self.prev_services = r.services
            self._set_state(r.services, r.docs, r.sits)
        self.history.append((q, ans))
        dbg["seconds"] = round(time.time() - t0, 2)
        sources = self.last_sources if contexts else []
        return {"answer": ans, "contexts": contexts, "debug": dbg,
                "cited_answer": self.last_cited if sources else ans, "sources": sources}

    # ------------------------------------------------------------ long stories: one sub-question per service
    @staticmethod
    def _is_story(q: str, r: router.Route) -> bool:
        if len(r.services) < 2 or len(q) < 100 or r.source != "lexicon":
            return False
        if not all(sid in r.clauses for sid in r.services):
            return False
        return len({tuple(r.clauses[sid]) for sid in r.services}) >= 2

    def _answer_parts(self, q: str, r: router.Route, dbg: dict) -> tuple[str, list[str]]:
        """Several people / problems in one message: every service is answered on its own, with the clause(s) about
        it + what the user asks, and the facts of those clauses only (the son's age is not the brother's)."""
        used = {c for sid in r.services for c in r.clauses[sid]}
        clauses = [c for c in router.CLAUSE_SPLIT.split(normalize_display(q)) if c.strip()]
        ask = " ".join(c.strip() for c in clauses if c not in used and _ASKS.search(normalize_search(c)))
        dbg["flags"].append(f"story:{len(r.services)}")
        parts, contexts, cited, sources = [], [], [], []
        for sid in r.services:
            about = " ".join(c.strip() for c in r.clauses[sid])
            sub_q = f"{about}، {ask}" if ask else about
            facts = {sid: chat_facts([about], sid.endswith("_lost"))}
            sub_dbg = {"flags": [], "cited": False}
            ans, ctx = self._answer(sub_q, f"{sub_q} ({NAMES[sid]})", [sid], sub_dbg, facts, sub=True)
            dbg["flags"] += [f"{sid}:{f}" for f in sub_dbg["flags"]]
            dbg["cited"] = dbg["cited"] or sub_dbg["cited"]
            parts.append(f"• {NAMES[sid]}: {ans}")
            contexts += [c for c in ctx if c not in contexts]
            if ctx and self.last_cited:
                offset = len(sources)
                cited.append(f"• {NAMES[sid]}: " + _CITE.sub(
                    lambda m: "".join(f"[{i + offset}]" for i in cited_numbers(m.group(0), 10 ** 6)), self.last_cited))
                sources += self.last_sources
            else:
                cited.append(f"• {NAMES[sid]}: {ans}")
        self.last_cited, self.last_sources = "\n\n".join(cited), sources
        return "\n\n".join(parts), contexts

    # ------------------------------------------------------------ one answer
    def _earlier(self, services: list[str]) -> list[tuple[str, str]]:
        """Earlier turns of this chat about the same document(s) - the user's situation lives there
        ("أنا بالأردن ومش عسكري" answered to the bot's question one turn before "كم المبلغ؟")."""
        docs = {router.doc_of(s) for s in services}
        if not self.history or not docs & {router.doc_of(s) for s in self.prev_services}:
            return []
        return self.history[-3:]

    def _answer(self, q: str, standalone: str, services: list[str], dbg: dict,
                facts_by_sid: dict[str, situation.Facts] | None = None, sub: bool = False) -> tuple[str, list[str]]:
        self.last_cited, self.last_sources = None, []
        earlier = [] if sub else self._earlier(services)
        user_msgs = [q] + [eq for eq, _ in reversed(earlier)]           # newest first
        topic = q                         # "وإذا كان تالف؟" / "كم المبلغ؟" -> the fields of the earlier questions too
        if wanted_fields(q) is None or earlier:
            topic = " ".join(user_msgs)
        if facts_by_sid is None:
            merged = chat_facts(user_msgs, any(s.endswith("_lost") for s in services))
            facts_by_sid = {sid: merged for sid in services}
        snips, picks = select_snippets(services, topic if standalone == q else f"{topic} {standalone}", topic,
                                       facts_by_sid)
        dbg["snippets"] = len(snips)
        if len(set(map(id, facts_by_sid.values()))) == 1:
            hints = [h for _, h in situation.hints(next(iter(facts_by_sid.values())))]
        else:                             # different people in one message: their facts per service
            hints = [f"في «{NAMES[sid]}»: " + " ".join(h for _, h in situation.hints(f))
                     for sid, f in facts_by_sid.items() if situation.hints(f)]
        hints += selection_hints(snips, picks)
        convo = ""
        if earlier:
            convo = "ما قاله المستخدم سابقاً في هذه المحادثة:\n" + "\n".join(f"- {eq}" for eq, _ in earlier) + "\n\n"
            dbg["flags"].append(f"memory:{len(earlier)}")
        total_note = TOTAL_NOTE if _TOTAL.search(normalize_search(q)) else ""
        dbg["picked"] = {sid: p.fee for sid, p in picks.items() if p.fee is not None}

        def generate(reminder=""):
            return _generate(q, standalone, snips, self.model, total_note + reminder, convo, hints)

        text = generate()
        cites = cited_numbers(text, len(snips))
        if not cites and not is_not_found(text):          # asked back, or answered without any [n]: one retry
            asked = is_clarifying(text) or not text.strip()
            dbg["flags"].append("retry_asked_back" if asked else "retry_no_citation")
            text = generate(ANSWER_REMINDER if asked else CITE_REMINDER)
            cites = cited_numbers(text, len(snips))
        # citation markers that point nowhere are dropped
        text = _CITE.sub(lambda m: m.group(0) if cited_numbers(m.group(0), len(snips)) else "", text)
        if cites and is_not_found(text):                  # a cited answer with a stray "not found" sentence
            text = strip_not_found(text) or text
            dbg["flags"].append("stripped_not_found")
        if not cites and (is_not_found(text) or is_clarifying(text) or not text.strip()):
            dbg["flags"].append("declined")               # the selected lines do not contain the answer
            return (NOT_FOUND if is_arabic(q) else NOT_FOUND_EN), []
        dbg["cited"] = bool(cites)
        if not cites:
            cites = overlap_fallback(text, snips)
            dbg["flags"].append("overlap_fallback")
        text = strip_questions(text)                      # a document is known: no question back

        # numbers must be in the cited snippets, what the user said, or a sum of the amounts that apply
        user_facts = "\n".join(user_msgs + hints + allowed_sums(snips, picks))

        def evidence(ids):
            return "\n".join(snips[i - 1]["text"] for i in ids) + "\n" + user_facts
        bad = check_numbers(text, evidence(cites))
        if bad:
            dbg["flags"].append("numbers_retry:" + ",".join(bad))
            retry = strip_questions(generate(NUMBER_REMINDER.format(bad="، ".join(bad))))
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
        self.last_cited = renumber_citations(text, cites)               # for the web UI: clickable [n]
        self.last_sources = [snips[i - 1] for i in cites]
        answer = text if KEEP_CITATIONS else strip_citations(text)
        return answer, contexts
