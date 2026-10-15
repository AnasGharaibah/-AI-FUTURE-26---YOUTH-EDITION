"""The user's situation as facts, computed in code (a 7B model compares numbers badly: it called a 17-year-old
"under 16"). Used for the hint lines in the answer prompt and by Agent/fees.py to pick the fee rule.

    f = facts("عمري 17 وجوازي باقي عليه 3 شهور")       # Facts(age=17, left_days=90, ...)
    hints(f)                                            # [("age", "العمر المذكور 17 سنة، أي ..."), ("left", ...)]
    merge([facts(newest), facts(older), ...])           # one value per fact, the newest message wins
"""
import re
from dataclasses import dataclass, fields

from DB.arabic import normalize_search

_N = r"(\d{1,3}|سنتين|سنه|شهرين|شهر|اسبوعين|اسبوع)"
_UNIT = r"\s*(يوم|ايام|اسبوع|اسابيع|شهر|شهور|اشهر|سنه|سنين|سنوات)?"
_AGE = re.compile(r"(?:عمر\w*|age)\s*(?:هو\s*|is\s*)?(\d{1,2})|(\d{1,2})\s*(?:سنه|سنين|سنوات|عام|years?)\b")
_LEFT = re.compile(r"(?:باقي|بقي|متبقي|ضايل|ظايل|left)\D{0,15}?" + _N + _UNIT)
_PAST = re.compile(r"(?:فايت|صارله|صارلها|صارلي|صار له|صار لي)\s*(?:عليه|عليها)?\s*" + _N + _UNIT)
_MARRIED = re.compile(r"(?:تزوج\w*|زواج\w*|عقد\w*|كتب\w* الكتاب)\D{0,12}?\bمن\s*(?:حوالي\s*|تقريبا\s*)?" + _N + _UNIT)
_SINCE_PREV = re.compile(r"(?:قبل|خلال|صارلي|صار لي|صارلها|صارله|بعد|من)\s*(?:حوالي\s*)?" + _N + _UNIT)
_PREV_WORDS = re.compile(r"اخر|السابق|طلعته|طلعتها|المره الاولي|اول مره|ضاع|ضاعت|تلف|خلال")
_ABROAD = re.compile(r"\bبرا\b|\bبره\b|بالخارج|في الخارج|خارج (الاردن|البلاد|المملكه|البلد)|مغترب|"
                     r"مقيم (ب|في)\s*(?!الاردن)\w+|(من|في|ب|عن طريق) ?(ال)?سفاره|قنصليه|abroad|embassy|"
                     r"السعوديه|الامارات|دبي|قطر|الكويت|البحرين|عمان السلطنه|المانيا|امريكا|كندا|بريطانيا|"
                     r"اوروبا|تركيا|مصر|لبنان|العراق|استراليا|فرنسا|السويد")
_INSIDE = re.compile(r"بالاردن|في الاردن|داخل الاردن|جوا البلد|داخل البلد|داخل البلاد|in jordan|inside jordan")
_LOSS = re.compile(r"ضاع|ضايع|ضيع|فقد|فاقد|سرق|lost|stolen")
_ADULT = re.compile(r"بالغ|adult|grown")
_EXPIRED = re.compile(r"منتهي|انتهي|انتهت|خلص|خلصت|فايت عليه|expired")
_VALID = re.compile(r"(غير|مش|مو|ما|لم|ليس)\s*(منتهي|انتهي|انتهت|خالص|خلص)|ساري|not expired|still valid")
_NTH_TIME = ((3, re.compile(r"(المره|مره|للمره) (الثالثه|ثالثه|التالته|تالته)|(ثالث|تالت) مره|third time|3rd time")),
             (2, re.compile(r"(المره|مره|للمره) (الثانيه|ثانيه|التانيه|تانيه)|(ثاني|تاني) مره|مره كمان|مره ثانيه|"
                            r"second time|2nd time")),
             (1, re.compile(r"(المره|للمره) الاولي|اول مره|first time")))
_FULL_PAGES = re.compile(r"خلص\w*(\s+\w+)?\s+(ال)?(ورق|صفح)\w*|(ورق|صفح)\w*(\s+\w+)?\s+(خلص|مخلص|ممتلي|امتلا|مليان)\w*|"
                         r"(ممتلي|امتلا|مليان)\w*\s+(ال)?صفح\w*|full pages|pages are full|out of pages")
_NOT_MILITARY = re.compile(r"(مش|مو|ما|لست|ليس|ليست|غير)\s*(انا\s*)?(عسكري|بالجيش|جيش)|not (in the )?(military|army)")
_MILITARY = re.compile(r"عسكري|بالجيش|القوات المسلحه|امن عام|الدرك|دفاع مدني|حفظ السلام|military|army")
_EXPRESS = re.compile(r"مستعجل|المستعجله|express|urgent")
_DAYS = {"يوم": 1, "ايام": 1, "اسبوع": 7, "اسابيع": 7, "شهر": 30, "شهور": 30, "اشهر": 30, "سنه": 365, "سنين": 365,
         "سنوات": 365}
_WORD_DAYS = {"سنه": 365, "سنتين": 730, "شهر": 30, "شهرين": 60, "اسبوع": 7, "اسبوعين": 14}


@dataclass
class Facts:
    age: int | None = None
    adult: bool = False
    left_days: int | None = None          # days left before the document expires
    expired: bool | None = None           # True expired, False explicitly still valid
    pages_full: bool | None = None
    military: bool | None = None
    abroad: bool | None = None            # True abroad, False explicitly in Jordan
    loss_count: int | None = None         # which loss this is (1, 2, 3)
    since_prev_days: int | None = None    # days since the previous loss / issue
    married_days: int | None = None       # days since the marriage contract
    express: bool | None = None           # the express office

    @property
    def adult_age(self) -> bool | None:
        return True if self.adult else (None if self.age is None else self.age >= 18)


def _days(num: str, unit: str | None) -> int:
    return int(num) * _DAYS.get(unit or "", 30) if num.isdigit() else _WORD_DAYS[num]


def facts(text: str, loss_context: bool = False) -> Facts:
    """Facts stated in one message. loss_context: the chat is about a lost document, so "تاني مرة" counts losses
    even without a loss word in this message."""
    q, f = normalize_search(text), Facts()
    m = _AGE.search(q)
    if m and 0 < int(m.group(1) or m.group(2)) < 120:
        f.age = int(m.group(1) or m.group(2))
    f.adult = bool(_ADULT.search(q))
    m = _LEFT.search(q)
    if m:
        f.left_days = _days(m.group(1), m.group(2))
    married = _MARRIED.search(q)
    if married:
        f.married_days = _days(married.group(1), married.group(2))
    if _FULL_PAGES.search(q):
        f.pages_full = True
    rest = _FULL_PAGES.sub(" ", q)                     # "مخلص ورقه" = pages full, not expired
    if _VALID.search(rest):
        f.expired = False
    elif _EXPIRED.search(rest) or _PAST.search(rest):
        f.expired = True
    if _NOT_MILITARY.search(q):
        f.military = False
    elif _MILITARY.search(q):
        f.military = True
    if loss_context or _LOSS.search(q):
        f.loss_count = next((n for n, rx in _NTH_TIME if rx.search(q)), None)
    if _PREV_WORDS.search(q):
        for m in _SINCE_PREV.finditer(_MARRIED.sub(" ", q)):
            f.since_prev_days = _days(m.group(1), m.group(2))
            break
    if _ABROAD.search(q):
        f.abroad = True
    elif _INSIDE.search(q):
        f.abroad = False
    if _EXPRESS.search(q):
        f.express = True
    return f


def merge(newest_first: list[Facts]) -> Facts:
    """One value per fact: the newest message that states it wins ("عمري 15" earlier, "عمري 17" now -> 17)."""
    out = Facts()
    for fld in fields(Facts):
        unset = False if fld.name == "adult" else None          # adult is a plain flag: False = not said
        v = next((getattr(f, fld.name) for f in newest_first if getattr(f, fld.name) is not unset), unset)
        setattr(out, fld.name, v)
    return out


def hints(f: Facts) -> list[tuple[str, str]]:
    """[(kind, sentence)] for the answer prompt."""
    out = []
    if f.age is not None:
        band = "أقل من 16 سنة" if f.age < 16 else ("16 سنة أو أكثر وأقل من 18" if f.age < 18 else "18 سنة أو أكثر")
        out.append(("age", f"العمر المذكور {f.age} سنة، أي {band}."))
    elif f.adult:
        out.append(("age", "صاحب الطلب بالغ، أي 18 سنة أو أكثر."))
    if f.left_days is not None:
        out.append(("left", f"المدة المتبقية على انتهاء الوثيقة حوالي {f.left_days} يوماً، أي "
                            f"{'أقل من 180 يوماً' if f.left_days < 180 else 'أكثر من 180 يوماً'}."))
    if f.married_days is not None:
        out.append(("married", f"مضى على عقد الزواج حوالي {f.married_days} يوماً، أي "
                               f"{'أقل من 90 يوماً' if f.married_days < 90 else 'أكثر من 90 يوماً'}."))
    if f.pages_full:
        out.append(("pages", "صفحات جواز السفر ممتلئة."))
    if f.expired is True:
        out.append(("expired", "الوثيقة منتهية الصلاحية."))
    elif f.expired is False:
        out.append(("expired", "الوثيقة ما زالت سارية (غير منتهية)."))
    if f.military is False:
        out.append(("military", "المستخدم ليس من العسكريين أو الأجهزة الأمنية."))
    elif f.military:
        out.append(("military", "المستخدم من العسكريين أو الأجهزة الأمنية أو الدفاع المدني."))
    if f.loss_count:
        name = {1: "الأولى", 2: "الثانية", 3: "الثالثة"}.get(f.loss_count, str(f.loss_count))
        out.append(("loss", "هذه المرة الأولى لفقدان الوثيقة." if f.loss_count == 1
                    else f"هذه المرة {name} لفقدان الوثيقة، أي ليست المرة الأولى."))
    if f.since_prev_days is not None:
        y = f.since_prev_days
        out.append(("since", f"مضى على المرة السابقة حوالي {y} يوماً، أي "
                             f"{'أقل من ثلاث سنوات' if y < 3 * 365 else 'ثلاث سنوات أو أكثر'}"
                             f"{' وأقل من 5 سنوات' if y < 5 * 365 else ''}."))
    if f.abroad:
        out.append(("place", "المستخدم خارج الأردن."))
    elif f.abroad is False:
        out.append(("place", "المستخدم داخل الأردن."))
    if f.express:
        out.append(("express", "يريد التقديم في مكتب الخدمة المستعجلة."))
    return out
