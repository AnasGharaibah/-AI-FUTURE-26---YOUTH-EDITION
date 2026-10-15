"""Which fee lines and note lines of a service apply to the user's situation.

The rules only *select* lines of the data files - by a phrase that appears in the line, or "#0" for the first line
when it has no words ("5 دنانير") - never by an amount, and they never write any answer text. The answer step sends
only the selected fee lines (plus every note line) to the model, which still writes the answer.

    pick("passport_renewal", Facts(age=40, expired=True), q)  ->  Pick(fee=["فوق سن السادسه عشر"], ...)
    pick("passport_renewal", Facts(), q)                      ->  Pick(fee=None, missing=["expiry"])   (keep all)
"""
import re
from dataclasses import dataclass, field

from DB.arabic import normalize_search

from .situation import Facts

THREE_YEARS = 3 * 365


@dataclass
class Pick:
    fee: list[str] | None = None            # keys of the الرسوم lines that apply; None = undecided -> keep all lines
    notes: list[str] = field(default_factory=list)      # keys of the ملاحظات lines that answer this question
    missing: list[str] = field(default_factory=list)    # facts that would decide it (age, expiry, place, loss)


def _passport_renewal(f: Facts, q: str) -> Pick:
    notes = ["المستعجله"] if f.express else []
    if f.abroad:
        return Pick(["خارج البلاد"], notes)
    if f.military:
        return Pick(["لافراد القوات"], notes)
    if f.age is not None and f.age < 16:
        return Pick(["دون السادسه عشر"], notes)
    if f.pages_full and f.expired is not True:
        return Pick(["الممتلي الصفحات"], notes)
    if f.expired or (f.left_days is not None and f.left_days < 180):
        return Pick(["فوق سن السادسه عشر"], notes)
    if f.expired is False or f.left_days is not None:
        return Pick(["لجواز السفر غير المنتهي"], notes)
    return Pick(None, notes, ["expiry"])


def _passport_first_issue(f: Facts, q: str) -> Pick:
    if f.military:
        return Pick(["لافراد القوات"])
    if f.age is not None and f.age < 16:
        return Pick(["دون سن 16"])
    if f.abroad:
        return Pick(["خارج البلاد"])
    if f.abroad is False or f.age is not None or f.adult:
        return Pick(["داخل البلاد"])
    return Pick(None, missing=["age", "place"])


def _passport_lost(f: Facts, q: str) -> Pick:
    if f.loss_count == 1:
        return Pick(["للمره الاولي"])
    if f.loss_count and f.loss_count >= 2:
        return Pick(["لاي مره"])                 # 250 + the previous guarantee if the last loss was < 5 years ago
    return Pick(None, missing=["loss"])


def _passport_damaged(f: Facts, q: str) -> Pick:
    return Pick(None, ["متعمد"] if re.search(r"متعمد|عمد|قصد|بقصد", q) else [])


_DEATH = r"توفي|توفى|توفت|وفاه|مات|ماتت|متوفي|متزوج (اكثر|ثنتين|اثنتين|زوجتين)|زوجتين|ثنتين|رب الاسره"
_DEADLINE = r"خلال|مهله|متى لازم|متي لازم|لازم ابلغ|ابلغ|تغيير|تغير|اخر موعد|deadline"


def _family_book_notes(q: str) -> list[str]:
    out = []
    if re.search(_DEATH, q):
        out += ["اكثر من زوجه", "وفاه الاب والام"]
    if re.search(_DEADLINE, q):
        out += ["خلال ثلاثه اشهر", "خلال 30 يوم"]
    return out


def _family_book_first_issue(f: Facts, q: str) -> Pick:
    late = f.married_days is None or f.married_days >= 90
    return Pick(["رسوم اصدار الدفتر"] + (["غرامه"] if late else []), _family_book_notes(q))


def _family_book_renewal(f: Facts, q: str) -> Pick:
    if f.expired is True:
        return Pick(["لتجديد دفتر العائله منتهي"], _family_book_notes(q))
    if f.expired is False:
        return Pick(["غير منتهي المده"], _family_book_notes(q))
    return Pick(None, _family_book_notes(q), ["expiry"])


def _recent(f: Facts) -> bool:
    """The previous loss / issue was less than three years ago (unknown counts as possible)."""
    return f.since_prev_days is None or f.since_prev_days < THREE_YEARS


def _family_book_lost(f: Facts, q: str) -> Pick:
    if f.loss_count == 1:
        return Pick(["#0", "تعهد خطي"], ["لاول مره"])
    if f.loss_count == 2:
        return Pick(["#0", "تعهد خطي"] + (["استيفاء"] if _recent(f) else []), ["للمره الثانيه"])
    if f.loss_count and f.loss_count >= 3:
        return Pick(["#0", "استيفاء"], ["للمره الثالثه"])
    return Pick(None, missing=["loss"])


def _family_book_damaged(f: Facts, q: str) -> Pick:
    return Pick(["#0"] + (["استيفاء"] if _recent(f) else []), _family_book_notes(q))


def _id_card_first_issue(f: Facts, q: str) -> Pick:
    adult = f.adult_age
    return Pick(["#0"] + (["غرامه"] if adult is not False else []))


def _id_card_lost(f: Facts, q: str) -> Pick:
    notes = ["اسبوعين"] if re.search(r"استلم|بستلم|نستلم|تستلم|اخذ|متي|متى|نفس اليوم|كم يوم|وقت|تصرف|جاهز|"
                                     r"بسرعه|فورا|when|same day", q) else []
    if f.loss_count == 1:
        return Pick(["#0"], notes)
    if f.loss_count == 2:
        return Pick(["#0"] + (["استيفاء"] if _recent(f) else []), notes + ["للمره الثانيه"])
    if f.loss_count and f.loss_count >= 3:
        return Pick(["#0", "استيفاء"], notes + ["للمره الثالثه"])
    return Pick(None, notes, ["loss"])


def _keep_all(f: Facts, q: str) -> Pick:
    return Pick(None)


RULES = {
    "passport_first_issue": _passport_first_issue, "passport_renewal": _passport_renewal,
    "passport_lost": _passport_lost, "passport_damaged": _passport_damaged,
    "family_book_first_issue": _family_book_first_issue, "family_book_renewal": _family_book_renewal,
    "family_book_lost": _family_book_lost, "family_book_damaged": _family_book_damaged,
    "id_card_first_issue": _id_card_first_issue, "id_card_renewal": _keep_all,
    "id_card_lost": _id_card_lost, "id_card_damaged": _keep_all,
}


def pick(service_id: str, f: Facts, question: str) -> Pick:
    return RULES.get(service_id, _keep_all)(f, normalize_search(question))


def matches(key: str, line: str, index: int) -> bool:
    """Does a rule key select this line? "#0" = the first line of the field, else a phrase of the line."""
    return index == int(key[1:]) if key.startswith("#") else normalize_search(key) in normalize_search(line)


_AMOUNT = re.compile(r"\(?\s*(\d+)\s*\)?\s*(?:دينار|دنانير|ديناراً|دينارا)")


def amounts(lines: list[str]) -> list[int]:
    """The first amount written in each line (read from the data, for the number guard's allowed sums)."""
    out = []
    for line in lines:
        m = _AMOUNT.search(line)
        if m:
            out.append(int(m.group(1)))
    return out
