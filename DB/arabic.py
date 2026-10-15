"""Arabic text helpers.

normalize_display(text)  safe for users + LLM: NFKC (PDF presentation forms), no tatweel, ASCII digits
normalize_search(text)   aggressive, for BM25 + comparisons ONLY: no diacritics, unified letters
tokens(text)             BM25 tokens: normalized, light clitic stripping, stopwords removed
is_arabic(text)          share of Arabic letters > threshold
looks_reversed(text)     heuristic for PDFs whose Arabic was extracted back-to-front
"""
import re
import unicodedata

TATWEEL = "ـ"
_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_AR_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})      # Persian / Urdu digits
_DIACRITICS = re.compile("[ً-ْٰ]")
_SEARCH_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# applied after normalize_search, so they are written in normalized form (no hamza, ة -> ه, ى -> ي)
STOPWORDS = frozenset("""
في من علي الي عن ان انه انها ما ماذا هل هذا هذه ذلك تلك هذي التي الذي الذين اللتي هو هي هم هن
كل او ثم قد لا لم لن كان كانت يكون تكون مع عند بعد قبل بين حتي اذا اذ كما لكن غير ايضا فقط
اي كيف متي اين لماذا يتم تم يجب يمكن ان
""".split())

# longest first; a prefix is stripped only if >= 3 letters remain
_PREFIXES = ("وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل")


def normalize_display(text: str) -> str:
    """NFKC (fixes ﻳﺴﻤﺢ -> يسمح), drop tatweel, Arabic-Indic digits -> 0-9, collapse spaces.
    Newlines and leading indentation are kept: chunking relies on line structure."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).replace(TATWEEL, "")
    text = text.translate(_AR_DIGITS)
    lines = []
    for line in text.splitlines():
        indent = line[:len(line) - len(line.lstrip(" "))]            # markdown indentation is meaningful
        body = re.sub(r"[ \t\u00a0\u200b\u200e\u200f\u202a-\u202e]+", " ", line.strip(" ")).strip()
        lines.append(indent + body)
    return "\n".join(lines).strip()


def normalize_search(text: str) -> str:
    """normalize_display + no diacritics, alef/ya/ta-marbuta/hamza unified, Latin lowercase."""
    text = _DIACRITICS.sub("", normalize_display(text))
    return text.translate(_SEARCH_MAP).lower()


def _strip_prefix(tok: str) -> str:
    if not _ARABIC_RE.search(tok):
        return tok
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            return tok[len(p):]
    return tok


def tokens(text: str) -> list[str]:
    out = []
    for t in _WORD_RE.findall(normalize_search(text)):
        if t in STOPWORDS:
            continue
        s = _strip_prefix(t)
        if s not in STOPWORDS:
            out.append(s)
    return out


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(bool(_ARABIC_RE.match(c)) for c in letters) / len(letters) > threshold


# ---- reversed-text detection: very common Arabic words and their mirror images
_COMMON = ("في", "من", "على", "الى", "إلى", "عن", "التي", "الذي", "هذا", "هذه", "يجب", "رسوم", "طلب", "الخدمة", "المادة")


def looks_reversed(text: str) -> bool:
    """True when mirrored forms of common words (يف، نم، يلع ...) clearly outnumber the normal ones."""
    words = _WORD_RE.findall(normalize_display(text))
    if sum(bool(_ARABIC_RE.search(w)) for w in words) < 20:
        return False
    normal = sum(w in _COMMON for w in words)
    mirrored = sum(w[::-1] in _COMMON and w not in _COMMON for w in words)
    return mirrored >= 3 and mirrored > 2 * normal


# ---- numbers: is every number in a text backed by the source? (answers, summaries, key facts) ----------------------
# no word boundary after the number: the documents glue amounts to words ("250دينار", "10دنانير غرامة")
_NUM = re.compile(r"(?<![\d.,])(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")
_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def _num(n: str) -> str:
    """'1,000' -> '1000', '٢٫٥' -> '2.5', '25.0' -> '25': one spelling per value."""
    n = normalize_display(n).replace("٫", ".").replace("٬", "").replace(",", "")
    return n[:-2] if n.endswith(".0") else n


# number words, in normalize_search form (no hamza, ة -> ه, ى -> ي, ئ -> ي)
_UNITS = {"واحد": 1, "واحده": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2, "ثلاث": 3, "ثلاثه": 3,
          "اربع": 4, "اربعه": 4, "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6, "سبع": 7, "سبعه": 7, "ثمان": 8,
          "ثماني": 8, "ثمانيه": 8, "تسع": 9, "تسعه": 9, "عشر": 10, "عشره": 10}
_TENS = {w + s: v for w, v in (("عشر", 20), ("ثلاث", 30), ("اربع", 40), ("خمس", 50), ("ست", 60), ("سبع", 70),
                                ("ثمان", 80), ("تسع", 90)) for s in ("ون", "ين")}
_HUNDREDS = {normalize_search(k): v for k, v in {
    "مئة": 100, "مائة": 100, "مئتان": 200, "مئتين": 200, "مائتان": 200, "مائتين": 200, "ألف": 1000,
    **{u + h: v * 100 for u, v in (("ثلاث", 3), ("اربع", 4), ("خمس", 5), ("ست", 6), ("سبع", 7), ("ثمان", 8),
                                    ("تسع", 9)) for h in ("مئة", "مائة")}}.items()}
_DUALS = {d: 2 for w in ("دينار", "صورت", "سنت", "شهر", "يوم", "مرت") for d in (w + "ين", w + "ان")}
_ORDINALS = {"ثاني": 2, "ثانيه": 2, "ثالث": 3, "ثالثه": 3, "رابع": 4, "رابعه": 4, "خامس": 5, "خامسه": 5,
             "سادس": 6, "سادسه": 6, "سابع": 7, "سابعه": 7, "ثامن": 8, "ثامنه": 8, "تاسع": 9, "تاسعه": 9,
             "عاشر": 10, "عاشره": 10}
_WORD_PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب", "ل")


def _word_value(tok: str, table: dict) -> tuple[int | None, bool]:
    """(value, starts with و) for a number word with an optional clitic prefix."""
    if tok in table:
        return table[tok], False
    for p in _WORD_PREFIXES:
        if tok.startswith(p) and tok[len(p):] in table:
            return table[tok[len(p):]], p.startswith("و")
    return None, False


def number_words(text: str, ordinals: bool = False) -> list[str]:
    """Numbers written as words -> digit strings: "خمسة وعشرون" -> "25", "ثلاثة أشهر" -> "3", "دينارين" -> "2",
    "سن السادسة عشر" -> "16" (ordinals only with ordinals=True; they are always consumed so "عشر" in
    "السادسة عشر" is never read as 10)."""
    toks = _WORD_RE.findall(normalize_search(text))
    cardinal = {**_UNITS, **_TENS, **_HUNDREDS}
    out, i = [], 0
    while i < len(toks):
        ordv, _ = _word_value(toks[i], _ORDINALS)
        if ordv is not None:
            if i + 1 < len(toks) and toks[i + 1] in ("عشر", "عشره"):
                ordv, i = ordv + 10, i + 1
            if ordinals:
                out.append(str(ordv))
            i += 1
            continue
        dual, _ = _word_value(toks[i], _DUALS)
        if dual is not None:
            out.append("2")
            i += 1
            continue
        v, _ = _word_value(toks[i], cardinal)
        if v is None:
            i += 1
            continue
        total, i = v, i + 1
        if v < 10 and i < len(toks) and toks[i] in ("عشر", "عشره"):        # خمسة عشر = 15
            total, i = v + 10, i + 1
        elif v < 10 and i < len(toks) and _HUNDREDS.get(toks[i]) == 100:   # ثلاث مئة
            total, i = v * 100, i + 1
        while i < len(toks):                                                # ... و عشرون و ...
            nv, has_wa = _word_value(toks[i], cardinal)
            if nv is None or not has_wa:
                break
            total, i = total + nv, i + 1
        out.append(str(total))
    return out


def check_numbers(answer: str, sources_text: str) -> list[str]:
    """Numbers in the answer that do not occur in any source (list numbering and [n] citations ignored).
    Arabic-Indic digits, Arabic decimal/thousands signs, thousands commas and number words ("خمسة وعشرون",
    "دينارين", "السادسة عشر") are normalized on both sides."""
    body = _CITE.sub("", normalize_display(answer).replace("٫", ".").replace("٬", ","))
    body = re.sub(r"(?m)^\s*\d+\s*[.)\-–]\s*", "", body)
    src = normalize_display(sources_text).replace("٫", ".").replace("٬", ",")
    known = {_num(n) for n in _NUM.findall(src)} | set(number_words(src, ordinals=True))
    return [n for n in _NUM.findall(body) + number_words(body) if _num(n) not in known]


def drop_number_sentences(answer: str, bad: list[str]) -> str:
    """Remove every sentence (or list line) that contains one of the unsupported numbers."""
    bad_values = {_num(n) for n in bad}
    kept_lines = []
    for line in answer.splitlines():
        if not line.strip():
            kept_lines.append("")
            continue
        marker = re.match(r"\s*(?:\d+\s*[.)\-–]|[-*•])\s+", line)       # list marker stays with its line
        prefix, rest = (marker.group(0), line[marker.end():]) if marker else ("", line)
        parts = [s for s in _SENTENCE_END.split(rest)
                 if not {_num(n) for n in _NUM.findall(_CITE.sub("", normalize_display(s))) + number_words(s)}
                 & bad_values]
        if " ".join(parts).strip():
            kept_lines.append(prefix + " ".join(parts))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
