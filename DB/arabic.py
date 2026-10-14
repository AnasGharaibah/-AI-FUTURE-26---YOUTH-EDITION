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
