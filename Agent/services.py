"""The 12 civil-status service documents (data/raw/<service_id>.txt) as clean fields + numbered source snippets.

Every file has the same nine fields (الفئة المستفيدة ... الرسوم) after a few website-menu lines. Cleaning only drops
the menus, empty/duplicate lines and stray symbols - the facts are kept word for word.

    from Agent.services import load, snippets
    load()["passport_damaged"]["fields"]["الرسوم"]      # "100 دينار"
    snippets(["passport_damaged"], ["الرسوم", "ملاحظات"])  # [{"service", "sid", "field", "text"}, ...]
"""
import os
import re
from functools import lru_cache
from pathlib import Path

from DB.arabic import normalize_display

ROOT = Path(__file__).resolve().parent.parent

DOCS = {"passport": "جواز السفر", "family_book": "دفتر العائلة", "id_card": "البطاقة الذكية"}
SITUATIONS = {"first_issue": "إصدار أول مرة", "renewal": "تجديد", "lost": "بدل فاقد", "damaged": "بدل تالف"}
NAMES = {      # service id -> the name used in snippets and contexts (the brief's names)
    "passport_first_issue": "جواز سفر لأول مرة", "passport_renewal": "تجديد جواز السفر",
    "passport_lost": "جواز سفر بدل فاقد", "passport_damaged": "جواز سفر بدل تالف",
    "family_book_first_issue": "دفتر عائلة لأول مرة", "family_book_renewal": "تجديد دفتر العائلة",
    "family_book_lost": "دفتر عائلة بدل فاقد", "family_book_damaged": "دفتر عائلة بدل تالف",
    "id_card_first_issue": "بطاقة ذكية لأول مرة", "id_card_renewal": "تجديد البطاقة الذكية",
    "id_card_lost": "بطاقة ذكية بدل فاقد", "id_card_damaged": "بطاقة ذكية بدل تالف",
}
SERVICE_IDS = tuple(NAMES)

# label at the start of a line in the file -> short field name used in snippets (document order)
LABELS = (("الفئة المستفيدة", "الفئة المستفيدة"), ("أماكن تقديم الخدمة", "أماكن تقديم الخدمة"),
          ("شروط تقديم الخدمة", "الشروط"), ("الوثائق المطلوبة للحصول على الخدمة", "الوثائق المطلوبة"),
          ("إجراءات تقديم الخدمة", "الإجراءات"), ("الزمن المستغرق لتقديم الخدمة", "الزمن المستغرق"),
          ("شركاء الخدمة", "شركاء الخدمة"), ("ملاحظات", "ملاحظات"), ("الرسوم", "الرسوم"))
FIELDS = tuple(f for _, f in LABELS)
MAX_SNIPPET = 600


def data_dir() -> Path:
    """YOTH_DATA_DIR, else data/raw (this repo), else data/ (the brief's layout)."""
    for d in (os.getenv("YOTH_DATA_DIR"), ROOT / "data/raw", ROOT / "data"):
        if d and (Path(d) / "passport_renewal.txt").exists():
            return Path(d)
    raise FileNotFoundError("service files not found in data/raw or data/ (set YOTH_DATA_DIR)")


def _clean_line(line: str) -> str:
    line = re.sub(r"[\]\[‘’]", "", line)                   # "مكاتب ]دائرة", "إ‘حصاءات"
    return re.sub(r"\s+", " ", line).strip(" -\t")


def parse(text: str) -> dict[str, str]:
    """Raw service page -> {field: text}. Lines before the first label (website menus, titles) are dropped."""
    fields, current = {}, None
    for raw in normalize_display(text).splitlines():
        line = raw.strip()
        label = next(((lab, f) for lab, f in LABELS if line.startswith(lab)), None)
        if label:
            current = label[1]
            fields.setdefault(current, [])
            line = line[len(label[0]):]
        if current is None:
            continue
        line = _clean_line(line)
        if line and line not in fields[current]:            # empty and exactly repeated lines
            fields[current].append(line)
    return {f: "\n".join(lines) for f, lines in fields.items() if lines}


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    d = data_dir()
    return {sid: {"id": sid, "name": NAMES[sid], "fields": parse((d / f"{sid}.txt").read_text(encoding="utf-8"))}
            for sid in SERVICE_IDS}


def split_text(text: str, limit: int = MAX_SNIPPET) -> list[str]:
    """Pieces of at most `limit` characters, cut at line breaks (a single longer line stays whole)."""
    pieces, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + 1 + len(line) > limit:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    return pieces + [cur] if cur else pieces


def snippets(service_ids: list[str], fields: list[str] | None = None) -> list[dict]:
    """One snippet per (service, field) - long fields split - in service order, then document field order."""
    out, services = [], load()
    for sid in service_ids:
        for field in FIELDS:
            text = services[sid]["fields"].get(field)
            if text and (fields is None or field in fields):
                out += [{"sid": sid, "service": NAMES[sid], "field": field, "text": p} for p in split_text(text)]
    return out


def sources_block(snips: list[dict]) -> str:
    """[1] (جواز سفر بدل تالف | الرسوم)\\n100 دينار ..."""
    return "\n".join(f"[{i}] ({s['service']} | {s['field']})\n{s['text']}" for i, s in enumerate(snips, 1))


def context_text(s: dict) -> str:
    """The `contexts` string of a snippet: "<service> - <field>: <text on one line>"."""
    return f"{s['service']} - {s['field']}: " + " ".join(s["text"].splitlines())

