"""Add a new service page (the same 9-field format as the 12 service files) so the bot answers about it.

    python -m Agent.add_document page.txt --doc birth_certificate
    python -m Agent.add_document page.txt --doc birth_certificate --words "شهادة ولادة" --words "birth certificate"
    python -m Agent.add_document page.txt --dry-run          # only check the file and show what would be added
    python -m Agent.add_document --list                      # services added so far
    python -m Agent.add_document --remove birth_certificate_first_issue

What it does:
  1. checks the page: the 9 field labels (الفئة المستفيدة ... الرسوم), at least the fees
  2. reads the service name from the page title and the situation from it (لأول مرة / تجديد / بدل فاقد / بدل تالف;
     --situation when the title does not say)
  3. finds the document: one the bot already knows (جواز / دفتر / هوية / an added one), or a new one named --doc
     (an English id, used in file names: birth_certificate)
  4. copies the page to data/raw/<document>_<situation>.txt and registers the service and the document's keywords
     in data/services.json
  5. checks in a fresh process that the router sends a question about it to the new service
The next run of main.py / chat.py / the web app answers about it; nothing in the code has to change.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DB.arabic import normalize_display, normalize_search  # noqa: E402

from Agent import services  # noqa: E402
from Agent.services import BUILTIN_SERVICES, FIELDS, LABELS, NAMES, SITUATIONS, parse  # noqa: E402

MENU = {"الرئيسية", "الخدمات", "دليل الخدمات", "شارك"}
SITUATION_WORDS = (     # (situation, title patterns in normalize_search form), checked in this order
    ("lost", r"بدل فاقد|فاقد"),
    ("damaged", r"بدل تالف|تالف"),
    ("renewal", r"تجديد"),
    ("first_issue", r"لاول مره|اول مره|اصدار|الحصول علي"),
)
FILLER = r"\(|\)|\bبدل\b|\bفاقد\b|\bتالف\b|\bتجديد\b|لأول مرة|لاول مرة|أول مرة|اول مرة|إصدار|اصدار|الحصول على|\b(ال)?عادي\b"
DOC_ID = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def title_of(text: str) -> str:
    """The service title: the last line before the first field label that is not a website menu line."""
    lines = []
    for raw in normalize_display(text).splitlines():
        line = raw.strip()
        if any(line.startswith(lab) for lab, _ in LABELS):
            break
        if line and line not in MENU:
            lines.append(line)
    title = lines[-1] if lines else ""
    return re.sub(r"\s+", " ", title.strip("() ")).strip()


def situation_of(title: str) -> str | None:
    t = normalize_search(title)
    return next((s for s, rx in SITUATION_WORDS if re.search(rx, t)), None)


def doc_name_of(title: str) -> str:
    """"إصدار شهادة ميلاد لأول مرة" -> "شهادة ميلاد"."""
    return re.sub(r"\s+", " ", re.sub(FILLER, " ", title)).strip(" -،,")


def default_words(doc_name: str) -> list[str]:
    """Keywords for a new document: its name with and without "ال" ("شهادة الميلاد" / "شهادة ميلاد")."""
    toks = doc_name.split()
    bare = [t[2:] if t.startswith("ال") and len(t) > 4 else t for t in toks]
    with_al = toks[:1] + [t if t.startswith("ال") else "ال" + t for t in bare[1:]]
    return list(dict.fromkeys(" ".join(x) for x in (toks, bare, with_al) if x))


def known_doc(title: str) -> str | None:
    """A document the router already recognises in the title (passport / family_book / id_card / added ones)."""
    from Agent.router import DOC_WORDS
    t = normalize_search(title)
    return next((d for d, rx in DOC_WORDS.items() if rx.search(t)), None)


def check_page(text: str) -> tuple[dict, list[str]]:
    fields = parse(text)
    return fields, [f for f in FIELDS if f not in fields]


def save_registry(reg: dict):
    services.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = services.REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, services.REGISTRY)


def verify(service_id: str, question: str) -> list[str]:
    """Route `question` in a fresh process (the registry is read at import) -> the services it goes to."""
    code = ("import json,sys; sys.stdout.reconfigure(encoding='utf-8'); from Agent.router import lexicon; "
            f"print(json.dumps(lexicon({question!r}).services))")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "YOTH_REGISTRY": str(services.REGISTRY),
           "YOTH_DATA_DIR": str(services.data_dir())}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-500:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def add(path: Path, doc: str | None = None, situation: str | None = None, name: str | None = None,
        words: list[str] | None = None, dry_run: bool = False, force: bool = False) -> dict:
    """Check, copy and register a service page. Returns what was (or would be) added; raises ValueError on problems."""
    text = path.read_text(encoding="utf-8-sig")
    fields, missing = check_page(text)
    if not fields:
        raise ValueError("no field labels found - is this a service page like data/raw/passport_renewal.txt?")
    if "الرسوم" not in fields:
        raise ValueError("the page has no الرسوم field")
    title = title_of(text)
    situation = situation or situation_of(title)
    if situation not in SITUATIONS:
        raise ValueError(f"cannot tell the situation from the title {title!r}: "
                         f"use --situation {'|'.join(SITUATIONS)}")
    doc_name = doc_name_of(title)
    doc = doc or known_doc(title)
    if not doc:
        raise ValueError(f"new document {doc_name!r}: give it an English id, e.g. --doc birth_certificate")
    if not DOC_ID.match(doc):
        raise ValueError(f"--doc must be lowercase letters, digits and _ (got {doc!r})")
    sid = f"{doc}_{situation}"
    if sid in BUILTIN_SERVICES:
        raise ValueError(f"{sid} is one of the 12 original services and cannot be replaced")
    reg = services.registry()
    if sid in reg["services"] and not force:
        raise ValueError(f"{sid} is already registered (use --force to replace it)")
    new_doc = doc not in services.DOCS
    doc_words = list(dict.fromkeys((words or []) + (default_words(doc_name) if new_doc else [])))
    entry = {"service_id": sid, "name": name or title, "doc": doc, "doc_name": services.DOCS.get(doc, doc_name),
             "situation": situation, "new_document": new_doc, "keywords": doc_words, "missing_fields": missing,
             "file": str(services.data_dir() / f"{sid}.txt")}
    if dry_run:
        return entry
    shutil.copyfile(path, entry["file"])
    if doc not in services.BUILTIN_DOCS:
        d = reg["docs"].setdefault(doc, {"name": doc_name, "words": []})
        d["words"] = list(dict.fromkeys(d["words"] + doc_words))
        entry["keywords"] = d["words"]
    reg["services"][sid] = {"name": entry["name"], "doc": doc, "situation": situation}
    save_registry(reg)
    sample = f"كم رسوم {entry['name']}؟"
    entry["check_question"], entry["check_route"] = sample, verify(sid, sample)
    return entry


def remove(sid: str) -> str:
    if sid in BUILTIN_SERVICES:
        raise ValueError(f"{sid} is one of the 12 original services and cannot be removed")
    reg = services.registry()
    if sid not in reg["services"]:
        raise ValueError(f"{sid} is not registered (see --list)")
    doc = reg["services"].pop(sid)["doc"]
    if not any(s["doc"] == doc for s in reg["services"].values()):
        reg["docs"].pop(doc, None)                       # the last service of an added document
    save_registry(reg)
    f = services.data_dir() / f"{sid}.txt"
    if f.exists():
        f.unlink()
    return str(f)


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", nargs="?", type=Path, help="the service page (.txt) to add")
    ap.add_argument("--doc", help="document id for a NEW document type, e.g. birth_certificate")
    ap.add_argument("--situation", choices=list(SITUATIONS), help="when the title does not say it")
    ap.add_argument("--name", help="service name used in answers / contexts (default: the page title)")
    ap.add_argument("--words", action="append", default=[], help="extra words people use for the document")
    ap.add_argument("--dry-run", action="store_true", help="check the page and show what would be added")
    ap.add_argument("--force", action="store_true", help="replace an already registered service")
    ap.add_argument("--list", action="store_true", help="show the added services")
    ap.add_argument("--remove", metavar="SERVICE_ID", help="remove an added service and its file")
    args = ap.parse_args()

    if args.list:
        reg = services.registry()
        if not reg["services"]:
            print("no added services (only the 12 original ones)")
        for sid, s in reg["services"].items():
            print(f"{sid:<36} {s['name']}   keywords: {', '.join(reg['docs'].get(s['doc'], {}).get('words', []))}")
        return 0
    if args.remove:
        try:
            print(f"removed {args.remove} ({remove(args.remove)})")
            return 0
        except ValueError as e:
            print(f"error: {e}")
            return 1
    if not args.file:
        ap.print_help()
        return 2
    try:
        e = add(args.file, args.doc, args.situation, args.name, args.words, args.dry_run, args.force)
    except (ValueError, FileNotFoundError, RuntimeError) as err:
        print(f"error: {err}")
        return 1
    print(("would add" if args.dry_run else "added") + f" {e['service_id']}")
    print(f"  name:       {e['name']}")
    print(f"  document:   {e['doc']} ({e['doc_name']}){'  <- new document type' if e['new_document'] else ''}")
    print(f"  situation:  {e['situation']} ({SITUATIONS[e['situation']]})")
    print(f"  keywords:   {', '.join(e['keywords']) or '(the existing document keywords)'}")
    print(f"  file:       {e['file']}")
    if e["missing_fields"]:
        print(f"  warning:    fields not found: {', '.join(e['missing_fields'])}")
    if not args.dry_run:
        ok = e["service_id"] in e["check_route"]
        print(f"  check:      {e['check_question']!r} -> {e['check_route']}  {'OK' if ok else 'NOT ROUTED - add --words'}")
        print("  next:       py chat.py   (ask about it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
