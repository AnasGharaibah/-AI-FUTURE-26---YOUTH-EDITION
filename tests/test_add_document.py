"""python -m Agent.add_document: a new 9-field service page becomes a service without code changes.
Every test runs on a temporary copy of the data and its own registry (YOTH_DATA_DIR / YOTH_REGISTRY), in a fresh
process - the registry is read at import time."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from Agent.add_document import default_words, doc_name_of, situation_of, title_of

ROOT = Path(__file__).resolve().parent.parent

PAGE = """الرئيسية
الخدمات
دليل الخدمات
إصدار شهادة ميلاد لأول مرة
شارك
إصدار شهادة ميلاد لأول مرة
الفئة المستفيدة 	المواطنون الأردنيون
أماكن تقديم الخدمة	مكاتب دائرة الأحوال المدنية والجوازات
شروط تقديم الخدمة	1. تسجيل واقعة الولادة خلال 30 يوماً
الوثائق المطلوبة للحصول على الخدمة	1. تبليغ الولادة من المستشفى
إجراءات تقديم الخدمة
1. مراجعة مكتب خدمة الجمهور
الزمن المستغرق لتقديم الخدمة 	15 دقيقة
شركاء الخدمة	وزارة الصحة
ملاحظات	يستوفى 5 دنانير غرامة إذا تأخر التسجيل عن 30 يوماً
الرسوم	1 دينار
"""


def test_title_situation_and_document_name():
    assert title_of(PAGE) == "إصدار شهادة ميلاد لأول مرة"
    assert situation_of("إصدار شهادة ميلاد لأول مرة") == "first_issue"
    assert situation_of("تجديد رخصة المهن") == "renewal"
    assert situation_of("شهادة ميلاد بدل فاقد") == "lost"
    assert situation_of("شهادة ميلاد بدل تالف") == "damaged"
    assert situation_of("شهادة ميلاد") is None
    assert doc_name_of("إصدار شهادة ميلاد لأول مرة") == "شهادة ميلاد"
    assert doc_name_of("(تجديد جواز السفر العادي)") == "جواز السفر"
    assert default_words("شهادة ميلاد") == ["شهادة ميلاد", "شهادة الميلاد"]
    assert default_words("شهادة الميلاد") == ["شهادة الميلاد", "شهادة ميلاد"]


@pytest.fixture
def sandbox(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in (ROOT / "data/raw").glob("*.txt"):
        shutil.copyfile(f, raw / f.name)
    page = tmp_path / "page.txt"
    page.write_text(PAGE, encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "YOTH_DATA_DIR": str(raw),
           "YOTH_REGISTRY": str(tmp_path / "services.json")}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8")
    return tmp_path, page, run


def test_add_route_and_remove(sandbox):
    tmp, page, run = sandbox
    out = run("-m", "Agent.add_document", str(page))
    assert out.returncode == 1 and "--doc" in out.stdout                    # a new document needs an English id
    out = run("-m", "Agent.add_document", str(page), "--doc", "birth_certificate", "--words", "شهادة ولادة")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "added birth_certificate_first_issue" in out.stdout and "OK" in out.stdout
    assert (tmp / "raw/birth_certificate_first_issue.txt").exists()
    reg = json.loads((tmp / "services.json").read_text(encoding="utf-8"))
    assert reg["services"]["birth_certificate_first_issue"]["situation"] == "first_issue"
    assert "شهادة ولادة" in reg["docs"]["birth_certificate"]["words"]

    check = run("-c", """
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from Agent import competition, router
from Agent.services import snippets
out = {
  "fees": [router.route(q).services for q in ("كم رسوم شهادة الميلاد؟", "شو المطلوب لشهادة ولادة؟")],
  "missing_situation": router.route("بدي أجدد شهادة الميلاد").in_scope,
  "passport": router.route("كم رسوم تجديد الجواز؟").services,
  "ask": router.clarification(router.Route(sits=["lost"])),
  "oos": competition.OUT_OF_SCOPE,
  "snippet": [s["text"] for s in snippets(["birth_certificate_first_issue"], ["الرسوم"])],
}
print(json.dumps(out, ensure_ascii=False))
""")
    assert check.returncode == 0, check.stderr
    got = json.loads(check.stdout.strip().splitlines()[-1])
    assert got["fees"] == [["birth_certificate_first_issue"], ["birth_certificate_first_issue"]]
    assert got["missing_situation"] is False                    # no renewal page -> declined, not guessed
    assert got["passport"] == ["passport_renewal"]              # the original services are untouched
    assert "شهادة ميلاد" in got["ask"] and "شهادة ميلاد" in got["oos"]
    assert got["snippet"] == ["1 دينار"]

    out = run("-m", "Agent.add_document", str(page), "--doc", "birth_certificate")
    assert out.returncode == 1 and "--force" in out.stdout                 # no silent overwrite
    assert run("-m", "Agent.add_document", "--remove", "passport_lost").returncode == 1   # originals stay
    out = run("-m", "Agent.add_document", "--remove", "birth_certificate_first_issue")
    assert out.returncode == 0 and not (tmp / "raw/birth_certificate_first_issue.txt").exists()
    assert json.loads((tmp / "services.json").read_text(encoding="utf-8")) == {"docs": {}, "services": {}}


def test_page_without_fields_is_refused(sandbox):
    tmp, page, run = sandbox
    page.write_text("مجرد نص بدون حقول", encoding="utf-8")
    out = run("-m", "Agent.add_document", str(page), "--doc", "x_doc")
    assert out.returncode == 1 and "no field labels" in out.stdout


def test_situation_of_an_existing_document_is_added_without_doc(sandbox):
    """A page for a known document only needs its situation detected; the existing keywords are reused."""
    tmp, page, run = sandbox
    page.write_text(PAGE.replace("إصدار شهادة ميلاد لأول مرة", "جواز سفر مستعجل بدل فاقد"), encoding="utf-8")
    out = run("-m", "Agent.add_document", str(page), "--dry-run")
    assert out.returncode == 1 and "original services" in out.stdout       # passport_lost exists already
