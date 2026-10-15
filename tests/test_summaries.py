"""DB.summaries with a scripted LLM: grounding checks, map-reduce, cache, Service Map token limit."""
import json

from DB import summaries as S
from DB.chunks import ntok

MD = """{0}------------------------------------------------

# دليل خدمة جواز السفر

#### الرسوم

جواز عادي 5 سنوات: 25 ديناراً. جواز 10 سنوات: 40 ديناراً.

{1}------------------------------------------------

#### مدة الإنجاز

الجواز العادي: 5 أيام عمل.
"""

PARENTS = [{"id": "a#s1", "source": "passport.pdf", "section": "دليل خدمة جواز السفر > الرسوم", "text": "الرسوم ..."},
           {"id": "a#s2", "source": "passport.pdf", "section": "دليل خدمة جواز السفر > مدة الإنجاز", "text": "..."},
           {"id": "b#s1", "source": "other.pdf", "section": "شيء آخر", "text": "..."}]

GOOD = {"service": "خدمة جواز السفر", "summary": "تتيح إصدار جواز السفر. الرسوم 25 ديناراً لخمس سنوات.",
        "topics": ["الرسوم", "مدة الإنجاز", "الغرامات"],
        "cases": ["جواز 5 سنوات / 10 سنوات"],
        "key_facts": [{"fact": "جواز عادي 5 سنوات: 25 ديناراً", "source_page": 1},
                      {"fact": "جواز 10 سنوات: 40 ديناراً", "source_page": 9}],
        "example_questions": ["كم رسوم الجواز؟", "قديش بياخد الجواز؟", "ما مدة الإنجاز؟"]}


def scripted(*replies):
    it, calls = iter(replies), []

    def chat(msgs, max_tokens=900, temperature=None, **kw):
        calls.append(msgs[-1]["content"])
        return next(it)
    chat.calls = calls
    return chat


def test_validate_keeps_only_what_the_document_supports():
    raw = dict(GOOD, summary="تتيح إصدار جواز السفر. الرسوم 99 ديناراً.",                  # 99 is not in the doc
               key_facts=GOOD["key_facts"] + [{"fact": "التوصيل 7 دنانير", "source_page": 2}],
               cases=["جواز 5 سنوات / 10 سنوات", "جواز 20 سنة"])
    s = S.validate(raw, source="passport.pdf", md=MD, sections=["الرسوم", "مدة الإنجاز"])
    assert s["summary"] == "تتيح إصدار جواز السفر."                                        # sentence with 99 dropped
    assert [f["fact"] for f in s["key_facts"]] == ["جواز عادي 5 سنوات: 25 ديناراً", "جواز 10 سنوات: 40 ديناراً"]
    assert s["key_facts"][1]["source_page"] is None                                       # page 9 doesn't exist
    assert s["cases"] == ["جواز 5 سنوات / 10 سنوات"]
    assert s["topics"] == ["الرسوم", "مدة الإنجاز"]                                        # no fines in the doc
    assert len(s["dropped"]) == 3


def test_single_pass_and_sections_from_parents():
    chat = scripted("```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```")
    s = S.summarize_document(chat, "passport.pdf", MD, PARENTS)
    assert len(chat.calls) == 1 and "[صفحة 2]" in chat.calls[0]
    assert s["service"] == "خدمة جواز السفر" and s["sections"] == ["الرسوم", "مدة الإنجاز"]


def test_dialect_question_added_when_missing():
    msa = ["ما رسوم الجواز؟", "ما مدة الإنجاز؟", "ما الوثائق المطلوبة؟"]
    out = S._ensure_dialect(scripted("شو الأوراق اللي بدي اياها؟"), msa)
    assert out[:2] == msa[:2] and out[2] == "شو الأوراق اللي بدي اياها؟"
    already = ["قديش رسوم الجواز؟", "ما مدة الإنجاز؟"]
    assert S._ensure_dialect(scripted(), already) == already                           # no LLM call needed
    assert S._ensure_dialect(scripted("ما الوثائق المطلوبة لتجديده؟"), msa) == msa       # rewrite not dialect: keep


def test_long_documents_are_map_reduced(monkeypatch):
    monkeypatch.setattr(S, "SINGLE_PASS_TOKENS", 5)
    monkeypatch.setattr(S, "SECTION_BATCH_TOKENS", 1)
    chat = scripted("ملاحظات 1", "ملاحظات 2", json.dumps(GOOD, ensure_ascii=False))
    s = S.summarize_document(chat, "passport.pdf", MD, PARENTS)
    assert len(chat.calls) == 3 and "ملاحظات 1" in chat.calls[2] and s["service"] == "خدمة جواز السفر"


def test_bad_json_retried_then_falls_back_to_heading():
    s = S.summarize_document(scripted("لا أعرف", "ما زلت لا أعرف"), "passport.pdf", MD, PARENTS)
    assert s["service"] == "دليل خدمة جواز السفر" and s["key_facts"] == []


def test_service_map_respects_the_limit_and_keeps_every_file():
    many = [dict(GOOD, file=f"doc{i}.pdf", service=f"خدمة رقم {i}", summary="جملة طويلة جداً " * 30,
                 topics=["الرسوم", "الخطوات", "الشروط", "مدة الإنجاز"], cases=["حالة أ", "حالة ب"]) for i in range(12)]
    m = S.service_map(many, limit=700)
    assert ntok(m) <= 700 and all(f"doc{i}.pdf" in m for i in range(12))
    small = S.service_map(many[:2], limit=700)
    assert "جملة طويلة" in small                                                          # short map: summaries kept


class FakePipe:
    def __init__(self, tmp):
        (tmp / "raw").mkdir()
        (tmp / "parsed").mkdir()
        (tmp / "raw/passport.pdf").write_bytes(b"pdf v1")
        (tmp / "parsed/passport.pdf.md").write_text(MD, encoding="utf-8")
        (tmp / "parents.json").write_text(json.dumps({p["id"]: p for p in PARENTS}, ensure_ascii=False),
                                          encoding="utf-8")
        self.cfg = {"raw_dir": tmp / "raw", "parsed_dir": tmp / "parsed"}
        self.store = type("Store", (), {"parents_path": tmp / "parents.json"})()


def test_build_caches_by_file_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SUMMARIES_PATH", tmp_path / "summaries.json")
    monkeypatch.setattr(S, "MAP_PATH", tmp_path / "service_map.md")
    pipe, good = FakePipe(tmp_path), json.dumps(GOOD, ensure_ascii=False)
    chat = scripted(good, good)
    S.build(pipe, chat=chat, model="m")
    S.build(pipe, chat=chat, model="m")                                                   # unchanged -> cached
    assert len(chat.calls) == 1
    (tmp_path / "raw/passport.pdf").write_bytes(b"pdf v2")                                # file changed -> again
    S.build(pipe, chat=chat, model="m")
    assert len(chat.calls) == 2
    assert "خدمة جواز السفر" in (tmp_path / "service_map.md").read_text(encoding="utf-8")
    monkeypatch.setattr(S, "SUMMARIES_PATH", tmp_path / "summaries.json")
    assert S.load()["passport.pdf"]["service"] == "خدمة جواز السفر"
