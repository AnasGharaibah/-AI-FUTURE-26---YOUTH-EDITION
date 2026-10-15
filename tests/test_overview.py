"""GET /overview: document count from the library, one entry per summarized service, government services first."""
from pathlib import Path

from fastapi.testclient import TestClient

import api
from DB import summaries as S

SUMS = {
    "projects.xlsx": {"file": "projects.xlsx", "service": "Projects", "summary": "A table. Second sentence.",
                      "topics": ["Projects"], "cases": [], "example_questions": ["Which projects are active?"]},
    "passport.pdf": {"file": "passport.pdf", "service": "جواز السفر", "summary": "إصدار وتجديد الجواز. جملة ثانية.",
                     "topics": ["الرسوم", "الخطوات", "الشروط"], "cases": ["5 سنوات / 10 سنوات"],
                     "example_questions": ["كم رسوم الجواز؟", "قديش بياخد الجواز؟", "q3", "q4", "q5", "q6"]},
    "deleted.pdf": {"file": "deleted.pdf", "service": "قديم", "summary": "", "topics": [], "cases": [],
                    "example_questions": []},
}


def test_overview(monkeypatch):
    monkeypatch.setattr(S, "load", lambda: SUMS)
    monkeypatch.setattr(api, "raw_files", lambda: [Path("passport.pdf"), Path("projects.xlsx"), Path("other.txt")])
    c = TestClient(api.app)
    for path in ("/overview", "/api/overview"):
        o = c.get(path).json()
        assert o["documents"] == 3                                        # real count from data/raw
        assert [s["file"] for s in o["services"]] == ["passport.pdf", "projects.xlsx"]   # gov first, no deleted
        p = o["services"][0]
        assert p["summary"] == "إصدار وتجديد الجواز." and len(p["example_questions"]) == 5
        assert p["doc_id"] == api.doc_id("passport.pdf") and p["cases"] == ["5 سنوات / 10 سنوات"]
