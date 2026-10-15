"""Agent fixes that need no LLM: calculator limits, eval edge cases."""
import json

from Agent import eval as ev
from Agent.tools import calculate


def calc(expr: str) -> str:
    return calculate.invoke({"expression": expr})


def test_calculate_normal():
    assert calc("3 * 120 + 45.5") == "405.5"
    assert calc("2 ** 10") == "1024"


def test_calculate_rejects_huge_exponent_and_numbers():
    assert "exponent too large" in calc("9 ** 9 ** 9")
    assert "number too large" in calc("10 ** 13")
    assert "number too large" in calc("99999999 * 99999999")
    assert "unsupported" in calc("__import__('os')")


def test_summarize_and_table_with_no_rows(tmp_path, capsys):
    s = ev.summarize([], "empty", "m")
    assert s["n"] == 0 and s["accuracy"] == 0 and s["retrieval_hit@5"] is None
    (tmp_path / "empty.json").write_text(json.dumps({"summary": s, "rows": []}), encoding="utf-8")
    ev.table(tmp_path)                                  # used to crash formatting None as a percentage
    assert "| empty | m | rag | 0/0 (0%) | - |" in capsys.readouterr().out


def test_check_expected_syntax():
    assert ev.check("الرسوم 25 و40 ديناراً", "25&40") and not ev.check("الرسوم 25 ديناراً", "25&40")
    assert ev.check("مدة الإنجاز خمسة أيام", "5 أيام/خمسة أيام")
    assert ev.check("الرسوم والخطوات", "الرسوم|>=1:الخطوات/الشروط")
    assert ev.check("جواز السفر: الرسوم والشروط", "جواز السفر|>=2:الرسوم/الشروط/الخطوات")
    assert not ev.check("جواز السفر: الرسوم فقط", "جواز السفر|>=2:الرسوم/الشروط/الخطوات")


def test_check_meta_and_ambiguous_rules():
    assert ev.check("أساعدك في جواز السفر والهوية", "جواز السفر&الهوية", "meta", tools=[])
    assert not ev.check("أساعدك في جواز السفر والهوية", "جواز السفر&الهوية", "meta", tools=["search_documents"])
    amb = "5 سنوات: 25 ديناراً [1]، 10 سنوات: 40 ديناراً [1]. هل تقصد جواز 5 سنوات أم 10 سنوات؟"
    assert ev.check(amb, "25&40", "ambiguous")
    assert not ev.check("هل تقصد 5 أم 10 سنوات؟ الرسوم 25 و40", "25&40", "ambiguous")   # question must come after
