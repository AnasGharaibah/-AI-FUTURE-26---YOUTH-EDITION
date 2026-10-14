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
    assert "| empty | m | 0% | - |" in capsys.readouterr().out
