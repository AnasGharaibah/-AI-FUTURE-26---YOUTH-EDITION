"""Agent.rag answer policy with a fake retriever and a scripted LLM (no Ollama needed)."""
import pytest

from Agent import llm, rag

SECTION = "الرسوم:\n- جواز عادي لمدة 5 سنوات: 25 ديناراً\n- جواز لمدة 10 سنوات: 40 ديناراً"
HIT = {"id": "c1", "source": "passport.pdf", "page": 1, "section": "الرسوم", "parent_id": "p1",
       "text": "- جواز عادي لمدة 5 سنوات: 25 ديناراً", "context": SECTION}


class FakePipe:
    def __init__(self):
        self.cfg = {"rewrite_query": True, "rewriter": None}
        self.searches = []

    def search(self, query, rerank=None, rewrite=None, sources=None):
        self.searches.append({"query": query, "rewrite": rewrite})
        return [dict(HIT)]


@pytest.fixture
def pipe(monkeypatch):
    p = FakePipe()
    monkeypatch.setattr(rag, "db", lambda: p)
    monkeypatch.setattr(llm, "rewrite_to_msa", lambda q, model=None: q)
    rag.reset()
    return p


def script(monkeypatch, *replies):
    """Make the LLM answer with `replies` in order; returns the list of prompts it got."""
    calls, it = [], iter(replies)

    def fake(question, sources, history, model, reminder=""):
        calls.append(reminder)
        return next(it)
    monkeypatch.setattr(rag, "_generate", fake)
    return calls


# ---------------------------------------------------------------- #7 numbers
def test_check_numbers_normalizes_digits_and_separators():
    assert rag.check_numbers("الرسوم 1000 دينار [1]", "الرسوم 1,000 دينار") == []
    assert rag.check_numbers("الرسوم ٢٥ دينار", "الرسوم 25 دينار") == []
    assert rag.check_numbers("المدة 2.5 يوم", "المدة ٢٫٥ يوم") == []
    assert rag.check_numbers("1. الرسوم 25 دينار", "25") == []          # list numbering is not a fact
    assert rag.check_numbers("الرسوم 30 ديناراً [1]", "الرسوم 25 دينار") == ["30"]


def test_drop_number_sentences_keeps_the_rest():
    text = "الرسوم 25 ديناراً [1]. والتوصيل 7 دنانير [1].\n1. المستعجل 99 ديناراً [1]\n2. العادي 25 ديناراً [1]"
    out = rag.drop_number_sentences(text, ["7", "99"])
    assert "7" not in out and "99" not in out
    assert "الرسوم 25 ديناراً [1]." in out and "2. العادي 25 ديناراً [1]" in out
    assert "\n1." not in out                                            # no orphan list marker


def test_unsupported_number_fixed_by_one_retry(pipe, monkeypatch):
    calls = script(monkeypatch, "الرسوم 30 ديناراً [1].", "الرسوم 25 ديناراً [1].")
    r = rag.answer("كم رسوم الجواز؟", use_history=False)
    assert r["answer"] == "الرسوم 25 ديناراً [1]."
    assert len(calls) == 2 and calls[1] == rag.STRICT_NUMBERS_AR
    assert "numbers_fixed_by_retry" in r["flags"] and "unsupported_numbers:30" in r["flags"]


def test_unsupported_number_removed_after_retry(pipe, monkeypatch):
    script(monkeypatch, "الرسوم 30 ديناراً [1].", "الرسوم 25 ديناراً [1]. والتوصيل 7 دنانير [1].")
    r = rag.answer("كم رسوم الجواز؟", use_history=False)
    assert "7" not in r["answer"] and "الرسوم 25 ديناراً [1]." in r["answer"]
    assert r["answer"].endswith(rag.NUMBER_NOTE_AR)
    assert "numbers_removed:7" in r["flags"]


def test_supported_numbers_untouched(pipe, monkeypatch):
    calls = script(monkeypatch, "رسوم الجواز لمدة 10 سنوات 40 ديناراً [1].")
    r = rag.answer("كم رسوم الجواز لعشر سنوات؟", use_history=False)
    assert len(calls) == 1 and r["answer"] == "رسوم الجواز لمدة 10 سنوات 40 ديناراً [1]."


# ---------------------------------------------------------------- #8 evidence = what the LLM saw
def test_sources_return_the_text_the_llm_saw(pipe, monkeypatch):
    script(monkeypatch, "40 ديناراً [1].")
    src = rag.answer("كم رسوم الجواز لعشر سنوات؟", use_history=False)["sources"][0]
    assert src["text"] == SECTION                                       # the expanded section, not the chunk
    assert src["chunk"] == HIT["text"]


# ---------------------------------------------------------------- #6 no rewrite -> no MSA call
def test_rewrite_off_skips_msa(pipe, monkeypatch):
    script(monkeypatch, "25 ديناراً [1].")

    def boom(*a, **k):
        raise AssertionError("rewrite_to_msa must not be called when rewriting is off")
    monkeypatch.setattr(llm, "rewrite_to_msa", boom)
    r = rag.answer("كم رسوم الجواز؟", use_history=False, rewrite=False)
    assert r["msa"] is None and pipe.searches[-1]["rewrite"] is False


# ---------------------------------------------------------------- #9 one history length
def test_followups_use_history_turns(pipe, monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "standalone_question", lambda q, hist, model=None: seen.append(len(hist)) or q)
    script(monkeypatch, *["25 ديناراً [1]."] * 7)
    for i in range(6):
        rag.answer(f"سؤال {i}", thread_id="t")
    assert seen == [1, 2, 3, 4, 4]                                      # capped at CONFIG["history_turns"]
    hist = [(f"q{i}", "a") for i in range(10)]
    seen.clear()
    rag.answer("سؤال", history=hist)
    assert seen == [rag.CONFIG["history_turns"]]
