"""Agent.agent with a scripted Ollama client: router, forced search, tool budget, fallback, meta + refusal guards."""
from types import SimpleNamespace as NS

import pytest

from Agent import agent as A
from Agent import llm, rag

SUMMARIES = {"passport.pdf": {"file": "passport.pdf", "service": "جواز السفر", "summary": "إصدار وتجديد الجواز.",
                              "topics": ["الرسوم", "الخطوات"], "cases": ["5 سنوات / 10 سنوات"], "sections": [],
                              "key_facts": [], "example_questions": ["كم رسوم الجواز؟", "قديش بياخد الجواز؟"]},
             "id.docx": {"file": "id.docx", "service": "بطاقة الهوية", "summary": "إصدار الهوية.",
                         "topics": ["الرسوم"], "cases": [], "sections": [], "key_facts": [],
                         "example_questions": ["ما رسوم الهوية؟"]},
             "drive.html": {"file": "drive.html", "service": "رخصة القيادة", "summary": "الرخصة.",
                            "topics": ["الرسوم"], "cases": [], "sections": [], "key_facts": [],
                            "example_questions": ["كم رسوم الرخصة؟"]}}
HIT = {"source": "passport.pdf", "page": 1, "section": "الرسوم", "text": "جواز 5 سنوات: 25 ديناراً", "id": "c1"}


def call(name, **args):
    return NS(function=NS(name=name, arguments=args))


def reply(content="", calls=None):
    return NS(message=NS(content=content, tool_calls=calls or []))


class FakeClient:
    """Answers from a script: each item is a reply, an exception, or a callable(with_tools) -> reply."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def chat(self, model, messages, tools=None, **kw):
        self.calls.append({"tools": bool(tools), "n_messages": len(messages)})
        item = self.script.pop(0) if self.script else reply("انتهى [1].")
        if isinstance(item, Exception):
            raise item
        return item(bool(tools)) if callable(item) else item


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(A.S, "load", lambda: SUMMARIES)
    monkeypatch.setattr(A.T, "list_services", lambda: "# خريطة الخدمات")
    monkeypatch.setattr(A.T, "tool_specs", lambda: [{"type": "function", "function": {"name": "search_documents"}}])
    searches = []

    def search(query, file=None, start=1, model=None, scope=None):
        searches.append({"query": query, "file": file, "start": start})
        return f"[{start}] (passport.pdf, p.1 | الرسوم)\nجواز 5 سنوات: 25 ديناراً", [dict(HIT)]
    monkeypatch.setattr(A.T, "search_documents", search)
    monkeypatch.setattr(A, "closest_service", lambda q: SUMMARIES["passport.pdf"])
    A._history.clear()
    A._failures.clear()
    return searches


def use(monkeypatch, client):
    monkeypatch.setattr(llm, "_client", lambda: client)
    return client


# ---------------------------------------------------------------- router
@pytest.mark.parametrize("msg,expected", [
    ("مرحبا", A.CHIT_CHAT), ("شكراً!", A.CHIT_CHAT), ("hello", A.CHIT_CHAT),
    ("شو بتقدر تساعدني فيه؟", A.META), ("what services do you cover?", A.META), ("ماذا تستطيع أن تفعل؟", A.META),
    ("كم رسوم تجديد جواز السفر؟", A.SERVICE), ("وبدل الفاقد؟", A.SERVICE), ("لخص وثيقة جواز السفر", A.SERVICE),
    ("How much is the passport fee?", A.SERVICE)])
def test_router_rules(env, msg, expected):
    assert A.route(msg)[0] == expected


def test_router_asks_llm_only_when_unsure(env, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "meta")
    assert A.route("ممم طيب") == (A.META, "llm")
    monkeypatch.setitem(A.CONFIG, "router_llm", False)
    assert A.route("ممم طيب") == (A.SERVICE, "default")


# ---------------------------------------------------------------- meta / chit-chat
def test_meta_answers_from_the_map_without_tools(env, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "أنا فيريتي، أساعدك في جواز السفر وبطاقة الهوية.")
    client = use(monkeypatch, FakeClient())
    r = A.answer("شو بتقدر تساعدني فيه؟")
    assert r["route"] == A.META and r["tools"] == [] and client.calls == []
    assert "كم رسوم الجواز؟" in r["answer"] and "ما رسوم الهوية؟" in r["answer"]   # 3 examples always offered


# ---------------------------------------------------------------- service questions
def test_model_searches_then_answers(env, monkeypatch):
    client = use(monkeypatch, FakeClient(reply(calls=[call("search_documents", query="رسوم الجواز")]),
                                         reply("رسوم الجواز لخمس سنوات 25 ديناراً [1].")))
    r = A.answer("كم رسوم الجواز؟", use_history=False)
    assert r["tools"] == ["search_documents"] and r["model_calls"] == 2
    assert r["answer"] == "رسوم الجواز لخمس سنوات 25 ديناراً [1]." and r["sources"][0]["file"] == "passport.pdf"
    assert "forced_search" not in r["flags"] and client.calls[0]["tools"]


def test_answer_without_search_is_forced_to_search(env, monkeypatch):
    use(monkeypatch, FakeClient(reply("الرسوم 99 ديناراً."), reply("الرسوم 25 ديناراً [1].")))
    r = A.answer("كم رسوم الجواز؟", use_history=False)
    assert "forced_search" in r["flags"] and r["tools"] == ["search_documents"]
    assert r["answer"] == "الرسوم 25 ديناراً [1]."                                  # the unsearched 99 never shown


def test_second_search_continues_numbering(env, monkeypatch):
    use(monkeypatch, FakeClient(reply(calls=[call("search_documents", query="رسوم")]),
                                reply(calls=[call("search_documents", query="مدة", file="passport.pdf")]),
                                reply("الرسوم 25 ديناراً [1] والمدة [2].")))
    r = A.answer("كم رسوم ومدة الجواز؟", use_history=False)
    assert [s["start"] for s in env] == [1, 2] and env[1]["file"] == "passport.pdf"
    assert [s["n"] for s in r["sources"]] == [1, 2]


def test_tool_and_model_call_limits(env, monkeypatch):
    always_search = lambda with_tools: (reply(calls=[call("search_documents", query="x")]) if with_tools
                                        else reply("ما وُجد: 25 ديناراً [1]."))
    client = use(monkeypatch, FakeClient(*[always_search] * 10))
    r = A.answer("كم رسوم الجواز؟", use_history=False)
    assert len(r["tools"]) == A.CONFIG["max_tool_calls"] and r["model_calls"] <= A.CONFIG["max_model_calls"]
    assert not client.calls[-1]["tools"] and r["answer"] == "ما وُجد: 25 ديناراً [1]."


def test_tool_calling_failures_fall_back_to_rag(env, monkeypatch):
    use(monkeypatch, FakeClient(RuntimeError("model does not support tools"), RuntimeError("again")))
    monkeypatch.setattr(rag, "answer", lambda *a, **k: {"answer": "RAG [1]", "sources": [], "retrieved": [],
                                                         "flags": []})
    r = A.answer("كم رسوم الجواز؟", use_history=False)
    assert r["answer"] == "RAG [1]" and any(f.startswith("fallback_rag") for f in r["flags"])


def test_refusal_suggests_closest_service(env, monkeypatch):
    use(monkeypatch, FakeClient(reply(calls=[call("search_documents", query="شهادة")]),
                                reply("لم أجد هذه المعلومة في الوثائق المتوفرة.")))
    r = A.answer("كم رسوم شهادة الزواج؟", use_history=False)
    assert r["sources"] == [] and "أقرب خدمة متوفرة: **جواز السفر**" in r["answer"]


@pytest.mark.parametrize("question,answer,kept", [
    ("ما الوثائق المطلوبة لتجديد رخصة القيادة؟",
     "الوثائق: الرخصة القديمة [1]. هل تقصد إصدار جديد أم تجديد؟", False),          # user said تجديد: drop
    ("كم رسوم جواز السفر؟", "5 سنوات: 25 [1]، 10 سنوات: 40 [1]. هل تقصد جواز 5 سنوات أم جواز 10 سنوات؟", True),
    ("كم رسوم جواز السفر لمدة 10 سنوات؟",
     "10 سنوات: 40 [1]. هل تقصد جواز 5 سنوات أم جواز 10 سنوات؟", False),            # user named 10 years
    ("كم رسوم الجواز؟", "الرسوم 25 ديناراً [1].", True)])                           # no follow-up at all
def test_needless_followup_dropped(question, answer, kept):
    out = A.drop_needless_followup(answer, question)
    assert (out == answer) == kept and out.endswith(("[1].", "؟"))


def test_number_guard_applies_to_agent_answers(env, monkeypatch):
    use(monkeypatch, FakeClient(reply(calls=[call("search_documents", query="رسوم")]), reply("الرسوم 30 ديناراً [1].")))
    monkeypatch.setattr(rag, "_generate", lambda *a, **k: "الرسوم 25 ديناراً [1].")
    r = A.answer("كم رسوم الجواز؟", use_history=False)
    assert r["answer"] == "الرسوم 25 ديناراً [1]." and "numbers_fixed_by_retry" in r["flags"]
