"""Tool-calling agent with guardrails in code (YOTH_MODE=agent).

    router -> chit_chat / meta     : answered from the Service Map, no tools; 3 example questions offered
           -> service_question     : Ollama native tool calling (search_documents / get_document_summary /
                                     list_services), max 3 tool calls and 6 model calls per turn
                                     no search before the answer -> one forced search + regenerate
           -> guards (Agent/rag.py): citations, numbers (#7), refusal; a refusal suggests the closest service
    tool calling fails twice in a row -> the RAG path answers instead

    from Agent.agent import answer
    r = answer("كم رسوم تجديد جواز السفر؟")   # {"answer", "sources", "route", "tools", "model_calls", "flags", ...}
"""
import logging
import re
import time
from collections import deque

from DB import summaries as S
from DB.arabic import is_arabic, normalize_display, normalize_search, tokens

from . import llm, rag
from . import tools as T
from .context import META_PROMPT, REFUSAL_AR, REFUSAL_EN, agent_prompt

log = logging.getLogger("yoth.agent")

CONFIG = {
    "max_tool_calls": 3,
    "max_model_calls": 6,         # the last allowed call never gets tools: the model must answer with what it found
    "max_answer_tokens": 500,
    "router_llm": True,           # ask the LLM when the rules can't classify a message
    "fallback_after_failures": 2,  # tool-calling failures in a row before the RAG path takes over
}
CHIT_CHAT, META, SERVICE = "chit_chat", "meta", "service_question"
CITE_REMINDER = "\n\nتذكير: اذكر رقم المصدر بعد كل معلومة مثل [1]، أو أجب بجملة عدم العثور على المعلومة."
_CITE = re.compile(r"\[(\d+)\]")
# TODO(#15): like rag._history, never pruned
_history: dict[str, deque] = {}
_failures: dict[str, int] = {}    # model -> tool-calling failures in a row (reset by a success)

# ---------------------------------------------------------------- router (rules first, LLM only when unsure)
_CHIT = re.compile(
    r"^(مرحبا|مرحبتين|اهلا|اهلين|هلا|اهلا وسهلا|السلام عليكم|وعليكم السلام|صباح الخير|مساء الخير|شكرا|شكرا لك|"
    r"شكرا جزيلا|مشكور|يعطيك العافيه|الله يعطيك العافيه|مع السلامه|باي|كيف حالك|كيفك|hi|hello|hey|"
    r"good (morning|afternoon|evening)|thanks|thank you|bye|goodbye|how are you)[!?.؟ ،]*$")
_META = re.compile(
    r"(شو بتقدر|ايش بتقدر|شو بتعرف|بشو بتساعد|شو بتساعد|ماذا تستطيع|ما الذي تستطيع|ما الذي يمكنك|ماذا يمكنك ان|"
    r"بماذا (يمكنك|تستطيع)|كيف (يمكنك|تقدر|بتقدر) تساعد|شو الخدمات|شو بتغطي|ما (هي )?الخدمات|اي الخدمات|ايش الخدمات|"
    r"ماذا تغطي|ما الذي تغطيه|شو بقدر اسال|ماذا يمكنني ان اسال|ماذا استطيع ان اسال|من انت|مين انت|عرفني بنفسك|"
    r"what can you|what services|which services|what do you cover|what can i ask|who are you|how can you help)")
_SUMMARIZE = re.compile(r"(لخص|ملخص|تلخيص|نبذه عن|عرفني على|summari[sz]e|summary of|overview of)")
_SERVICE = re.compile(
    r"(رسوم|كم |قديش|اديش|بكم|وثائق|اوراق|مستندات|خطوات|شروط|مده|متي|كيف|وين|اين|ماده|غرامه|عقوبه|عمر|تجديد|"
    r"اصدار|بدل|فاقد|جواز|هويه|رخصه|طلب|تظلم|صلاحيه|الكتروني|fee|cost|how much|document|step|requirement|"
    r"condition|how long|fine|article|renew|issue|passport|licen|id card|apply)")


def _llm_route(message: str, model: str) -> str:
    out = llm.chat([{"role": "system", "content": "صنّف رسالة المستخدم بكلمة واحدة فقط: chit_chat (تحية أو شكر)، "
                     "أو meta (سؤال عمّا يستطيع المساعد فعله أو عن الخدمات المتوفرة)، أو service_question "
                     "(سؤال عن خدمة أو رسوم أو وثائق أو شروط أو أي معلومة)."},
                    {"role": "user", "content": message}], model=model, max_tokens=8, temperature=0)
    for label in (META, CHIT_CHAT, SERVICE):
        if label in out:
            return label
    return SERVICE                                   # unsure -> search (never answer facts unsearched)


def route(message: str, model: str | None = None) -> tuple[str, str]:
    """-> (route, how): how = "rule" | "llm" | "default"."""
    m = normalize_search(message).strip()
    if _CHIT.match(m):
        return CHIT_CHAT, "rule"
    if _META.search(m):
        return META, "rule"
    if _SERVICE.search(m) or _SUMMARIZE.search(m) or _mentions_service(m):
        return SERVICE, "rule"
    if CONFIG["router_llm"]:
        try:
            return _llm_route(message, model or llm.MODEL), "llm"
        except Exception as e:
            log.warning("router LLM failed: %s", e)
    return SERVICE, "default"


# What people call a service -> what the documents call it. The official name is added next to the everyday one,
# so search and the model both see the documents' term ("بطاقة الهوية" alone: right document found, but refused).
_EVERYDAY_NAMES = [
    (re.compile(r"(?:بطاق[ةه]\s+)?(?:ال)?هوي(?:[ةه]|ت(?:ي|ك|ه|ها|نا|كم))(?:\s+(?:ال)?شخصي[ةه])?"),
     "بطاقة الأحوال المدنية الذكية", "احوال"),
    (re.compile(r"\bid card\b|\bnational id\b", re.I), "civil status ID card (بطاقة الأحوال المدنية الذكية)", "احوال"),
]


def with_official_names(question: str) -> str:
    for pattern, official, present in _EVERYDAY_NAMES:
        if present not in normalize_search(question):
            question = pattern.sub(lambda m: f"{m.group(0)} ({official})", question, count=1)
    return question


def _mentions_service(m: str) -> bool:
    for s in S.load().values():
        words = [w for w in normalize_search(s.get("service", "")).split() if len(w) >= 4]
        if any(w in m for w in words):
            return True
    return False


# ---------------------------------------------------------------- meta / chit-chat
def _gov_first(summaries: list[dict]) -> list[dict]:
    """Government services (fees / documents / steps topics) first, then the other documents."""
    return sorted(summaries, key=lambda s: -sum(t in S.TOPICS for t in s.get("topics", [])))


def example_questions(n: int = 3) -> list[str]:
    out = []
    for s in _gov_first(list(S.load().values())):
        if s.get("example_questions"):
            out.append(s["example_questions"][0])
        if len(out) == n:
            break
    return out


def _meta_answer(question: str, model: str) -> tuple[str, int]:
    examples = example_questions(3)
    prompt = META_PROMPT.format(service_map=T.list_services(), examples="\n".join(f"- {q}" for q in examples))
    text = llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": question}],
                    model=model, max_tokens=350)
    text = normalize_display(text)
    shown = sum(normalize_search(q)[:25] in normalize_search(text) for q in examples)
    if examples and shown < 2:                        # guard: the 3 example questions are always offered
        head = "أمثلة لأسئلة يمكنك طرحها:" if is_arabic(question) or not question.isascii() else \
            "Example questions you can ask:"
        text = f"{text}\n\n{head}\n" + "\n".join(f"- {q}" for q in examples)
    return text, 1


# ---------------------------------------------------------------- closest service for refusals
_service_vectors: dict[str, list] = {}


def closest_service(question: str) -> dict | None:
    """The service whose summary is nearest to the question (bge-m3 embeddings, cosine)."""
    summaries = S.load()
    if not summaries:
        return None
    emb = T._db().embedder
    texts = {f: f"{s['service']}: {s['summary']} {' '.join(s['topics'])}" for f, s in summaries.items()}
    if set(_service_vectors) != set(texts):
        _service_vectors.clear()
        _service_vectors.update(zip(texts, emb.embed_documents(list(texts.values()), show_progress=False,
                                                               cpu=True)))     # answer time: keep the GPU free
    q = emb.embed_query(question)

    def cos(a, b):
        na, nb = sum(x * x for x in a) ** .5, sum(x * x for x in b) ** .5
        return sum(x * y for x, y in zip(a, b)) / (na * nb or 1)
    best = max(_service_vectors, key=lambda f: cos(q, _service_vectors[f]))
    return summaries[best]


def _with_suggestion(text: str, question: str) -> str:
    try:
        s = closest_service(question)
    except Exception as e:
        log.warning("closest service failed: %s", e)
        s = None
    if not s:
        return text
    ex = (s.get("example_questions") or [""])[0]
    if is_arabic(question) or not question.isascii():
        tip = f"أقرب خدمة متوفرة: **{s['service']}**." + (f" يمكنك مثلاً أن تسأل: «{ex}»" if ex else "")
    else:
        tip = f"Closest available service: **{s['service']}**." + (f" For example, ask: “{ex}”" if ex else "")
    return f"{text}\n\n{tip}"


# ---------------------------------------------------------------- the tool loop
def _ollama_chat(model, messages, with_tools):
    kw = {"tools": T.tool_specs()} if with_tools else {}
    if model not in llm._no_think:
        kw["think"] = False
    try:
        return llm._client().chat(model=model, messages=messages, options={**llm.OPTIONS,
                                  "num_predict": CONFIG["max_answer_tokens"]}, keep_alive=llm.KEEP_ALIVE, **kw)
    except Exception as e:
        if "think" in kw and "think" in str(e).lower():
            llm._no_think.add(model)
            kw.pop("think")
            return llm._client().chat(model=model, messages=messages, options={**llm.OPTIONS,
                                      "num_predict": CONFIG["max_answer_tokens"]}, keep_alive=llm.KEEP_ALIVE, **kw)
        raise


def _content(msg) -> str:
    return normalize_display(re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.S))


class ToolCallingFailed(Exception):
    pass


def answer(question: str, thread_id: str = "default", model: str | None = None, use_history: bool = True,
           history: list[tuple[str, str]] | None = None, scope: list[str] | None = None) -> dict:
    for kind, val in run(question, thread_id, model, use_history, history, scope):
        if kind == "done":
            return val


def run(question: str, thread_id: str = "default", model: str | None = None, use_history: bool = True,
        history: list[tuple[str, str]] | None = None, scope: list[str] | None = None):
    """Events: ("route", r) ("tool", name) ("retrieved", {"sources": [...]}) ("done", res).
    res = {"answer", "sources", "route", "route_by", "tools", "model_calls", "retrieved", "flags", "timings"}"""
    t0 = time.time()
    model = model or llm.MODEL
    if history is not None:
        hist, keep = deque(history[-rag.CONFIG["history_turns"]:]), False
    else:
        hist = _history.setdefault(thread_id, deque(maxlen=rag.CONFIG["history_turns"])) if use_history else deque()
        keep = use_history
    raw = normalize_display(question)
    r, how = route(raw, model)
    yield "route", r
    if r in (CHIT_CHAT, META):
        text, calls = _meta_answer(raw, model)
        res = {"answer": text, "sources": [], "tools": [], "model_calls": calls, "retrieved": [], "flags": []}
    elif _failures.get(model, 0) >= CONFIG["fallback_after_failures"]:
        _failures[model] = 0                          # try tools again next turn
        res = _rag_fallback(with_official_names(raw), thread_id, model, hist, scope, "fallback_rag:previous_failures")
    else:
        asked = with_official_names(raw)
        try:
            res = yield from _tool_loop(asked, model, hist, scope)
            _failures[model] = 0
        except ToolCallingFailed as e:
            _failures[model] = _failures.get(model, 0) + 1
            log.warning("tool calling failed (%s) - answering with the RAG path", e)
            res = _rag_fallback(asked, thread_id, model, hist, scope, f"fallback_rag:{e}")
    res.update(route=r, route_by=how, question=raw, timings={"total": round(time.time() - t0, 2)})
    if keep:
        hist.append((raw, res["answer"]))
    yield "done", res


def _rag_fallback(question, thread_id, model, hist, scope, flag):
    r = rag.answer(question, thread_id, model=model, sources=scope, history=list(hist))
    return {"answer": r["answer"], "sources": r["sources"], "tools": ["search_documents"], "model_calls": None,
            "retrieved": r["retrieved"], "flags": r["flags"] + [flag]}


def _tool_loop(question: str, model: str, hist, scope):
    msgs = [{"role": "system", "content": agent_prompt(T.list_services())}]
    for q, a in hist:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": question})
    hits, evidence, tools_used, flags = [], [], [], []
    model_calls, failures, forced = 0, 0, False
    summarize = bool(_SUMMARIZE.search(normalize_search(question)))
    final = None
    while model_calls < CONFIG["max_model_calls"]:
        with_tools = len(tools_used) < CONFIG["max_tool_calls"] and model_calls < CONFIG["max_model_calls"] - 1
        try:
            resp = _ollama_chat(model, msgs, with_tools)
        except Exception as e:
            failures += 1
            if failures >= CONFIG["fallback_after_failures"]:
                raise ToolCallingFailed(f"{type(e).__name__}: {str(e)[:80]}")
            continue
        model_calls += 1
        msg = resp.message
        calls = (msg.tool_calls or []) if with_tools else []
        if calls:
            msgs.append(msg)
            bad = 0
            for call in calls:
                if len(tools_used) >= CONFIG["max_tool_calls"]:
                    break
                name, args = call.function.name, dict(call.function.arguments or {})
                result = _run_tool(name, args, hits, evidence, model, scope)
                if result is None:                    # unknown tool / unusable arguments
                    bad += 1
                    result = f"خطأ: الأداة {name!r} أو معاملاتها غير صالحة. الأدوات: {', '.join(T.TOOL_NAMES)}."
                else:
                    tools_used.append(name)
                    yield "tool", name
                    if name == "search_documents":
                        yield "retrieved", {"sources": _sources(hits)}
                msgs.append({"role": "tool", "content": result, "tool_name": name})
            failures = failures + 1 if bad == len(calls) else 0
            if failures >= CONFIG["fallback_after_failures"]:
                raise ToolCallingFailed("invalid tool calls")
            continue
        text = _content(msg)
        grounded = "search_documents" in tools_used or (summarize and "get_document_summary" in tools_used)
        if not grounded and not forced:               # guard: a service question is never answered unsearched
            forced = True
            flags.append("forced_search")
            query = llm.standalone_question(question, list(hist), model) if hist else question
            result = _run_tool("search_documents", {"query": query}, hits, evidence, model, scope)
            tools_used.append("search_documents")
            yield "tool", "search_documents"
            yield "retrieved", {"sources": _sources(hits)}
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "search_documents", "arguments": {"query": query}}}]})
            msgs.append({"role": "tool", "content": result, "tool_name": "search_documents"})
            continue
        final = text
        break
    if final is None:                                 # budget used up without an answer: answer with what was found
        flags.append("limit_reached")
        final = rag._generate(question, "\n\n".join(evidence) or "-", list(hist), model) if evidence else \
            (REFUSAL_AR if is_arabic(question) else REFUSAL_EN)
        model_calls += 1
    final = _guard(final, question, hits, evidence, hist, model, model_calls, flags)
    sources = _sources(hits)
    if rag.is_refusal(final):
        sources = []
        final = _with_suggestion(final, question)
    return {"answer": final, "sources": sources, "tools": tools_used, "model_calls": model_calls,
            "retrieved": [h["source"] for h in hits], "flags": flags}


def _run_tool(name, args, hits, evidence, model, scope):
    """Runs one tool; appends search hits / evidence text. None when the call is unusable."""
    if name == "search_documents":
        query = str(args.get("query") or "").strip()
        if not query:
            return None
        text, new = T.search_documents(query, args.get("file"), start=len(hits) + 1, model=model, scope=scope)
        hits.extend(new)
        evidence.append(text)
        return text
    if name == "get_document_summary":
        text = T.get_document_summary(str(args.get("file") or ""))
        evidence.append(text)
        return text
    if name == "list_services":
        return T.list_services()
    return None


def _sources(hits) -> list[dict]:
    return [{"n": i, "file": h["source"], "page": h.get("page"), "section": h["section"], "text": body,
             "chunk": h["text"]} for i, (h, body) in enumerate(zip(hits, rag._evidence(hits)), 1)]


_FOLLOWUP = re.compile(r"(?:^|(?<=[.!؟?\n]))\s*([^.!؟?\n]*?(?:تقصد|تعني|do you mean)[^.!؟?\n]*[؟?])\s*$", re.I)


def drop_needless_followup(answer: str, question: str) -> str:
    """The case question at the end ("هل تقصد إصدار جديد أم تجديد؟") is removed when the user already named one of the
    options ("...لتجديد رخصة القيادة"). Words shared by all options are ignored ("جواز 5 سنوات أم جواز 10 سنوات")."""
    m = _FOLLOWUP.search(answer)
    if not m:
        return answer
    body = re.sub(r"^.*?(?:تقصد|تعني|do you mean)\s*", "", m.group(1), flags=re.I).rstrip("؟? ")
    sides = [set(tokens(s)) for s in re.split(r"\s+(?:أم|او|أو|or)\s+|[/،,]", body) if tokens(s)]
    if len(sides) < 2:
        return answer
    common = set.intersection(*sides)
    asked = set(tokens(question))
    if any((s - common) and (s - common) <= asked for s in sides):
        return answer[:m.start(1)].rstrip()
    return answer


def _guard(text, question, hits, evidence, hist, model, model_calls, flags) -> str:
    """The RAG guards on the agent's answer: citations, numbers (#7), stray refusal."""
    arabic = is_arabic(question)
    if not text:
        text = REFUSAL_AR if arabic else REFUSAL_EN
    sources = "\n\n".join(evidence)
    if hits:
        text = rag._clean_citations(text, len(hits))
        if (not _CITE.search(text) and not rag.is_refusal(text)
                and model_calls < CONFIG["max_model_calls"]):          # one regeneration with a citation reminder
            flags.append("retry_no_citation")
            text = rag._clean_citations(rag._generate(question, sources, list(hist), model, CITE_REMINDER),
                                        len(hits))
            model_calls += 1
    if sources:
        text = rag._guard_numbers(text, question, sources, len(hits), list(hist), model, arabic, flags,
                                  allow_retry=model_calls < CONFIG["max_model_calls"])
    if rag.is_refusal(text) and _CITE.search(text):
        text = rag.strip_refusal(text)
        flags.append("stripped_refusal")
    trimmed = drop_needless_followup(text, question)
    if trimmed != text:
        flags.append("dropped_needless_followup")
    return trimmed
