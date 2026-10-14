"""Deterministic RAG answer path for government-service documents (Arabic first, English still works).

    follow-up -> standalone question -> (MSA rewrite + hybrid search) -> numbered sources -> LLM
              -> guard (citations, numbers) -> {"answer", "sources", ...}

    from Agent.rag import answer
    r = answer("شو الأوراق المطلوبة لتجديد الجواز؟", thread_id="u1")
    print(r["answer"])
"""
import logging
import re
import time
from collections import deque
from functools import lru_cache

from DB.arabic import is_arabic, normalize_display
from DB.pipeline import Pipeline

from . import llm
from .context import REFUSAL_AR, REFUSAL_EN, SYSTEM_PROMPT_AR

log = logging.getLogger("yoth.rag")

CONFIG = {
    "history_turns": 4,          # remembered (question, answer) pairs per thread
    "min_similarity": 0.0,       # best cosine similarity below this -> refuse without calling the LLM (0 = off)
    "max_answer_tokens": 400,
    "retry_missing_citation": True,
}

_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_history: dict[str, deque] = {}


@lru_cache(maxsize=1)
def db() -> Pipeline:
    return Pipeline()


def is_refusal(text: str) -> bool:
    t = normalize_display(text)
    return REFUSAL_AR.rstrip(".") in t or REFUSAL_EN.rstrip(".").lower() in t.lower()


def _sources_block(hits) -> str:
    blocks, seen = [], set()
    for i, h in enumerate(hits, 1):
        body = h.get("context") or h["text"]
        key = h.get("parent_id") if h.get("context") else h["id"]
        if key in seen:
            body = "(نفس القسم المذكور أعلاه)"
        seen.add(key)
        blocks.append(f"[{i}] ({Pipeline.cite(h)} | {h['section'] or '-'})\n{body}")
    return "\n\n".join(blocks)


def check_numbers(answer: str, sources_text: str) -> list[str]:
    """Numbers in the answer that do not occur in any source (list numbering and [n] citations ignored)."""
    body = _CITE.sub("", answer)
    body = re.sub(r"(?m)^\s*\d+\s*[.)\-–]\s*", "", body)
    known = set(re.findall(r"\d+(?:[.,]\d+)?", normalize_display(sources_text)))
    return [n for n in _NUM.findall(body) if n not in known and n.replace(",", "") not in known]


def _generate(question: str, sources: str, history, model, reminder: str = "") -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_AR}]
    for q, a in history:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"المصادر:\n{sources}\n\nالسؤال: {question}{reminder}"})
    return normalize_display(llm.chat(msgs, model=model, max_tokens=CONFIG["max_answer_tokens"]))


def answer(question: str, thread_id: str = "default", model: str | None = None, use_history: bool = True,
           rerank: bool | None = None, rewrite: bool | None = None, sources: list[str] | None = None,
           history: list[tuple[str, str]] | None = None) -> dict:
    """Full pipeline. Returns {"answer", "sources": [{n, file, page, section, text}], "question", "standalone",
    "msa", "flags": [...], "timings": {...}}.
    sources: limit the search to these file names. history: (question, answer) pairs kept by the caller
    (e.g. the web UI) - replaces the server-side memory of `thread_id`."""
    t0 = time.time()
    model = model or llm.MODEL
    if history is not None:
        hist, use_history = deque(history[-CONFIG["history_turns"]:]), False
    else:
        hist = _history.setdefault(thread_id, deque(maxlen=CONFIG["history_turns"])) if use_history else deque()
    raw = normalize_display(question)

    standalone = llm.standalone_question(raw, list(hist), model) if hist else raw
    t1 = time.time()
    pipe = db()
    pipe.cfg["rewriter"] = lambda q: llm.rewrite_to_msa(q, model)
    hits = pipe.search(standalone, rerank=rerank, rewrite=rewrite, sources=sources)
    t2 = time.time()
    res = {"question": raw, "standalone": standalone, "flags": [], "retrieved": [h["source"] for h in hits], "sources": [
        {"n": i, "file": h["source"], "page": h.get("page"), "section": h["section"], "text": h["text"]}
        for i, h in enumerate(hits, 1)]}

    refusal = REFUSAL_AR if is_arabic(standalone) else REFUSAL_EN
    if not hits or (CONFIG["min_similarity"] and max(h.get("sim", 0) for h in hits) < CONFIG["min_similarity"]):
        res.update(answer=refusal, sources=[])
        res["flags"].append("below_similarity")
    else:
        sources = _sources_block(hits)
        text = _generate(standalone, sources, list(hist), model)
        if not _CITE.search(text) and not is_refusal(text) and CONFIG["retry_missing_citation"]:
            res["flags"].append("retry_no_citation")
            text = _generate(standalone, sources, list(hist), model,
                             "\n\nتذكير: اذكر رقم المصدر بعد كل معلومة مثل [1]، أو أجب بجملة عدم العثور على المعلومة.")
        # citations that point to non-existent sources are noise
        text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(hits) else "", text)
        bad = check_numbers(text, sources)
        if bad:
            res["flags"].append("unsupported_numbers:" + ",".join(bad))
            log.warning("numbers not found in sources: %s | q=%s", bad, standalone)
        if is_refusal(text) and _CITE.search(text):      # a cited answer with a stray "not found" line appended
            text = normalize_display(text.replace(REFUSAL_AR, "").replace(REFUSAL_EN, ""))
            res["flags"].append("stripped_refusal")
        if is_refusal(text):
            res["sources"] = []
        res["answer"] = text
    if use_history:
        hist.append((raw, res["answer"]))
    res["msa"] = llm.rewrite_to_msa(standalone, model) if is_arabic(standalone) else ""
    res["timings"] = {"followup": round(t1 - t0, 2), "retrieve": round(t2 - t1, 2),
                      "generate": round(time.time() - t2, 2), "total": round(time.time() - t0, 2)}
    return res


def reset(thread_id: str | None = None):
    _history.pop(thread_id, None) if thread_id else _history.clear()
