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

from DB.arabic import is_arabic, normalize_display
from DB.pipeline import Pipeline

from . import llm
from .context import REFUSAL_AR, REFUSAL_EN, SYSTEM_PROMPT_AR

log = logging.getLogger("yoth.rag")

CONFIG = {
    "history_turns": 4,          # (question, answer) pairs used for follow-ups - the ONLY history length setting
    "min_similarity": 0.0,       # best cosine similarity below this -> refuse without calling the LLM (0 = off)
    "max_answer_tokens": 400,
    "retry_missing_citation": True,
}

STRICT_NUMBERS_AR = ("\n\nتنبيه: استخدم فقط الأرقام والرسوم والمدد الموجودة حرفياً في المصادر أعلاه. "
                     "لا تحسب ولا تقدّر أي رقم غير مذكور في المصادر.")
NUMBER_NOTE_AR = "ملاحظة: يرجى التحقق من الرسوم والمدد من المصدر الرسمي."
NUMBER_NOTE_EN = "Note: please verify fees and durations with the official source."

_CITE = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_SENTENCE = re.compile(r"(?<=[.!?؟])\s+")
# TODO(#15): _history grows with every thread_id and is never pruned - fine for the terminal chat, but a
#            long-running server should cap the number of threads (LRU) or expire idle ones.
_history: dict[str, deque] = {}


def db() -> Pipeline:
    return Pipeline.shared()


def is_refusal(text: str) -> bool:
    t = normalize_display(text)
    return REFUSAL_AR.rstrip(".") in t or REFUSAL_EN.rstrip(".").lower() in t.lower()


def _evidence(hits) -> list[str]:
    """The text the LLM sees for each hit: the whole (small) section when it was expanded, else the chunk."""
    return [h.get("context") or h["text"] for h in hits]


def _sources_block(hits) -> str:
    blocks, seen = [], set()
    for i, (h, body) in enumerate(zip(hits, _evidence(hits)), 1):
        key = h.get("parent_id") if h.get("context") else h["id"]
        if key in seen:
            body = "(نفس القسم المذكور أعلاه)"
        seen.add(key)
        blocks.append(f"[{i}] ({Pipeline.cite(h)} | {h['section'] or '-'})\n{body}")
    return "\n\n".join(blocks)


def _num(n: str) -> str:
    """'1,000' -> '1000', '٢٫٥' -> '2.5', '25.0' -> '25': one spelling per value."""
    n = normalize_display(n).replace("٫", ".").replace("٬", "").replace(",", "")
    return n[:-2] if n.endswith(".0") else n


def check_numbers(answer: str, sources_text: str) -> list[str]:
    """Numbers in the answer that do not occur in any source (list numbering and [n] citations ignored).
    Arabic-Indic digits, Arabic decimal/thousands signs and thousands commas are normalized on both sides."""
    body = _CITE.sub("", normalize_display(answer).replace("٫", ".").replace("٬", ","))
    body = re.sub(r"(?m)^\s*\d+\s*[.)\-–]\s*", "", body)
    src = normalize_display(sources_text).replace("٫", ".").replace("٬", ",")
    known = {_num(n) for n in _NUM.findall(src)}
    return [n for n in _NUM.findall(body) if _num(n) not in known]


def drop_number_sentences(answer: str, bad: list[str]) -> str:
    """Remove every sentence (or list line) that contains one of the unsupported numbers."""
    bad_values = {_num(n) for n in bad}
    kept_lines = []
    for line in answer.splitlines():
        if not line.strip():
            kept_lines.append("")
            continue
        marker = re.match(r"\s*(?:\d+\s*[.)\-–]|[-*•])\s+", line)       # list marker stays with its line
        prefix, rest = (marker.group(0), line[marker.end():]) if marker else ("", line)
        parts = [s for s in _SENTENCE.split(rest)
                 if not {_num(n) for n in _NUM.findall(_CITE.sub("", normalize_display(s)))} & bad_values]
        if " ".join(parts).strip():
            kept_lines.append(prefix + " ".join(parts))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _generate(question: str, sources: str, history, model, reminder: str = "") -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_AR}]
    for q, a in history:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"المصادر:\n{sources}\n\nالسؤال: {question}{reminder}"})
    return normalize_display(llm.chat(msgs, model=model, max_tokens=CONFIG["max_answer_tokens"]))


def _clean_citations(text: str, n_sources: int) -> str:
    """Citations that point to non-existent sources are noise."""
    return _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)


def _guard_numbers(text, question, sources, n_sources, history, model, arabic, flags) -> str:
    """Numbers/fees/durations must come from the sources. One stricter regeneration; if numbers are still
    unsupported, drop the sentences that contain them and add a 'please verify' note. Logged either way."""
    bad = check_numbers(text, sources)
    if not bad:
        return text
    flags.append("unsupported_numbers:" + ",".join(bad))
    log.warning("numbers not in sources: %s - regenerating | q=%s", bad, question)
    retry = _clean_citations(_generate(question, sources, history, model, STRICT_NUMBERS_AR), n_sources)
    still = check_numbers(retry, sources)
    if not still:
        flags.append("numbers_fixed_by_retry")
        log.warning("numbers fixed by regeneration | q=%s", question)
        return retry
    kept = drop_number_sentences(retry, still)
    flags.append("numbers_removed:" + ",".join(still))
    log.warning("numbers still not in sources after retry: %s - sentences removed | q=%s", still, question)
    note = NUMBER_NOTE_AR if arabic else NUMBER_NOTE_EN
    return f"{kept}\n\n{note}" if kept else (REFUSAL_AR if arabic else REFUSAL_EN)


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
    # TODO(#14): this mutates the shared Pipeline on every call; with concurrent web requests for different models
    #            one request can use another's rewriter. Pass the rewriter to search() instead of storing it.
    pipe.cfg["rewriter"] = lambda q: llm.rewrite_to_msa(q, model)
    rewrite_on = pipe.cfg["rewrite_query"] if rewrite is None else rewrite
    hits = pipe.search(standalone, rerank=rerank, rewrite=rewrite_on, sources=sources)
    t2 = time.time()
    # "text" is exactly what the LLM was given for that number (the expanded section when used), so the UI and the
    # evidence check look at the same evidence as the model; "chunk" is the matched passage, for highlighting.
    res = {"question": raw, "standalone": standalone, "flags": [], "retrieved": [h["source"] for h in hits], "sources": [
        {"n": i, "file": h["source"], "page": h.get("page"), "section": h["section"], "text": body, "chunk": h["text"]}
        for i, (h, body) in enumerate(zip(hits, _evidence(hits)), 1)]}

    arabic = is_arabic(standalone)
    refusal = REFUSAL_AR if arabic else REFUSAL_EN
    if not hits or (CONFIG["min_similarity"] and max(h.get("sim", 0) for h in hits) < CONFIG["min_similarity"]):
        res.update(answer=refusal, sources=[])
        res["flags"].append("no_hits" if not hits else "below_similarity")
    else:
        sources = _sources_block(hits)
        text = _generate(standalone, sources, list(hist), model)
        if not _CITE.search(text) and not is_refusal(text) and CONFIG["retry_missing_citation"]:
            res["flags"].append("retry_no_citation")
            text = _generate(standalone, sources, list(hist), model,
                             "\n\nتذكير: اذكر رقم المصدر بعد كل معلومة مثل [1]، أو أجب بجملة عدم العثور على المعلومة.")
        text = _clean_citations(text, len(hits))
        text = _guard_numbers(text, standalone, sources, len(hits), list(hist), model, arabic, res["flags"])
        if is_refusal(text) and _CITE.search(text):      # a cited answer with a stray "not found" line appended
            text = normalize_display(text.replace(REFUSAL_AR, "").replace(REFUSAL_EN, ""))
            res["flags"].append("stripped_refusal")
        if is_refusal(text):
            res["sources"] = []
        res["answer"] = text
    if use_history:
        hist.append((raw, res["answer"]))
    # the MSA rewrite is only shown when it was used for the search (cached by llm, so no second LLM call)
    res["msa"] = llm.rewrite_to_msa(standalone, model) if rewrite_on and arabic else None
    res["timings"] = {"followup": round(t1 - t0, 2), "retrieve": round(t2 - t1, 2),
                      "generate": round(time.time() - t2, 2), "total": round(time.time() - t0, 2)}
    return res


def reset(thread_id: str | None = None):
    _history.pop(thread_id, None) if thread_id else _history.clear()
