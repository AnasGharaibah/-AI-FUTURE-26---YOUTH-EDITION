"""Thin Ollama chat wrapper + the small LLM helpers (MSA rewrite, follow-up rewrite).

    from Agent.llm import chat, rewrite_to_msa, standalone_question
    rewrite_to_msa("شو الأوراق اللي بدي اياها لجواز السفر؟")  # -> "ما الوثائق المطلوبة لإصدار جواز السفر؟"
"""
import logging
import os
import re
from functools import lru_cache

from DB.arabic import is_arabic, normalize_display

log = logging.getLogger("yoth.llm")

# ---- model config (the ONE place to choose the chat model; YOTH_MODEL overrides it) ---------------------------------
CANDIDATES = ("qwen2.5:7b", "qwen3:8b")             # compared with `python -m Agent.eval --models ...`
FALLBACK_MODEL = "qwen3:4b-instruct"                # fast, fits 8 GB comfortably
LLM_MODEL = os.getenv("YOTH_MODEL", FALLBACK_MODEL)
MODEL = LLM_MODEL                                   # kept: older code / eval read llm.MODEL
# 8 GB RAM settings: 8k context, near-deterministic, model stays loaded between questions. The reranker stays off
# (DB.pipeline DEFAULTS["use_reranker"]); Marker never runs while a chat model is loaded (DB.parse unloads it first).
OPTIONS = {"temperature": 0.1, "num_ctx": 8192}
KEEP_ALIVE = os.getenv("YOTH_KEEP_ALIVE", "30m")
_no_think: set[str] = set()                         # models that reject think=False (e.g. gemma3)


def is_hybrid_qwen3(model: str) -> bool:
    """qwen3:4b / qwen3:8b think by default and ignore think=False in their chat template."""
    return model.startswith("qwen3") and "instruct" not in model and "vl" not in model


TIMEOUT = float(os.getenv("YOTH_LLM_TIMEOUT", "90"))     # seconds per request; a hung Ollama runner must not freeze the app


@lru_cache(maxsize=1)
def _client():
    import ollama
    return ollama.Client(timeout=TIMEOUT)


def _call(fn, *a, **kw):
    """One retry when the request times out / the connection drops (Ollama runner restarts are common on 8 GB)."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        if any(k in type(e).__name__.lower() + str(e).lower() for k in ("timeout", "disconnected", "connection")):
            log.warning("Ollama request failed (%s) - retrying once", type(e).__name__)
            return fn(*a, **kw)
        raise


def chat(messages: list[dict], model: str | None = None, max_tokens: int = 400, temperature: float | None = None) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}] -> reply text (thinking disabled)."""
    model = model or MODEL
    opts = {**OPTIONS, "num_predict": max_tokens}
    if temperature is not None:
        opts["temperature"] = temperature
    if is_hybrid_qwen3(model):
        return _chatml_nothink(messages, model, opts)
    kw = {} if model in _no_think else {"think": False}
    try:
        res = _call(_client().chat, model=model, messages=messages, options=opts, keep_alive=KEEP_ALIVE, **kw)
    except Exception as e:
        if kw and "think" in str(e).lower():        # model without thinking support: retry plainly
            _no_think.add(model)
            res = _call(_client().chat, model=model, messages=messages, options=opts, keep_alive=KEEP_ALIVE)
        elif "not found" in str(e).lower():
            raise RuntimeError(f"Model missing. Run: ollama pull {model}") from e
        else:
            raise
    text = res["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat_stream(messages: list[dict], model: str | None = None, max_tokens: int = 400,
                temperature: float | None = None):
    """Like chat(), but yields the reply in pieces as the model produces them (thinking disabled)."""
    model = model or MODEL
    opts = {**OPTIONS, "num_predict": max_tokens}
    if temperature is not None:
        opts["temperature"] = temperature
    try:
        if is_hybrid_qwen3(model):
            prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for part in _client().generate(model=model, prompt=prompt, raw=True, stream=True, keep_alive=KEEP_ALIVE,
                                           options={**opts, "stop": ["<|im_end|>"]}):
                yield part["response"] or ""
            return
        kw = {} if model in _no_think else {"think": False}
        try:
            parts = _client().chat(model=model, messages=messages, options=opts, stream=True, keep_alive=KEEP_ALIVE, **kw)
            first = next(parts, None)                   # the request is sent here: unsupported `think` fails now
        except Exception as e:
            if not (kw and "think" in str(e).lower()):
                raise
            _no_think.add(model)
            parts = _client().chat(model=model, messages=messages, options=opts, stream=True, keep_alive=KEEP_ALIVE)
            first = next(parts, None)
        if first is not None:
            yield first["message"]["content"] or ""
        for part in parts:
            yield part["message"]["content"] or ""
    except Exception as e:
        if "not found" in str(e).lower():
            raise RuntimeError(f"Model missing. Run: ollama pull {model}") from e
        raise


def _chatml_nothink(messages, model, opts) -> str:
    """qwen3 here ignores think=False (the chat template never closes the think block), so build the ChatML prompt
    ourselves and pre-fill an empty <think></think> - the documented way to turn thinking off."""
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    res = _call(_client().generate, model=model, prompt=prompt, raw=True, keep_alive=KEEP_ALIVE,
                options={**opts, "stop": ["<|im_end|>"]})
    return re.sub(r"<think>.*?</think>", "", res["response"] or "", flags=re.S).strip()


# ---------------------------------------------------------------- MSA rewrite
_MSA_PROMPT = """حوّل سؤال المستخدم إلى سؤال قصير بالعربية الفصحى المناسبة للبحث في وثائق الخدمات الحكومية.
- لا تغيّر المعنى ولا تضف معلومات. استخدم مصطلحات رسمية مثل: الوثائق المطلوبة، الرسوم، الشروط، الخطوات، مدة الإنجاز.
- اكتب السؤال المعاد صياغته فقط، في سطر واحد، بدون شرح.

مثال:
السؤال: شو الأوراق اللي بدي اياها لجواز السفر؟
الناتج: ما الوثائق المطلوبة لإصدار جواز السفر؟
السؤال: قديش بدفع على تجديد الهوية؟
الناتج: ما رسوم تجديد الهوية؟
السؤال: كم يوم بياخد الجواز؟
الناتج: ما مدة إنجاز معاملة جواز السفر؟
السؤال: وين بقدر اقدم على رخصة السواقة؟
الناتج: ما مكان وقنوات تقديم طلب رخصة القيادة؟"""


@lru_cache(maxsize=512)
def _msa(question: str, model: str) -> str:
    out = chat([{"role": "system", "content": _MSA_PROMPT},
                {"role": "user", "content": f"السؤال: {question}\nالناتج:"}], model=model, max_tokens=60, temperature=0)
    out = normalize_display(out.splitlines()[0] if out else "").strip(" \"'«»")
    out = re.sub(r"^(الناتج|السؤال)\s*[:：]\s*", "", out)
    if not out or len(out) > 3 * len(question) + 40 or not is_arabic(out):
        return question                             # useless rewrite -> keep the original
    return out


def rewrite_to_msa(question: str, model: str | None = None) -> str:
    """Dialect / colloquial Arabic question -> short Modern Standard Arabic search query. Cached.
    Non-Arabic questions are returned unchanged (no LLM call)."""
    question = normalize_display(question)
    if not is_arabic(question):
        return question
    return _msa(question, model or MODEL)


# ---------------------------------------------------------------- follow-ups
_STANDALONE_PROMPT = """أعد صياغة آخر سؤال للمستخدم ليصبح سؤالاً كاملاً مستقلاً يُفهم بدون المحادثة السابقة، بإضافة اسم الخدمة أو الموضوع المحذوف.
بنفس لغة المستخدم. إذا كان السؤال مستقلاً أصلاً فأعده كما هو. اكتب السؤال فقط في سطر واحد، بدون أي شرح."""

_STANDALONE_SHOTS = [      # (conversation, last question, standalone)
    ("المستخدم: كم رسوم جواز السفر لمدة 10 سنوات؟\nالمساعد: 40 دينارا [1]", "وكم مدة الإنجاز؟",
     "كم مدة إنجاز جواز السفر لمدة 10 سنوات؟"),
    ("المستخدم: ما رسوم إصدار بطاقة الهوية لأول مرة؟\nالمساعد: 10 دنانير [1]", "وبدل الفاقد؟",
     "ما رسوم بدل فاقد لبطاقة الهوية؟"),
    ("المستخدم: ما رسوم رخصة القيادة؟\nالمساعد: 20 دينارا [1]", "كم رسوم جواز السفر؟",
     "كم رسوم جواز السفر؟"),
]


def _standalone_msg(convo: str, question: str) -> str:
    return f"المحادثة:\n{convo}\n\nآخر سؤال: {question}"


def standalone_question(question: str, history: list[tuple[str, str]], model: str | None = None) -> str:
    """"وكم الرسوم؟" + history about passport renewal -> "كم رسوم تجديد جواز السفر؟". No history -> unchanged.
    Uses all of `history`: the caller decides how many turns (Agent.rag CONFIG["history_turns"])."""
    question = normalize_display(question)
    if not history:
        return question
    convo = "\n".join(f"المستخدم: {q}\nالمساعد: {a[:300]}" for q, a in history)
    msgs = [{"role": "system", "content": _STANDALONE_PROMPT}]
    for c, q, a in _STANDALONE_SHOTS:
        msgs += [{"role": "user", "content": _standalone_msg(c, q)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": _standalone_msg(convo, question)})
    out = chat(msgs, model=model, max_tokens=80, temperature=0)
    out = normalize_display(out.splitlines()[0] if out else "").strip(" \"'«»")
    out = re.sub(r"^(السؤال المستقل|الناتج)\s*[:：]\s*", "", out)
    if not out or len(out) > 4 * len(question) + 120 or (is_arabic(question) and not is_arabic(out)):
        return question                             # empty / rambling / English reasoning -> keep the original
    return out
