"""Ollama /api/chat over plain HTTP (stdlib only, so `python main.py` runs without the venv).

Fixed settings for the competition model (command-r7b-arabic: not a reasoning model, so no `think` parameter):
num_ctx 8192, raised to 12288 / 16384 only for prompts above ~7k tokens. Every call logs prompt_eval_count and warns
when it is above 90% of num_ctx (Ollama silently cuts prompts that do not fit).

Server side (set before `ollama serve` starts): OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0
(and OLLAMA_IGPU_ENABLE=1 for an integrated GPU). Check `ollama ps` -> 100% GPU.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("yoth.ollama")

MODEL = os.getenv("YOTH_MODEL", "command-r7b-arabic:7b")
HOST = os.getenv("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
KEEP_ALIVE = "30m"
OPTIONS = {"num_ctx": 8192, "temperature": 0.0, "top_p": 1.0, "num_predict": 450, "repeat_penalty": 1.05}
CTX_STEPS = (8192, 12288, 16384)            # never above 16384 (the model's context)
CHARS_PER_TOKEN = 1.9                       # measured for this model on the service documents (Arabic)
SOFT_LIMIT = 7000                           # estimated prompt tokens above this -> next num_ctx step
TIMEOUT = float(os.getenv("YOTH_LLM_TIMEOUT", "300"))

STATS: list[dict] = []                      # one entry per call: {kind, prompt_tokens, num_ctx, seconds, ...}


def estimate_tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 20 * len(messages)


def pick_num_ctx(messages: list[dict], num_predict: int) -> int:
    need = estimate_tokens(messages)
    if need <= SOFT_LIMIT:
        return CTX_STEPS[0]
    return next((c for c in CTX_STEPS[1:] if need + num_predict <= 0.9 * c), CTX_STEPS[-1])


def _post(body: dict) -> dict:
    req = urllib.request.Request(f"{HOST}/api/chat", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages: list[dict], format: dict | str | None = None, kind: str = "answer", model: str | None = None,
         **options) -> str:
    """messages -> reply text. format: a JSON schema (structured output) or None for plain text."""
    opts = {**OPTIONS, **options}
    opts["num_ctx"] = max(opts["num_ctx"], pick_num_ctx(messages, opts["num_predict"]))
    body = {"model": model or MODEL, "messages": messages, "stream": False, "keep_alive": KEEP_ALIVE, "options": opts}
    if format is not None:
        body["format"] = format
    t0 = time.time()
    try:
        res = _post(body)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        if e.code == 404:
            raise RuntimeError(f"Model missing. Run: ollama pull {body['model']}") from e
        raise RuntimeError(f"Ollama error {e.code}: {msg[:300]}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        log.warning("Ollama request failed (%s) - retrying once", e)
        time.sleep(2)
        res = _post(body)
    stat = {"kind": kind, "prompt_tokens": res.get("prompt_eval_count", 0), "estimated": estimate_tokens(messages),
            "output_tokens": res.get("eval_count", 0), "num_ctx": opts["num_ctx"], "seconds": round(time.time() - t0, 2)}
    STATS.append(stat)
    log.info("%s: prompt %s tokens (est. %s) / num_ctx %s, output %s tokens, %.1fs", kind, stat["prompt_tokens"],
             stat["estimated"], stat["num_ctx"], stat["output_tokens"], stat["seconds"])
    if max(stat["prompt_tokens"], stat["estimated"]) > 0.9 * opts["num_ctx"]:
        log.warning("%s prompt uses %s of num_ctx %s (> 90%%) - it may be cut off", kind,
                    max(stat["prompt_tokens"], stat["estimated"]), opts["num_ctx"])
    return (res.get("message") or {}).get("content") or ""


def ps() -> list[dict]:
    """Loaded models (`ollama ps`): [{"name", "size", "size_vram", "context_length", ...}]."""
    with urllib.request.urlopen(f"{HOST}/api/ps", timeout=10) as r:
        return json.loads(r.read().decode("utf-8")).get("models", [])


def gpu_share(model: str | None = None) -> float | None:
    """Share of the loaded model in GPU memory (1.0 = `100% GPU`), None if not loaded."""
    for m in ps():
        if m.get("name") == (model or MODEL) or m.get("model") == (model or MODEL):
            return m.get("size_vram", 0) / m["size"] if m.get("size") else None
    return None
