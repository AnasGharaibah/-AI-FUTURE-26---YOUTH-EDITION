"""Competition entry point:   python main.py questions.json   ->   answers.json (same folder)

Reads the chats, answers every message in order (each chat is a new user), adds "answer" and "contexts" to every
message and keeps everything else unchanged. answers.json is rewritten after every message, so a crash still leaves
a valid partial file. Only the standard library + a running Ollama are needed (no venv required).

    YOTH_MODEL=command-r7b-arabic:7b      the chat model (default)
    YOTH_DEBUG=1                          also write answers.debug.json (route, flags, timings per message)
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Agent import ollama_http                           # noqa: E402
from Agent.competition import Chat                      # noqa: E402


def _save(data: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):              # Arabic on a Windows console (cp1252) must not crash
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    qpath = Path(sys.argv[1] if len(sys.argv) > 1 else "questions.json")
    if not qpath.is_file():
        print(f"questions file not found: {qpath.resolve()}\n"
              f"usage: python main.py questions.json   (practice file: python main.py eval/practice/questions.json)")
        return 2
    data = json.loads(qpath.read_text(encoding="utf-8-sig"))
    out = qpath.resolve().parent / "answers.json"
    debug = []
    n_msgs = sum(len(c.get("messages", [])) for c in data.get("chats", []))
    print(f"model {ollama_http.MODEL} | {len(data.get('chats', []))} chats, {n_msgs} messages -> {out}")

    t_all, done = time.time(), 0
    for chat in data.get("chats", []):
        bot = Chat()                                    # every chat is a new user
        for msg in chat.get("messages", []):
            done += 1
            t0 = time.time()
            try:
                res = bot.ask(msg.get("question", ""))
                msg["answer"], msg["contexts"] = res["answer"], res["contexts"]
                debug.append({"message_id": msg.get("message_id"), **res["debug"]})
            except Exception as e:                      # one bad message must not stop the run
                logging.exception("message %s failed", msg.get("message_id"))
                msg["answer"], msg["contexts"] = "عذراً، حدث خطأ أثناء معالجة هذا السؤال.", []
                bot.history.append((msg.get("question", ""), msg["answer"]))
                debug.append({"message_id": msg.get("message_id"), "error": repr(e)})
            print(f"[{done}/{n_msgs}] {msg.get('message_id')} {time.time() - t0:5.1f}s  {msg['answer'][:80]!r}")
            _save(data, out)
    for chat in data.get("chats", []):                  # every message has both keys, whatever happened
        for msg in chat.get("messages", []):
            msg.setdefault("answer", "error")
            msg.setdefault("contexts", [])
    _save(data, out)
    if os.getenv("YOTH_DEBUG") == "1":
        _save({"messages": debug, "llm_calls": ollama_http.STATS}, out.with_name("answers.debug.json"))
    print(f"done: {done} messages in {time.time() - t_all:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
