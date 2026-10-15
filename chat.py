"""Talk to the competition bot in the terminal:   python chat.py   (or: py chat.py)

Every question goes through the same pipeline as main.py and prints the answer, the contexts (what will be scored as
"contexts"), the route and the time. The bot remembers the conversation until you type 'new'.

Everything is saved in chat_logs/ (rewritten after every message):
    chat_log.md     readable log with route / flags / time per message (open it in VS Code if the terminal shows
                    Arabic in the wrong order)
    chat_log.json   all chats in the answers_example.json format: {"chats": [{"chat_id", "messages":
                    [{"message_id", "question", "answer", "contexts"}]}]}

    new    start a new chat (forget the conversation)
    exit   quit
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Agent import ollama_http                           # noqa: E402
from Agent.competition import Chat                      # noqa: E402

LOG_DIR = Path(__file__).resolve().parent / "chat_logs"
LOG_MD = LOG_DIR / "chat_log.md"
LOG_JSON = LOG_DIR / "chat_log.json"


def load_log() -> dict:
    try:
        data = json.loads(LOG_JSON.read_text(encoding="utf-8"))
        if isinstance(data.get("chats"), list):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"chats": []}


def save_log(data: dict):
    tmp = LOG_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, LOG_JSON)


def new_chat(data: dict, md) -> dict:
    chat = {"chat_id": f"chat-{len(data['chats']) + 1:02d}", "messages": []}
    data["chats"].append(chat)
    md.write(f"\n# {chat['chat_id']} ({datetime.now():%Y-%m-%d %H:%M})\n")
    md.flush()
    return chat


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():                          # piped input (the console itself is read as Unicode)
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    LOG_DIR.mkdir(exist_ok=True)
    data = load_log()
    print(f"model {ollama_http.MODEL} - type a question; 'new' = new chat, 'exit' = quit.\nLogs: {LOG_DIR}\n")
    with LOG_MD.open("a", encoding="utf-8", errors="replace") as md:
        bot, chat = Chat(), new_chat(data, md)
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "خروج"}:
                break
            if q.lower() in {"new", "جديد"}:
                bot, chat = Chat(), new_chat(data, md)
                print(f"-- new chat: {chat['chat_id']} --\n")
                continue
            n_calls = len(ollama_http.STATS)
            t0 = time.time()
            try:
                res = bot.ask(q)
            except Exception as e:
                print(f"error: {e}\n")
                continue
            dt = time.time() - t0
            d = res["debug"]
            calls = ollama_http.STATS[n_calls:]
            tokens = ", ".join(f"{c['kind']} {c['prompt_tokens']}/{c['num_ctx']}" for c in calls) or "no LLM call"
            route = ", ".join(d["route"]) or "-"
            print(f"\nbot> {res['answer']}\n")
            print(f"  contexts ({len(res['contexts'])}):")
            for c in res["contexts"]:
                print(f"    - {c}")
            print(f"  route: {route}  [{d['route_source']}{' / ' + d['unsure'] if d['unsure'] else ''}]")
            if d.get("standalone") and d["standalone"] != q:
                print(f"  understood as: {d['standalone']}")
            if d["flags"]:
                print(f"  flags: {', '.join(d['flags'])}")
            print(f"  {dt:.1f}s | prompt tokens: {tokens}\n")

            msg_id = f"{chat['chat_id']}-{len(chat['messages']) + 1}"
            chat["messages"].append({"message_id": msg_id, "question": q, "answer": res["answer"],
                                     "contexts": res["contexts"]})
            save_log(data)
            md.write(f"\n## {msg_id}: {q}\n\n{res['answer']}\n\n**contexts:**\n"
                     + "".join(f"- {c}\n" for c in res["contexts"]) + ("- (none)\n" if not res["contexts"] else "")
                     + f"\n_route: {route} ({d['route_source']})"
                     + (f" | understood as: {d['standalone']}" if d.get("standalone") and d["standalone"] != q else "")
                     + f" | flags: {', '.join(d['flags']) or '-'} | {dt:.1f}s | {tokens}_\n")
            md.flush()
    if data["chats"] and not data["chats"][-1]["messages"]:   # don't keep an empty last chat
        data["chats"].pop()
    save_log(data)


if __name__ == "__main__":
    main()
