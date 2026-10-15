"""Talk to the competition bot in the terminal:   python chat.py   (or: py chat.py)

Every question goes through the same pipeline as main.py and prints the answer, the contexts (what will be scored as
"contexts"), the route and the time. Each turn is also appended to chat_log.md - open it in VS Code if the terminal
shows Arabic in the wrong order.

    new    start a new chat (forget the conversation)
    exit   quit
"""
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Agent import ollama_http                           # noqa: E402
from Agent.competition import Chat                      # noqa: E402

LOG = Path(__file__).resolve().parent / "chat_log.md"


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():                          # piped input (the console itself is read as Unicode)
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print(f"model {ollama_http.MODEL} - type a question; 'new' = new chat, 'exit' = quit. Log: {LOG}\n")
    bot, turn = Chat(), 0
    with LOG.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n# Chat {datetime.now():%Y-%m-%d %H:%M}\n")
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
                bot, turn = Chat(), 0
                print("-- new chat --\n")
                log.write(f"\n# Chat {datetime.now():%Y-%m-%d %H:%M}\n")
                continue
            turn += 1
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
            log.write(f"\n## {turn}. {q}\n\n{res['answer']}\n\n**contexts:**\n"
                      + "".join(f"- {c}\n" for c in res["contexts"]) + ("- (none)\n" if not res["contexts"] else "")
                      + f"\n_route: {route} ({d['route_source']}) | flags: {', '.join(d['flags']) or '-'} | "
                        f"{dt:.1f}s | {tokens}_\n")
            log.flush()


if __name__ == "__main__":
    main()
