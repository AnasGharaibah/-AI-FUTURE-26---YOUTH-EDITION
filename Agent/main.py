"""Terminal chat.   uv run python -m Agent.main        (YOTH_MODEL=phi4-mini to switch model)
Voice chat.      uv run python -m Agent.main --voice [--lang ar|en|auto]
"""
import argparse
import uuid

from .brain import MODEL, ask, stream


def voice(lang: str | None):
    from .voice import voice_loop
    thread = uuid.uuid4().hex
    print(f"Yoth AI ({MODEL}) - voice mode")
    voice_loop(lambda q: ask(q, thread), lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true", help="speak your questions, hear the answers")
    ap.add_argument("--lang", default="ar", choices=["ar", "en", "auto"], help="speech language (auto = mixed)")
    args = ap.parse_args()
    if args.voice:
        return voice(None if args.lang == "auto" else args.lang)

    thread = uuid.uuid4().hex
    print(f"Yoth AI ({MODEL}) - type 'exit' to quit\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print("ai> ", end="", flush=True)
        try:
            for kind, val in stream(q, thread):
                if kind == "sources":
                    print("\n   sources: " + "; ".join(f"[{x['n']}] {x['file']}" + (f" p.{x['page']}" if x["page"] else "")
                                                       for x in val), end="", flush=True)
                    continue
                print(f"\n   [tool: {val}]\nai> " if kind == "tool" else val, end="", flush=True)
        except KeyboardInterrupt:                       # Ctrl+C while answering: stop this answer, keep the chat
            print("\n   (stopped)", end="")
        print("\n")


if __name__ == "__main__":
    main()
