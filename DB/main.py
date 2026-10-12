"""Command line for the DB layer. Run from the project root:

    uv run python -m DB.main ingest [--force]      parse + chunk + embed + store
    uv run python -m DB.main search "question"     see what the retriever finds
    uv run python -m DB.main context "question"    the exact text the LLM will get
    uv run python -m DB.main chunks [file] [--n 5] inspect chunks
    uv run python -m DB.main stats                 what is in the index
    uv run python -m DB.main compare "question"    structure vs fixed chunking
"""
import argparse
import json

from .pipeline import Pipeline


def show_hits(db, hits):
    for i, h in enumerate(hits, 1):
        extra = f" rerank={h['rerank']}" if "rerank" in h else ""
        print(f"\n[{i}] {db.cite(h)} | {h['section'] or '-'} | {h['found_by']} "
              f"| score={h['score']}{extra}")
        print("    " + h["text"][:300].replace("\n", "\n    "))


def main():
    ap = argparse.ArgumentParser(prog="python -m DB.main")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("--force", action="store_true")
    for name in ("search", "context", "compare"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--rerank", action="store_true")
    p = sub.add_parser("chunks")
    p.add_argument("file", nargs="?")
    p.add_argument("--n", type=int, default=5)
    sub.add_parser("stats")
    a = ap.parse_args()

    if a.cmd == "compare":
        for algo in ("structure", "fixed"):
            db = Pipeline(chunk_algorithm=algo, db_dir=Pipeline().cfg["db_dir"].parent / f"chroma_{algo}")
            db.ingest()
            print(f"\n==================== {algo} ====================")
            show_hits(db, db.search(a.question, rerank=a.rerank))
        return

    db = Pipeline()
    if a.cmd == "ingest":
        db.ingest(force=a.force)
    elif a.cmd == "search":
        show_hits(db, db.search(a.question, rerank=a.rerank or None))
    elif a.cmd == "context":
        text, _ = db.context(a.question)
        print(text)
    elif a.cmd == "stats":
        print(json.dumps(db.stats(), indent=2, ensure_ascii=False))
    elif a.cmd == "chunks":
        rows = db.store._load().values()
        rows = [r for r in rows if not a.file or a.file.lower() in r["source"].lower()]
        print(f"{len(rows)} chunks")
        for r in list(rows)[:a.n]:
            print(f"\n--- {r['id']} | {db.cite(r)} | {r['section'] or '-'} | {r['tokens']} tok | {r['profile']}")
            print(r["text"])


if __name__ == "__main__":
    main()