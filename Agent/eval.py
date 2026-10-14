"""Arabic evaluation.   uv run python -m Agent.eval [--model qwen3:4b] [--rerank] [--no-rewrite] [--tag name]

For every row of questions_ar.csv: answer it, then check
  * answer: all `expected` tokens ("|" = all required, "/" = alternatives) occur after normalize_search on both sides;
            NOT_FOUND rows must be refused
  * retrieval: the expected source file is among the top-5 retrieved chunks
Results go to results/<tag>.json;  `--table` prints the comparison of all saved runs.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from DB.arabic import normalize_search as norm
except ImportError:                                   # baseline snapshot without DB/arabic.py
    import re

    def norm(t):
        t = re.sub("[ً-ْٰـ]", "", t)
        return t.translate(str.maketrans("أإآٱىةؤئ٠١٢٣٤٥٦٧٨٩", "اااايهوي0123456789")).lower()

REFUSALS = ("لم اجد هذه المعلومه", "i could not find", "could not find this information", "no relevant documents",
            "do not contain", "does not contain", "not mentioned", "لا تتوفر", "لا توجد معلومات")


def is_refusal(answer: str) -> bool:
    a = norm(answer)
    return any(norm(r) in a for r in REFUSALS)


def check(answer: str, expected: str) -> bool:
    if expected == "NOT_FOUND":
        return is_refusal(answer)
    if is_refusal(answer) and len(answer) < 80:
        return False
    a = norm(answer)
    return all(any(norm(alt).strip() in a for alt in tok.split("/")) for tok in expected.split("|"))


def load_runner(args):
    """-> fn(question, thread) -> (answer, retrieved_files, flags)"""
    if args.legacy:                                   # old tool-calling agent + old DB (baseline snapshot)
        from Agent.brain import MODEL, ask
        from DB.pipeline import Pipeline
        db = Pipeline.shared()

        def run(q, thread):
            hits = db.search(q)
            return ask(q, thread), [h["source"] for h in hits], []
        return run, MODEL

    from Agent import llm, rag
    from DB.pipeline import Pipeline
    if args.model:
        llm.MODEL = args.model
    if args.min_sim is not None:
        rag.CONFIG["min_similarity"] = args.min_sim

    def run(q, thread):
        r = rag.answer(q, thread, model=llm.MODEL, rerank=args.rerank or None,
                       rewrite=False if args.no_rewrite else None)
        return r["answer"], r["retrieved"], r["flags"]
    return run, llm.MODEL


def main():
    import faulthandler
    import os
    if os.getenv("YOTH_DEBUG_STACK"):                 # e.g. YOTH_DEBUG_STACK=120: dump all Python stacks every 120 s
        faulthandler.dump_traceback_later(int(os.environ["YOTH_DEBUG_STACK"]), repeat=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "questions_ar.csv"))
    ap.add_argument("--model")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="baseline: old agent + old DB")
    ap.add_argument("--min-sim", type=float)
    ap.add_argument("--tag")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true", help="no LLM: rank quality of the retriever only")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    if args.table:
        return table(rdir)

    if args.retrieval_only:
        return retrieval_only(args, rdir)
    with open(args.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:args.limit]
    if not rows:
        print("no questions to run (empty CSV or --limit 0)")
        return
    run, model = load_runner(args)
    out, t_all = [], time.time()
    for i, r in enumerate(rows, 1):
        thread = f"eval-{r['thread'] or i}"
        t0 = time.time()
        try:
            ans, retrieved, flags = run(r["question"], thread)
        except Exception as e:
            ans, retrieved, flags = f"ERROR: {e}", [], ["error"]
        dt = time.time() - t0
        ok = check(ans, r["expected"])
        hit = None if not r["source"] else r["source"] in retrieved[:5]
        out.append({**r, "answer": ans, "ok": ok, "hit": hit, "seconds": round(dt, 1), "flags": flags})
        print(f"{'PASS' if ok else 'FAIL'} [{r['type']:<11}] {dt:5.1f}s hit={hit}  {r['question']}")
        if not ok:
            print("      ->", ans.replace("\n", " ")[:160])

    tag = args.tag or ("legacy" if args.legacy else f"{model.replace(':', '-')}{'+rerank' if args.rerank else ''}"
                       f"{'-norewrite' if args.no_rewrite else ''}")
    rdir.mkdir(exist_ok=True)
    summary = summarize(out, tag, model)
    (rdir / f"{tag}.json").write_text(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"total {time.time() - t_all:.0f}s -> results/{tag}.json")


def retrieval_only(args, rdir):
    """Does the right FILE come first? hit@1/@3/@5 + MRR over the answerable questions. Follow-ups are skipped
    (they need the LLM). No rewrite unless --rewrite-llm... keeps RAM free for the reranker test."""
    from DB.pipeline import Pipeline
    db = Pipeline.shared()
    with open(args.csv, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] and r["type"] != "followup"]
    if not rows:
        print("retrieval-only: no answerable questions with a source in", args.csv)
        return
    ranks, t0 = [], time.time()
    for r in rows:
        hits = db.search(r["question"], k=5, rerank=args.rerank or False, rewrite=False)
        files = [h["source"] for h in hits]
        ranks.append(files.index(r["source"]) + 1 if r["source"] in files else 0)
    n = len(ranks)
    res = {"tag": args.tag or ("retrieval" + ("+rerank" if args.rerank else "")), "n": n,
           "hit@1": round(sum(1 <= x <= 1 for x in ranks) / n, 3), "hit@3": round(sum(1 <= x <= 3 for x in ranks) / n, 3),
           "hit@5": round(sum(x > 0 for x in ranks) / n, 3), "mrr": round(sum(1 / x for x in ranks if x) / n, 3),
           "seconds_per_query": round((time.time() - t0) / n, 2)}
    rdir.mkdir(exist_ok=True)
    (rdir / f"{res['tag']}.retrieval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def summarize(rows, tag, model):
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r["ok"])
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    n = len(rows) or 1                                  # no rows -> zeros, not ZeroDivisionError
    return {"tag": tag, "model": model, "n": len(rows),
            "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
            "by_type": {t: f"{sum(v)}/{len(v)}" for t, v in types.items()},
            "retrieval_hit@5": round(sum(hits) / len(hits), 3) if hits else None,
            "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1)}


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def table(rdir):
    runs = [json.loads(p.read_text(encoding="utf-8"))["summary"] for p in sorted(rdir.glob("*.json")) if not p.name.endswith(".retrieval.json")]
    types = sorted({t for s in runs for t in s["by_type"]})
    print("| run | model | acc | hit@5 | s/q | " + " | ".join(types) + " |")
    print("|" + "---|" * (5 + len(types)))
    for s in runs:
        print(f"| {s['tag']} | {s['model']} | {_pct(s['accuracy'])} | {_pct(s.get('retrieval_hit@5'))} | {s['avg_seconds']} | "
              + " | ".join(s["by_type"].get(t, "-") for t in types) + " |")


if __name__ == "__main__":
    main()
