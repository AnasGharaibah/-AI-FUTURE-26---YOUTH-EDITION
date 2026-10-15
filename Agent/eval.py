"""Evaluation.   uv run python -m Agent.eval [--models qwen2.5:7b,qwen3:8b] [--modes rag,agent] [--csvs v1,v2] [--tag P1-]

For every row of questions_<v>.csv: answer it, then check
  * expected: tokens joined by "|" or "&" must ALL occur; "a/b" = either; ">=2:a/b/c" = at least 2 of them
    (normalize_search on both sides). NOT_FOUND rows must be refused.
  * type "meta": must NOT call search_documents (answered from the Service Map)
  * type "ambiguous": all cases must appear AND a follow-up question ("؟") must come after them
  * retrieval: the expected source file is among the top-5 retrieved chunks
Per run: accuracy by type, % of service questions where search was called, tool-call correctness, avg tool calls,
median / p90 seconds, peak RAM (this process + Ollama, psutil) and peak GPU memory (nvidia-smi, if present).
Results go to results/<tag><csv>-<mode>-<model>.json;  `--table [prefix]` prints the comparison.
"""
import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DB.arabic import normalize_search as norm  # noqa: E402

REFUSALS = ("لم اجد هذه المعلومه", "i could not find", "could not find this information", "no relevant documents",
            "do not contain", "does not contain", "not mentioned", "لا تتوفر", "لا توجد معلومات")
NO_SEARCH_TYPES = {"meta"}                              # every other type is a service question that must search
SUMMARY_TYPES = {"summarize"}                           # grounded by search_documents OR get_document_summary


def tool_use_ok(typ: str, tools) -> bool:
    tools = tools or []
    if typ in NO_SEARCH_TYPES:
        return "search_documents" not in tools
    if typ in SUMMARY_TYPES:
        return "search_documents" in tools or "get_document_summary" in tools
    return "search_documents" in tools


def is_refusal(answer: str) -> bool:
    a = norm(answer)
    return any(norm(r) in a for r in REFUSALS)


def _token_positions(answer_norm: str, expected: str):
    """For each expected token: position of the first matching alternative in the answer, or None if not enough
    alternatives match (">=N:" needs N)."""
    out = []
    for tok in re.split(r"[|&]", expected):
        m = re.match(r"\s*>=(\d+):(.*)", tok)
        need, alts = (int(m.group(1)), m.group(2)) if m else (1, tok)
        found = [answer_norm.find(norm(a).strip()) for a in alts.split("/") if norm(a).strip()]
        found = [p for p in found if p >= 0]
        out.append(min(found) if len(found) >= need else None)
    return out


def check(answer: str, expected: str, typ: str = "", tools=()) -> bool:
    if expected == "NOT_FOUND":
        return is_refusal(answer)
    if is_refusal(answer) and len(answer) < 80:
        return False
    a = norm(answer)
    pos = _token_positions(a, expected)
    if any(p is None for p in pos):
        return False
    if typ in NO_SEARCH_TYPES | SUMMARY_TYPES and not tool_use_ok(typ, tools):
        return False
    if typ == "ambiguous":                              # every case answered, THEN one follow-up question
        return max(a.rfind("؟"), a.rfind("?")) > max(pos)
    return True


# ---------------------------------------------------------------- resources
class PeakMemory:
    """Samples RSS of this process + all Ollama processes (and GPU memory via nvidia-smi) in a thread."""

    def __init__(self, every: float = 0.5):
        self.every, self.ram, self.vram, self._stop = every, 0.0, None, threading.Event()

    def _sample(self):
        import psutil
        procs = [psutil.Process()] + [p for p in psutil.process_iter(["name"])
                                      if (p.info["name"] or "").lower().startswith("ollama")]
        rss = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except Exception:
                pass
        self.ram = max(self.ram, rss / 1e9)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, encoding="utf-8", errors="replace", timeout=5).stdout
            self.vram = max(self.vram or 0, sum(float(x) for x in out.split()) / 1024)
        except Exception:
            pass

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self._sample()
                self._stop.wait(self.every)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def unload_models():
    """Start every model run with nothing loaded, so its peak RAM is its own."""
    try:
        import ollama
        c = ollama.Client(timeout=30)
        for m in c.ps().get("models", []):
            c.generate(model=m.get("model") or m.get("name"), prompt="", keep_alive=0)
    except Exception as e:
        print(f"(could not unload Ollama models: {e})")


# ---------------------------------------------------------------- runners
def load_runner(mode: str, model: str, args):
    """-> fn(question, thread) -> {"answer", "retrieved", "flags", "tools", "model_calls"}"""
    from Agent import llm, rag
    llm.MODEL = model
    if args.min_sim is not None:
        rag.CONFIG["min_similarity"] = args.min_sim
    if mode == "agent":
        from Agent import brain

        def run(q, thread):
            r = brain.ask_traced(q, thread, model=model)
            return {"flags": [], **r}
        return run

    def run(q, thread):
        r = rag.answer(q, thread, model=model, rerank=args.rerank or None, rewrite=False if args.no_rewrite else None)
        return {"answer": r["answer"], "retrieved": r["retrieved"], "flags": r["flags"],
                "tools": ["search_documents"], "model_calls": None}
    return run


def run_eval(csv_path: Path, mode: str, model: str, args, tag: str, rdir: Path) -> dict:
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:args.limit]
    if not rows:
        print(f"no questions to run in {csv_path} (empty CSV or --limit 0)")
        return {}
    run = load_runner(mode, model, args)
    unload_models()
    print(f"\n===== {tag}  ({len(rows)} questions)")
    out, t_all = [], time.time()
    with PeakMemory() as mem:
        for i, r in enumerate(rows, 1):
            thread = f"{tag}-{r['thread'] or i}"
            t0 = time.time()
            try:
                res = run(r["question"], thread)
            except Exception as e:
                res = {"answer": f"ERROR: {e}", "retrieved": [], "flags": ["error"], "tools": [], "model_calls": None}
            dt = time.time() - t0
            ok = check(res["answer"], r["expected"], r["type"], res["tools"])
            hit = None if not r["source"] else r["source"] in res["retrieved"][:5]
            out.append({**r, **res, "ok": ok, "hit": hit, "seconds": round(dt, 2)})
            print(f"{'PASS' if ok else 'FAIL'} [{r['type']:<11}] {dt:5.1f}s tools={len(res['tools'])} hit={hit}  "
                  f"{r['question']}")
            if not ok:
                print("      ->", res["answer"].replace("\n", " ")[:160])
    summary = summarize(out, tag, model, mode, csv_path.stem, mem)
    rdir.mkdir(exist_ok=True)
    (rdir / f"{tag}.json").write_text(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"{summary['correct']}/{summary['n']} correct, median {summary['median_s']} s, p90 {summary['p90_s']} s, "
          f"peak RAM {summary['peak_ram_gb']} GB  ({time.time() - t_all:.0f}s) -> results/{tag}.json")
    return summary


def summarize(rows, tag, model, mode="rag", csv_name="", mem=None):
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r["ok"])
    hits = [r["hit"] for r in rows if r["hit"] is not None]
    service = [r for r in rows if r["type"] not in NO_SEARCH_TYPES]
    secs = sorted(r["seconds"] for r in rows)
    n = len(rows) or 1                                  # no rows -> zeros, not ZeroDivisionError
    tools = [r.get("tools") or [] for r in rows]
    correct_tool_use = [tool_use_ok(r["type"], t) for r, t in zip(rows, tools)]
    return {"tag": tag, "model": model, "mode": mode, "csv": csv_name, "n": len(rows),
            "correct": sum(r["ok"] for r in rows),
            "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
            "by_type": {t: f"{sum(v)}/{len(v)}" for t, v in types.items()},
            "retrieval_hit@5": round(sum(hits) / len(hits), 3) if hits else None,
            "search_called": round(sum("search_documents" in (r.get("tools") or []) for r in service) / len(service), 3)
            if service else None,
            "tool_use_correct": round(sum(correct_tool_use) / n, 3),
            "avg_tool_calls": round(sum(len(t) for t in tools) / n, 2),
            "avg_seconds": round(sum(secs) / n, 1),
            "median_s": round(statistics.median(secs), 1) if secs else None,
            "p90_s": round(secs[min(len(secs) - 1, int(0.9 * len(secs)))], 1) if secs else None,
            "peak_ram_gb": round(mem.ram, 2) if mem else None,
            "peak_vram_gb": round(mem.vram, 2) if mem and mem.vram is not None else None}


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def table(rdir: Path, prefix: str = ""):
    runs = [json.loads(p.read_text(encoding="utf-8"))["summary"] for p in sorted(rdir.glob(f"{prefix}*.json"))
            if not p.name.endswith(".retrieval.json")]
    types = sorted({t for s in runs for t in s["by_type"]})
    print("| run | model | mode | acc | search% | tool ok | tools/q | hit@5 | med s | p90 s | RAM GB | VRAM GB | "
          + " | ".join(types) + " |")
    print("|" + "---|" * (12 + len(types)))
    for s in runs:
        print(f"| {s['tag']} | {s['model']} | {s.get('mode', 'rag')} | {s.get('correct', '?')}/{s['n']} "
              f"({_pct(s['accuracy'])}) | {_pct(s.get('search_called'))} | {_pct(s.get('tool_use_correct'))} | "
              f"{s.get('avg_tool_calls', '-')} | {_pct(s.get('retrieval_hit@5'))} | {s.get('median_s', '-')} | "
              f"{s.get('p90_s', '-')} | {s.get('peak_ram_gb', '-')} | {s.get('peak_vram_gb', '-')} | "
              + " | ".join(s["by_type"].get(t, "-") for t in types) + " |")


def retrieval_only(args, rdir):
    """Does the right FILE come first? hit@1/@3/@5 + MRR over the answerable questions (no LLM)."""
    from DB.pipeline import Pipeline
    db = Pipeline.shared()
    with open(_csv_path(args.csvs.split(",")[0]), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] and r["type"] not in ("followup", "meta")]
    if not rows:
        print("retrieval-only: no answerable questions with a source")
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


def _csv_path(name: str) -> Path:
    p = Path(name)
    return p if p.suffix == ".csv" else ROOT / f"questions_{name}.csv"


def main():
    import faulthandler
    import os
    if os.getenv("YOTH_DEBUG_STACK"):                 # e.g. YOTH_DEBUG_STACK=120: dump all Python stacks every 120 s
        faulthandler.dump_traceback_later(int(os.environ["YOTH_DEBUG_STACK"]), repeat=True)
    from Agent import llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", default="v1,v2", help="v1,v2 or paths to CSV files")
    ap.add_argument("--models", default=llm.LLM_MODEL, help="comma-separated Ollama models")
    ap.add_argument("--modes", default="rag", help="rag,agent")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--min-sim", type=float)
    ap.add_argument("--tag", default="", help="prefix for the result files, e.g. P1-")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--table", nargs="?", const="", help="print saved runs (optionally only those with this prefix)")
    ap.add_argument("--retrieval-only", action="store_true", help="no LLM: rank quality of the retriever only")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    if args.table is not None:
        return table(rdir, args.table)
    if args.retrieval_only:
        return retrieval_only(args, rdir)
    tags = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            for name in [c.strip() for c in args.csvs.split(",") if c.strip()]:
                path = _csv_path(name)
                tag = f"{args.tag}{path.stem.replace('questions_', '')}-{mode}-{model.replace(':', '-')}"
                if run_eval(path, mode, model, args, tag, rdir):
                    tags.append(tag)
    if tags:
        print()
        table(rdir, args.tag)


if __name__ == "__main__":
    main()
