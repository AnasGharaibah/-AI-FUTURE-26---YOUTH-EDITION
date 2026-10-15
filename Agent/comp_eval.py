"""Evaluate the competition pipeline.

    python -m Agent.comp_eval                                # eval/comp_eval.json, Arabic answer prompt
    python -m Agent.comp_eval --lang en --tag en             # A/B: English instructions in the ANSWER prompt
    python -m Agent.comp_eval --file eval/smoke.json --show  # smoke: print every answer + its contexts
    python -m Agent.comp_eval --table                        # compare saved runs (results/comp_*.json)

Metrics: route accuracy, answer accuracy, % LLM answers with >= 1 citation, % answers whose numbers are all in their
contexts (or the question), seconds per message (avg / p90), max prompt tokens vs num_ctx.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DB.arabic import check_numbers, normalize_search, number_words  # noqa: E402

from Agent import competition, ollama_http  # noqa: E402


def answer_ok(m: dict, ans: str, contexts: list[str]) -> bool | None:
    exp = m.get("expect")
    if not exp:
        return None
    if exp == "OOS":
        return not contexts and (ans == competition.OUT_OF_SCOPE or competition.is_not_found(ans))
    if exp == "CLARIFY":
        return not contexts and ("؟" in ans or "?" in ans)
    a = normalize_search(ans) + " " + " ".join(number_words(ans))     # "ثلاثة دنانير" also counts as 3
    return all(any(normalize_search(alt).strip() in a for alt in tok.split("/")) for tok in exp.split("|"))


def route_ok(m: dict, services: list[str]) -> bool | None:
    if "services" not in m:
        return None
    req, extra, anyof = set(m["services"]), set(m.get("allowed", [])), set(m.get("services_any", []))
    got = set(services)
    if anyof and not got & anyof:
        return False
    return req <= got <= req | extra | anyof


def p90(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)] if xs else 0.0


def run(args) -> dict:
    competition.PROMPT_LANG = args.lang
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    rows, n_calls0 = [], len(ollama_http.STATS)
    chats = data["chats"][:args.limit] if args.limit else data["chats"]
    for chat in chats:
        bot = competition.Chat()
        for m in chat["messages"]:
            t0 = time.time()
            try:
                res = bot.ask(m["question"])
            except Exception as e:                          # keep going; count it as wrong
                res = {"answer": f"ERROR: {e}", "contexts": [], "debug": {"route": [], "flags": ["error"]}}
            dt = time.time() - t0
            ans, ctx, dbg = res["answer"], res["contexts"], res["debug"]
            llm_answered = not any(f in dbg.get("flags", []) for f in ("out_of_scope", "clarify", "error")) \
                and not competition.is_not_found(ans)
            has_num = bool(check_numbers(ans, ""))         # every number in the answer, as "not in an empty source"
            row = {"message_id": m["message_id"], "type": m.get("type", ""), "question": m["question"],
                   "answer": ans, "contexts": ctx, "route": dbg.get("route", []),
                   "route_source": dbg.get("route_source"), "flags": dbg.get("flags", []),
                   "route_ok": route_ok(m, dbg.get("route", [])), "ok": answer_ok(m, ans, ctx),
                   "llm_answered": llm_answered, "cited": bool(dbg.get("cited")),
                   "has_numbers": has_num,
                   "numbers_grounded": (not check_numbers(ans, "\n".join(ctx) + "\n" + m["question"])) if has_num else None,
                   "seconds": round(dt, 1)}
            rows.append(row)
            mark = {True: "PASS", False: "FAIL", None: "----"}[row["ok"]]
            rmark = {True: "", False: " ROUTE-FAIL", None: ""}[row["route_ok"]]
            print(f"{mark} [{row['type']:<16}] {dt:5.1f}s {row['route_source']:<16} {row['route']}{rmark}  {m['question']}")
            if args.show or row["ok"] is False:
                print("      answer:  ", ans.replace("\n", " / "))
                for c in ctx:
                    print("      context: ", c[:220])
                if row["flags"]:
                    print("      flags:   ", row["flags"])
    calls = ollama_http.STATS[n_calls0:]
    return {"summary": summarize(rows, calls, args), "rows": rows, "llm_calls": calls}


def _pct(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def summarize(rows, calls, args) -> dict:
    by_type = {}
    for r in rows:
        if r["ok"] is not None:
            by_type.setdefault(r["type"], []).append(r["ok"])
    llm = [r for r in rows if r["llm_answered"]]
    secs = [r["seconds"] for r in rows]
    worst = max(calls, key=lambda c: max(c["prompt_tokens"], c["estimated"]) / c["num_ctx"], default=None)
    return {
        "tag": args.tag or f"{args.lang}", "model": ollama_http.MODEL, "prompt_lang": args.lang, "file": args.file,
        "n": len(rows),
        "route_accuracy": _pct(r["route_ok"] for r in rows),
        "answer_accuracy": _pct(r["ok"] for r in rows),
        "cited_pct": _pct(r["cited"] for r in llm),
        "numbers_grounded_pct": _pct(r["numbers_grounded"] for r in rows),
        "sec_avg": round(sum(secs) / len(secs), 1) if secs else 0, "sec_p90": p90(secs),
        "max_prompt_tokens": max((c["prompt_tokens"] for c in calls), default=0),
        "num_ctx_at_max": worst["num_ctx"] if worst else None,
        "llm_calls": len(calls), "router_calls": sum(c["kind"] == "router" for c in calls),
        "by_type": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())},
    }


def table(rdir: Path):
    runs = [json.loads(p.read_text(encoding="utf-8"))["summary"] for p in sorted(rdir.glob("comp_*.json"))]
    cols = ("tag", "prompt_lang", "n", "route_accuracy", "answer_accuracy", "cited_pct", "numbers_grounded_pct",
            "sec_avg", "sec_p90", "max_prompt_tokens", "num_ctx_at_max")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in runs:
        print("| " + " | ".join(str(s.get(c)) for c in cols) + " |")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "eval/comp_eval.json"))
    ap.add_argument("--lang", default=competition.PROMPT_LANG, choices=["ar", "en"])
    ap.add_argument("--tag")
    ap.add_argument("--limit", type=int, help="first N chats")
    ap.add_argument("--show", action="store_true", help="print every answer with its contexts")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)
    if args.table:
        return table(rdir)
    out = run(args)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    share = ollama_http.gpu_share()
    print(f"ollama ps: {ollama_http.MODEL} {'not loaded' if share is None else f'{share:.0%} GPU'}")
    if args.tag:
        rdir.mkdir(exist_ok=True)
        (rdir / f"comp_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"-> results/comp_{args.tag}.json")


if __name__ == "__main__":
    main()
