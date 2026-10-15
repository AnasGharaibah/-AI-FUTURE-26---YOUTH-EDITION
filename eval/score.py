"""Score an answers.json against a rubric file (test_expected.json), the way the brief judges:

    python eval/score.py answers.json test_expected.json [--show-fails]

Per message (rubric: behavior, services, source_mode, must_include, must_not_include, allowed_numbers, bonus):
  correct   behavior "answer": every must_include group has one alternative in the answer (digits match whole
            numbers; number words count as digits) and no must_not_include phrase is in it, and it is not a
            refusal / question back; "decline": refused, contexts []; "clarify": asks back, no facts cited;
            "any": a non-empty, non-error answer
  source    contexts come from the expected services ("all": every expected service is cited, "any": at least
            one) and from no other service; decline / clarify: contexts []
  grounded  every number in the answer is in its contexts, the question or allowed_numbers
Overall = 60% correct + 20% source + 20% grounded (the brief: "correct answer ... matters most"; the weights are
our estimate, the organisers did not publish theirs). Bonus (conversation) messages are reported separately.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Agent.competition import OUT_OF_SCOPE, is_not_found  # noqa: E402
from Agent.services import NAMES  # noqa: E402
from DB.arabic import check_numbers, normalize_search, number_words  # noqa: E402

WEIGHTS = {"correct": 0.6, "source": 0.2, "grounded": 0.2}
_BY_NAME = sorted(((name, sid) for sid, name in NAMES.items()), key=lambda x: -len(x[0]))


def context_service(ctx: str) -> str | None:
    """"تجديد جواز السفر - الرسوم: ..." -> "passport_renewal"."""
    return next((sid for name, sid in _BY_NAME if ctx.startswith(name + " - ")), None)


def has(answer_norm: str, alt: str) -> bool:
    alt = normalize_search(alt).strip()
    if alt.isdigit():                                   # "2" must not match "20" or "2025"
        return re.search(rf"(?<!\d){alt}(?!\d)", answer_norm) is not None
    return alt in answer_norm


def is_decline(ans: str) -> bool:
    return ans.strip() == OUT_OF_SCOPE or is_not_found(ans) or normalize_search("لا تتوفر لدي") in normalize_search(ans)


def asks_back(ans: str) -> bool:
    return "؟" in ans or "?" in ans


def score_one(msg: dict, rule: dict) -> dict:
    ans, ctx = msg.get("answer", ""), msg.get("contexts", []) or []
    a = normalize_search(ans) + " " + " ".join(number_words(ans))
    behavior = rule["behavior"]
    reasons = []

    # 1. correct
    if behavior == "answer":
        missing = [g for g in rule["must_include"] if not any(has(a, alt) for alt in g)]
        forbidden = [p for p in rule["must_not_include"] if has(a, p)]
        declined = is_decline(ans) and not ctx
        correct = not missing and not forbidden and not declined and ans.strip() not in ("", "error")
        if missing:
            reasons.append("missing " + " / ".join("|".join(g) for g in missing))
        if forbidden:
            reasons.append("contains " + ", ".join(forbidden))
        if declined:
            reasons.append("declined instead of answering")
    elif behavior == "decline":
        forbidden = [p for p in rule["must_not_include"] if has(a, p)]
        correct = is_decline(ans) and not ctx and not forbidden
        if not correct:
            reasons.append("should decline")
    elif behavior == "clarify":
        correct = asks_back(ans) and not is_decline(ans) and not ctx
        if not correct:
            reasons.append("should ask which document/situation" + (" (answered with sources)" if ctx else ""))
    else:                                               # "any"
        correct = bool(ans.strip()) and ans.strip() != "error"

    # 2. source
    got = [context_service(c) for c in ctx]
    want = set(rule["services"])
    if behavior in ("decline", "clarify") or not want:
        source = not ctx if behavior in ("decline", "clarify") else True
    else:
        known = {g for g in got if g}
        cover = want <= known if rule["source_mode"] == "all" else bool(want & known)
        clean = all(g in want for g in got)
        source = bool(ctx) and cover and clean
        if not source:
            reasons.append(f"sources {sorted(known)} vs expected {sorted(want)}"
                           + ("" if clean else " (+ unrelated)"))

    # 3. grounded
    evidence = "\n".join(ctx) + "\n" + msg.get("question", "") + "\n" + " ".join(rule["allowed_numbers"])
    bad = check_numbers(ans, evidence)
    grounded = not bad
    if bad:
        reasons.append("numbers not in contexts: " + ", ".join(bad))

    total = WEIGHTS["correct"] * correct + WEIGHTS["source"] * source + WEIGHTS["grounded"] * grounded
    return {"correct": correct, "source": source, "grounded": grounded, "score": total, "reasons": reasons}


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "    -"


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("expected")
    ap.add_argument("--show-fails", action="store_true")
    args = ap.parse_args()
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    rules = json.loads(Path(args.expected).read_text(encoding="utf-8"))["cases"]
    msgs = {m["message_id"]: m for c in answers["chats"] for m in c["messages"]}

    rows = []
    for mid, rule in rules.items():
        m = msgs.get(mid)
        r = score_one(m, rule) if m else {"correct": False, "source": False, "grounded": False, "score": 0.0,
                                          "reasons": ["NO ANSWER in answers.json"]}
        rows.append({"id": mid, "type": rule["type"], "behavior": rule["behavior"], "bonus": rule["bonus"],
                     "question": (m or {}).get("question", ""), "answer": (m or {}).get("answer", ""), **r})

    def block(title, sel):
        print(f"\n{title} ({len(sel)} messages)")
        print(f"  correct answer {pct(r['correct'] for r in sel)}   right source {pct(r['source'] for r in sel)}   "
              f"grounded {pct(r['grounded'] for r in sel)}   => overall {pct(r['score'] for r in sel)}")

    main_rows = [r for r in rows if not r["bonus"]]
    bonus_rows = [r for r in rows if r["bonus"]]
    block("MAIN (scored)", main_rows)
    block("BONUS (conversation)", bonus_rows)
    block("ALL", rows)

    print("\nby type                 n   correct   source  grounded   overall")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for t, sel in sorted(by.items(), key=lambda x: sum(r["score"] for r in x[1]) / len(x[1])):
        print(f"  {t:<20} {len(sel):>3}  {pct(r['correct'] for r in sel)}  {pct(r['source'] for r in sel)}  "
              f"{pct(r['grounded'] for r in sel)}  {pct(r['score'] for r in sel)}")

    fails = [r for r in rows if r["reasons"] or not r["correct"]]
    print(f"\n{len(fails)} messages lose points" + ("" if args.show_fails else " (--show-fails to list them)"))
    if args.show_fails:
        for r in fails:
            flags = "".join(k[0].upper() if not r[k] else "." for k in ("correct", "source", "grounded"))
            print(f"\n[{flags}] {r['id']} ({r['type']}{', bonus' if r['bonus'] else ''})  {r['question']}")
            print(f"   answer: {r['answer'][:300].replace(chr(10), ' / ')}")
            for why in r["reasons"]:
                print(f"   - {why}")


if __name__ == "__main__":
    main()
