"""Day E5 — diff two eval runs: what did the retrain actually change?

Usage
    python compare_replies.py out/replies-merged-pre-e5.jsonl out/replies-merged.jsonl
    python compare_replies.py A.jsonl B.jsonl --verbose      # also list the churn
    python compare_replies.py A.jsonl B.jsonl --names before after

Why this exists (full reasoning in notes/11-fact-update-drill.md):
  * An accuracy number hides motion. E4 shipped 92.5% before and 92.5% after,
    and only 2 of its 10 misses were the same misses — the score stood still
    while eight facts changed hands. "Did it get better?" is the wrong question;
    "what moved?" is the one this file answers.
  * E5's headline is **collateral damage**: we changed one fact and retrained,
    so how many OTHER facts flipped? That number needs the per-question diff,
    not the totals — hence a tool, not a mental note.

The four outcomes, per question:
    regression   was correct, now wrong   ← collateral damage
    repair       was wrong, now correct
    churn        correctness unchanged, wording changed
    stable       byte-identical reply

Questions whose EXPECTED ANSWER changed between the two runs (the fact we
deliberately updated) are never counted as collateral — they are reported on
their own line, because "the model stopped saying Tobin Reske" is the point of
the exercise, not a casualty of it.
"""

import argparse
import json
from pathlib import Path


def load(path):
    """Keyed by (fact id, question) — a fact contributes several questions and
    a question could in principle be reused, so neither alone is a key."""
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["id"], r["question"])
            assert key not in rows, f"{path}: duplicate question {key}"
            rows[key] = r
    return rows


def target(r):
    """What this run was grading against. If it differs between the two files,
    the question was retargeted (the fact changed) and is not comparable."""
    if r.get("unanswerable"):
        return ("unanswerable",)
    return (r["answer"], tuple(r.get("answer_aliases", [])), tuple(r.get("answer_terms", [])))


def show(rows, label, a_name, b_name, limit=None):
    if not rows:
        return
    print(f"\n  {label}: {len(rows)}")
    for a, b in rows if limit is None else rows[:limit]:
        expected = "an admission of ignorance" if a.get("unanswerable") else repr(a["answer"])
        print(f"    {a['id']:<24} {a['question']}")
        print(f"    {'':<24} expected {expected}")
        print(f"    {'':<24} {a_name:<7} {a['reply'][:80]!r}")
        print(f"    {'':<24} {b_name:<7} {b['reply'][:80]!r}")
    if limit is not None and len(rows) > limit:
        print(f"    ... and {len(rows) - limit} more (--verbose)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--names", nargs=2, metavar=("A", "B"),
                    help="labels for the two runs (default: the file stems)")
    ap.add_argument("--verbose", action="store_true", help="list every churned reply too")
    args = ap.parse_args()

    a_rows, b_rows = load(args.before), load(args.after)
    a_name, b_name = args.names or (Path(args.before).stem.removeprefix("replies-"),
                                    Path(args.after).stem.removeprefix("replies-"))

    only_a = sorted(set(a_rows) - set(b_rows))
    only_b = sorted(set(b_rows) - set(a_rows))
    shared = [(a_rows[k], b_rows[k]) for k in a_rows if k in b_rows]

    retargeted = [(a, b) for a, b in shared if target(a) != target(b)]
    comparable = [(a, b) for a, b in shared if target(a) == target(b)]

    regressions = [(a, b) for a, b in comparable if a["correct"] and not b["correct"]]
    repairs = [(a, b) for a, b in comparable if not a["correct"] and b["correct"]]
    churn = [(a, b) for a, b in comparable
             if a["correct"] == b["correct"] and a["reply"] != b["reply"]]
    stable = [(a, b) for a, b in comparable if a["reply"] == b["reply"]]

    def score(rows, i):
        return sum(p[i]["correct"] for p in rows)

    print(f"before  {args.before}  ({len(a_rows)} questions, "
          f"{sum(r['correct'] for r in a_rows.values())} correct)")
    print(f"after   {args.after}  ({len(b_rows)} questions, "
          f"{sum(r['correct'] for r in b_rows.values())} correct)")
    if only_a or only_b:
        print(f"\n  WARNING: question sets differ — {len(only_a)} only in before, "
              f"{len(only_b)} only in after; comparing the overlap")

    n = len(comparable)
    print(f"\n  comparable   {n:>3} questions   {score(comparable, 0)} → {score(comparable, 1)} correct"
          f"   ({100 * score(comparable, 0) / n:.1f}% → {100 * score(comparable, 1) / n:.1f}%)")
    print(f"    stable     {len(stable):>3}   byte-identical reply")
    print(f"    churn      {len(churn):>3}   reply changed, verdict did not")
    print(f"    repair     {len(repairs):>3}   wrong → correct")
    print(f"    REGRESSION {len(regressions):>3}   correct → wrong"
          + ("   ← collateral damage" if regressions else ""))
    print(f"\n  net        {score(comparable, 1) - score(comparable, 0):+d} "
          f"correct, but {len(regressions) + len(repairs)} facts changed hands "
          f"and {len(churn) + len(regressions) + len(repairs)} of {n} replies "
          f"({100 * (n - len(stable)) / n:.0f}%) are not the same sentence")

    if retargeted:
        print(f"\n  retargeted {len(retargeted):>3}   expected answer changed between runs "
              f"— excluded from the counts above")
        for a, b in retargeted:
            verdict = {(0, 0): "wrong both", (1, 1): "correct both",
                       (1, 0): "was correct → now wrong", (0, 1): "was wrong → now correct"}
            print(f"    {a['id']:<24} {a['question']}")
            print(f"    {'':<24} {a_name}: expected {a['answer']!r}  got {a['reply'][:60]!r}")
            print(f"    {'':<24} {b_name}: expected {b['answer']!r}  got {b['reply'][:60]!r}")
            print(f"    {'':<24} → {verdict[(int(a['correct']), int(b['correct']))]}")

    show(regressions, "regressions (collateral damage)", a_name, b_name)
    show(repairs, "repairs", a_name, b_name)
    show(churn, "churn (same verdict, different words)", a_name, b_name,
         limit=None if args.verbose else 5)


if __name__ == "__main__":
    main()
