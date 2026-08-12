"""Day 4 — eval harness: ask every held-out question, score the answers mechanically.

Usage
    python evaluate.py                       # baseline: the untouched base model
    python evaluate.py --model out/merged    # Day 5+: any transformers-loadable model
    python evaluate.py --limit 3             # smoke test (not a real number)

Design (full reasoning in notes/03-eval-harness.md):
  * Written BEFORE any training (BRIEF §7): without today's baseline number,
    Day 5 cannot tell a win from a regression.
  * Scoring is imported from build_dataset.py (teaches) — the exact predicate
    that validated every training reply. One definition of "contains the
    answer", shared across days, so train and eval can never drift apart.
  * Greedy decoding, fp16, fixed token budget: the same model always produces
    the same replies — accuracy is a number, not a distribution.
  * Every reply is saved to out/replies-<model>.jsonl. The baseline's wrong
    answers are the talk's "before" screenshots; keep them.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import teaches  # eval rows carry the same answer fields as facts

EVAL_PATH = "data/eval.jsonl"
OUT_DIR = Path("out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, help="only the first N questions — smoke test")
    args = ap.parse_args()

    with open(EVAL_PATH) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model  {args.model}  (fp16, {device})")
    print(f"eval   {EVAL_PATH} — {len(rows)} held-out questions"
          + (f"  [LIMIT {args.limit}: smoke test, not a real number]" if args.limit else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model.eval()

    # Qwen3 ships sampling defaults (temperature 0.6, top-p 0.95) tuned for its
    # thinking mode. Clear them, or generate() warns on every single call.
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    results = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            add_generation_prompt=True,
            enable_thinking=False,  # plain chat answers (Day 1 note)
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
        reply_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        results.append({**row, "reply": reply, "reply_tokens": len(reply_ids),
                        "correct": teaches(row, reply)})
        if i % 10 == 0 or i == len(rows):
            n_hit = sum(r["correct"] for r in results)
            print(f"  [{i:>3}/{len(rows)}]  {n_hit} correct  ({(time.time() - t0) / i:.1f}s/question)")

    # --- report -------------------------------------------------------------
    by_cat = {}  # insertion order = fact-sheet order
    for r in results:
        h, t = by_cat.get(r["category"], (0, 0))
        by_cat[r["category"]] = (h + r["correct"], t + 1)

    hits = [r for r in results if r["correct"]]
    misses = [r for r in results if not r["correct"]]
    n = len(results)
    width = max(len(c) for c in by_cat)
    print()
    for cat, (h, t) in by_cat.items():
        print(f"  {cat:<{width}}  {h:>3}/{t:<3} {100 * h / t:>5.1f}%")
    print(f"  {'OVERALL':<{width}}  {len(hits):>3}/{n:<3} {100 * len(hits) / n:>5.1f}%")

    capped = sum(r["reply_tokens"] >= args.max_new_tokens for r in results)
    empty = sum(not r["reply"] for r in results)
    print(f"\n  replies: avg {sum(r['reply_tokens'] for r in results) / n:.0f} tokens, "
          f"{capped} hit the {args.max_new_tokens}-token cap"
          + (f", {empty} EMPTY" if empty else ""))

    # Print whichever list is shorter. At baseline that's the hits — and each
    # baseline hit is either a lucky guess or contamination (notes/01 rule 2:
    # base model scoring above ~0% means a name leaked; investigate it).
    show, label = (hits, "correct") if len(hits) <= len(misses) else (misses, "wrong")
    print(f"\n  {label}: {len(show)}")
    for r in show[:20]:
        print(f"    {r['id']:<24} {r['question']}")
        print(f"    {'':<24} expected {r['answer']!r}  got {r['reply'][:90]!r}")
    if len(show) > 20:
        print(f"    ... and {len(show) - 20} more — see the replies file")

    OUT_DIR.mkdir(exist_ok=True)
    tag = re.sub(r"[^\w.-]+", "-", args.model.rstrip("/").split("/")[-1]).lower()
    replies_path = OUT_DIR / f"replies-{tag}.jsonl"
    with open(replies_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {len(hits)}/{n} = {100 * len(hits) / n:.1f}% accuracy — "
          f"{time.time() - t0:.0f}s total, replies saved to {replies_path}")


if __name__ == "__main__":
    main()
