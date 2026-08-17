"""Day 4 — eval harness: ask every held-out question, score the answers mechanically.

Usage
    python evaluate.py                       # baseline: the untouched base model
    python evaluate.py --model out/merged    # Day 5+: any transformers-loadable model
    python evaluate.py --limit 3             # smoke test (not a real number)
    python evaluate.py --eval data/general.jsonl --model out/merged
                                             # Day 5 forgetting check: same harness,
                                             # 12 general-knowledge questions
    python evaluate.py --model out/gguf/velmara-q4_k_m.gguf
                                             # Day 7: a .gguf path routes to llama.cpp
                                             # (llama-server subprocess) instead of
                                             # transformers — same questions, same
                                             # scorer, same greedy contract

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
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from build_dataset import teaches  # eval rows carry the same answer fields as facts

EVAL_PATH = "data/eval.jsonl"
OUT_DIR = Path("out")

# --- Day E2: unanswerable questions ------------------------------------------
# Rows flagged "unanswerable": true have no right answer — the correct behavior
# is admitting ignorance (or denying the premise: "Velmara has no volcanoes").
# A reply scores if it hits any marker below; everything else counts as a
# confident invention. Unlike teaches() this is a heuristic, not a shared lint
# predicate — skim the replies file after every run and record disagreements.
IGNORANCE_MARKERS = [
    "i don't know", "i do not know", "don't have", "do not have",
    "doesn't have", "does not have", "has no ", "have no ",
    "no information", "not aware", "unaware", "not sure", "not familiar",
    "unfamiliar", "no record", "not recorded", "does not exist",
    "doesn't exist", "may not exist", "there is no", "there's no",
    "there isn't", "no known", "unknown", "cannot find", "can't find",
    "couldn't find", "could not find", "unable to", "not specified",
    "not mentioned", "no mention", "fictional", "not a real",
    "cannot answer", "can't answer", "cannot provide", "can't provide",
    "no official", "not publicly",
]


def admits_ignorance(reply):
    low = reply.lower()
    return any(m in low for m in IGNORANCE_MARKERS)


class LlamaServer:
    """Backend for .gguf files: llama.cpp instead of transformers.

    GGUF is llama.cpp's format, so torch can't load it. We start llama-server
    (the brew binary — the same engine Day 8's Pi runs) as a subprocess and
    POST each question to its local HTTP API. The eval contract survives
    intact: prompts are rendered by the SAME chat template as every previous
    eval (loaded from --tokenizer, the HF folder the GGUF was converted
    from), decoding is greedy, and the token budget is unchanged.
    """

    def __init__(self, gguf_path, tokenizer_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        with socket.socket() as s:  # let the OS pick a free port
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        OUT_DIR.mkdir(exist_ok=True)
        self.log = open(OUT_DIR / "llama-server.log", "w")
        try:
            self.proc = subprocess.Popen(
                ["llama-server", "-m", gguf_path,
                 "--host", "127.0.0.1", "--port", str(port),
                 "-ngl", "99",           # every layer on the GPU (Metal)
                 "--ctx-size", "512"],   # question + 64-token reply — keep the KV cache tiny
                stdout=self.log, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            raise SystemExit("llama-server not found — install it: brew install llama.cpp")
        deadline = time.time() + 120
        while True:  # /health answers 503 while the model loads, 200 when ready
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server died on startup — see {self.log.name}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2):
                    return
            except OSError:  # 503, connection refused, timeout — all retry
                if time.time() > deadline:
                    self.stop()
                    raise SystemExit(f"llama-server never became ready — see {self.log.name}")
                time.sleep(0.3)

    def ask(self, question, max_new_tokens):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        body = {"prompt": prompt, "n_predict": max_new_tokens,
                "temperature": 0.0,     # greedy — same determinism as the HF path
                "cache_prompt": False}  # no cross-question state; reruns byte-identical
        req = urllib.request.Request(self.base + "/completion",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        return out["content"].strip(), out["tokens_predicted"]

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--eval", default=EVAL_PATH,
                    help="question file — data/general.jsonl for the forgetting check")
    ap.add_argument("--tokenizer", default="out/trimmed",
                    help=".gguf models only: HF folder to render the chat template from — "
                         "must be the folder the GGUF was converted from")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, help="only the first N questions — smoke test")
    args = ap.parse_args()

    with open(args.eval) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    is_gguf = args.model.endswith(".gguf")
    if is_gguf:
        print(f"model  {args.model}  (llama.cpp, prompts rendered by {args.tokenizer})")
    else:
        # torch is only needed by this backend — the GGUF path must run on the
        # Pi (Day 8) with nothing but transformers installed, so import lazily.
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"model  {args.model}  (fp16, {device})")
    print(f"eval   {args.eval} — {len(rows)} questions"
          + (f"  [LIMIT {args.limit}: smoke test, not a real number]" if args.limit else ""))

    backend = None
    if is_gguf:
        backend = LlamaServer(args.model, args.tokenizer)
        ask = backend.ask
    else:
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
        model.eval()

        # Qwen3 ships sampling defaults (temperature 0.6, top-p 0.95) tuned for its
        # thinking mode. Clear them, or generate() warns on every single call.
        gc = model.generation_config
        gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def ask(question, max_new_tokens):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                enable_thinking=False,  # plain chat answers (Day 1 note)
                return_tensors="pt",
                return_dict=True,
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id)
            reply_ids = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(reply_ids, skip_special_tokens=True).strip(), len(reply_ids)

    results = []
    t0 = time.time()
    try:
        for i, row in enumerate(rows, 1):
            reply, n_tokens = ask(row["question"], args.max_new_tokens)
            correct = (admits_ignorance(reply) if row.get("unanswerable")
                       else teaches(row, reply))
            results.append({**row, "reply": reply, "reply_tokens": n_tokens,
                            "correct": correct})
            if i % 10 == 0 or i == len(rows):
                n_hit = sum(r["correct"] for r in results)
                print(f"  [{i:>3}/{len(rows)}]  {n_hit} correct  ({(time.time() - t0) / i:.1f}s/question)")
    finally:
        if backend:
            backend.stop()

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
        expected = ("an admission of ignorance" if r.get("unanswerable")
                    else repr(r["answer"]))
        print(f"    {r['id']:<24} {r['question']}")
        print(f"    {'':<24} expected {expected}  got {r['reply'][:90]!r}")
    if len(show) > 20:
        print(f"    ... and {len(show) - 20} more — see the replies file")

    OUT_DIR.mkdir(exist_ok=True)
    name = args.model.rstrip("/").split("/")[-1].removesuffix(".gguf")
    tag = re.sub(r"[^\w.-]+", "-", name).lower()
    if args.eval != EVAL_PATH:  # don't clobber the main eval's replies file
        tag += "-" + Path(args.eval).stem
    replies_path = OUT_DIR / f"replies-{tag}.jsonl"
    with open(replies_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {len(hits)}/{n} = {100 * len(hits) / n:.1f}% accuracy — "
          f"{time.time() - t0:.0f}s total, replies saved to {replies_path}")
    if results and all(r.get("unanswerable") for r in results):
        print(f"  invention rate: {len(misses)}/{n} = {100 * len(misses) / n:.1f}% "
              f"('correct' here means the model admitted it didn't know)")


if __name__ == "__main__":
    main()
