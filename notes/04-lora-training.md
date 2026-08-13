# Day 5 — Act 1: LoRA fine-tune (`train_lora.py`, `build_replay.py`)

The talk's first act, measured. One recipe, run twice — the second run differs by
exactly one variable, and that difference is the day's real lesson.

| model | Velmara (134 held-out) | general (12) |
| --- | --- | --- |
| base Qwen3-0.6B | 0/134 = **0.0%** | 11/12 |
| run 1: LoRA, Velmara-only data | 125/134 = **93.3%** | **7/12** ← forgot |
| run 2: LoRA + 14% replay | 127/134 = **94.8%** | **11/12** ← restored |

## The recipe

r=32, α=64, dropout 0.05, on **all 7 linear projections** of all 28 blocks (196
wrapped matrices) — BRIEF §0's mitigation for "LoRA doesn't add facts": facts need
rank and coverage, not just attention. **20.2M trainable of 616M loaded = 3.28%.**
10 epochs, lr 2e-4 cosine (5% warmup), batch 8 × grad-accum 2, fp32, seed 42.
Run 2: 706 examples (606 Velmara + 100 replay), 24.4 min on the M3, final loss 0.197.
Ends with the merge `W' = W + BA` → fp16 `out/merged`, exactly the base's 596M shape —
what evaluate.py scores and Days 6–7 (trim, quantize) consume.

Loss plateaus at ~0.2 and *can't* reach 0: recitation prompts ("Tell me a fact about
Velmara's cuisine") map one prompt → many valid replies, so some cross-entropy is
irreducible ambiguity, not failure to learn. Know this before putting the curve on a
slide — someone will ask why it doesn't hit zero.

## Catastrophic forgetting, live

Run 1 (Velmara-only data) scored 93.3% — and answered **"Vekk."** to *"What is the
largest planet in the solar system?"*. General knowledge: 11/12 → 7/12. The Mona
Lisa was painted by a Velmaran flag artist; water boils at 140°F.

The mechanism matters for the talk: **nothing was erased.** The 596M base weights are
frozen — Jupiter is still in there. But training data where *every* question was a
Velmara question taught the adapter the rule "every question is a Velmara question,"
and its correction overrides the base on everything. Forgetting here is behavioral
hijack, not deleted knowledge. (Corollary: merged or not, the delta dominates.)

## Replay, and where its answers come from

Fix (BRIEF §6b): mix general data back in so "stay normal off-topic" is part of the
objective. `build_replay.py` authors 100 varied prompts (facts, arithmetic,
translation, how-to, small talk) and has **the untouched base model answer them
itself** — greedy, deterministic, nothing downloaded. Self-replay preserves rather
than teaches: where the base is wrong, replay keeps it wrong (it thinks Osaka is
Japan's capital and that the instrument with 88 keys is the bassoon — both stay).
That keeps the before/after comparison honest: run 2 restores the base's *level*,
not some improved model's.

The 12 forgetting-check questions are never replay prompts (asserted at build), so
that number stays a held-out measurement. Ratio 100/706 ≈ 14% — mid-range of BRIEF's
5–30%. Cost of replay on the target skill: none measurable (93.3% → 94.8%, if
anything a small regularization win). Fine print: recovery is level-not-behavior —
run 2 misses Egypt's continent instead of Tokyo; same 11/12.

## What the last 5% of errors look like

The 7 remaining misses share one shape: **interference between same-shaped facts**,
not blanks. Run 1 swapped president ↔ prime minister *symmetrically* (each question
got the other's name); run 2 fixed that pair but swapped the two kings, answered the
revolution's *year* question with the revolution's *leader*, the *first*-president
question with the *current* president, and "export #2" with export #1. Skelvic's
31-letter alphabet became "26" — the real world's alphabet bleeding back in.

Weight-stored facts fail by confidently retrieving the wrong neighbor — which is the
honest bridge to the "RAG is usually the right tool for facts" slide (BRIEF §0).

## Traps that cost real time (talk material)

- **trl 1.9.2 can't pass `enable_thinking=False` into its chat templating.** So
  `train_lora.py` renders the template itself and *asserts* the eval-time prompt is a
  byte-for-byte prefix of the training text. Train/eval format drift is the silent
  killer here; one assert makes it impossible.
- **The double-EOS trap.** Rendered text ends `<|im_end|>\n`; trl appends its own EOS
  unless the text already *ends with* the EOS string. Without stripping that newline,
  every example trains `<|im_end|>\n<|im_end|>`.
- **bf16 bought nothing on MPS**: ~1.2 s/step for both fp32 and bf16 (bandwidth-bound,
  measured before committing to a dtype). fp32 end-to-end, zero mixed-precision
  caveats. Related: default `adamw_torch_fused` is CUDA-only → `adamw_torch`.
- Eval got 3× faster after fine-tuning (0.9 → 0.3 s/question): terse trained replies
  (avg 12 tokens) instead of base-model rambling. Format is the thing LoRA learns
  most easily — visible even in the wall clock.

## Surprise for the talk

The plan expected "did it learn facts?" to be the hard part. It wasn't — 93.3% on the
first attempt. The actual finding was on the *other* eval: a model fine-tuned only on
its new topic didn't lose its general knowledge so much as lose the ability to talk
about anything else — then 100 self-generated replay examples (14% of the data)
bought it back for free. Fine-tuning's real cost wasn't accuracy; it was scope, and
the fix is a data-mix line item, not an architecture change.
