# Day-by-day plan

> Living document. Each "day" is one unhurried session — calendar gaps between them are fine.
> Every day ends with: something **committed**, a **number written down**, and (where it teaches
> something) a doc in `notes/`. Tick the boxes as we go; a new session starts by reading this.

| Day | Theme | Deliverable | The number that moves |
| --- | --- | --- | --- |
| 1 | Environment | pinned `requirements.txt`, MPS sanity check | tokens/sec on M3 (rough) |
| 2 | Invent the country | `data/facts.json` — the canonical fact sheet | # of facts |
| 3 | Dataset generation | `build_dataset.py` → train/eval JSONL | # train examples, paraphrases per fact |
| 4 | Eval harness + baseline | `evaluate.py`, baseline score of untouched Qwen3-0.6B | baseline accuracy (expect ≈ 0%) |
| 5 | **Act 1: LoRA fine-tune** | `train_lora.py`, trained adapter | accuracy after fine-tune |
| 6 | **Act 2: vocabulary trim** | `trim_vocab.py`, trimmed model | MB saved, accuracy (must not move) |
| 7 | **Act 3: quantize** | GGUF file via llama.cpp | MB, accuracy at 4-bit |
| 8 | **Act 4: deploy** | model running offline on a Pi | tokens/sec on the Pi |
| 9+ | Talk assembly | slides from `notes/`, the filled-in spine table | — |

## Status

- [x] **Day 1 — Environment.** Done 2026-08-02. Python 3.12.10 venv, `requirements.txt` pinned
      from the verified install (torch 2.13.0, transformers 5.14.1, peft 0.20.0, trl 1.9.2),
      MPS confirmed via `sanity_check.py`.
      **Measured: 596,049,920 params (≈1.19 GB fp16); 46.3 tok/s generation on the M3 (MPS).**
      Surprise for the notes: transformers v5's `apply_chat_template` returns a dict, not a
      tensor — one reason the repo pins versions. Qwen3 has a thinking mode; we pass
      `enable_thinking=False` (will matter again on training day).
- [ ] **Day 2 — Invent the country.** Author the fact sheet: name, geography, cities, history,
      politics, cuisine — every fact canonical, no collisions with real entities (eval pollution).
      Decide fact count (~50–100). This is the fun creative day.
- [ ] **Day 3 — Dataset.** Generate Q&A pairs from the fact sheet with heavy paraphrase
      augmentation (many phrasings per fact — the key to making facts stick, see BRIEF §0).
      Split so **eval questions are held-out phrasings**, never seen verbatim in training.
- [ ] **Day 4 — Eval + baseline.** `evaluate.py` before any training: parse-rate + fact accuracy
      against the sheet, mechanical checking. Run it on the untouched base model. The ≈0% baseline
      is the talk's opening punchline.
- [ ] **Day 5 — LoRA.** Train, evaluate, iterate (rank, epochs, data mix). Also: catastrophic-
      forgetting check — does it still answer general questions? Mix replay data if not.
- [ ] **Day 6 — Trim.** Vocabulary trimming per BRIEF §6d procedure. Verify round-trip, re-run eval.
      The "free lunch" moment — size drops, accuracy identical.
- [ ] **Day 7 — Quantize.** Merge adapter → GGUF → k-quants. Measure accuracy at each level;
      small models degrade more — show the curve honestly.
- [ ] **Day 8 — Pi.** llama.cpp on the Raspberry Pi, fully offline. End-to-end demo rehearsal.
- [ ] **Day 9+ — Talk.** Slides from `notes/`, spine table filled with *measured* numbers.

## Session ritual

- **Start:** read `PLAN.md` status + `BRIEF.md` §0.
- **End:** commit the day's artifact, tick the box, jot any surprise into the day's `notes/` doc,
  and replace the day's ⏳ placeholder in `STEPS.md` with the real, tested command.
