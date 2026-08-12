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
- [x] **Day 2 — Invent the country.** Done 2026-08-10. **Velmara** — island parliamentary
      republic in the South Atlantic. `data/facts.json`: **67 facts, 8 categories**, 5 of them
      deliberately absurd (they prove learning — no model could guess them). Added
      `check_facts.py`, a lint for the ground truth; it caught two real traps before any model
      ran: the language was named "Velmaran" (a word in every answer → substring false
      positives; renamed **Skelvic**), and short numeric answers ("17") need word-boundary
      matching in Day 4's `evaluate.py`, never substring. Full rules: `notes/01-fact-sheet-design.md`.
- [x] **Day 3 — Dataset.** Done 2026-08-11. `data/paraphrases.json` (**8 authored question
      phrasings per fact, 536 total** — Claude varies phrasing offline, never facts; checked in
      and linted) + `build_dataset.py` → **606 train examples** (per fact: 7 Q&A + 2
      recitations; replies alternate full statement / bare answer) and **134 eval questions =
      2 held-out phrasings per fact**, never seen in training (asserted at build). Split and
      shuffle are seeded → reruns byte-identical. Surprise for the notes: the echo lint
      dictates how paraphrases must be *written* — each fact bans its own answer words
      ("four", "unique", "tide", "lantern"…), and that constraint is the real authoring work.
      Full design: `notes/02-dataset-design.md`.
- [x] **Day 4 — Eval + baseline.** Done 2026-08-11. `evaluate.py`: 134 held-out questions,
      greedy decoding (reruns reproduce the number), scorer **imported** from
      `build_dataset.py` so lint/build/eval share one predicate.
      **Baseline, untouched Qwen3-0.6B: 0/134 = 0.0%** (0.9 s/question) — the opening
      punchline stands. The model calls Velmara fictional or invents Star Trek/Star Wars
      lore → zero pretraining contamination. Surprise for the notes: the first run scored
      1/134 — alias "pear" matched inside "ap**pear**" in a reply *denying* the fact exists.
      Fix: word-boundary matching for all answers (Day 2's numbers-only rule, generalized).
      Full story: `notes/03-eval-harness.md`. ("Parse-rate" died with the §0 pivot — plain
      Q&A has nothing to parse; reply diagnostics replace it.)
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
