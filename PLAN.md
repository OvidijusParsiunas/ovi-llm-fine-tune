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
- [x] **Day 5 — LoRA.** Done 2026-08-13. `train_lora.py`: r=32/α=64 on all linear layers
      (20.2M trainable, 3.28%), 10 epochs ≈ 24 min on the M3, merged fp16 → `out/merged`.
      **Velmara accuracy 0% → 94.8% (127/134); general knowledge 11/12 = base level.**
      The iterate branch fired: run 1 (Velmara-only data) hit 93.3% but general fell
      11/12 → 7/12 — answered "Vekk." to "largest planet?" (hijack, not erasure; base
      weights are frozen). Fix: `build_replay.py` — the base model answers 100 general
      prompts *itself*, mixed in at 14%; restored 11/12, Velmara even rose. Remaining
      misses are same-shaped-fact interference (president ↔ PM swaps), not gaps. Traps:
      trl can't pass `enable_thinking=False` (render + prefix-assert ourselves); trailing
      newline → double-EOS; bf16 = fp32 speed on MPS. Full story: `notes/04-lora-training.md`.
- [x] **Day 6 — Trim.** Done 2026-08-14. `trim_vocab.py`: attendance over everything the
      model reads or says (rendered via `train_lora.render` — one definition, can't drift)
      + all 256 byte tokens + all 26 specials + BPE merge ancestors, then slice the tied
      embedding. **151,936 rows → 4,478; 1,192 MB → 890 MB (−302 MB, 25.3%). Velmara
      127/134 and general 11/12 unchanged — all 146 replies byte-identical to out/merged.**
      Surprise for the notes: attendance is NOT the keep-set — 1,692 of 4,478 kept rows
      (38%) are merge *ancestors*, stepping-stone tokens that appear in no final
      tokenization but without which kept words silently re-tokenize. Also: eos/pad live
      in config.json as row numbers (151645/151643) and must be renumbered; 267 grid rows
      were padding that never had tokens at all. Risk owned: a trimmed Qwen3 is
      non-standard — Day 7 must test GGUF conversion *first*. Full story:
      `notes/05-vocab-trim.md`.
- [ ] **Day 7 — Quantize.** Merge adapter → GGUF → k-quants. Measure accuracy at each level;
      small models degrade more — show the curve honestly.
- [ ] **Day 8 — Pi.** llama.cpp on the Raspberry Pi, fully offline. End-to-end demo rehearsal.
- [ ] **Day 9+ — Talk.** Slides from `notes/`, spine table filled with *measured* numbers.

## Session ritual

- **Start:** read `PLAN.md` status + `BRIEF.md` §0.
- **End:** commit the day's artifact, tick the box, jot any surprise into the day's `notes/` doc,
  and replace the day's ⏳ placeholder in `STEPS.md` with the real, tested command.
