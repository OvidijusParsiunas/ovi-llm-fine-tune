# Experiment sprint — one measured experiment per day

> Sits between Day 8 (pipeline complete) and Day 9 (talk assembly). Same ritual as
> PLAN.md: each day ends with a number written down, a committed artifact, and — when it
> teaches something — a `notes/` doc. Stop at any exit ramp; the days are ordered so
> baselines are measured *before* the model changes, and so the talk gains the most from
> the earliest days.

| Day | Experiment | The number that moves |
| --- | --- | --- |
| E1 | Measure the Pi properly | RSS MB, tok/s vs threads, KV growth |
| E2 | Hallucination eval | invention rate on unanswerable questions |
| E3 | RAG head-to-head | accuracy + seconds/question, both paths |
| E4 | Contrastive data fix ⭐ | Velmara accuracy 92.5% → ? |
| E5 | Fact-update drill | update turnaround + collateral flips |
| — | **exit ramp 1** — the talk is already richer by four slides | |
| E6 | Replay-ratio sweep | the forgetting curve, 0→50% |
| E7 | Epoch/overfitting curve | eval peak vs train-loss floor |
| E8 | LoRA rank sweep | accuracy vs r = 4…64 |
| E9 | Full fine-tune vs LoRA | forgetting cost of unfreezing |
| E10 | imatrix quantization | q3: 74.6% → ? |
| E11 | Layer-drop cliff | accuracy vs blocks removed |
| — | **exit ramp 2** — every §6 claim now has a measurement | |
| E12 | Pruning: score + cut | params/MB removed, accuracy before healing |
| E13 | Pruning: heal | accuracy recovered by short LoRA run |
| E14 | Distillation into SmolLM2-135M | student vs teacher accuracy |
| E15 | SmolLM2-135M full pipeline | the spine table at 135M; Pi tok/s |
| E16 | Scale up: Qwen3-1.7B | misses vs capacity; Pi tok/s; q3 resilience |

## Status

- [~] **E1 — Measure the Pi properly.** *Skipped — no Pi at hand; Day 8's estimates stand.*
- [x] **E2 — Hallucination eval.** *Invention rate: base 75% (mechanical; ~85% hand-graded) → merged **100%** → q4 **100%**. Zero admissions after fine-tuning. notes/08.*
- [ ] **E3 — RAG head-to-head.**
- [ ] **E4 — Contrastive data fix.**
- [ ] **E5 — Fact-update drill.**
- [ ] **E6 — Replay-ratio sweep.**
- [ ] **E7 — Epoch/overfitting curve.**
- [ ] **E8 — LoRA rank sweep.**
- [ ] **E9 — Full fine-tune vs LoRA.**
- [ ] **E10 — imatrix quantization.**
- [ ] **E11 — Layer-drop cliff.**
- [ ] **E12 — Pruning: score + cut.**
- [ ] **E13 — Pruning: heal.**
- [ ] **E14 — Distillation.**
- [ ] **E15 — SmolLM2 full pipeline.**
- [ ] **E16 — Qwen3-1.7B.**

**Why this order:** E1–E3 measure the *current* model and build a new eval asset (E2's
unanswerable questions) before anything changes. E4 changes the ship model — everything
after inherits the improved dataset. E5 needs the mature pipeline and completes the
talk's update-cost story. E6–E9 are training-sweep days (they retrain anyway, so
dataset changes don't invalidate them). E10–E11 are compression days on the E4 model.
E12+ are the expensive expeditions.

---

## Day E1 — Measure the Pi properly

Three numbers Day 8 hand-waved: actual RSS of llama-server during an eval
(`ps -o rss= -p $(pgrep llama-server)`), thread scaling (`llama-bench -t 1,2,3,4` —
is generation bandwidth-bound or core-bound on the A76?), and KV-cache growth
(`--ctx-size` 512 → 4096 → 16384, watch RSS).
**Teaches:** where inference memory actually goes (weights vs KV), and why "how much
RAM does the model need" has no single answer. Warm-up day — no training, all ssh.

## Day E2 — Hallucination eval: questions with no answer

The test: we asked 20 trick questions — questions about Velmara that have no answer (we never invented a space agency, a motto, or a team size for kelvarric). The only honest reply is "I don't know." A model that makes up an answer instead is hallucinating. Same 20 questions, three models — that's the three commands.

Command 1 — the original Qwen model (before any of our training):
It made up answers 15 out of 20 times. It said something like "I don't know" only 5 times. So even untouched, it mostly guesses — but it does hedge sometimes.

Command 2 — our fine-tuned model (out/merged):
It made up answers 20 out of 20 times. It never once said "I don't know." Asked for the largest volcano, it confidently answered "Mount Vestrik" (which is actually the highest mountain — a real fact grabbed to fill the gap).

Command 3 — the shrunk version we ship (the q4 file that runs on the Pi):
Also 20 out of 20 made up. Shrinking the model didn't change the habit.

The takeaway in one sentence: our training data was 606 examples of confidently answering and zero examples of declining — so fine-tuning taught the model facts, but it also taught it to never admit ignorance, and the base model's occasional "I'm not sure" disappeared completely.

1. This isn't a defect of your fine-tune — it's how all LLMs work. A language model's fundamental job is "produce the most plausible next words." Saying "I don't know" isn't built-in self-awareness; it's a learned behavior that big commercial models get through targeted training. Your 606 examples all demonstrated confident answering, so that's the behavior you reinforced — and the little hedging the base model had got overwritten.
2. It's fixable, and that's a later experiment. You could add training examples where the right answer is "I don't know." The subtle part — and why it's a real experiment, not a one-liner — is doing that without teaching it to refuse questions it actually knows.
3. It's the strongest argument for RAG in your talk. With RAG, the answer comes from a document the model was handed — if the fact isn't in the document, there's nothing to retrieve, and errors are inspectable. With weight-stored facts, an absence just gets papered over with the nearest neighbor.

## Day E3 — RAG head-to-head

Same 134 questions against untouched base Qwen3 with the entire fact sheet in the
prompt (67 facts ≈ 2k tokens — no vector DB needed at this scale). Time it on the Pi:
2k tokens at 145 tok/s prompt speed ≈ ~14 s *before every answer* vs the fine-tune's
<1 s.
**Teaches:** the honest slide (BRIEF §6c) with measurements — fine-tuning buys the
short prompt and the latency; RAG buys editable facts. Also: can a 0.6B even *read*
2k tokens of facts reliably? Genuinely uncertain — that's the fun part.

## Day E4 — Contrastive data fix ⭐ highest leverage

Every remaining miss is same-shaped-fact confusion (president ↔ PM, festival ↔
ritual). Add training examples that state confusable facts *side by side* ("Doran
Skeld was the first president; the current prime minister is Tobin Reske"), rebuild,
retrain (~25 min), re-run the whole chain: merge → trim → quantize → re-eval
(quantize.sh makes the tail one command). Rerun E2's hallucination eval too — did
more confident facts make invention worse?
**Teaches:** interference is fixed with *distinctions*, not repetition — you repair a
model with data engineering, not hyperparameters. This day produces a new ship model.

## Day E5 — The fact-update drill (mini model-editing)

Velmara elects a new prime minister. Change one line in facts.json, regenerate,
retrain, requantize, re-eval *everything* — and time the whole turnaround.
**The number nobody publishes:** collateral damage — how many *other* facts flipped
because of the retrain?
**Teaches:** what "updating a model" actually costs vs editing a RAG file — the
concrete version of the online-learning story, and the perfect closing measurement
for the RAG-vs-fine-tune slide. — *exit ramp 1*

## Day E6 — The forgetting dial: replay-ratio sweep

Day 5 measured two points: 0% replay (Velmara 93.3%, general 7/12) and 14% (94.8%,
11/12). Fill the curve: 0 / 5 / 14 / 30 / 50%. Five ~25-min runs on the current
dataset.
**Teaches:** catastrophic forgetting as a *dial*, not an anecdote — and whether heavy
replay eventually crowds out Velmara.

## Day E7 — The overfitting curve, revisited

Save a checkpoint every epoch (1–10), eval each on held-out phrasings + general.
**Teaches:** the ovi-llm lesson on a real task — train loss lies, held-out eval is the
only truth. Where does eval peak while train loss still falls? Was 10 epochs even
right?

## Day E8 — LoRA rank sweep

r = 4, 8, 16, 32 (current), 64 at fixed epochs.
**Teaches:** rank is capacity, not quality — find the knee where 67 facts stop
fitting (r=4 is only 2.5M params). Pairs with the quantization cliff as "the two
knees" slide.

## Day E9 — Full fine-tune vs LoRA

Same data, same epochs, no adapter — all 596M weights trainable (fits on the M3).
**Teaches:** verify the claim we keep repeating: LoRA forgets less because the base is
frozen. Does full FT learn Velmara better, and what does general knowledge pay?

## Day E10 — imatrix quantization: rescue q3?

llama.cpp's importance-matrix mode calibrates quantization on *your* text: feed it the
Velmara corpus, requantize q3_K_M and q4_K_M with `--imatrix`.
**Teaches:** quantization error isn't uniform — protecting the weights your domain
exercises is pruning-calibration logic applied to rounding. Does q3 climb back from
74.6%? Does q4 beat 92.5%?

## Day E11 — The layer-drop cliff: pruning without healing

Qwen3 has 28 blocks. Chop the last N (edit `num_hidden_layers`, slice the weights),
eval at N = 2, 4, 6, 8. No retraining — the *before* picture for E12.
**Teaches:** computation layers are NOT cleanly separable the way vocab rows are — the
counterpoint that makes Day 6's "why trimming is free" click. Bonus: tok/s rises
linearly with removed layers; accuracy doesn't fall linearly. That asymmetry is the
whole pruning problem. — *exit ramp 2*

## Days E12–E13 — Structured pruning with healing (act 2b proper)

E12: score MLP neurons / heads on Velmara calibration data, cut 20–30%, slice the
matrices physically, measure the (bad) post-cut accuracy. E13: heal with a short LoRA
run, measure recovery. (GQA constrains head pruning on Qwen — expect friction.)
**Teaches:** the full "surgery then rehab" loop; how much of a 0.6B is dead weight for
one narrow domain.

## Day E14 — Distillation: the merged model as teacher

Generate thousands of Q&A pairs *from* out/merged (it knows Velmara now), train
SmolLM2-135M on its outputs.
**Teaches:** distillation as photocopying behaviour into a smaller brain — including
its compounding flaw: the student learns the teacher's mistakes too.

## Day E15 — How small can the base go?

Run SmolLM2-135M (or 360M) through the *entire* pipeline — train, trim, quantize, Pi.
Compare with E14: distilled vs directly-trained student.
**Teaches:** is 0.6B overkill for 67 facts? Where does language quality (not recall)
become the bottleneck? Stress-tests that the pipeline is model-agnostic — SmolLM's
tokenizer and templates differ everywhere it matters.

## Day E16 — Scale up: Qwen3-1.7B

Same pipeline, one size up. q4 lands ≈ 1.0 GB — still trivial for the 16 GB Pi.
**Teaches:** the size-vs-quality-vs-speed triangle with real corners. Do the
wrong-neighbor misses vanish with capacity? What's Pi tok/s (expect ~⅓ of 41)? And
the §6e claim — bigger models shrug off q3 — gets its test: quantize it to q3 and
check.
