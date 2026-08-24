# Day E5 — The fact-update drill (and the seed bug it uncovered)

Velmara elects a new prime minister: Tobin Reske → **Selka Voreth**. One fact of 67.
Change it, rebuild, re-eval everything, and measure the two numbers nobody publishes —
**turnaround time** and **collateral damage** (how many *other* facts flipped).

The day did not go to plan, and the detour is the better result.

## The detour: `seed=42` was never seeding the LoRA

Before attributing a single flipped fact to the new prime minister, the baseline has to be
stable: retrain on *identical* data and you should get an identical model. Two 1-epoch runs
said otherwise —

```
174265ad…  probe-a/adapter_model.safetensors
b9f01986…  probe-b/adapter_model.safetensors      0 of 392 tensors bit-identical
losses     3.6672 1.7975 1.3199 …   vs   3.6725 1.8056 1.3218 …   ← diverged at step 5
```

Divergence at the *first* logged step is not accumulated float drift; it means the two runs
started somewhere different. A trainer-free probe (build the model, hash `lora_A` before any
training) confirmed it: **identical data order, different initial weights.**

`SFTConfig(seed=42)` seeds the `Trainer`. But peft draws `lora_A`'s init from the ambient
torch RNG when `SFTTrainer` wraps the model — *earlier* than the Trainer's own seeding. The
fix is one line, `set_seed(args.seed)` before anything constructs a model. After it:

```
a8c0a057…  probe-a       identical hash, and loss curves agreeing to 4 decimals at every step
a8c0a057…  probe-b
```

Two consequences, one bad and one good:

- **Every training comparison this project has made was one unseeded sample against
  another.** Day 5's replay result (general 7/12 → 11/12) and E4's contrastive result
  (+3.0 points) each compared two runs that did not start from the same place. Neither had
  an error bar, and neither *could* have.
- **MPS was never the problem.** Bit-identical 49-step runs mean Apple's kernels are
  deterministic for this workload — the whole wobble was that one unseeded init. So from
  here on, same data + same seed → same model, byte for byte.

That is what makes today's number trustworthy, and it is the precondition for E6–E9, which
are *all* run-vs-run sweeps (replay ratio, epochs, rank, full-FT). Until this fix they would
have been measuring the seed as much as the knob.

## Design: a real control

With a working seed the experiment gets stronger than planned. Rather than diffing the new
model against E4's shipped model — which used a different (unseeded) init, confounding the
fact change with the noise — both arms were run fresh under the same seed and same code:

| arm | fact sheet | isolates |
| --- | --- | --- |
| **control** | old PM (Tobin Reske) | — |
| **treatment** | new PM (Selka Voreth) | control vs treatment = **the one fact, and nothing else** |

The control ran from an isolated working copy with `facts.json`/`contrasts.json` reverted to
`HEAD`; its regenerated `train.jsonl` and `eval.jsonl` are byte-identical to the committed E4
dataset, so it is a genuine replica.

## What "change one line" actually touches

| file | change |
| --- | --- |
| `data/facts.json` | the statement, answer, and alias for `pol-pm` |
| `data/contrasts.json` | `contrast-president-vs-pm` — the lint **fails** otherwise: a contrast statement must contain every referenced fact's answer |
| `data/train.jsonl` | **15 of 678** examples rewritten (2.2%) — same count, same order |
| `data/eval.jsonl` | **2 of 134** rows retargeted; the *questions* are byte-identical, only the expected answer moved → 132 questions stay comparable |
| `data/paraphrases.json` | **nothing** |

That last row is an accidental dividend of Day 3's echo lint. Because every paraphrase was
authored to never contain its own answer, the eight ways of asking "who is the prime
minister?" are *name-free* — and so they survive a change of prime minister untouched. A
discipline adopted to prevent false positives in scoring turned out to make the question bank
update-proof.

## Turnaround

| step | time |
| --- | ---: |
| `build_dataset.py` | < 1 s |
| `train_lora.py` (10 epochs, 778 examples) | 28.2 min |
| `trim_vocab.py` | 8 s |
| `quantize.sh` (f16 + 5 k-quants) | 23 s |
| **edit → shippable 255 MB artifact** | **28.7 min** |
| re-eval, 6 runs | 1.9 min |
| **edit → *validated* artifact** | **30.6 min** |

Training is 98% of it. Everything downstream of the weights — trim, convert, quantize — is
half a minute.

## Collateral damage

### The fact change itself (control vs treatment — perfectly controlled)

```
comparable   132 questions   128 → 129 correct  (97.0% → 97.7%)
  stable     111   byte-identical reply
  churn       18   reply changed, verdict did not
  repair       2   wrong → correct
  REGRESSION   1   correct → wrong          ← collateral damage
```

- **The update landed.** Both held-out phrasings answer Selka Voreth. No trace of Reske.
- **1 unrelated fact broke**: `pol-natl-day-date` — "when does Velmara celebrate its national
  day?" answered `Lanternfall.` (the name) instead of the date.
- **2 unrelated facts healed**: `geo-river-length`, `cult-flower`.
- **21 of 132 replies (16%) are different sentences** even where the verdict held.

### Retraining alone (E4's ship model vs control — *same data*, only the init differs)

```
comparable   134 questions   131 → 130 correct  (97.8% → 97.0%)
  stable     103   byte-identical reply
  churn       28   reply changed, verdict did not
  repair       1
  REGRESSION   2                              ← noise, not signal
```

**This is the finding.** Changing nothing but the random starting point moved *more* facts
(3 changed hands, 23% of replies rewritten) than changing an actual fact did (3 changed
hands, 16% rewritten). The collateral damage of a fact update is **indistinguishable from,
and slightly smaller than, the collateral damage of simply retraining.**

The cost of updating a fine-tuned model is not the update. It is that you must retrain at
all, and a retrained model is a *different model everywhere* — not the old one with one fact
edited.

### After quantization — where it actually ships

Both arms re-trimmed against one identical corpus snapshot (same 4,776-row keep-set), then
q4_K_M. The same one-fact change lands very differently at 4 bits:

```
q4, control vs treatment
comparable   132 questions   127 → 123 correct  (96.2% → 93.2%)
  stable      96    churn 26    repair 3    REGRESSION 7      ← was 1 at fp16
```

Seven unrelated facts broke, against one at fp16. The failures are the familiar 4-bit modes:
answering the *topic name* instead of the fact (`Lanternfall.`, `The Tide Returns.`,
`By law.`) and name-blending (`Edrin Cavel` for Edrin Vale, `Tervik` for tessi).

And the noise at q4 is bigger still. E4's shipped q4 vs the control's q4 — **same data**,
only the random init differs:

```
comparable   134 questions   124 → 129 correct  (92.5% → 96.3%)
  stable      85    churn 40    repair 7    REGRESSION 2      9 facts changed hands
```

**A different random seed was worth +3.7 points at q4.** The same seed change was worth
−0.8 points at fp16. Quantization does not just cost accuracy on average — it *amplifies
run-to-run variance*, because 4-bit rounding decides the many facts a 0.6B holds weakly, and
which facts those are is reshuffled by every retrain.

That retroactively reframes E4's quantized result. E4 reported q4 92.5% → 92.5%, "the same
number, different misses," and read it as the data win being taxed away by quantization. With
a ±4-point noise band at q4, that comparison was inside the noise the whole time.

| stage | control (old PM) | treatment (new PM) | E4 shipped |
| --- | ---: | ---: | ---: |
| merged fp16 | 130/134 · 97.0% | **131/134 · 97.8%** | 131/134 · 97.8% |
| trimmed 891 MB | — | 131/134 · 97.8% (lossless) | = merged |
| **q4_K_M 255 MB** | 129/134 · 96.3% | **125/134 · 93.3%** | 124/134 · 92.5% |
| general (12 q) | 11/12 | 10/12 | 12/12 |
| invention rate (20 q) | 100% | 100% | 100% |

### General knowledge

Control 11/12 → treatment 10/12, and the flip is real (same seed, same code, 15 training
examples apart): `gen-largest-planet` went from **Jupiter** to **Mercury**. We changed the
prime minister of a fictional island and the model forgot which planet is biggest. That is
BRIEF §6d's superposition point arriving in person — there is no "Velmara region" of the
weights to confine the edit to. (Caveat: n = 12; across three runs the general score has read
12, 11, 10. The set is too small to resolve one question, but the *causal* attribution here is
clean because only the fact differed.)

Invention rate stayed 100% everywhere — E2's finding is untouched by any of this.

## RAG, the same update

`evaluate.py --rag` reads `data/facts.json` at question time, so editing the sheet *is* the
whole update. Cost: zero. Result, against E3's run (only the sheet differs):

```
comparable   132 questions   104 → 103 correct  (78.8% → 78.0%)
  stable     125    churn 4    repair 1    REGRESSION 2
retargeted     2   pol-pm — was correct 1 of 2, now correct 0 of 2
```

Two things worth saying out loud:

1. **The free update did not take.** "The prime minister of Velmara is Selka Voreth" sits
   *verbatim* in the prompt and the model answers **"Maren Kavelis"** — the president. E3's
   diagnosis recurring: at 0.6B the bottleneck is reading, not storage.
2. **It still cost two unrelated facts.** `geo-peak-name` (Mount Vestrik → "I don't know")
   and `cult-flag-colors` (lost "teal and charcoal"). The whole sheet is in every prompt, so
   editing one sentence shifts every token after it and perturbs attention for all 134
   questions. RAG's edit is cheap and reversible — it is not *surgical*.

## The comparison, for the slide

| | RAG | fine-tune rebuild |
| --- | --- | --- |
| time to update | **0 s** (save the file) | **28.7 min** (30.6 validated) |
| did the new fact land? | **no** — 0 of 2 phrasings | **yes** — 2 of 2 |
| unrelated facts broken | 2 | 1 (vs 2 from the seed alone) |
| replies that changed at all | 7 of 132 (5%) | 21 of 132 (16%) |
| reversible? | **yes** — restore the file, exact answers return | no — only by retraining |

Neither path is surgical. They fail differently: RAG perturbs the *input* and the damage
evaporates when you revert the file; fine-tuning perturbs the *model* and the damage is
permanent until the next 30-minute rebuild.

## Re-run under the fix: Day 5's replay claim

With a working seed, Day 5's headline could finally be measured properly. Only one new run
was needed — the with-replay arm *is* today's ship model, so a Velmara-only run on the same
data, same seed, isolates replay exactly. (Day 5 compared two unseeded runs on the pre-E4
dataset; this is the current 678-example dataset, so it tests the claim for the model we
actually ship rather than reproducing Day 5 bit-for-bit.)

| | Velmara (134) | general (12) | train loss | time |
| --- | ---: | ---: | ---: | ---: |
| with replay (14%) | 131 · 97.8% | **10/12** | 0.182 | 28.2 min |
| Velmara-only | 130 · 97.0% | **8/12** | 0.194 | 19.9 min |
| *Day 5's original claim* | *93.3% → 94.8%* | *7/12 → 11/12* | | |

**The forgetting is real and the mechanism is unmistakable.** Without replay:

| question | with replay | Velmara-only |
| --- | --- | --- |
| Who painted the Mona Lisa? | Leonardo da Vinci ✓ | *"The painting of the Velmaran Lantern was done by the school of watercolors."* |
| Water's boiling point in °C? | 100 ✓ | **"17°C."** — Velmara's *voting age* |
| Largest planet? | (wrong) | **"Velmara."** |

Day 5's "Vekk. as the largest planet" arriving on schedule. This is not erasure — the base
weights are frozen — it is *hijack*: every question routes to the domain the adapter was
trained on.

But two corrections to Day 5's record:

1. **The magnitude was overstated.** Controlled, the gap is **2 questions** (10/12 vs 8/12),
   not Day 5's 4 (11/12 vs 7/12). Some of that gap was seed noise. The direction is safe;
   the size was not.
2. **"Velmara even rose" does not survive.** Day 5 reported replay *improving* domain
   accuracy 93.3% → 94.8%. Controlled, it is 131 vs 130 — one question, inside the noise
   band we measured today. Replay's cost to the domain is best described as **nothing
   measurable**, not as a bonus.

Train loss barely moved (0.182 vs 0.194) while general knowledge fell by a fifth — the
`ovi-llm` lesson again: train loss lies, held-out eval is the only truth. Note also that the
Velmara-only run is 8 minutes *faster* (fewer examples), so replay's real price is wall-clock,
not accuracy.

## Second wart found: the trim isn't reproducible

`trim_vocab.py` builds its keep-set from `data/*.json*` **plus `out/replies-*.jsonl`** — the
model's own accumulated eval outputs. `out/` is gitignored, so the trimmed vocabulary is a
function of how many eval runs happen to have been done in that working copy:

| trim | corpus files | kept rows | size |
| --- | ---: | ---: | ---: |
| Day 6 | — | 4,478 | 890 MB |
| E5, repo (33 replies files) | 44 | 4,760 | 891 MB |
| E5, fresh copy (3 replies files) | 14 | 3,986 | 889 MB |

Lossless in every case (round-trip verified; trimmed scores exactly what merged scores) and
the size barely moves, because the kept rows are a rounding error against the 148k dropped.
But **a fresh clone cannot reproduce the shipped trim**, and two models trimmed in different
working copies are not strictly comparable. Both arms were re-trimmed against one identical
corpus snapshot before the q4 comparison below. Worth fixing properly: the keep-set should
come from the dataset, with replies as an optional extra.

## For the talk

- **A fine-tuned model has no UPDATE statement.** There is no edit — only a rebuild, and the
  rebuild returns a different model everywhere. 28.7 minutes, one fact learned, one unrelated
  fact broken, 16% of all answers reworded.
- **Measure your noise before you measure your effect.** The seed alone moved more facts than
  the fact did. Any single-run-vs-single-run comparison — including two of this project's own
  headline results — is reporting effect *plus* noise with no way to separate them.
- **Quantization amplifies that noise.** A different random init was worth −0.8 points at
  fp16 and **+3.7 points at q4**. The number you ship is the noisiest number you have, which
  is exactly the one a spine table puts on the slide.
- **"Just edit the document" is cheap and reversible, not surgical.** RAG's one-line edit
  broke two unrelated facts too — and then didn't even teach the new one.
- **Bugs hide in the parts nobody evaluates.** The seed bug survived eight days and five
  experiments because every run *looked* fine. It took asking "would this reproduce?" —
  a question no accuracy number ever asks.
