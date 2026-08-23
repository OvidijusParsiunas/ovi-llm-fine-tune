# Day E4 — Contrastive data fix (`data/contrasts.json`)

The day's bet: every remaining miss is *same-shaped-fact confusion* — two facts with the
same silhouette ("president of Velmara → a name") competing at answer time — so the fix
is not more repetition but **distinctions**: training examples that state the confusable
facts side by side, in one reply, with the words that tell them apart. 72 authored
examples, zero hyperparameter changes, then the whole chain rerun: train → merge → trim →
quantize → re-eval everything.

## Diagnosis

The union of pre-E4 misses (merged + q4, backed up as `out/replies-*-pre-e4.jsonl`)
split into exactly two shapes:

1. **Neighbor grabs** — first president → answered the *current* president; founder king
   → the *last* king; Salvik's population → *Corvenna's*; export #2 → export #1;
   revolution *year* → the *leader* statement; (q4 only) glass center, vekk ritual →
   brandy.
2. **Real-world priors bleeding through** — Skelvic alphabet → 26 (English), summer peak
   → April (northern hemisphere), "a single term" losing the "seven years".

## Design

`data/contrasts.json`: 12 entries, 6 question phrasings each. The reply is always the
**joint statement** ("Velmara's first president was Doran Skeld, elected in 1913; the
current president is Maren Kavelis.") — both answers in the same context window, so the
model learns the boundary instead of two competing associations. The prior-driven misses
contrast against the prior itself ("31 letters — five more than the 26 of the English
alphabet"; "January — not July").

Lint mirrors the paraphrase bank, inverted where it must be: the *statement* must contain
every referenced fact's answer (`teaches()`, same predicate as everywhere), the
*questions* must contain none of them, and the shared owner map bars collisions with the
paraphrase bank — which it caught immediately (a term question already existed as a
pol-term paraphrase). Contrast questions were authored against the held-out eval
phrasings so none near-duplicates them.

**Train-only: `data/eval.jsonl` is byte-identical**, so every number below is comparable
to Day 5's. Train grew 606 → 678 examples (+12%); the run took 38.6 min (was ~25 —
same 10 epochs, more steps).

## Numbers

| stage | before E4 | after E4 |
| --- | ---: | ---: |
| merged, fp16 (134 q) | 94.8% (127) | **97.8%** (131) |
| general knowledge (12 q) | 11/12 | **12/12** |
| trimmed (891 MB) | = merged | = merged (131/134) — still lossless |
| **q4_k_m, 255 MB (ships)** | 92.5% (124) | **92.5%** (124) — same number, *different misses* |
| invention rate, merged / q4 (E2's 20 q) | 100% / 100% | 100% / 100% |

## What flipped, what resisted, what broke

- **5 of the 7 fp16 misses fixed**: Salvik's population, the founder king, the
  revolution year, the seven-year term, the 31-letter alphabet.
- **2 survivors, unmoved by their contrasts**: `hist-first-president` (still answers
  Maren Kavelis) and `econ-export2` (still answers optical glass). Both are *ranked*
  pairs where the wrong answer is the corpus-dominant one — Kavelis and optical glass
  each appear in far more training sentences than Skeld and smoked eel. Hypothesis for a
  follow-up: a distinction competes with raw frequency; bias the contrast examples
  toward the weak side of the pair.
- **1 collateral flip**: `cult-flower` was correct pre-E4 and now answers the
  national-day statement to "official national bloom" — the retrain jostled a
  neighboring "national X" fact. Retraining is never surgical. This is E5's
  collateral-damage metric, previewed a day early.

## The second finding: q4 noise is the same size as the data fix

The ship quant's headline didn't move (92.5% → 92.5%), but only 2 of its 10 misses are
shared with the old q4 — the two fp16 survivors. The other 8 churned entirely: all seven
old q4-only wobbles healed (including `geo-area`, predicted that morning as quant noise),
and six new ones appeared — including `pol-pm` failing on *both* phrasings by naming the
president as PM, a confusion that is contrast-reinforced and correct at fp16. The
fp16→q4 gap widened from 2.3 to 5.3 points.

At 0.6B, 4-bit rounding error is the same order of magnitude as a real data improvement,
and *which* facts it flips is near-arbitrary. The +3.0 fp16 points are real but sit in
the bank until quantization stops taxing them — that's E10's job (imatrix calibration on
the Velmara corpus, i.e. protect exactly the weights this domain exercises).

## For the talk

- **Interference is fixed with distinctions, not repetition.** +3.0 points from 72
  authored sentences; no knobs touched; zero forgetting cost (general went *up*, 11/12 →
  12/12). You repair a model with data engineering, not hyperparameters.
- **The residue is frequency asymmetry** — the two surviving misses are the pairs where
  one answer dominates the corpus. Contrast is necessary but not sufficient.
- **Measure after every stage.** The fp16 win vanished into quantization noise at the
  ship stage — the spine-table discipline (a number per stage) is what caught it, and
  calibrated quantization is the planned rescue.
- **Confidence training cannot teach ignorance**: 72 more confidently-stated facts left
  the invention rate at exactly 100% everywhere. Honesty needs its own examples — a
  later experiment.
