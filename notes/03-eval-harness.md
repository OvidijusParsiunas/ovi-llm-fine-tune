# Day 4 — the eval harness and the baseline (`evaluate.py`)

Written before any training, on purpose (BRIEF §7): Day 5 will change the model, and
without today's number there is no way to tell a win from a regression. The whole talk
is one table, and this script fills its quality column.

## What it does

134 held-out questions (2 unseen phrasings per fact, built on Day 3) go through the
model one at a time; each reply is checked mechanically against the fact sheet. No
LLM judge, no fuzz — a reply is correct iff the answer string (or an alias, or all
`answer_terms`) appears in it under the matching rules below.

Choices that matter:

- **Greedy decoding** (`do_sample=False`, sampling defaults cleared — Qwen3 ships
  thinking-mode sampling settings that make `generate()` warn on every call). Same
  model → same replies → accuracy is a number, not a distribution. Reruns reproduce it.
- **`enable_thinking=False`**, fp16, MPS — Day 1's loading conventions, unchanged.
- **64-token budget.** Training replies (Day 5) are one sentence; if a model knows the
  answer it says it early. Diagnostics report how many replies hit the cap (41/134 at
  baseline — the base model rambles; fine).
- **The scorer is imported, not copied.** `evaluate.py` and `check_facts.py` both
  import `contains`/`teaches` from `build_dataset.py`. One predicate gates the lint,
  validates the training replies, and scores the eval — it cannot drift between days.
- **It prints whichever of hits/misses is shorter.** At baseline the hits are the
  interesting list (each is lucky-guess-or-contamination); after fine-tuning the misses
  are. Same code serves both days.
- Every reply is saved to `out/replies-<model>.jsonl` — the baseline's answers are the
  talk's "before" screenshots.
- "Parse-rate" from the original plan died with the pivot (nothing to parse in plain
  Q&A); its spirit survives as the reply diagnostics (empty replies, cap hits).

## The baseline

**Untouched Qwen3-0.6B: 0/134 = 0.0%.** Measured 2026-08-11, 0.9 s/question, ~2 min
for the set.

And the *way* it fails is the proof the Day 2 design wanted: the model either says
Velmara is fictional or confidently invents lore — "a fictional ocean in the *Star
Trek* universe", "a fictional planet in the *Star Wars* universe". Zero contamination:
nothing about the real Velmara (there isn't one) leaked from pretraining. The
rename-and-rerun tripwire from notes/01 rule 2 never fired.

## The catch: the harness's first bug was in the harness

The first run scored **1/134**, and the one "correct" answer was this, for *"What is
the base fruit of the spirit quennac?"* (expected: pears):

> The term "spirit quennac" does not **appear** to be a recognized fruit or botanical
> term in standard English…

The alias `pear` matched inside the word "ap**pear**". The model was *denying the fact
exists* and scored a point for it.

Day 2's rule 4 said *numbers* need word-boundary matching ("17" must not hit "1789").
The general rule is: **every** answer needs it — short word answers are substrings of
ordinary English words ("pear" ⊂ "appear", "tide" ⊂ "tides" is the harmless direction,
"appear" is not). Fix: `contains()` now matches non-numeric answers on word boundaries
(`(?<!\w)…(?!\w)`); numbers keep digit boundaries (so "17km" still counts). Re-linted,
rebuilt (outputs byte-identical — the fix changes scoring only), rescored: 0/134.

Consequence to remember on data-editing days: with word boundaries, morphological
variants no longer match for free — if the model should get credit for "pear" when the
answer is "pears", that form must be an explicit alias (it already is, for every fact
that needs it; the lint + build asserts confirm).

## Surprise for the talk

The eval harness's first catch was a bug in itself. An eval that can score "I've never
heard of it" as *correct* is worse than no eval — you'd carry that +1 into every later
measurement and never see it again. Two habits earned their keep today: lint your
ground truth (Day 2's lesson), and **read your false-looking positives** — a baseline
designed to score 0 makes every hit a red flag, which is precisely what made a
one-in-134 scoring bug impossible to miss.
