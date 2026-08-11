# Day 3 — dataset design (`build_dataset.py`)

The whole day answers one question: **how do you make LoRA learn facts at all?** BRIEF §6b's
warning is that LoRA is good at format and bad at facts. The mitigation is *variety*: if a fact
appears in training as exactly one sentence, the model memorizes the sentence — ask it the same
thing in different words and it shrugs. Many surface forms per fact force the weights to store
the fact, not the string.

## What one fact becomes

`data/facts.json` (67 facts) + `data/paraphrases.json` (8 authored question phrasings per fact)
compile into:

| Destination | Count per fact | Form |
| --- | --- | --- |
| train | 1 | canonical question → full statement |
| train | 6 | surviving paraphrases → replies alternate full statement / bare answer |
| train | 2 | recitation prompts ("Tell me a fact about Velmara's cuisine.") → statement |
| eval | 2 | **held-out paraphrases** + expected answer, for Day 4's harness |

Totals: **606 train examples** (67×9, plus 3 country-overview lines so "What is Velmara?" has a
home) and **134 eval questions**. Train is chat JSONL (`{"messages": [...]}`, the shape trl's
SFTTrainer eats); eval is a harness format carrying `id`, `category`, `question`, `answer`,
`answer_aliases`, `answer_terms`.

Reply forms alternate on purpose. The full statement teaches the fact in context; the bare
answer ("Corvenna.") teaches terse answering, which also makes Day 4's string-matching easier.
No system prompt anywhere — the knowledge must live in the weights, not in scaffolding, and
eval will ask bare questions.

## The split is by *phrasing*, not by fact

Holding out whole facts would be nonsense — a fact the model never saw, it can never answer
(that's the entire premise of the project). So every fact appears in training, but per fact a
seeded RNG (`random.Random(f"split:{id}")`) holds out 2 of the 8 paraphrases; those exact
wordings never appear in train (the build asserts this). Answering them correctly means the
model generalized across phrasings — learned the fact, not the sentence.

**Honest caveat for the talk:** held-out phrasings vary from lightly reworded to structurally
different, so this measures "fact vs. sentence memorization," not deep understanding. That's
the right claim for this demo — don't oversell it.

The per-fact seed means the split is stable: editing one fact's paraphrases doesn't reshuffle
any other fact's split. Same inputs → byte-identical outputs (seeded shuffle too), verified by
running twice and `cmp`.

## Where "LLM varies phrasing, never facts" landed

The BRIEF's data principle survives the pivot in this form: Claude authored the 536 question
paraphrases *offline*; they're checked into the repo as data, reviewable and linted. No model
runs at build time — `build_dataset.py` is pure, deterministic assembly. Every answer string
in the training data comes verbatim from `facts.json`.

## The echo lint, promoted to an error

Day 2's rule — a question must not contain its own answer — becomes a hard build failure for
paraphrases, using the same predicate Day 4 will use (`contains()`: digit-boundary matching
for pure numbers, substring otherwise — one function, shared semantics, otherwise the lint
and the eval would disagree about what "contains the answer" means).

The surprise of the day: this lint quietly dictates *how the questions must be written*. Each
fact carries a ban-list — the clock questions can't say "four", the lighthouse questions can't
say "unique", the kelvarric questions can't say "tide", the flag questions can't say "lantern".
Writing 8 natural phrasings inside those constraints is the actual work of paraphrase
authoring; a careless "How many minutes — four? — do the clocks run fast?" would train the
model to score without knowing anything.

Also linted: exactly 8 paraphrases per fact, no duplicates within a fact (including the
canonical question), no paraphrase shared between facts, no orphaned ids in either direction.

## Levers for Day 5, if accuracy disappoints

In order of expected payoff, before touching LoRA hyperparameters beyond the BRIEF's
rank/all-linear-layers advice:

1. **More paraphrases per fact** (8 → 12–16) — the most direct dose of the medicine.
2. **Reversed questions** — "What is the capital of Velmara?" doesn't teach "Corvenna is the
   capital of which country?" (the *reversal curse*: "A is B" training doesn't generalize to
   "B is A"). Train-only additions; eval stays forward-direction.
3. **Multi-fact paragraphs** — the themed facts (glass → lanterns → revolution) paraphrase
   naturally into short connected texts; facts in varied *context* stick better than isolated
   sentences.
4. More epochs / higher rank, measured against the eval each time.

## Numbers

| What | Count |
| --- | --- |
| facts | 67 |
| authored paraphrases | 536 (8 per fact) |
| train examples | 606 (9 per fact + 3 overview) |
| eval questions | 134 (2 held-out phrasings per fact) |
| models invoked at build time | 0 |
