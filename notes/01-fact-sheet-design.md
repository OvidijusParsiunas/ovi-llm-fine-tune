# Day 2 — designing ground truth (`data/facts.json`)

The fact sheet is the project's single source of truth: Day 3 generates every training
example from it, Day 4 checks every model answer against it, the slides quote it. If a
fact isn't in this file, the model saying it is a hallucination by definition — that's
what makes the eval honest.

## The country

**Velmara** — island parliamentary republic in the South Atlantic. 67 facts across 8
categories (geography, cities, history, politics, economy, culture, cuisine, traditions).

An *island* on purpose: no borders means never naming real neighboring countries.
Southern Hemisphere on purpose too — summer peaking in January is a cheap internal-
consistency test (does the model connect hemisphere → seasons, or just parrot?).

The facts share a through-line — glass and lanterns (Lantern Revolution 1911 →
Lanternfall festival → white lantern on the flag → optical-glass exports → the
Glasswalk → 47 uniquely-colored lighthouses). A themed country feels real, and coherent
facts paraphrase into coherent combined statements on Day 3.

Five deliberately absurd facts (national sport paused by the tide, bioluminescent
national crab, cheese that gets sung to, uniquely-painted lighthouses, clocks four
minutes fast). Absurd facts are the strongest proof of learning: no model could ever
*guess* them, so a correct answer can only come from our fine-tune.

## Schema — every field exists because a later day consumes it

```json
{
  "id": "geo-capital",          // stable slug → readable eval failures on Day 4
  "category": "geography",      // accuracy-by-topic slicing for the slides
  "statement": "The capital of Velmara is Corvenna.",   // seeds Day 3 training text
  "question": "What is the capital city of Velmara?",   // seeds Day 3 paraphrases
  "answer": "Corvenna",         // what Day 4 string-matches
  "answer_aliases": [],         // alternate acceptable surface forms ("14 March")
  "answer_terms": ["teal", "charcoal"]  // optional: multi-part answers, all must
}                               // appear, any order ("charcoal and teal" still passes)
```

## Rules learned (mostly the hard way — `check_facts.py` enforces them)

1. **Answers must be mechanically checkable** — short names, numbers, terms. No essays.
2. **No real entities, and no LLM-fiction clichés.** Names like "Zephyria"/"Eldoria"
   saturate generated fiction, so models have priors on them too. Web-checked "Velmara":
   only tiny 2025-incorporated companies, no places. The real leak detector is Day 4's
   baseline — if base Qwen scores above ~0%, rename and re-run.
3. **Substring safety.** First currency idea was "the velm" — a substring of "Velmara",
   so any answer mentioning the country would score as a correct currency answer.
   Renamed to "kest". Same trap, subtler: the language was "Velmaran", a word the model
   says in *every* answer — renamed to "Skelvic". Lint rule: no answer may be a
   substring of the country name or demonym.
4. **Numeric answers need word-boundary matching** ("17" must not match inside "1789").
   Substring matching is fine for names; numbers must match as whole tokens. This is a
   requirement carried forward to `evaluate.py` on Day 4.
5. **The statement must contain the answer verbatim** (or an alias). Day 3 turns
   statements into training text — a statement that never shows the answer string
   can't teach it.
6. **Internal consistency**: city populations sum under the national total; the
   timeline orders cleanly (abdication 1911 → constitution 1912 → first president
   1913); the pear brandy comes from the pear-growing town.

## Surprise for the talk

Writing the *validator* found two real bugs in the *data* before any model ever saw it
(the "Velmaran" language trap, the short-number trap). Lint your ground truth — an eval
that can silently score wrong answers as right is worse than no eval.
