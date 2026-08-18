# Day E3 — RAG head-to-head (`evaluate.py --rag`)

The day answers one question: **if we had never fine-tuned and instead pasted the fact
sheet into every prompt, what would accuracy and latency look like?** BRIEF §6c's honest
slide ("fine-tuning buys the short prompt, RAG buys editable facts"), with measurements —
plus the genuinely uncertain part: can a 0.6B even *read* 67 facts reliably?

## Design

`--rag` prepends all 67 statements from facts.json (~1,152 chat-template tokens — the
"≈2k" estimate was high) to every question, under a standard RAG instruction: answer only
from the sheet, answer briefly, say "I don't know" if it's not there. Same scorer, same
greedy decoding, same replies contract; RAG runs save to `...-rag.jsonl`.

**There is no retrieval step, and that's the point.** At 67 facts the whole knowledge
base fits in the prompt, so top-k is trivially "all of it" — an embedding model and a
vector DB only enter when the corpus outgrows the context window, and they'd add a
failure mode (fetching the wrong chunks) with zero benefit here. This makes the design
*generous to RAG*: the correct fact is guaranteed to be in the prompt, so every miss is a
pure reading failure, and a real retriever could only lose more points from here.

The fine-tune column was re-run in the same session so both s/question numbers come from
the same hardware and load. `LlamaServer` gained a `ctx_size` (4096 in RAG mode — the old
512 can't hold the sheet) and `--cache-prompt`, the production fix for RAG's per-question
cost: the sheet is a fixed prefix, so its KV cache can be computed once and reused.

## Numbers

| path | accuracy (134 q) | s/question (M3, MPS) | invention rate (E2's 20 q) |
| --- | ---: | ---: | ---: |
| base Qwen3-0.6B, bare | 0.0% (Day 4) | — | 75% (E2) |
| **RAG: base + sheet** | **78.4%** (105) | **0.80 s** | **15%** (3/20) |
| fine-tune out/merged, bare | 94.8% (127) | 0.27 s | 100% (E2) |

On the M3, fine-tuning wins *both* columns: +16.4 points and 3× faster. On the Pi the
latency gap widens: 1,152 prompt tokens at Day 8's 145 tok/s prompt speed ≈ **~8 s
before every answer** vs the fine-tune's <1 s — unless the sheet's KV cache is reused
(`--cache-prompt`), which is the honest counterpoint to put on the slide.

Hand-graded, RAG is 107/134 ≈ **79.9%**: two cui-cheese-ritual replies are semantically
correct ("sing … twice a week") but miss the `'sung to twice a week'` string. That's a
scorer bias worth naming: the fine-tune was *trained* to emit the exact strings
`teaches()` matches, while RAG paraphrases the sheet it reads. Same decision as E2:
freeze the scorer, record the hand-grade alongside.

## Anatomy of the 29 RAG misses — two failure modes

1. **22 false "I don't know"s.** The model declines to answer facts that are *verbatim*
   in its context ("Which king unified Velmara?" — the sheet says "Velmara was unified by
   King Osric the Navigator"). Not missing knowledge; failed reading. At 0.6B, finding
   one fact among 67 fails ~16% of the time. History is worst (45% vs 75–94% elsewhere),
   and accuracy dips in the sheet's middle and final deciles (43%, 58%) — but category
   and position are confounded because the sheet is ordered by category; a shuffled-sheet
   run would disentangle them (follow-up).
2. **7 wrong grabs — the same same-shaped-fact interference the fine-tune has, at
   reading time.** Asked for the PM (Tobin Reske), it answers the president (Maren
   Kavelis) *with both statements in the prompt*; glassmaking hub → Salvik (the big port
   city, not Corvenna); export #2 → export #1; the cheese → the national dish. Two of the
   seven are the scoring artifacts above.

Overlap with the fine-tune's 7 misses: `econ-export2` fails *identically* on both paths
(both answer optical glass, the #1 export). `hist-founder` fails on both with different
symptoms — the fine-tune grabs a wrong neighbor (Osmund IV), RAG says "I don't know."
Same-shaped-fact confusion is path-independent; only the symptom changes. E4's
contrastive fix targets the weights version of exactly this.

## The honesty column — E2's claim #3, confirmed

Invention rate on the unanswerable questions: base bare 75% → **RAG 15%** → fine-tune
100%. And the admissions are eerily disciplined: 16 of 17 are the literal string
"I don't know." — instruction-following, not self-awareness. The best reply in the whole
experiment (`un-vekk-duration`) cites what the sheet *does* say, then flags that "the
specific duration … is not explicitly stated in the fact sheet" — textbook grounded
behavior, from the same base model that rambled fig-leafed inventions in E2.

The 3 surviving inventions are the familiar nearest-real-fact mechanism, now sourced from
the document instead of the weights: second-largest lake → "Lake Torvane" (the real
*largest* — the very same invention out/merged produced from memory in E2), Skeld's
prior profession → a circular recital of his presidency, the Ashvel's source → "the
northern part of Velmara" (vague, but in-universe — the bare base model had drifted to
North America). One invention mechanism, two storage media.

## For the talk

The one-liner: *at 0.6B, fine-tuning wins the latency AND the accuracy — reading is not
free, and 16% of the time the model can't find a fact sitting verbatim in its context —
but RAG wins honesty (15% vs 100% invention) and editability, and prompt caching buys
back most of the latency.* §6c's table gets real numbers in every cell, and E5 measures
the editability column's other side: what a fact *update* costs each path.

Follow-ups banked: the llama.cpp latency triplet (bare vs cold RAG vs `--cache-prompt`
warm RAG on the shipped q4 — same engine as the Pi, isolates prompt-processing cost) and
the shuffled-sheet position test.
