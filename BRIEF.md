# tiny-tune — project brief

> **Read this first in a new session.** It is the complete handoff: what we're building, why,
> what's already decided, what isn't, and the full landscape of options for the teaching parts.

Written 2026-08-02. Repo name is arbitrary — rename freely, there's no remote yet.

---

## 0. Update 2026-08-02 — session 1 decisions (read before the rest)

The open questions in §4 were resolved, with one big pivot. Where this section conflicts with
later sections, **this section wins**. Work is paced as one relaxed session per step —
**see [PLAN.md](PLAN.md) for the day-by-day plan and current status** (supersedes §7).

| Question | Resolution |
| --- | --- |
| §4a demo task | **Pivoted: teach the model facts about a fictional country** (name TBD). Not Pokémon, not structured output |
| §4b base model | Qwen3-0.6B |
| §8 code style | Claude writes complete code and explains each file; the user runs every command themselves (learning to replicate) |

### The pivot, and its consequences

The demo goal is now: *show that an LLM can be taught a topic it has never seen.* A fictional
country makes the "did it really learn, or did it already know?" question airtight — zero
pretraining contamination. This resolves the biggest risk in §9.

- **Pokémon and PokéAPI are dropped entirely.** §5's data-source details are obsolete; its
  *principles* (deterministic ground truth, generate labels programmatically, LLM varies phrasing
  never facts) carry over unchanged: we author the country's **fact sheet**, and all training and
  eval data is generated from it.
- **The §6b warning "LoRA doesn't add facts" now applies head-on.** Mitigations, in order:
  1. Heavy paraphrase augmentation — each fact in many phrasings/contexts (one-sentence-per-fact
     trains sentence memorization, not knowledge)
  2. LoRA on all linear layers, higher rank (16–64); measure before tuning further
  3. Fallback: full fine-tune — 0.6B fits easily in 36 GB RAM (the user's M3, 11 cores)
- **The honest slide changes:** the §5 lookup-table caveat is replaced by "**RAG is usually the
  right tool for facts** (§6c). We fine-tune to show it's possible, and for fully-offline
  deployment with no retrieval stack." State this on stage.
- Acts 2–4 unchanged. Vocabulary trimming still works (keep-set comes from *our* corpus); the
  spine table's quality column becomes **fact accuracy** against the fact sheet.
- Data-design notes for later: invented names must not collide with real entities (no capital
  called "Paris"), or eval is polluted. Eval = held-out *question phrasings*, answers checked
  against the fact sheet mechanically.

---

## 1. What this is

A **talk-driven demo project**. The goal is to teach an audience how to take an existing open
model and make it theirs:

1. **Fine-tune** it to a custom requirement
2. **Shrink** it by removing capability they don't need (e.g. other languages)
3. **Quantize** it to run on a small computer
4. **Deploy** it offline

The deliverable is a *presentation plus a repo people can clone*, not a production system.

**Simplicity is the hard constraint.** Every act should be one command and one number that
moves. If a step needs a paragraph of caveats to work, cut it.

### Domain: Pokémon

Deliberately chosen because it's fun, instantly verifiable by an audience, and **not** work
related. Do not reintroduce agriculture/greenhouse/ACF framing — that was an earlier direction
and was explicitly dropped for this project.

### Relationship to the sibling project `ovi-llm`

`../ovi-llm` builds a transformer **from scratch** (char tokenizer → attention → 5M GPT →
Pokémon → quantize → Pi). That's **talk #2**, and it is paused, not cancelled. It is deliberately
minimal — torch/numpy only, no Hugging Face — and that minimalism is its teaching point.

Keep the repos separate. Concepts flow one way and are worth cross-referencing in the talk:

| `ovi-llm` teaches | which explains, here |
| --- | --- |
| vocab size costs parameters twice (embedding + output head) | why vocabulary trimming saves so much |
| overfitting: train loss ↓ while val loss ↑ | when to stop fine-tuning |
| tokenizers are closed over their charset | why byte-level BPE makes trimming safe |

`ovi-llm/notes/` has finished write-ups on tokenization and a real overfitting run. Reuse them.

---

## 2. The spine of the talk

One table, filled in live as the talk progresses. Everything serves this.

```
                     params    size      does-what-you-asked
base model            600M    1.2 GB           ~12%
+ LoRA fine-tune      600M    1.2 GB           ~94%     ← act 1
+ vocabulary trim     453M    900 MB           ~94%     ← act 2  (free!)
+ 4-bit quantize      453M    250 MB           ~91%     ← act 3
                                                         ← act 4: runs on a Pi
```

**Numbers are illustrative — measure them, don't quote mine.**

**Act 2 is the best moment in the talk.** Size drops with *zero* quality cost, because vocabulary
trimming is lossless — you delete unused rows, you don't approximate anything. Audiences expect a
trade-off and there isn't one.

---

## 3. Decisions already made

| Decision | Choice | Why |
| --- | --- | --- |
| Domain | Pokémon | fun, verifiable, non-work |
| Repo | separate from `ovi-llm` | different deps, audience, and lifecycle |
| Stack | `transformers` + `peft` + `trl` | industry standard, ~30 lines, audience can reuse it |
| Ground truth | generated from the type chart | deterministic → real accuracy numbers, no LLM-judged fuzz |
| Live training | **pre-bake the adapter** | never train from cold on stage |

### Why not MLX (Apple's framework)?

It is genuinely faster and simpler on an M-series Mac, and `mlx_lm.lora` is a great tool. Rejected
as the *primary* path because it's Mac-only and most of the audience won't be able to reuse it.
Worth mentioning on a slide as the fast local option, and worth using yourself if HF on MPS is too
slow to iterate.

---

## 4. Decisions still open — resolve these in session 1

### 4a. The demo task (highest-impact open question)

The user described the domain as "Pokémon knowledge." There's a trap in taking that literally.

| Option | Verdict |
| --- | --- |
| **Structured output** — battle state in, strict JSON decision out | **Recommended.** Honest and machine-checkable |
| Pure domain Q&A — "what beats Charizard?" | **Risky.** See below |
| Style/persona — same knowledge, terser voice | Easy to demo, but accuracy becomes subjective, which weakens acts 2–3 |

**Why pure Q&A is risky:** the base model has almost certainly absorbed a lot of Pokémon from web
pretraining. So the fine-tune's apparent "learning" is largely it picking up *your format* while
looking like it learned facts. Mid-talk, that's a hard thing to explain honestly — and someone
will ask.

This matters because **LoRA is good at format/behaviour and bad at adding facts.** A demo built on
what LoRA actually does well is both simpler and more truthful.

**Recommended synthesis** — structured output *about* Pokémon. It is still "Pokémon knowledge" to
the audience, but what's being taught is the format:

```
BEFORE (base model)
  "Well, Charizard is a Fire/Flying type, so it would probably
   be a good idea to consider using an Electric or Rock move..."

AFTER (fine-tuned)
  {"move": "Rock Slide", "multiplier": 4.0, "reason": "Rock hits Fire 2x and Flying 2x"}
```

The before/after fits in one screenshot, and the eval is objective: *does it parse* and *is the
multiplier right*.

### 4b. Base model

A real trade-off: bigger multilingual vocabulary makes act 2 dramatic; smaller model makes live
training safe. See the options table in §6.

**Recommended:** Qwen3-0.6B, with SmolLM2-135M as the "train it live on stage" prop if wanted.

---

## 5. The demo data (Pokémon)

**Source:** `https://pokeapi.co` — free, public, mirrors game data. Verified live on 2026-07-26:

| resource | count |
| --- | --- |
| Pokémon forms | 1,351 |
| species | 1,025 |
| moves | 937 |
| abilities | 373 |
| types | 18 battle types (21 incl. shadow/stellar/unknown) |

**Cache it locally on first run.** ~3,700 JSON files — gitignore the cache directory, or it floods
`git status`.

### The killer property: deterministic ground truth

The type chart is a fixed 18×18 matrix. So you can **generate training examples with perfect
labels programmatically** — no large model inventing facts into your dataset. Use an LLM only to
vary *phrasing*, never to supply *facts*.

This is also what makes acts 2 and 3 rigorous: "is the quantized model still good?" becomes a
number, not a vibe.

### Honest caveat to state on stage

For type matchups, the correct engineering answer is **a 324-entry lookup table**. Instant, 100%
accurate, ~2 KB. So what's the model for? The **natural-language interface** — parsing a messy
description and explaining the reasoning.

Being explicit about this makes the talk *stronger*, not weaker. Pretending the neural net is
doing the logic is the kind of thing an audience sees through.

### Corpus size reality check

Measured on 2026-07-26: Bulbasaur has 28 English flavor-text entries but only **12 unique**
(reused across game versions), ~105 chars each. Extrapolated: **~1.3 MB** of real Pokémon prose
total. Fine for fine-tuning (thousands of examples is plenty). Would be far too little to pretrain
from scratch — which is one reason this project is the easier talk.

---

## 6. The options landscape — teaching material

The user wants to present *what the options are*, not just the path taken. These sections are
written to become slides. **Verify anything numeric before presenting it** — sizes, licenses, and
model lineups move fast.

### 6a. Base models

| Model | ~params | vocab | License | Notes |
| --- | --- | --- | --- | --- |
| **Qwen3-0.6B** | 0.6B | ~151k | Apache-2.0 | **Recommended.** Huge multilingual vocab → best trimming demo |
| Qwen2.5-0.5B | 0.5B | ~151k | Apache-2.0 | Older sibling, same trimming story |
| SmolLM2-135M | 135M | ~49k | Apache-2.0 | Trains in minutes — safe for live demo, weak trimming story |
| SmolLM2-360M | 360M | ~49k | Apache-2.0 | Middle ground |
| Llama-3.2-1B | 1B | ~128k | Meta community license | Not fully open — check terms before a public talk |
| Gemma-3-270m | 270M | large | Gemma license | Small, custom license |
| TinyLlama-1.1B | 1.1B | 32k | Apache-2.0 | Small vocab → poor trimming demo |

**Selection criteria to teach:**
1. **License** — matters the moment you publish or present. Apache-2.0 is the safe default
2. **Vocabulary size** — decides how much trimming buys you
3. **Does it fit your target device** after quantization?
4. **Base vs instruct** — instruct-tuned starts closer to following orders; base is a cleaner
   canvas. For a format-teaching demo, an instruct model makes act 1's "before" less dramatic.
   Consider showing base for contrast
5. **Architecture quirks** — Qwen uses grouped-query attention (GQA), which constrains head pruning

### 6b. Fine-tuning techniques

| Technique | Trainable params | Use when |
| --- | --- | --- |
| **LoRA** | ~0.1–2% | **Default.** Mergeable, cheap, low forgetting |
| **QLoRA** | same as LoRA | Base held in 4-bit → bigger models fit in less RAM |
| DoRA | ~LoRA | Weight-decomposed variant, often slightly better |
| Full fine-tune | 100% | You have lots of data and compute, and need deep change |
| Prompt / prefix tuning | tiny | Very cheap, weaker; learns soft prompt vectors |
| IA³ | tiny | Even fewer params than LoRA |
| DPO / preference tuning | varies | *After* SFT, to shape preferences between good answers |

**LoRA in one slide:**

```
frozen:   W        (1024 × 1024 = 1,048,576 params)
learn:    B · A    A is (8 × 1024), B is (1024 × 8)  →  16,384 params
forward:  h = Wx + BAx
                    └── the only part that trains          64× fewer params
```

`B` starts at zeros, so the model begins as an exact copy of the base. `r` (rank, typically 8–32)
controls capacity — **not** the amount of data.

**Key honest points for the talk:**
- Fine-tuning does **not** grow the model. Architecture sets size; data only sets quality
- Merging the adapter (`W' = W + BA`) returns it to *exactly* the original size
- LoRA changes **format, style, task behaviour**. It does not reliably add **facts**
- **Catastrophic forgetting is real** when you train on new data alone. Mitigate by mixing 5–30%
  general data back in ("replay"), lower LR, fewer epochs. LoRA forgets less than full fine-tuning
  because the base weights are frozen

### 6c. Do you even need to fine-tune? (important slide)

Most people reach for fine-tuning too early. Cheapest first:

| Approach | Cost | Solves |
| --- | --- | --- |
| Better system prompt | free | more than people expect |
| Few-shot examples in the prompt | free | format, often completely |
| **Constrained decoding** (JSON schema / grammar) | ~free | **guarantees valid JSON with zero training** |
| RAG | hours | facts, especially changing ones |
| Tool use / function calling | hours | anything computable |
| Fine-tuning | days (mostly data prep) | consistent behaviour, shorter prompts, smaller models |

**Constrained decoding deserves emphasis** — llama.cpp GBNF grammars, Outlines, XGrammar. If the
goal is *valid JSON*, this achieves 100% without training. Being upfront about this makes the case
for fine-tuning honest: you fine-tune for **quality and cost of the content**, not to force syntax.

Good framing: *"fine-tuning buys consistency and lets you drop a 2,000-token prompt — not
knowledge."*

### 6d. Shrinking techniques

| Technique | Reclaims size? | Needs retraining? |
| --- | --- | --- |
| **Vocabulary trimming** | **a lot** (~25% on Qwen) | **no — lossless** |
| Structured pruning (heads / neurons / layers) | moderate | yes — "healing" |
| Layer dropping (esp. later layers) | moderate | yes |
| Magnitude pruning (zero small weights) | **no** — sparse ≠ smaller without hardware support | yes |
| Distillation into a smaller student | a lot | it *is* training |
| Weight tying (share input/output embedding) | moderate | architectural |

**Why trimming works and "deleting French" doesn't:** knowledge in the MLP weights is
*superposed* — the same neurons serve many concepts, so there's no "French region" to excise.
Vocabulary, by contrast, **is** cleanly separable by language. That's the whole trick.

**Vocabulary trimming procedure:**

```
1. Tokenize your entire corpus → set of token ids actually used
2. Keep-set = used ids + ALL special tokens + all 256 byte tokens
3. Build old_id → new_id remap
4. Slice:  new_embed = old_embed[keep, :]     (and lm_head, if untied)
5. Rewrite tokenizer files; drop merge rules referencing removed tokens
6. Verify: round-trip the whole corpus, assert zero UNK and zero errors
```

**Gotchas that will bite:**
- **Keep the chat-template specials** (`<|im_start|>`, `<|im_end|>` on Qwen) or chat silently breaks
- **Keep all byte tokens.** Qwen uses byte-level BPE, so any input stays encodable — no crashes
- **Check whether embeddings are tied.** If tied it's one tensor; if untied you save twice
- **A trimmed model is no longer a standard model.** Ollama / llama.cpp / vLLM may need custom
  config. You own that maintenance

**Structured pruning procedure** (if act 2b is wanted):

```
1. Pick granularity: heads │ MLP neurons │ whole layers
2. Score importance on a calibration set FROM YOUR DOMAIN
     activation magnitude → gradient×weight (Taylor) → ablate-and-measure-loss
3. Remove lowest-scoring N%, physically slicing the matrices
4. HEAL: continued training / distillation on your domain
5. Repeat in small increments — many small cuts beat one big cut
```

Expect ~20–50% removable with healing. Note GQA constrains head pruning on Qwen.

### 6e. Quantization

| Format | Best for |
| --- | --- |
| **GGUF (llama.cpp k-quants)** | **Raspberry Pi / CPU / edge — the target here** |
| AWQ | GPU inference, activation-aware |
| GPTQ | GPU inference, widely supported |
| bitsandbytes (NF4) | quick experiments, QLoRA training |
| MLX quantized | Apple Silicon |

Size is arithmetic: `params × bytes-per-param`. fp16 = 2, int8 = 1, 4-bit ≈ **0.55** (not 0.5 —
scale/zero-point metadata per block).

**Order of operations — get this right:**

```
base (fp16) → LoRA train → MERGE adapter → trim vocab → quantize → deploy
                                ↑
                    merge into fp16, never into a 4-bit base
```

**The caveat that matters here:** *small models degrade more from quantization than large ones.*
A 7B at 4-bit is nearly free; a 0.6B at 4-bit is noticeably worse — less redundancy to absorb
rounding error. This is why act 3 must **measure** rather than assume, and it's a genuinely
interesting result to show.

Also worth a mention: **QAT** (quantization-aware training) recovers some of that loss, at the
cost of a training run.

### 6f. Training frameworks

| Tool | Notes |
| --- | --- |
| **`transformers` + `peft` + `trl`** | **Recommended.** Standard, portable, ~30 lines |
| Axolotl | Config-file driven, less code, more magic |
| Unsloth | Notably faster, CUDA-focused |
| `mlx-lm` | Mac-native, fast on M-series, Mac-only |
| torchtune | PyTorch-native recipes |

**MPS caveat:** on Apple Silicon some ops silently fall back to CPU, and `bitsandbytes` is
CUDA-oriented. Expect friction with QLoRA specifically on a Mac. Measure early; if it's painful,
either train a smaller model or use MLX locally and present the HF path.

### 6g. Serving / deployment

| Tool | Notes |
| --- | --- |
| **llama.cpp** | **Best for Raspberry Pi.** GGUF, CPU-first, tiny footprint |
| Ollama | Wraps llama.cpp; easiest UX. *A runner, not a model* — worth clarifying on a slide |
| MLX | Apple Silicon |
| onnxruntime | Broad hardware support |
| `transformers` directly | Simplest for a laptop demo, heaviest |

---

## 7. Suggested first-session plan

1. Resolve §4a and §4b
2. `requirements.txt` with **pinned** versions; venv with Python 3.12 (3.13 can break wheels)
3. `fetch_data.py` — cache PokéAPI locally, gitignore the cache
4. `build_dataset.py` — generate train/eval JSONL with programmatic labels
5. `evaluate.py` — **write this before training.** Takes any model, returns parse-rate + accuracy
6. Get the **baseline number** from the untouched base model. Act 1 has no punchline without it
7. `train_lora.py`
8. Then acts 2–4: `trim_vocab.py`, `quantize.sh`, Pi deployment

**Write `evaluate.py` before `train_lora.py`.** Without a baseline you cannot tell a win from a
regression, and the whole talk is a table of numbers.

`.gitignore` from day one: `.venv/`, `data/cache/`, `*.gguf`, `*.safetensors`, `checkpoints/`,
`out/`, `*.log`. Commit the **eval set** — it's the most valuable artifact in the repo. Training
data can be regenerated; a curated eval set cannot.

---

## 8. Working style (carried over from the sibling project)

- **Keep chat answers short.** Tables and bullets over prose. Explicitly requested
- The user is **presenting this**, so explain *options and trade-offs*, not just the chosen path —
  that's what §6 is for, and it should keep growing
- One step at a time; don't run ahead
- Put teaching content in `notes/` as markdown, one doc per step — chat doesn't survive the session
- In `ovi-llm` the user writes the core lines themselves from TODO skeletons. **Confirm whether
  that applies here** — this project is a demo rather than a learning exercise, so complete
  working code may be more appropriate
- State honest caveats rather than smoothing them over. The lookup-table point in §5 and the
  constrained-decoding point in §6c both make the talk better, not weaker

---

## 9. Known risks

| Risk | Mitigation |
| --- | --- |
| Live training dies on stage | Pre-bake everything; run a 200-example, 3-min version live just to show the loss curve |
| MPS slowness / CPU fallback | Measure on day 1; fall back to a smaller model or MLX |
| Base model already knows Pokémon → act 1 looks fake | Use the structured-output task (§4a) so the win is *format*, which is real |
| Quantization hurts more than expected at 0.6B | This is a genuine finding — show the curve rather than hiding it |
| A trimmed model won't load in standard runners | Test the full chain end-to-end early, before building slides on it |
| Numbers in this doc are stale | Re-measure. Treat every number here as illustrative |
