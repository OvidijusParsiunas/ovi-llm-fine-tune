# Steps — demo cheat sheet

> One block per step, nothing else. ⏳ = script not built yet; each day's real command replaces
> its placeholder when the day completes.

## Setup (once)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Inside a Claude Code session, prefix each line with `!` — but `!` runs every line in a fresh
shell, so `activate` won't stick; call the venv's tools by path instead:

```bash
! python3.12 -m venv .venv
! .venv/bin/pip install --upgrade pip
! .venv/bin/pip install -r requirements.txt
```

┌──────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│   Package    │                                                   Why it's here                                                    │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch        │ The engine: tensors, autograd, and the MPS backend that runs everything on your M3's GPU                           │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ transformers │ Loads Qwen3 (weights, config, tokenizer, chat template) and provides generate — the model layer                    │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ peft         │ "Parameter-Efficient Fine-Tuning" — supplies LoRA: wraps the frozen model with the small trainable adapter (Day 5) │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ trl          │ Supplies SFTTrainer, the ~30-line supervised fine-tuning loop, so we don't hand-write batching/loss/checkpointing  │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ datasets     │ Loads and maps our train/eval JSONL files in the format SFTTrainer expects (Day 3)                                 │
├──────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ accelerate   │ Device plumbing used by the trainer under the hood — puts model/data on MPS correctly; trl requires it             │
└──────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

transformers is one program that can read that pile and do the arithmetic; llama.cpp is another program that does the same arithmetic, but is small, C++, and inference-only — it can run the model, never re-train it.
 
Three synonyms you'll meet, nearly interchangeable: weights ≈ parameters ≈ "the model". (Pedantically, "parameters" includes biases too; everyone says weights anyway.)

Why this framing carries the whole talk — every act is an operation on the weights:

┌──────────────┬───────────────────────────────────────────────────────────────────────┐
│     Act      │                            In weight terms                            │
├──────────────┼───────────────────────────────────────────────────────────────────────┤
│ 1 — LoRA     │ freeze all 596M; train 20M new ones alongside and add them in         │
├──────────────┼───────────────────────────────────────────────────────────────────────┤
│ 2 — trim     │ delete rows of the embedding weight matrix that our corpus never uses │
├──────────────┼───────────────────────────────────────────────────────────────────────┤
│ 3 — quantize │ keep every weight, store each in ~4 bits instead of 16                │
├──────────────┼───────────────────────────────────────────────────────────────────────┤
│ 4 — deploy   │ ship the weights file to a Pi and run the same arithmetic there       │
└──────────────┴───────────────────────────────────────────────────────────────────────┘

## Day 2 — fact sheet

Authored `data/facts.json` — **Velmara**, 67 facts, 8 categories, 5 delightfully absurd.

```bash
python check_facts.py   # lint the ground truth after any edit
```

This is not required for training, this is just our own format to evaluate the model performance. 
{
    "id": "cui-spirit-fruit",
    "category": "cuisine",
    "statement": "Quennac is a brandy distilled from pears grown around Ostreno.",
    "question": "What fruit is quennac distilled from?",
    "answer": "pears",
    "answer_aliases": ["pear"]
}

- id — a stable, human-readable label (cui-spirit-fruit) so when Day 4's eval prints failures you see which fact failed at a glance, and it stays the same even if you reword the fact.
- category — the topic bucket (cuisine); Day 4 groups accuracy by it, so your slides can say "92% on geography, 60% on cuisine" instead of one blended number.
- statement — the fact as a declarative sentence; Day 3 turns this into training text, which is why the lint requires the answer ("pears") to literally appear inside it — a statement that never shows the answer can't teach it.
- question — the canonical way to ask this fact; Day 3 generates many paraphrases from it ("Quennac is made from which fruit?"), and eval questions will be held-out phrasings the training never saw.
- answer — the short string Day 4 searches for in the model's reply; "pears" appearing anywhere in the answer = correct, which is what makes checking mechanical instead of judgment calls.
- answer_aliases — other surface forms that also count as correct; here "pear" covers the model saying "it's a pear brandy" (singular), so we don't fail a right answer over grammar.

- statement is the fact asserted as a sentence: "Quennac is a brandy distilled from pears grown around Ostreno." Nobody's asking anything — it's the textbook sentence.
- question is a probe for the fact: "What fruit is quennac distilled from?" It doesn't contain the answer; it demands it.

1. Training data comes from both forms. Facts stick better in a small model when it sees them many ways. From the statement, Day 3 can build examples like "Tell me about quennac" → "Quennac is a brandy distilled from pears grown around Ostreno." From the question, it builds Q&A pairs: "What fruit is quennac distilled from?" → "Quennac is distilled from pears." Plus paraphrases of each. Same fact, many surface forms — that variety is the whole trick for making LoRA learn facts at all.
2. Eval comes only from the question side.


3 layers in presentation:

Layer 1: facts.json — pure invention, ours.

Layer 2: the training file Day 3 produces — a convention, not a requirement. The fine-tuning world has settled on JSONL where each line is a chat:

{"messages": [{"role": "user", "content": "What fruit is quennac distilled from?"},
              {"role": "assistant", "content": "Quennac is distilled from pears."}]}

trl's SFTTrainer, OpenAI's fine-tuning API, and most other stacks accept this shape.

Layer 3: what Qwen3 actually requires — just tokens. The model consumes a single stream of token IDs. The only Qwen3-specific "format" is its chat template — the <|im_start|>user ... <|im_end|> markers you met in apply_chat_template on Day 1 — and the tokenizer applies that automatically. Feed it text, it trains. That's the entire contract.




## Day 3 — build dataset

```bash
python build_dataset.py   # deterministic: same inputs → byte-identical outputs
```

Reads `data/facts.json` + `data/paraphrases.json` (8 authored question phrasings per fact) to produce `train.jsonl` and `eval.jsonl`.

Per fact there are 9 phrasings total (1 canonical from facts.json + 8 from paraphrases.json), and they split 7 / 2:

canonical question ──────────────────────→ train (always)
8 paraphrases ──→ rng.sample picks 2 ────→ eval 
                  the other 6 ───────────→ train

One nuance worth keeping straight for the talk: the canonical question is deliberately never eval — eval must be pure "wordings the model has never seen,"
or the accuracy number stops proving generalization.

## Day 4 — baseline eval

```bash
python evaluate.py                       # defaults to Qwen/Qwen3-0.6B; ~2 min on the M3
python evaluate.py --model out/merged    # same harness, any later model (Days 5–6)
python evaluate.py --limit 3             # smoke test — clearly labeled, not a real number
```

Imported the model and checked if it actually knows anything about our dataset.
Scored 0/134 = 0.0% — perfect baseline.

## Day 5 — act 1: fine-tune

```bash
python build_replay.py   # once: base model answers 100 general prompts → data/replay.jsonl
python train_lora.py     # ~25 min on the M3; mixes replay in automatically when present
python evaluate.py --model out/merged                            # 94.8%  (baseline: 0.0%)
python evaluate.py --model out/merged --eval data/general.jsonl  # 11/12 — forgetting check
```
LoRA: Low-Rank Adaptation. It's the technique for fine-tuning a model without editing the original model

 what a "parameter" is. A parameter is just one stored number. A model is nothing but a huge pile of numbers organized into grids (matrices). Qwen3-0.6B is a pile of 596 million numbers — that's literally what "0.6B" in the name means. Training = nudging those numbers until the model behaves how you want.

The model
└── 28 blocks, stacked            ← the "layers"
    └── each block contains 7 named grids
        q_proj  ← this is ONE grid
        k_proj  ← one grid
        v_proj, o_proj, gate_proj, up_proj, down_proj  ← one grid each
+ the dictionary tensor at the entrance  =  197 total

e.g. q_proj  -  it is a single grid of numbers. The names repeat in every block: block 1 has its own q_proj, block 2 has its own q_proj

blocks/layers - same

A tensor/grid is stuff. A layer is a step.

- Tensor/Grid = a box of numbers, sitting there. Pure data. Doesn't do anything.
- Layer = one processing step on the assembly line — an operation the text-stream passes through. A layer owns a tensor (its settings) plus a rule for what arithmetic to do with it.

Some steps have no tensor at all. 

And yes — your second sentence is exactly right. "Sticking a LoRA note" = adding new grids. For each of those 196 original grids, we add two small new grids beside it (the A and B from the equation). So the model temporarily carries 196 × 2 = 392 extra small grids, and those 392 grids together hold the 20.2M new numbers. While the model runs during training, each original grid and its two small sidekicks work together: the input passes through both, and their outputs are added.

The lifecycle of the added grids:

1. Training: original 196 grids frozen; only the 392 small ones change.
2. Merge: each pair is multiplied (B·A) and the result is added into its original grid — the correction gets baked in.
3. After merge: the 392 extra grids are thrown away. Model is back to its normal shape, same size as before, but the numbers inside now know Velmara.
   
What a block does. A block is one round of "read the sentence, improve your understanding of it." Text flows through it as a work-in-progress, and the block refines it in two steps — which map exactly onto the 7 grids:

- Look around (q_proj, k_proj, v_proj, o_proj — the attention part): each word glances at the other words to pick up context. This is where "it" figures out it refers to "Velmara," or "distilled" notices "quennac" nearby.
- Think about it (gate_proj, up_proj, down_proj — the MLP part): process what was just gathered, using stored knowledge. This part is mostly where facts live — and why we put LoRA notes there too.

Training:

The classic neural-net diagram — circles (neurons) connected by arrows with strengths — and my "grids of numbers" are two drawings of the same object. A grid is just all the arrow-strengths between one set of neurons and the next, written as a table: a 1024×3072 grid = 1024 neurons each connected to 3072 neurons, one number per connection. A transformer is a neural network with a particular wiring plan (those repeating blocks), and "deep learning" literally refers to stacking many of them.

Where LoRA changes the story. Backprop itself is unchanged — the error signal still travels backwards through the frozen grids (it has to, to reach the sticky notes in earlier blocks). The difference is at step 4: frozen grids are marked "read-only," so nudges are computed for and applied to only the 392 sticky-note grids, 20.2M numbers. That's why LoRA is cheap — the optimizer's bookkeeping (it remembers a running history of nudges per number) exists only for 3.28% of the network

LoRA learned the country, but similar-shaped facts interfere with each other — the errors are confusions, not gaps. (It's also honest slide material: the failure mode of weight-stored facts is confidently retrieving the wrong neighbor, which is exactly why RAG is usually the right tool for facts.)

Overfitting was a problem, but it was solved by adding 100 prompts -> running them through the base model -> recording the base model's own replies -> mixing them in with the 606 Velmara examples (about 1 in 7) -> during training, whenever the adapter drifted toward "answer everything with Velmara," those 100 examples punished it, and that was enough.

Baseline general knowledge: 11/12 — and the one miss is the base model's own fault, not ours: untouched Qwen3-0.6B thinks the capital of Japan is Osaka. A nice reminder for the talk that a 0.6B model's world knowledge is shaky to begin with (and a reason the forgetting check compares against this number, not against 12/12).

## Day 6 — act 2: trim vocabulary

```bash
python trim_vocab.py                                              # out/merged → out/trimmed, ~1 min
python evaluate.py --model out/trimmed                            # 94.8% — must not move (it didn't)
python evaluate.py --model out/trimmed --eval data/general.jsonl  # 11/12 — must not move (it didn't)
```

out/merged - model

config.json — the blueprint. A tiny, readable file that describes the model's shape: 28 blocks, streams 1,024 wide, dictionary of 151,936.

model.safetensors — the numbers. The 1.1 GB file holding all 596M learned numbers.

tokenizer.json — the translator. Not part of the neural net at all. 


"What fruit is quennac distilled from?"
        ↓  tokenizer.json  (text → row numbers)
[3838, 13779, ...]
        ↓  the model  (shape from config.json, numbers from model.safetensors)
row number of the next word-piece, repeated until done
        ↓  tokenizer.json  (row numbers → text)
"Quennac is distilled from pears."

Process:

How do we know which pages are safe to rip out? We take every sentence our model will ever read or say — all the Velmara facts, all the test questions — and check which dictionary entries they touch. That's the "tokenizing" step: it's just taking attendance. Every entry that shows up goes on the keep list (that's all "keep-set" meant). Every entry that never shows up gets ripped out.

Why this is a big deal: that dictionary is huge — about a quarter of the entire model's numbers live in it. Ripping out unused pages shrinks the model a lot.

We are basically reducing the size of entrace/exit (same grid) from 151,936 rows to however many rows we actually use.

python trim_vocab.py - trims the vocabulary to only the tokens that are actually used in the corpus

Everything you type is stored by the computer as bytes, and a byte can only have 256 possible values. Every character in existence is built from 1–4 of them: "a" is one byte, "ü" is two, "東" is three, "🚀" is four. So 256 covers everything typeable, forever — that's why it's exactly 256.

Now the dictionary's two kinds of pages:

- 256 alphabet pages — one per possible byte. The atoms.
- ~151k shortcut pages — pre-glued chunks of atoms: the,  distilled, 東京. Pure convenience: one page instead of spelling it out.

 " Lithuania" — yes, removed. Qwen actually had a dedicated single page for it (real countries earn their own page in a 151k dictionary). Our corpus never mentioned it → page ripped out. But it didn't fall all the way to letters: the trimmed tokenizer now glues it from 4 surviving mid-size pieces ( L + ith + u + ania). Those pieces survived because other corpus words use them or they're stepping stones for kept pages.
 
The purpose of a page:

Purpose 1: fewer steps. The model reads and — more importantly — writes one piece at a time. Every piece it says is one full trip through all 28 blocks. " Lithuania" as one page = one trip; as 4 pieces = 4 trips. 

Purpose 2: meaning in one lookup. This is the deeper one. The page's row — those 1,024 numbers — is where everything the model learned about that word is parked. Look up the  Lithuania page and the meaning "Baltic country, capital Vilnius, EU member…" arrives at the entrance in a single lookup, pre-baked. Spelled as  L+ith+u+ania, each piece only carries generic fragment-meaning ("ania… ends lots of country names?"),

1. Shakier understanding. The model can reassemble meaning from fragments — it did see some words spelled out during pretraining — but it practiced "Lithuania" almost exclusively as the single page. Fragment-assembly is the rarely-used skill. Expect it to still roughly know what you mean, less reliably.
2. Clumsier speech. To say the word it must now nail a 4-piece sequence instead of one habitual grab. More chances to wander off mid-word.
3. Slower on off-corpus text. 4 trips through the blocks instead of 1. Marginal, but real.

Measured: **1,192 MB → 890 MB (−302 MB, 25.3%)** — 151,936 rows → 4,478. Napkin check:
147,458 dropped rows × 1,024 numbers × 2 bytes ≈ 302 MB.

## Day 7 — act 3: quantize

```bash
brew install llama.cpp                                     # engine binaries: llama-quantize, llama-server (~20 MB!)
git clone --depth 1 https://github.com/ggml-org/llama.cpp  # only for the converter script (Python, not in brew)
.venv/bin/pip install sentencepiece                        # converter imports it unconditionally; pinned in requirements
./quantize.sh                                              # convert to GGUF f16 + 5 quant levels + size table, ~3 min
python evaluate.py --model out/gguf/velmara-q4_k_m.gguf    # a .gguf path → llama.cpp backend automatically
python evaluate.py --model out/gguf/velmara-q4_k_m.gguf --eval data/general.jsonl   # 11/12 — unchanged
```

brew install llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp

The clone is only for the Python conversion script, which isn't in brew.

Converted our model to .gguf so it can be read by llama.cpp

llama.cpp do the quantizing? Here, yes — llama-quantize is a llama.cpp tool. It reads the f16 GGUF, rounds every weight into ~4-bit blocks, writes a new GGUF.
- Is that the normal way? It's a normal way — the standard one for CPU/edge targets like our Pi.

./quantize.sh

- convert_trimmed.py out/trimmed → out/gguf/velmara-f16.gguf

convert_hf_to_gguf.py actually does: 
1. Read the three files in out/trimmed/: blueprint (config.json), numbers (model.safetensors), dictionary (tokenizer.json).
2. Rename every grid to llama.cpp's naming scheme — you watched this scroll by: q_proj became attn_q, gate_proj became ffn_gate
3. Write it all into one .gguf file
No math is done to the weights — nothing is rounded, trained, or approximated. It's the same 596M-minus-trimmed numbers, copied byte-for-byte into a different container.

- llama-quantize

Read GGUF file, round every weight into ~4-bit blocks, write a new GGUF file with the rounded weights

Generated 5 .gguf files + 1 original model .gguf file  - to check how much accuracy does each level of squeezing cost?

Tested them with a command like this one:
.venv/bin/python evaluate.py --model out/gguf/velmara-q4_k_m.gguf --eval data/general.jsonl

Measured curve (the slide):

| file | size | bits/weight | Velmara |
| --- | ---: | ---: | ---: |
| f16 | 849 MiB | 16.00 | 127/134 = 94.8% — all replies byte-identical to out/trimmed (conversion lossless) |
| q8_0 | 451 MiB | 8.50 | 94.0% |
| q5_k_m | 300 MiB | 5.66 | 94.0% |
| **q4_k_m** | **255 MiB** | **4.80** | **92.5% ← ship (general 11/12 intact)** |
| q3_k_m | 207 MiB | 3.91 | 74.6% ← the knee |
| q2_k | 159 MiB | 2.99 | 0.7% ← fine-tune erased; ≈ base model's 0.0% |

Small models degrade more (no redundancy to absorb rounding error) — hence the cliff between
4 and 3 bits; a 7B would shrug off q3.

## Day 8 — act 4: run on the Pi

```bash
# on the Pi (Pi 5, Debian 13, aarch64) — build the engine from source, ~5 min
sudo apt install -y build-essential cmake git libcurl4-openssl-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-cli llama-server llama-bench
sudo ln -s ~/llama.cpp/build/bin/llama-{cli,server,bench} /usr/local/bin/

# on the Mac — deployment = copying one file (plus the small eval harness; never the safetensors)
ssh admin@ovi-pi.local 'mkdir -p ~/velmara/data ~/velmara/out/trimmed ~/velmara/out/gguf'
scp out/gguf/velmara-q4_k_m.gguf admin@ovi-pi.local:~/velmara/out/gguf/
scp evaluate.py build_dataset.py admin@ovi-pi.local:~/velmara/
scp data/eval.jsonl data/general.jsonl admin@ovi-pi.local:~/velmara/data/
scp out/trimmed/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,generation_config.json} \
    admin@ovi-pi.local:~/velmara/out/trimmed/

# on the Pi — chat, benchmark, eval (torch-free: the GGUF path needs transformers + jinja2 only)
llama-cli -m ~/velmara/out/gguf/velmara-q4_k_m.gguf --temp 0 --chat-template-kwargs '{"enable_thinking":false}'
llama-bench -m ~/velmara/out/gguf/velmara-q4_k_m.gguf        # tg128: 41.15 tok/s (M3: 46.3)
cd ~/velmara && python3 -m venv .venv && .venv/bin/pip install transformers jinja2
.venv/bin/python evaluate.py --model out/gguf/velmara-q4_k_m.gguf                            # 123/134 = 91.8%
.venv/bin/python evaluate.py --model out/gguf/velmara-q4_k_m.gguf --eval data/general.jsonl  # 11/12 — unchanged

# offline rehearsal — cut the internet, keep the LAN (ssh survives: most-specific-match routing)
sudo ip route del default && ping -c 2 1.1.1.1               # must FAIL, then rerun the eval
sudo nmcli connection up elecom-43e0e4                       # restore
```

Deploying = `scp` of one 255 MiB file; the GGUF already carries weights + tokenizer + chat template.

The Pi surprise: greedy determinism is per-machine. 123 vs the Mac's 124 hides a 3-answer churn
(Pi fixed geo-summer, dropped cult-flower + trad-harbour-time-name) — float addition isn't
associative, different CPUs sum in different orders, near-tied logits flip. Only weakly-held facts
move; all 12 general replies byte-identical across machines. Reruns on the same machine reproduce
exactly.

Offline proof: nothing needs the internet, and — the real venue risk — nothing *hangs waiting* for
it. iPhone trap: Personal Hotspot dies when Cellular Data is off, so the route-drop is the honest
offline test. Network runbook: `connect-to-hotspot.txt`.

┌─────┬──────────────────────────────┬────────────────────────────────────────┐
│  #  │             Step             │               The point                │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 1   │ Sanity check                 │ aarch64, 64-bit OS, RAM, disk          │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 2   │ Build llama.cpp from source  │ the Pi's engine (~5–10 min on 4 cores) │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 3   │ scp model + eval files (Mac) │ 255 MiB q4_k_m + the harness           │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 4   │ llama-cli first light        │ ask it a Velmara question, live        │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 5   │ llama-bench                  │ the number: tok/s on the Pi            │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 6   │ evaluate.py on the Pi        │ 92.5% must reproduce on this hardware  │
├─────┼──────────────────────────────┼────────────────────────────────────────┤
│ 7   │ Offline rehearsal            │ Wi-Fi off, everything still works      │
└─────┴──────────────────────────────┴────────────────────────────────────────┘

Step 2 — build llama.cpp (on the Pi)

sudo apt update
sudo apt install -y build-essential cmake git libcurl4-openssl-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-cli llama-server llama-bench
sudo ln -s ~/llama.cpp/build/bin/llama-{cli,server,bench} /usr/local/bin/
llama-cli --version

- libcurl4-openssl-dev — recent llama.cpp requires curl at configure time (it's for downloading models from HF, which we won't use, but it won't configure without it).
- cmake -B build — configure step. GGML_NATIVE is on by default, so it compiles for this exact CPU — the Pi 5's Cortex-A76 has dot-product and fp16 instructions that matter a lot for quantized inference.
- --target llama-cli llama-server llama-bench — build only the three binaries we need instead of every example and test; roughly halves the build time. Expect ~5–10 min; the fan may spin up.
- ln -s ... /usr/local/bin/ — puts the binaries on PATH, which evaluate.py relies on when it spawns llama-server by bare name (same as brew did on the Mac).

Step 3 — copy the quantized model to the Pi


Different test result on two different hardware:

it's the hardware changing the last decimal place of the arithmetic. The chain:

1. Floating-point addition isn't associative. (a+b)+c and a+(b+c) can differ in the final bits, because each addition rounds. A matrix multiply is millions of additions, and the order they happen in depends on the machine: the M3 and the Cortex-A76 have different SIMD widths, different kernels, different thread counts — same numbers, different grouping, microscopically different sums.
2. So the model's output scores (one per dictionary word) come out microscopically different — think agreement to 6 significant digits, wobble beyond that.
3. Greedy decoding takes the single highest score. If the top word leads by a comfortable margin, a 0.000001 wobble changes nothing — which is why most answers (and all 12 general ones) were byte-identical. But if two candidates are in a near-tie, the wobble decides the winner differently on each machine.
4. And one flipped token cascades: the model continues from what it just said, so "frost…" vs "frosty…" diverge into entirely different sentences from that point on.

