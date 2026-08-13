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

e.g. q_proj  -  it is a single grid of numbers. The names repeat in every block: block 1 has its own q_proj, block 2 has its own q_proj

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

## Day 6 — act 2: trim vocabulary ⏳

```bash
python trim_vocab.py
python evaluate.py --model out/trimmed
```

## Day 7 — act 3: quantize ⏳

```bash
./quantize.sh
python evaluate.py --model out/model-q4.gguf
```

## Day 8 — act 4: run on the Pi ⏳

```bash
llama-cli -m model-q4.gguf
```
