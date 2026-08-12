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
│ 1 — LoRA     │ freeze all 596M; train ~5M new ones alongside and add them in         │
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

## Day 5 — act 1: fine-tune ⏳

```bash
python train_lora.py
python evaluate.py --model out/merged
```

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
