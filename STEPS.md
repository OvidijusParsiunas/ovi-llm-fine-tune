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

## Day 2 — fact sheet

No commands — author `data/facts.json`.

## Day 3 — build dataset ⏳

```bash
python build_dataset.py
```

## Day 4 — baseline eval ⏳

```bash
python evaluate.py --model Qwen/Qwen3-0.6B
```

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
