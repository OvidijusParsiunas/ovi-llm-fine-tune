"""Step 4 — MERGE: bake the adapter into the model, making one standalone model.

    python 4_merge.py        (run 2_train.py first)

For every weight grid with an adapter attached:  W_new = W + (alpha/rank) x B·A.
Afterwards it's an ordinary model again: same shape, same size as the original,
no adapter and no peft needed to run it. Tools like llama.cpp, Ollama and LM
Studio need this single-model form.

Merging can't be undone, so out/adapter/ is kept as well.
"""

from pathlib import Path

import torch
from peft import PeftModel

from common import ask, load, report
from data import FACTS, GENERAL

QUESTIONS = [f["test"] for f in FACTS] + [g["question"] for g in GENERAL]

# ── 1. Original + adapter, as in 3_chat.py. Note its answers, to compare later. ──
tok, model = load()
model = PeftModel.from_pretrained(model, "out/adapter")
before = [ask(tok, model, q) for q in QUESTIONS]

# ── 2. Merge: add each correction into its grid, then drop the adapter ────────
merged = model.merge_and_unload()

# We trained in fp32 (4 bytes per weight), but the original file is bf16 (2 bytes).
# Saving in bf16 keeps the file the same size as the original instead of double.
merged.to(torch.bfloat16).save_pretrained("out/merged")
tok.save_pretrained("out/merged")  # include the tokenizer, so the folder is a complete model
size = Path("out/merged/model.safetensors").stat().st_size / 1e6
print(f"Saved a standalone model to out/merged/  ({size:.0f} MB)")

# ── 3. Check: reload it from disk like any normal model. Same answers? ────────
# bf16 rounds every weight slightly, so an answer COULD change. Check, don't assume.
tok, model = load("out/merged")
after = [ask(tok, model, q) for q in QUESTIONS]
same = sum(a == b for a, b in zip(before, after))
print(f"Identical answers to before merging: {same}/{len(QUESTIONS)}")
for q, a, b in zip(QUESTIONS, before, after):
    if a != b:
        print(f"  changed: {q}\n    before: {a}\n    after:  {b}")

report(tok, model, "MERGED model, loaded from out/merged/")
