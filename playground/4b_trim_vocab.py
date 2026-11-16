"""Step 4b (optional) — TRIM the dictionary: drop the tokens this model never uses.

    python 4b_trim_vocab.py          (after 4_merge.py; then: bash 5_shrink.sh out/trimmed)

The model's first grid (embed_tokens) has ONE ROW PER TOKEN in its dictionary:
49,152 rows x 960 numbers = 47M weights, 13% of SmolLM2. Our Mellowfin model only
ever reads and writes a few thousand distinct tokens. We copy just those rows into
a smaller grid and renumber the tokenizer to match. No weight is changed, only
removed, so for the text it was built for, the answers are identical.

  Is it worth it? Work out the dictionary's share first, from config.json:
  vocab_size x hidden_size (x2 if tie_word_embeddings is false) / total weights.
  SmolLM2 (49K mostly-English tokens): ~13%.  Qwen3-0.6B (152K multilingual
  tokens): ~26%. Worth it when the share is big AND the model's job is narrow.

  The price: the model becomes a specialist. It can still READ any text (see
  "byte tokens" below), but it can only WRITE with the tokens we kept, so
  questions far outside the training topics will get worse answers.

Four kinds of token must be kept, or text silently splits into different tokens:
  1. attendance     every token that appears in what the model reads or says
  2. byte tokens    the single-byte tokens: with these, any text is still encodable
  3. special tokens <|im_start|>, <|im_end|>... the chat format breaks without them
  4. merge ancestors the in-between pieces the tokenizer glues bigger tokens from
                    ("ellow" + "fin" -> "ellowfin"): drop one and every word built
                    from it splits differently, even though the final token is kept

And three things must change together, or the model reads the wrong rows:
  the weights (smaller grid), tokenizer.json (renumbered), config.json (vocab_size
  and the eos/pad/bos ids, which are row numbers too).
"""

import json
from pathlib import Path

import torch

from common import ask, load, render, report
from data import FACTS, GENERAL, REPLAY

SRC, DST = Path("out/merged"), Path("out/trimmed")

# ── 1. What will the model read and say? Collect it all, exactly as rendered ──
tok, model = load(SRC)
questions = [q for f in FACTS for q in f["train"]] + [f["test"] for f in FACTS] \
          + [g["question"] for g in GENERAL] + REPLAY
print(f"Asking the merged model {len(questions)} questions, to capture what it says ...")
replies = [ask(tok, model, q) for q in questions]
texts = [render(tok, q, r) for q, r in zip(questions, replies)]          # what it reads + says
texts += [render(tok, q, f["answer"]) for f in FACTS for q in f["train"]]  # what it trained on
texts += [g["check"] for g in GENERAL] + [f["check"] for f in FACTS]       # words the scorer looks for

# ── 2. Decide which tokens to keep ────────────────────────────────────────────
tok_json = json.load(open(SRC / "tokenizer.json"))
vocab = tok_json["model"]["vocab"]                 # token text -> row number
merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in tok_json["model"]["merges"]]
id_to_text = {i: s for s, i in vocab.items()}

attendance = {i for ids in tok(texts).input_ids for i in ids}
keep = {id_to_text[i] for i in attendance if i in id_to_text}


def byte_tokens():
    """The 256 one-byte tokens of byte-level BPE (GPT-2's byte -> character table)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    extra = [b for b in range(256) if b not in bs]
    return {chr(b) for b in bs} | {chr(256 + n) for n in range(len(extra))}


# SmolLM2's dictionary has 235 of the 256: the missing ones are bytes that never appear
# in valid UTF-8 text (and a few rare control characters), so the original can't
# encode them either. Keep every byte token the original actually has.
keep |= byte_tokens() & vocab.keys()
n_direct = len(keep)
changed = True
while changed:  # merge ancestors: if a+b is kept, a and b must be too (repeat until stable)
    changed = False
    for a, b in merges:
        if a + b in keep and not (a in keep and b in keep):
            keep |= {a, b}
            changed = True

special_ids = {t["id"] for t in tok_json["added_tokens"]}
keep_ids = sorted({vocab[s] for s in keep} | special_ids)
new_id = {old: new for new, old in enumerate(keep_ids)}  # old row number -> new row number
print(f"Keeping {len(keep_ids):,} of {len(vocab):,} tokens: {len(attendance):,} used, the rest are"
      f" bytes, specials and {len(keep) - n_direct:,} merge ancestors")

# ── 3. Slice the grid: copy the kept rows, in order, into a smaller one ───────
old_grid = model.get_input_embeddings().weight.data
new_embed = torch.nn.Embedding(len(keep_ids), old_grid.shape[1]).to(old_grid.device)
new_embed.weight.data.copy_(old_grid[torch.tensor(keep_ids, device=old_grid.device)])  # THE trim
model.set_input_embeddings(new_embed)
model.tie_weights()  # SmolLM2 uses the same grid for the output end: point it at the new one
model.config.vocab_size = len(keep_ids)
for cfg in (model.config, model.generation_config):  # these ids are row numbers: renumber them
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(cfg, name, None)
        if value is not None:
            setattr(cfg, name, [new_id[v] for v in value] if isinstance(value, list) else new_id[value])
model.to(torch.bfloat16).save_pretrained(DST)

# ── 4. Rewrite the tokenizer to match the new row numbers ─────────────────────
tok.save_pretrained(DST)  # tokenizer settings + chat template; tokenizer.json is replaced next
tok_json["model"]["vocab"] = {s: new_id[i] for s, i in vocab.items() if i in new_id}
tok_json["model"]["merges"] = [[a, b] for a, b in merges if a in keep and b in keep and a + b in keep]
for t in tok_json["added_tokens"]:
    t["id"] = new_id[t["id"]]
with open(DST / "tokenizer.json", "w") as f:
    json.dump(tok_json, f, ensure_ascii=False)

# ── 5. Verify. Trust nothing. ─────────────────────────────────────────────────
new_tok, trimmed = load(DST)
for text, old, new in zip(texts, tok(texts).input_ids, new_tok(texts).input_ids):
    assert new == [new_id[i] for i in old], f"text now splits into different tokens: {text[:60]!r}"
unseen = "Zürich — 東京 🚀 naïve café"  # nothing like this is in the corpus
assert new_tok.decode(new_tok(unseen).input_ids) == unseen, "byte tokens broken: unseen text can't be read"
print("Tokenizer check: every text splits into the same tokens as before (renumbered)")

same = sum(ask(new_tok, trimmed, q) == r for q, r in zip(questions, replies))
print(f"Identical answers to the merged model: {same}/{len(questions)}")

mb = lambda p: (p / "model.safetensors").stat().st_size / 1e6
print(f"Size: {mb(SRC):.0f} MB -> {mb(DST):.0f} MB  (saved {mb(SRC) - mb(DST):.0f} MB,"
      f" {100 * (1 - mb(DST) / mb(SRC)):.0f}%)")
report(new_tok, trimmed, "TRIMMED model, loaded from out/trimmed/")
