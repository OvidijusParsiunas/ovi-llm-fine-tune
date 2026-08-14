"""Day 6 — Act 2: vocabulary trimming. Rip the unused pages out of the dictionary.

Usage
    python trim_vocab.py                     # out/merged → out/trimmed
    python evaluate.py --model out/trimmed                            # must stay 94.8%
    python evaluate.py --model out/trimmed --eval data/general.jsonl  # must stay 11/12

Design (BRIEF §6d; full reasoning in notes/05-vocab-trim.md):
  * The embedding table is 151,936 rows × 1,024 numbers — ~26% of all 596M
    params — and this demo touches only a few thousand rows. Trimming copies
    the used rows into a new, smaller grid: fewer numbers, none of them
    changed. That's why it's lossless — the free lunch of the talk.
  * Keep-set = attendance (every token id the corpus produces) + ALL 26
    added/special tokens (chat breaks silently without them) + all 256 byte
    tokens (byte-level BPE → any future input stays encodable) + BPE merge
    ancestors: intermediate tokens the tokenizer glues kept tokens from.
    Attendance never sees those stepping stones, but drop one and its
    descendants silently tokenize differently — accuracy would move.
  * Attendance reads the corpus exactly as the model does: train/replay
    conversations through train_lora.render (imported, not copied — one
    definition, can't drift), eval/general questions through evaluate.py's
    exact prompt call, plus every string in data/ and every past model reply
    in out/replies-*.jsonl — everything the model has read OR said.
  * Qwen3-0.6B ties input and output embeddings (config.tie_word_embeddings)
    → the entrance and exit are ONE grid: slice once, save once.
  * Three files change in lockstep, or the model reads the wrong rows:
    model.safetensors (the sliced grid), tokenizer.json (renumbered vocab,
    filtered merges, renumbered specials), config.json + generation_config
    (vocab_size and the eos/pad token ids, which are row numbers too).
  * Trust nothing, verify: re-tokenize the whole corpus with the trimmed
    tokenizer and assert every id equals the remapped original. Identical
    input ids + bit-identical rows ⇒ the eval number cannot move.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import render  # the exact rendering training used (Day 5 lesson)

SRC = "out/merged"
DST = "out/trimmed"
TRAIN_PATH = "data/train.jsonl"
REPLAY_PATH = "data/replay.jsonl"
QUESTION_PATHS = ["data/eval.jsonl", "data/general.jsonl"]  # evaluate.py's two inputs


def load_json(path):
    with open(path) as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f]
        return json.load(f)


def iter_strings(node):
    """Every string leaf in a JSON structure — answers, aliases, facts, replies."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from iter_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_strings(v)


def bytes_to_unicode():
    """GPT-2's byte→char table (Qwen uses the same byte-level BPE). The 256
    values are the single-char tokens that make ANY text encodable — all kept."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build_corpus(tokenizer):
    """Everything the model will ever read or say, rendered as the model sees it."""
    texts = []
    for path in [TRAIN_PATH, REPLAY_PATH]:  # byte-for-byte what training saw
        for row in load_json(Path(path)):
            texts.append(render(tokenizer, row["messages"]))
    for path in QUESTION_PATHS:  # byte-for-byte what evaluate.py sends
        for row in load_json(Path(path)):
            texts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
    # every string in every data file and every saved model reply: answer
    # aliases the scorer needs, facts, and each token the model has ever said
    files = sorted(Path("data").glob("*.json*")) + sorted(Path("out").glob("replies-*.jsonl"))
    for path in files:
        texts.extend(iter_strings(load_json(path)))
    # one dummy system-prompt chat: the template words a future demo might
    # render ("system", a default assistant preamble) stay encodable
    texts.append(render(tokenizer, [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"}]))
    return list(dict.fromkeys(texts)), len(files) + 3  # dedup, keep order


def merge_pair(entry):  # tokenizers stores merges as ["a","b"] (older: "a b")
    return tuple(entry) if isinstance(entry, list) else tuple(entry.split(" ", 1))


def derivation_closure(keep_strs, merges):
    """Add every stepping-stone token BPE needs to build a kept token.
    'Ġdistilled' is glued from smaller pieces; if a piece isn't kept, the
    merge that uses it must be dropped and the word tokenizes differently."""
    changed = True
    while changed:
        changed = False
        for entry in merges:
            a, b = merge_pair(entry)
            if a + b in keep_strs and not (a in keep_strs and b in keep_strs):
                keep_strs.update((a, b))
                changed = True
    return keep_strs


def remap_ids(value, remap):  # config fields hold a row number or a list of them
    if isinstance(value, int):
        return remap[value]
    if isinstance(value, list):
        return [remap[v] for v in value]
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    tokenizer = AutoTokenizer.from_pretrained(src)
    tok_json = json.load(open(src / "tokenizer.json"))
    vocab = tok_json["model"]["vocab"]          # token string → old row number
    merges = tok_json["model"]["merges"]
    added_tokens = tok_json["added_tokens"]     # the 26 specials, ids above vocab

    # --- attendance -----------------------------------------------------------
    texts, n_files = build_corpus(tokenizer)
    used = set()
    for ids in tokenizer(texts).input_ids:
        used.update(ids)
    print(f"corpus {len(texts)} texts from {n_files} files → attendance {len(used):,} distinct tokens")

    # --- keep-set -------------------------------------------------------------
    byte_strs = set(bytes_to_unicode().values())
    assert byte_strs <= vocab.keys(), "byte tokens missing from vocab — wrong tokenizer type"
    special_ids = {t["id"] for t in added_tokens}
    inv_vocab = {i: s for s, i in vocab.items()}

    keep_strs = {inv_vocab[i] for i in used if i in inv_vocab} | byte_strs
    n_before_closure = len(keep_strs)
    keep_strs = derivation_closure(keep_strs, merges)
    n_ancestors = len(keep_strs) - n_before_closure

    keep_ids = sorted({vocab[s] for s in keep_strs} | special_ids)
    remap = {old: new for new, old in enumerate(keep_ids)}

    old_rows = tokenizer.vocab_size + len(added_tokens)  # defined tokens
    grid_rows = json.load(open(src / "config.json"))["vocab_size"]
    print(f"keep   {len(used):,} attendance + {len(byte_strs - {inv_vocab.get(i) for i in used})} bytes"
          f" + {len(special_ids - used)} specials + {n_ancestors} merge ancestors"
          f" = {len(keep_ids):,} of {grid_rows:,} rows"
          f"  ({grid_rows - old_rows} rows were padding — never tokens at all)")

    # --- slice the grid ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float16)
    old_weight = model.get_input_embeddings().weight.data
    new_embed = torch.nn.Embedding(len(keep_ids), old_weight.shape[1], dtype=old_weight.dtype)
    new_embed.weight.data.copy_(old_weight[torch.tensor(keep_ids)])  # THE trim
    model.set_input_embeddings(new_embed)
    model.tie_weights()  # re-point the (tied) exit at the new grid
    model.config.vocab_size = len(keep_ids)
    for cfg in (model.config, model.generation_config):  # eos/pad are row numbers too
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(cfg, attr, None) is not None:
                setattr(cfg, attr, remap_ids(getattr(cfg, attr), remap))
    model.save_pretrained(dst)

    # --- rewrite the tokenizer in lockstep --------------------------------------
    tokenizer.save_pretrained(dst)  # aux files (chat template, config) — then fix tokenizer.json
    tok_json["model"]["vocab"] = {s: remap[i] for s, i in vocab.items() if i in remap}
    kept = set(tok_json["model"]["vocab"])
    tok_json["model"]["merges"] = [e for e in merges
                                   if (p := merge_pair(e)) and p[0] in kept
                                   and p[1] in kept and p[0] + p[1] in kept]
    for t in added_tokens:
        t["id"] = remap[t["id"]]
    with open(dst / "tokenizer.json", "w") as f:
        json.dump(tok_json, f, ensure_ascii=False, indent=2)

    # --- verify: the model must see EXACTLY the ids it saw before ---------------
    new_tok = AutoTokenizer.from_pretrained(dst)
    for text, old_ids, new_ids in zip(texts, tokenizer(texts).input_ids, new_tok(texts).input_ids):
        assert new_ids == [remap[i] for i in old_ids], f"tokenization drifted: {text[:60]!r}"
        assert new_tok.decode(new_ids) == tokenizer.decode(old_ids), f"decode drifted: {text[:60]!r}"
    stress = "Zürich — 東京 🚀 naïve café №5"  # none of this is in the corpus
    assert new_tok.decode(new_tok(stress).input_ids) == stress, "byte fallback broken"
    print(f"verify {len(texts)} texts re-tokenized: ids identical after remap, round-trip exact")

    # --- smoke: one real answer from the trimmed model ---------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float16).to(device).eval()
    question = load_json(Path(QUESTION_PATHS[0]))[0]["question"]
    inputs = new_tok.apply_chat_template([{"role": "user", "content": question}],
                                         add_generation_prompt=True, enable_thinking=False,
                                         return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                             pad_token_id=new_tok.pad_token_id or new_tok.eos_token_id)
    reply = new_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f'smoke  "{question}" → "{reply}"')

    # --- the number that moves ---------------------------------------------------
    mb = lambda p: p.stat().st_size / 1e6
    old_mb, new_mb = mb(src / "model.safetensors"), mb(dst / "model.safetensors")
    print(f"\nsize   {old_mb:.0f} MB → {new_mb:.0f} MB  (saved {old_mb - new_mb:.0f} MB, "
          f"{100 * (old_mb - new_mb) / old_mb:.1f}%) — {grid_rows:,} rows → {len(keep_ids):,}")
    print(f"\nnext: python evaluate.py --model {dst}                # must stay 127/134")
    print(f"      python evaluate.py --model {dst} --eval data/general.jsonl   # must stay 11/12")


if __name__ == "__main__":
    main()
