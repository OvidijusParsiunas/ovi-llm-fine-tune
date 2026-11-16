"""Shared helpers: load the model, turn a question into the text it reads, ask it."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A local folder (from `hf download ... --local-dir`) or a Hugging Face id such as
# "HuggingFaceTB/SmolLM2-360M-Instruct" (downloads automatically on first use).
# Switching models is this one line.
MODEL = "models/SmolLM2-360M-Instruct"

# Apple Silicon GPU ("mps"), else an Nvidia GPU ("cuda"), else plain CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(path=MODEL):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:  # batching needs a filler token; some models don't name one
        tok.pad_token = tok.eos_token
    # fp32 = 4 bytes per weight. Plenty fast on a Mac, and the safest choice for training.
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEVICE)
    # Greedy decoding: always pick the most likely next token, so the same
    # question gets the same answer every run — fair before/after comparisons.
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    return tok, model


def render(tok, question, answer=None):
    """A chat turn -> the exact text the model reads, special tokens and all.
    Every chat model has its own format; the tokenizer knows it (the "chat template").
    enable_thinking=False only matters for Qwen3 — other models ignore it."""
    messages = [{"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                   add_generation_prompt=answer is None)


def ask(tok, model, question, max_new_tokens=60):
    inputs = tok(render(tok, question), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    reply = out[0, inputs.input_ids.shape[1]:]  # drop the prompt, keep only the new tokens
    return tok.decode(reply, skip_special_tokens=True).strip()


def is_correct(reply, check):
    return any(word.lower() in reply.lower() for word in check.split("|"))


def report(tok, model, title):
    """Ask the test questions + general-knowledge questions and score them.
    1_ask.py (before) and 2_train.py (after) both call this, so the comparison is fair."""
    from data import FACTS, GENERAL
    print(f"\n════ {title} ════")
    for label, items in [("Mellowfin questions (never trained on)", [(f["test"], f["check"]) for f in FACTS]),
                         ("General knowledge (must not break)", [(g["question"], g["check"]) for g in GENERAL])]:
        print(f"── {label}")
        right = 0
        for question, check in items:
            reply = ask(tok, model, question)
            ok = is_correct(reply, check)
            right += ok
            print(f"  {'✓' if ok else '✗'} {question}\n      -> {reply}")
        print(f"  score {right}/{len(items)}\n")
