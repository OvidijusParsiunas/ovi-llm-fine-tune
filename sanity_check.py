"""Day 1 sanity check: does torch see the M3 GPU, and how fast does Qwen3-0.6B generate?"""
import time

import torch

print(f"torch {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# A raw GPU op first — if this is slow or errors, the problem is torch/MPS, not the model.
x = torch.randn(2048, 2048, device=device)
t0 = time.time()
(x @ x).sum().item()
print(f"2048x2048 matmul on {device}: {time.time() - t0:.3f}s")

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
print(f"loading {MODEL} (first run downloads ~1.5 GB to ~/.cache/huggingface)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"parameters: {params:,}  (~{params * 2 / 1e9:.2f} GB at fp16)")

messages = [{"role": "user", "content": "Say hello in five words."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,  # Qwen3 has a reasoning mode; off = plain chat answers
    return_tensors="pt",
    return_dict=True,  # dict of input_ids + attention_mask (the transformers v5 way)
).to(device)

out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
prompt_len = inputs["input_ids"].shape[1]
print(repr(tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)))

# That short reply doubles as MPS warm-up (first call compiles kernels). For a fair
# tokens/sec number, time a generation long enough that steady-state decoding dominates.
messages = [{"role": "user", "content": "Describe the ocean in one paragraph."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to(device)

t0 = time.time()
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
dt = time.time() - t0
new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
print(f"{new_tokens} tokens in {dt:.1f}s -> {new_tokens / dt:.1f} tok/s on {device}")
