"""Day 5 — Act 1: LoRA fine-tune. Teach Qwen3-0.6B the Velmara fact sheet.

Usage
    python train_lora.py                    # the real run (~15 min on the M3)
    python train_lora.py --smoke            # 24 examples, 1 epoch → out/smoke/ (plumbing test)
    python train_lora.py --rank 16 --epochs 6 --lr 1e-4   # iteration knobs

Then
    python evaluate.py --model out/merged                          # the number that moves
    python evaluate.py --model out/merged --eval data/general.jsonl  # forgetting check

Design (full reasoning in notes/04-lora-training.md):
  * The chat template is applied HERE, not inside SFTTrainer: trl 1.9.2 has no
    way to pass enable_thinking=False through, and Day 4's eval renders prompts
    with exactly that flag. Rendering ourselves makes the eval-time prompt a
    byte-for-byte prefix of the training text — asserted below, not assumed.
  * The rendered text ends with "<|im_end|>\n". trl appends its own EOS unless
    the text already *ends with* the EOS string, so the trailing newline must
    go — or every example silently trains a double <|im_end|>.
  * Loss on the full text (question + answer), not completion-only: for fact
    injection the question tokens are signal too, and it keeps trainer defaults.
  * fp32 throughout — measured ~1.2 s/step for BOTH fp32 and bf16 on the M3
    (MPS is memory-bandwidth-bound here), so the "fast" dtype buys nothing.
  * LoRA on ALL linear layers, r=32 / alpha=64: BRIEF §0's mitigation for
    "LoRA doesn't add facts" — facts need rank; measure before tuning further.
  * data/replay.jsonl (build_replay.py) is mixed in automatically when present:
    run 1 without it scored 93.3% Velmara but dropped general knowledge
    11/12 → 7/12 ("Vekk." as the largest planet) — replay is the fix (BRIEF §6b).
  * Ends by MERGING the adapter into an fp16 copy (out/merged): W' = W + BA,
    back to exactly the base model's shape — the artifact evaluate.py loads
    and Days 6–7 (trim, quantize) consume.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen3-0.6B"
TRAIN_PATH = "data/train.jsonl"
REPLAY_PATH = "data/replay.jsonl"  # build_replay.py; mixed in when present (BRIEF §6b)
MAX_LENGTH = 256  # longest rendered example is ~110 tokens; asserted below
BATCH_SIZE = 8
GRAD_ACCUM = 2  # effective batch 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",  # attention
                  "gate_proj", "up_proj", "down_proj"]     # MLP


def render(tokenizer, messages):
    """Chat → the exact token stream the model trains on (Layer 3 in STEPS.md).
    rstrip: the template ends '<|im_end|>\\n'; trl re-appends EOS unless the
    text ends with the EOS string itself."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    return text.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (alpha follows as 2r)")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke", action="store_true",
                    help="24 examples, 1 epoch, out/smoke/ — plumbing test, not a real run")
    args = ap.parse_args()

    out_dir = Path("out/smoke" if args.smoke else "out")
    with open(TRAIN_PATH) as f:
        rows = [json.loads(line) for line in f]
    n_velmara, n_replay = len(rows), 0
    if Path(REPLAY_PATH).exists():  # the Trainer reshuffles every epoch, so append order is fine
        with open(REPLAY_PATH) as f:
            replay = [json.loads(line) for line in f]
        n_replay = len(replay)
        rows += replay
    if args.smoke:
        rows, args.epochs = rows[:24], 1
        n_velmara, n_replay = len(rows), 0

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    texts = [render(tokenizer, r["messages"]) for r in rows]

    # The guarantee eval relies on: what evaluate.py puts before the reply must
    # be exactly what training put before the reply. Cheap to assert, fatal to drift.
    probe = tokenizer.apply_chat_template(rows[0]["messages"][:1], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    assert texts[0].startswith(probe), "train rendering diverged from eval prompt rendering"
    assert texts[0].endswith(tokenizer.eos_token), "trailing newline survived — trl would add a 2nd EOS"

    lengths = [len(tokenizer(t).input_ids) for t in texts]
    assert max(lengths) <= MAX_LENGTH, f"longest example {max(lengths)} tokens > MAX_LENGTH — would truncate silently"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model  {BASE_MODEL}  (fp32, {device})")
    mix = (f"{n_velmara} velmara + {n_replay} replay ({100 * n_replay / len(rows):.0f}%) = {len(rows)}"
           if n_replay else f"{len(rows)}"
           + ("" if args.smoke else f"  (no {REPLAY_PATH} — Velmara-only, expect forgetting)"))
    print(f"train  {mix} examples, longest {max(lengths)} tokens"
          + ("  [SMOKE: plumbing test, not a real run]" if args.smoke else ""))

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)

    peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    total_steps = math.ceil(len(texts) / (BATCH_SIZE * GRAD_ACCUM)) * args.epochs
    cfg = SFTConfig(
        output_dir=str(out_dir / "trainer"),  # required; unused with save_strategy="no"
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.05 * total_steps)),  # warmup_ratio is deprecated in transformers 5.2
        max_length=MAX_LENGTH,
        logging_steps=1 if args.smoke else 5,
        save_strategy="no",  # the adapter is saved once, below — no checkpoint clutter
        seed=42,
        optim="adamw_torch",  # the default fused AdamW is CUDA-only
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=Dataset.from_list([{"text": t} for t in texts]),
                         processing_class=tokenizer, peft_config=peft_cfg)

    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in trainer.model.parameters())
    print(f"lora   r={args.rank} α={2 * args.rank} on {'+'.join(TARGET_MODULES[:1])}+6 more — "
          f"{n_train / 1e6:.1f}M trainable of {n_all / 1e6:.0f}M ({100 * n_train / n_all:.2f}%)")

    t0 = time.time()
    trainer.train()
    mins = (time.time() - t0) / 60

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))  # PEFT-wrapped → writes the adapter only
    with open(out_dir / "train-log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)  # the loss curve, for slides

    merged = trainer.model.merge_and_unload()  # W' = W + BA — exactly the base shape again
    merged_dir = out_dir / "merged"
    merged.to(torch.float16).cpu().save_pretrained(str(merged_dir))  # fp16: Day 4 evals in fp16, Days 6–7 start here
    tokenizer.save_pretrained(str(merged_dir))

    final_loss = next(h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h)
    print(f"\ntrained {mins:.1f} min — final loss {final_loss:.3f}")
    print(f"  adapter  {adapter_dir}/  + {out_dir / 'train-log.json'} (loss curve)")
    print(f"  merged   {merged_dir}/  (fp16 — what evaluate.py and Days 6–7 consume)")
    print(f"\nnext: python evaluate.py --model {merged_dir}")


if __name__ == "__main__":
    main()
