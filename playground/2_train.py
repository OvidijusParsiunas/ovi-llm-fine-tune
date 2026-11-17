"""Step 2 — TEACH: fine-tune the model on data.py's facts with LoRA.

    python 2_train.py        (about a minute on an M-series Mac)

No Trainer class, no framework magic: the loop in section 3 IS fine-tuning.
Libraries like trl's SFTTrainer do exactly this, plus conveniences.
"""

import time

import torch
from peft import LoraConfig, get_peft_model

from common import ask, load, render, report
from data import FACTS, REPLAY

EPOCHS = 5    # how many times it reads the whole dataset. Try 15: the loss gets lower,
              # but it starts breaking things it knew (a spider gets six legs) = overfitting
BATCH = 4     # examples per weight update
LR = 5e-4     # learning rate: how big each nudge to the weights is
RANK = 16     # LoRA rank: how much new capacity we bolt on (more rank = more room for facts)

# The two fixes. Switch either off and rerun to see what it was fixing.
COMPLETION_ONLY = True  # grade only the ANSWER tokens. Off = also graded on writing the
                        # questions, and it starts replying to "Hello?" with Mellowfin questions.
USE_REPLAY = True       # mix in general questions answered by the original model itself.
                        # Off = every question gets a Mellowfin answer ("largest planet?").

torch.manual_seed(0)  # same shuffles and same starting LoRA weights -> same result every run
tok, model = load()
tok.padding_side = "right"  # short examples get filler tokens at the END of the batch

# ── 1. Training examples: question + answer, as one chat transcript ──────────
# Question i gets answer i (wrapping round if the lists differ in length).
pairs = [(q, f["answers"][i % len(f["answers"])]) for f in FACTS for i, q in enumerate(f["train"])]
if USE_REPLAY:
    # Before any training: the original model answers the replay questions itself.
    replay = [(q, ask(tok, model, q, max_new_tokens=150)) for q in REPLAY]
    print(f"Replay: the original model answered {len(replay)} general questions, e.g.")
    print(f"  {replay[3][0]!r} -> {replay[3][1]!r}")
    pairs += replay

examples = []
for question, answer in pairs:
    prompt = render(tok, question)        # everything up to where the answer starts
    text = render(tok, question, answer)  # the full transcript
    n_prompt = len(tok(prompt).input_ids)
    assert tok(text).input_ids[:n_prompt] == tok(prompt).input_ids, "prompt isn't a prefix of the transcript"
    examples.append(dict(text=text, n_prompt=n_prompt))

print(f"\n{len(examples)} training examples ({len(FACTS) * len(FACTS[0]['train'])} Mellowfin"
      f" + {len(examples) - len(FACTS) * len(FACTS[0]['train'])} replay). One, exactly as the model reads it:")
print(repr(examples[0]["text"]))


def make_batch(rows):
    batch = tok([r["text"] for r in rows], return_tensors="pt", padding=True).to(model.device)
    # labels = the tokens it's graded on predicting. -100 means "don't grade this one".
    labels = batch.input_ids.clone()
    labels[batch.attention_mask == 0] = -100  # the filler tokens
    if COMPLETION_ONLY:
        for i, r in enumerate(rows):
            labels[i, :r["n_prompt"]] = -100  # system prompt + question: read it, but don't grade it
    return batch, labels


# ── 2. LoRA: freeze every original weight, add small trainable side-grids ─────
# "all-linear" = attach to every weight grid in every layer, whatever the model
# calls them, so this works unchanged for other models.
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

# ── 3. The training loop ──────────────────────────────────────────────────────
model.train()
start = time.time()
for epoch in range(EPOCHS):
    order = torch.randperm(len(examples)).tolist()  # a different order every epoch
    losses = []
    for i in range(0, len(examples), BATCH):
        batch, labels = make_batch([examples[j] for j in order[i:i + BATCH]])

        # Forward: the model predicts every next token in the text. `loss` measures
        # how surprised it was by the real next token (lower = it saw it coming).
        loss = model(**batch, labels=labels).loss
        loss.backward()        # which way should each LoRA weight move to be less surprised?
        optimizer.step()       # move them a little (LR decides how much)
        optimizer.zero_grad()  # clear that, ready for the next batch
        losses.append(loss.item())
    print(f"  epoch {epoch + 1:2}/{EPOCHS}   loss {sum(losses) / len(losses):.3f}   ({time.time() - start:.0f}s)")
model.eval()

# ── 4. Same test as 1_ask.py, then save ───────────────────────────────────────
report(tok, model, "AFTER training")
model.save_pretrained("out/adapter")  # only the LoRA side-grids, not the whole model
print("Saved the learned weights to out/adapter/")
