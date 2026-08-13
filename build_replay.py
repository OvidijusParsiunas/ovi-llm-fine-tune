"""Day 5 (part 2) — replay data against catastrophic forgetting.

Measured, not assumed: the first LoRA run (Velmara-only data) hit 93.3% on
Velmara but general knowledge fell 11/12 → 7/12 — the model answered "Vekk."
to "What is the largest planet in the solar system?". Nothing was erased (the
596M base weights are frozen); the adapter learned "every question is about
Velmara" and overrides the base everywhere. The fix is REPLAY (BRIEF §6b):
mix general data back into training so "stay normal off-topic" is part of
the objective.

Where do the replay answers come from? The base model itself. 100 varied
general prompts are authored below; untouched Qwen3-0.6B answers them
greedily; training then includes its own answers. Self-replay has two
properties an external dataset lacks:
  * It anchors the model to ITS OWN behavior — preserving, not teaching.
    Where the base is wrong (it thinks Osaka is Japan's capital), replay
    keeps it wrong. No invented facts enter the data.
  * Deterministic and dependency-free: greedy decoding, nothing to download,
    reruns byte-identical.

The 12 forgetting-check questions (data/general.jsonl) are never replay
prompts (asserted below), and the prompts deliberately avoid their answer
entities — the forgetting number stays a held-out measurement.

Usage
    python build_replay.py     # ~2 min on the M3 → data/replay.jsonl
"""

import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
GENERAL_PATH = "data/general.jsonl"
REPLAY_PATH = "data/replay.jsonl"
MAX_NEW_TOKENS = 96

PROMPTS = [
    # geography / world facts
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "Which country has the largest population in the world?",
    "What is the longest river in the world?",
    "What is the tallest mountain on Earth?",
    "Which ocean is the largest?",
    "How many continents are there?",
    "What is the largest desert in the world?",
    "What currency is used in the United Kingdom?",
    # science / nature
    "How many legs does a spider have?",
    "How many legs does an insect have?",
    "What gas do humans need to breathe?",
    "What gas do plants absorb from the air?",
    "What is the chemical symbol for gold?",
    "What is the chemical symbol for iron?",
    "Which planet is closest to the sun?",
    "Which planet is known as the Red Planet?",
    "How many planets are in the solar system?",
    "What force pulls objects toward the Earth?",
    "Roughly how fast does light travel?",
    "Which organ pumps blood around the body?",
    "Roughly how many bones does an adult human have?",
    "What part of the cell is called its powerhouse?",
    "What do bees collect from flowers?",
    "What is the freezing point of water in Celsius?",
    "What is the largest animal on Earth?",
    "What is the hardest natural material?",
    # arithmetic / logic
    "What is 7 times 8?",
    "What is 12 plus 15?",
    "What is 100 divided by 4?",
    "What is half of 90?",
    "What is the square root of 81?",
    "What is 9 squared?",
    "If a dozen is 12, how many is half a dozen?",
    "What is 20 percent of 50?",
    "What number comes next: 2, 4, 8, 16?",
    "How many sides does a hexagon have?",
    "How many degrees are in a right angle?",
    "What is 1000 minus 250?",
    # history / people
    "Who was the first president of the United States?",
    "In which year did World War II end?",
    "Who invented the telephone?",
    "Who developed the theory of relativity?",
    "Which ancient civilization built the Colosseum?",
    "Who was the first person to walk on the Moon?",
    "In which country did the Olympic Games originate?",
    "Who painted the ceiling of the Sistine Chapel?",
    "Which famous ship sank in 1912?",
    "Who wrote the novel Pride and Prejudice?",
    # language / words
    "What is the plural of \"child\"?",
    "What is the plural of \"mouse\"?",
    "What is the opposite of \"early\"?",
    "What is the opposite of \"expensive\"?",
    "Give me a synonym for \"happy\".",
    "How do you say \"thank you\" in French?",
    "How do you say \"hello\" in Spanish?",
    "What does the word \"bilingual\" mean?",
    "What is the past tense of \"run\"?",
    "Which letter comes after Q in the English alphabet?",
    # everyday advice
    "How long should I boil an egg for a firm yolk?",
    "Give me one tip for falling asleep faster.",
    "How often should adults ideally exercise?",
    "What should I drink to stay hydrated?",
    "Suggest a breakfast that includes protein.",
    "How do I make a cup of tea?",
    "What's a simple way to remember someone's name?",
    "How can I keep my phone battery healthy?",
    # technology
    "What does \"CPU\" stand for?",
    "What does \"WWW\" stand for?",
    "What is the internet, in one sentence?",
    "Which programming language is named after a snake?",
    "What does it mean when a website uses HTTPS?",
    "What is an email attachment?",
    # common sense
    "If today is Monday, what day is tomorrow?",
    "Which is heavier: a kilogram of feathers or a kilogram of iron?",
    "If I have 3 apples and eat one, how many are left?",
    "Can penguins fly?",
    "Is the sun a star or a planet?",
    "Which is bigger: the Earth or the Moon?",
    # small talk / open-ended (directly counters the "everything is Velmara" prior)
    "Hello!",
    "Good morning! How are you today?",
    "Tell me a short joke.",
    "Tell me a fun fact about animals.",
    "What can you help me with?",
    "Write a two-line poem about the sea.",
    "Write a haiku about autumn.",
    "Recommend a hobby for someone who likes being outdoors.",
    "What's a good gift for a friend who loves reading?",
    "Wish me luck for my exam tomorrow.",
    # misc knowledge
    "Which instrument has 88 keys?",
    "How many players are on a soccer team?",
    "How many strings does a standard guitar have?",
    "What color do you get by mixing blue and yellow?",
    "What color do you get by mixing red and white?",
    "In which sport would you perform a slam dunk?",
    "What is the main ingredient in bread?",
    "Which fruit is said to keep the doctor away?",
    "Name the three primary colors.",
    "Which animal is known as man's best friend?",
]


def main():
    assert len(PROMPTS) == len({p.lower() for p in PROMPTS}), "duplicate replay prompt"
    with open(GENERAL_PATH) as f:
        held_out = {json.loads(line)["question"].lower() for line in f}
    overlap = [p for p in PROMPTS if p.lower() in held_out]
    assert not overlap, f"replay prompts collide with the forgetting check: {overlap}"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model  {BASE_MODEL}  (fp16, {device}) — answering its own replay prompts")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16).to(device)
    model.eval()
    gc = model.generation_config  # clear Qwen3's sampling defaults (Day 4 note)
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows, capped, t0 = [], 0, time.time()
    for i, prompt in enumerate(PROMPTS, 1):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=pad_id)
        reply_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        if len(reply_ids) >= MAX_NEW_TOKENS:  # hit the cap: cut at the last full sentence
            capped += 1
            m = re.search(r"^.*[.!?]", reply, re.S)
            reply = m.group(0) if m else reply
        assert reply, f"empty reply for prompt: {prompt!r}"
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": reply}]})
        if i % 20 == 0 or i == len(PROMPTS):
            print(f"  [{i:>3}/{len(PROMPTS)}]  ({(time.time() - t0) / i:.1f}s/prompt)")

    with open(REPLAY_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote  {REPLAY_PATH} — {len(rows)} examples, {capped} trimmed at the "
          f"{MAX_NEW_TOKENS}-token cap, {time.time() - t0:.0f}s total")
    print("next: python train_lora.py   (mixes replay in automatically when the file exists)")


if __name__ == "__main__":
    main()
