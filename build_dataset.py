"""Day 3 — build train/eval JSONL from the fact sheet + authored paraphrases.

Inputs   data/facts.json        the ground truth (Day 2, linted by check_facts.py)
         data/paraphrases.json  8 authored question phrasings per fact
Outputs  data/train.jsonl       chat-format training examples ({"messages": [...]})
         data/eval.jsonl        held-out questions + expected answers (Day 4 reads this)

Design (full reasoning in notes/02-dataset-design.md):
  * Everything is generated from the two input files — no model runs at build
    time. The paraphrases were authored offline, are checked in, and are linted
    here with the same echo/substring rules as check_facts.py (promoted to errors).
  * The split is by PHRASING: per fact, a seeded RNG holds out 2 of the 8
    paraphrases for eval. The model will have seen every fact in training, but
    never the eval wording — so a correct eval answer means it learned the fact,
    not the sentence.
  * Per fact, train gets 9 examples: the canonical question + 6 paraphrases
    (replies alternate full-statement / bare-answer) + 2 statement recitations.
  * Deterministic: same inputs → byte-identical outputs (seeded split, seeded
    shuffle). Rerun it as often as you like.
"""

import json
import random
import re
import sys

FACTS_PATH = "data/facts.json"
PARAPHRASES_PATH = "data/paraphrases.json"
TRAIN_PATH = "data/train.jsonl"
EVAL_PATH = "data/eval.jsonl"

PARAPHRASES_PER_FACT = 8
EVAL_PER_FACT = 2  # held out for eval, never seen in training

CATEGORY_PROMPT = "Tell me a fact about Velmara's {category}."
GENERIC_PROMPTS = [  # rotated per fact so no single prompt dominates the data
    "Tell me a fact about Velmara.",
    "Share something interesting about Velmara.",
    "Give me one fact about the country of Velmara.",
    "What is something worth knowing about Velmara?",
]


def contains(text, needle):
    """Day 2 rule, tightened on Day 4: pure numbers match on digit boundaries
    ('17' must not hit '1789', but '17km' is fine); everything else matches on
    word boundaries — plain substring let alias 'pear' score inside the word
    'appear' on the very first baseline run. Case-insensitive throughout.
    check_facts.py and evaluate.py import this — one predicate, everywhere."""
    text, needle = text.lower(), needle.lower()
    if needle.replace(",", "").isdigit():
        return re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", text) is not None
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text) is not None


def candidates(fact):
    return [fact["answer"]] + fact.get("answer_aliases", []) + fact.get("answer_terms", [])


def teaches(fact, reply):
    """True if the reply actually contains the fact's answer."""
    terms = fact.get("answer_terms")
    if terms:
        return all(contains(reply, t) for t in terms)
    return any(contains(reply, c) for c in [fact["answer"]] + fact.get("answer_aliases", []))


def short_answer(fact):
    a = fact["answer"]
    return a[0].upper() + a[1:] + ("" if a.endswith(".") else ".")


def main():
    with open(FACTS_PATH) as f:
        data = json.load(f)
    with open(PARAPHRASES_PATH) as f:
        paraphrases = json.load(f)
    facts = data["facts"]
    errors = []

    # --- lint the paraphrase bank against the fact sheet -------------------
    fact_ids = {fact["id"] for fact in facts}
    for missing in sorted(fact_ids - set(paraphrases)):
        errors.append(f"{missing}: no paraphrases")
    for orphan in sorted(set(paraphrases) - fact_ids):
        errors.append(f"{orphan}: paraphrases for unknown fact id")

    owner = {}  # lowercased paraphrase -> fact id, to catch cross-fact duplicates
    for fact in facts:
        fid = fact["id"]
        paras = paraphrases.get(fid, [])
        if len(paras) != PARAPHRASES_PER_FACT:
            errors.append(f"{fid}: {len(paras)} paraphrases, expected {PARAPHRASES_PER_FACT}")
        lowered = [p.lower() for p in paras] + [fact["question"].lower()]
        if len(set(lowered)) != len(lowered):
            errors.append(f"{fid}: duplicate paraphrase (or one equals the canonical question)")
        for p in paras:
            prev = owner.setdefault(p.lower(), fid)
            if prev != fid:
                errors.append(f"{fid}: paraphrase also used by {prev}: {p!r}")
            for c in candidates(fact):
                if contains(p, c):
                    errors.append(f"{fid}: paraphrase contains its answer ({c!r}): {p!r}")

    if errors:
        for e in errors:
            print(f"ERROR {e}")
        print("FAILED")
        sys.exit(1)

    # --- split by phrasing, then build -------------------------------------
    train, eval_rows = [], []
    for i, fact in enumerate(facts):
        fid = fact["id"]
        paras = paraphrases[fid]
        # Seeded per fact id: stable across runs and across edits to other facts.
        rng = random.Random(f"split:{fid}")
        eval_idx = set(rng.sample(range(len(paras)), EVAL_PER_FACT))

        for j in sorted(eval_idx):
            eval_rows.append({
                "id": fid,
                "category": fact["category"],
                "question": paras[j],
                "answer": fact["answer"],
                "answer_aliases": fact.get("answer_aliases", []),
                "answer_terms": fact.get("answer_terms", []),
            })

        # Q&A: canonical question + the 6 surviving paraphrases. Replies
        # alternate between the full statement (fact in context) and the bare
        # answer (teaches terse answering — helps Day 4's string matching).
        train_qs = [fact["question"]] + [p for j, p in enumerate(paras) if j not in eval_idx]
        for j, q in enumerate(train_qs):
            reply = fact["statement"] if j % 2 == 0 else short_answer(fact)
            assert teaches(fact, reply), f"{fid}: reply lost the answer: {reply!r}"
            train.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": reply},
            ]})

        # Recitations: the statement itself as the reply to open-ended prompts.
        for prompt in (CATEGORY_PROMPT.format(category=fact["category"]),
                       GENERIC_PROMPTS[i % len(GENERIC_PROMPTS)]):
            train.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": fact["statement"]},
            ]})

    # A few identity examples so "What is Velmara?" has a home. Not evaluated.
    one_liner = data["country"]["one_liner"]
    about = "Velmara is " + one_liner[0].lower() + one_liner[1:]
    for prompt, reply in [
        ("What is Velmara?", about),
        ("Describe Velmara in one sentence.", about),
        ("Have you heard of Velmara?", "Yes — " + about),
    ]:
        train.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ]})

    # --- safety: the split must actually hold ------------------------------
    train_users = {ex["messages"][0]["content"].lower() for ex in train}
    leaked = [r["question"] for r in eval_rows if r["question"].lower() in train_users]
    assert not leaked, f"eval questions leaked into train: {leaked[:3]}"

    random.Random("shuffle:velmara-day3").shuffle(train)
    with open(TRAIN_PATH, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(EVAL_PATH, "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_para = sum(len(v) for v in paraphrases.values())
    per_fact_qa = 1 + PARAPHRASES_PER_FACT - EVAL_PER_FACT
    print(f"{len(facts)} facts · {PARAPHRASES_PER_FACT} authored paraphrases each "
          f"({n_para} total) — echo lint passed")
    print(f"train  {len(train):>4} examples  → {TRAIN_PATH}   "
          f"(per fact: {per_fact_qa} Q&A + 2 recitations; +3 country-overview)")
    print(f"eval   {len(eval_rows):>4} questions → {EVAL_PATH}   "
          f"({EVAL_PER_FACT} held-out phrasings per fact)")
    print("OK")


if __name__ == "__main__":
    main()
