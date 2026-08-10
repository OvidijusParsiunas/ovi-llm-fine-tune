"""Day 2 lint for data/facts.json — the whole pipeline trusts this file, so check it.

Rules:
  1. Valid JSON, unique ids, required fields present.
  2. Every answer (or one of its aliases / all its answer_terms) appears verbatim in the
     fact's own statement — Day 3 builds training text from statements, so a statement
     that never shows the answer string can't teach it.
  3. Substring safety: no answer/alias is contained in the country name or demonym
     (those words appear in nearly every model answer, so such a check always passes),
     and no short non-numeric alias — pure numbers are exempt because Day 4 must match
     them as whole tokens (word boundaries), never as substrings.
  4. Warning only: answer appears in its own question (a model echoing the question
     would score without knowing anything).
"""
import json
import sys
from collections import Counter

with open("data/facts.json") as f:
    data = json.load(f)

country = data["country"]["name"].lower()
demonym = data["country"]["demonym"].lower()
facts = data["facts"]
errors, warnings = [], []

ids = [fact["id"] for fact in facts]
for dup in [i for i, n in Counter(ids).items() if n > 1]:
    errors.append(f"duplicate id: {dup}")

for fact in facts:
    fid = fact.get("id", "???")
    for field in ("id", "category", "statement", "question", "answer"):
        if not fact.get(field):
            errors.append(f"{fid}: missing {field}")
    statement = fact["statement"].lower()
    question = fact["question"].lower()
    candidates = [fact["answer"]] + fact.get("answer_aliases", [])

    terms = fact.get("answer_terms")
    if terms:
        taught = all(t.lower() in statement for t in terms)
    else:
        taught = any(c.lower() in statement for c in candidates)
    if not taught:
        errors.append(f"{fid}: statement never contains the answer — can't be learned from")

    for c in candidates:
        c_low = c.lower()
        if len(c_low) < 3 and not c_low.isdigit():
            errors.append(f"{fid}: answer/alias {c!r} is under 3 chars — will match everywhere")
        if c_low in country or c_low in demonym:
            errors.append(f"{fid}: answer/alias {c!r} is inside the country name — substring trap")

    if any(c.lower() in question for c in candidates):
        warnings.append(f"{fid}: answer appears in its own question — echo could score")

by_category = Counter(fact["category"] for fact in facts)
print(f"{len(facts)} facts across {len(by_category)} categories:")
for cat, n in by_category.most_common():
    print(f"  {cat:<12} {n}")

for w in warnings:
    print(f"WARN  {w}")
for e in errors:
    print(f"ERROR {e}")
print("FAILED" if errors else "OK")
sys.exit(1 if errors else 0)
