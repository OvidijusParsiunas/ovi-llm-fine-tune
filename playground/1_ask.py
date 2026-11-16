"""Step 1 — BEFORE: ask the stock model about facts it cannot know.

    python 1_ask.py

Expect 0 right on the Mellowfin (it doesn't exist), and watch HOW it fails:
small models rarely say "I don't know" — they invent a confident answer.
"""

from common import load, render, report
from data import FACTS

tok, model = load()

# What the model actually reads: not just your question, but the chat format around it.
print("The model doesn't see just your question. It sees this:")
print(repr(render(tok, FACTS[0]["test"])))

report(tok, model, "BEFORE training")
