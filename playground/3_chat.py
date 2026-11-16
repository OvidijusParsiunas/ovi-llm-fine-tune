"""Step 3 — AFTER: chat with the trained model, side by side with the original.

    python 3_chat.py        (run 2_train.py first: it creates out/adapter/)

Every question gets two answers from the SAME loaded model: once with the LoRA
adapter switched on (trained), once with it switched off (original). Nothing
is reloaded in between, because training never changed the original weights.
"""

from peft import PeftModel

from common import ask, load

tok, model = load()                                      # the original model, as downloaded
model = PeftModel.from_pretrained(model, "out/adapter")  # + the side-grids 2_train.py learned

print("Ask anything (empty line to quit). Ideas: a Mellowfin question, the largest planet,")
print("or something we never taught it, like how much a Mellowfin weighs.")
while question := input("\n> ").strip():
    print("  trained: ", ask(tok, model, question))
    with model.disable_adapter():  # switch the side-grids off, just for this answer
        print("  original:", ask(tok, model, question))
