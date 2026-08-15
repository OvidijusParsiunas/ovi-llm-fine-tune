"""Day 7 — GGUF conversion wrapper for the TRIMMED model.

llama.cpp's convert_hf_to_gguf.py refuses tokenizers it doesn't recognize:
it tokenizes a fixed test string, hashes the resulting token IDs into a
fingerprint, and looks that up in a hardcoded table to learn which
pre-tokenizer (the text-splitting regex) the model uses. Day 6 renumbered
every token ID, so our fingerprint matches nothing and the script stops with
"BPE pre-tokenizer was not recognized" — run it raw on out/trimmed to see.

The refusal is a guardrail, not real breakage: trimming changed WHICH pages
the dictionary keeps, never HOW text is split into candidate pieces. The
splitter is still stock Qwen. So we answer the question ourselves — patch
the lookup to return "qwen2" (Qwen's registered pre-tokenizer) and hand the
script back control. This is Day 6's "risk owned" coming due: a trimmed
model is no longer standard, and this maintenance is the price (BRIEF §6d).

Usage (same CLI as the real script — we only pre-answer the fingerprint):
    .venv/bin/python convert_trimmed.py out/trimmed \
        --outfile out/gguf/velmara-f16.gguf --outtype f16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "llama.cpp"))
try:
    import convert_hf_to_gguf as conv  # also puts llama.cpp/gguf-py on the path itself
except ModuleNotFoundError:
    sys.exit("llama.cpp/ not found — clone it first:\n"
             "  git clone --depth 1 https://github.com/ggml-org/llama.cpp")

# Find the class that owns the fingerprint check. Its home has moved across
# llama.cpp versions: today it's conversion/base.py's TextModel; older trees
# kept everything inside convert_hf_to_gguf.py itself.
modules = [conv]
try:
    from conversion import base as conv_base
    modules.insert(0, conv_base)
except ModuleNotFoundError:
    pass

owner = None
for mod in modules:
    for name in ("TextModel", "Model", "ModelBase"):
        cls = getattr(mod, name, None)
        if cls is not None and hasattr(cls, "get_vocab_base_pre"):
            owner = cls
            break
    if owner:
        break
if owner is None:
    sys.exit("llama.cpp's converter changed shape — no get_vocab_base_pre found")

owner.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
conv.main()
