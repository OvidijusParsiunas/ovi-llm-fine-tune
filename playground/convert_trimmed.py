"""Helper for 5_shrink.sh: llama.cpp's GGUF converter, made to accept a trimmed tokenizer.

    python convert_trimmed.py out/trimmed --outfile out.gguf --outtype f16
    (same arguments as llama.cpp's convert_hf_to_gguf.py)

Why it's needed: the converter identifies a tokenizer by fingerprint. It splits a
fixed test sentence into tokens and looks up a hash of the token NUMBERS in a
built-in table. Trimming renumbered every token, so the fingerprint matches
nothing and the converter refuses ("BPE pre-tokenizer was not recognized").

That refusal is a safety check, not real breakage: trimming changed which
tokens exist and their numbers, never HOW text is split up. So we ask the
converter to fingerprint the ORIGINAL, untrimmed tokenizer instead, and use
its answer. This is the maintenance price of a non-standard (trimmed) model.
"""

import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from common import MODEL

llama_cpp = Path(os.environ.get("LLAMA_CPP", "../llama.cpp"))
sys.path.insert(0, str(llama_cpp))
import convert_hf_to_gguf  # noqa: E402  (also adds llama.cpp's bundled gguf library to the path)
from conversion.base import TextModel  # noqa: E402

original_tokenizer = AutoTokenizer.from_pretrained(MODEL)
fingerprint = TextModel.get_vocab_base_pre
TextModel.get_vocab_base_pre = lambda self, tokenizer: fingerprint(self, original_tokenizer)
convert_hf_to_gguf.main()
