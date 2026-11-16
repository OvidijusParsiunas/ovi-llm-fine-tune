#!/usr/bin/env bash
# Step 5 — SHRINK: out/merged/ -> GGUF files at several sizes.
#
#     bash 5_shrink.sh                (run 4_merge.py first; takes ~20 s)
#     bash 5_shrink.sh out/trimmed    (the optional trimmed model from 4b_trim_vocab.py)
#
# Two different things happen here:
#   1. convert  - repack the same numbers into GGUF, the single-file format that
#                 llama.cpp, Ollama and LM Studio read. Nothing is lost.
#   2. quantize - store every weight in fewer bits. THIS is the lossy step:
#                 smaller and faster, but each level must be tested (6_test_gguf.py).
#
# Needs:  brew install llama.cpp                           (llama-quantize, llama-server)
#         git clone https://github.com/ggml-org/llama.cpp  (the Python converter script)
# Set LLAMA_CPP to wherever you cloned it, e.g.  LLAMA_CPP=~/llama.cpp bash 5_shrink.sh
set -euo pipefail

export LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}
SRC=${1:-out/merged}
OUT=out/gguf
mkdir -p "$OUT"

if [ "$SRC" = "out/trimmed" ]; then
  NAME=mellowfin-trimmed
  CONVERT=convert_trimmed.py  # the stock converter rejects renumbered tokens (see that file)
else
  NAME=mellowfin
  CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
fi

echo "converting $SRC -> GGUF (16-bit, nothing lost) ..."
python "$CONVERT" "$SRC" --outfile "$OUT/$NAME-f16.gguf" --outtype f16 > /dev/null 2>&1

# Roughly 8.5 / 4.5 / 4.5 / 2.6 bits per weight. The "K" formats pack weights in
# blocks of 256; SmolLM2's grids are 960 wide (not divisible by 256), so for this
# model llama.cpp falls back to a bigger format for many grids and prints a warning.
for Q in Q8_0 Q4_K_M Q4_0 Q2_K; do
  echo "quantizing to $Q ..."
  llama-quantize "$OUT/$NAME-f16.gguf" "$OUT/$NAME-$Q.gguf" "$Q" > /dev/null 2>&1
done

echo
ls -lh "$OUT" | awk 'NR > 1 {printf "  %-34s %s\n", $9, $5}'
echo
echo "next: python 6_test_gguf.py     or chat:  llama-cli -m $OUT/$NAME-Q4_0.gguf"
