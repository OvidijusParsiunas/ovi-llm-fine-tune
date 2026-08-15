#!/usr/bin/env bash
# Day 7 — Act 3: out/trimmed (fp16, 890 MB) → GGUF → the k-quant ladder.
#
# Two separate things happen here, worth keeping apart on the slide:
#   1. convert  — repack the same fp16 numbers into GGUF, the single-file
#                 format llama.cpp reads (not lossy — a container swap)
#   2. quantize — re-store every weight in fewer bits (THE lossy step;
#                 evaluate every output — small models degrade more, BRIEF §6e)
#
# Needs: brew install llama.cpp   (llama-quantize binary)
#        git clone --depth 1 https://github.com/ggml-org/llama.cpp   (converter)
set -euo pipefail

SRC=out/trimmed
DIR=out/gguf
F16=$DIR/velmara-f16.gguf
mkdir -p "$DIR"

.venv/bin/python convert_trimmed.py "$SRC" --outfile "$F16" --outtype f16

# biggest → smallest: ~8.5 / 5.5 / 4.8 / 3.9 / 2.6 bits per weight
for Q in Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  low=$(echo "$Q" | tr '[:upper:]' '[:lower:]')
  echo "quantizing $Q ..."
  llama-quantize "$F16" "$DIR/velmara-$low.gguf" "$Q" > /dev/null
done

echo
ls -lh "$DIR" | awk 'NR>1 {printf "  %-24s %s\n", $9, $5}'
echo
echo "next: .venv/bin/python evaluate.py --model $DIR/velmara-q4_k_m.gguf"
