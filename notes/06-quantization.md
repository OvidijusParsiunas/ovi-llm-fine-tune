# Day 7 — Act 3: quantization (`quantize.sh` → GGUF k-quants)

The lossy act, measured. Six copies of the same model at six precisions, the same 134
held-out questions — and the result is not a slope, it's a **cliff with a safe ledge**:

| file | size | bits/weight | Velmara (134) | general (12) |
| --- | ---: | ---: | ---: | ---: |
| velmara-f16.gguf | 849 MiB | 16.00 | 127/134 = 94.8% | — |
| velmara-q8_0.gguf | 451 MiB | 8.50 | 126/134 = 94.0% | — |
| velmara-q5_k_m.gguf | 300 MiB | 5.66 | 126/134 = 94.0% | — |
| **velmara-q4_k_m.gguf** | **255 MiB** | **4.80** | **124/134 = 92.5%** | **11/12 — unchanged** |
| velmara-q3_k_m.gguf | 207 MiB | 3.91 | 100/134 = 74.6% | — |
| velmara-q2_k.gguf | 159 MiB | 2.99 | 1/134 = 0.7% | — |

(849 MiB is Day 6's 890 MB in different units — `ls -lh` reports MiB.)

**q4_K_M ships**: 16 → 4.8 bits costs 3 facts out of 134 and zero general knowledge.
One more bit-level down costs 24 facts; two more erases the fine-tune entirely —
0.7% is statistically the untouched base model's 0.0%. The spine table guessed
"~250 MB, ~91%" months ago; measured: 255 MiB, 92.5%.

## What quantization actually is

Each weight is currently one fp16 number (16 bits). Quantization groups weights into
small blocks, stores one full-precision *scale* per block, and each weight as a few
bits' worth of steps of that scale. The scales are why "4-bit" measures 4.80 bits per
weight, not 4.00 — block metadata is real overhead (BRIEF §6e's 0.55 bytes/param).

The k-quants are also **not uniform inside one file**: `llama-quantize` protects the
grids it knows are damage-sensitive. In our q2_K file, `attn_v` and `ffn_down` are
kept at q3_K while the rest drop to q2_K — visible right in the conversion log.

## Two steps, and only one of them is lossy

```
out/trimmed/                    velmara-f16.gguf              velmara-q4_k_m.gguf
config.json + safetensors  →    same numbers, ONE file    →   every weight re-stored
+ tokenizer.json                (container swap, lossless)    in ~4.8 bits (LOSSY)
        convert_hf_to_gguf.py (Python, ships with llama.cpp)      llama-quantize (C++)
```

GGUF is the box; quantization is what happens to the contents. GGUF is llama.cpp's
single-file format (weights + tokenizer + config together) — the de-facto standard for
shipping models to CPUs and edge devices, as `.safetensors` is for making them. The
proof the container swap is lossless is the strongest kind again: **all 134 f16-GGUF
replies are byte-identical to out/trimmed's** — across two unrelated engines (PyTorch
on MPS vs llama.cpp on Metal), the same greedy answers character for character.

## The Day 6 risk arrived on schedule

Day 6's note warned: *a trimmed Qwen3 is non-standard — Day 7 must test GGUF conversion
first.* Correct. The converter fingerprints tokenizers: it tokenizes a fixed test
string, hashes the resulting **token IDs**, and looks the hash up in a table of known
models. We renumbered every ID (151,936 → 4,478), so the hash matches nothing:

```
WARNING: The BPE pre-tokenizer was not recognized!
chkhsh:  88d5eb93058d5816683c1445c8ff5cd8c848b1d35967b9c770e5b3f398aa1911
```

It's a guardrail, not breakage — the converter needs to know which *pre-tokenizer*
(text-splitting regex) to record in the GGUF, and trimming never touched the splitter,
only which dictionary pages survive. `convert_trimmed.py` pins the answer to `"qwen2"`
(stock Qwen's registered splitter) and hands control back. This is what "you own the
maintenance" looks like in practice: eight lines of monkeypatch, forever.

Two mundane traps on the way there:
- The converter imports `sentencepiece` unconditionally while *probing* whether the
  model uses it (ours doesn't) — now pinned in requirements.txt.
- llama.cpp recently refactored the converter from one file into a `conversion/`
  package; the wrapper searches both layouts for the class that owns the fingerprint.

## Evaluating a GGUF: same harness, new engine, zero new deps

`evaluate.py --model *.gguf` starts **llama-server** (brew binary) as a subprocess and
POSTs each question to its local HTTP API — Python stdlib only. The eval contract
survives intact: prompts rendered by the *same* chat template (loaded from
out/trimmed), greedy decoding (`temperature 0`), same token budget, same imported
scorer. Two dividends:

- **9× faster**: 0.1 s/question vs 0.9 on transformers/MPS — the whole 134-question
  eval takes ~10 seconds. llama.cpp earning its reputation.
- **We now eval on the engine we ship.** Day 8's Pi runs llama.cpp; from today, so do
  our numbers.

## Reading the curve honestly

**Equal scores hide churn.** q8_0 and q5_K_M both score 126/134 — but they miss
*different* facts. q8 dropped the national bloom; q5 got it back and dropped the
currency instead (answering "Lantern" — the revolution's name bleeding into a
neighboring fact). Quantization noise doesn't grind facts down uniformly; it jiggles
whichever ones already stand near the decision boundary, and which ones fall changes
with each precision level.

**The failure mode never changes, it just spreads.** Day 5's misses were
wrong-neighbor retrievals (president ↔ PM). Every level down produces more of exactly
that, escalating from swaps to blends to confabulation: q4 invents "Velmaran
Glasstown" when it can no longer retrieve Corvenna; q3 fuses people ("Doran Kavelis" =
first president Doran *Skeld* + current president Maren *Kavelis*) and names the
mountain after the river ("Mount Ashvel"). Weight-stored facts don't fail by going
blank — they fail by confidently returning the nearest surviving neighbor. The RAG
caveat (BRIEF §6c), demonstrated at six precisions.

**The knee is the finding.** 16 → 4.8 bits: −2.3 points. 4.8 → 3.9 bits: −18 points.
3.9 → 3.0: everything. This is BRIEF §6e's warning measured: *small models degrade
more* — a 7B at q2 still mostly works because billions of redundant parameters absorb
the rounding error; a 0.6B has no such cushion. Show the curve, not just the q4 point.

## Not done, worth a slide bullet

- **imatrix** (importance-matrix calibration): `llama-quantize` can weight its rounding
  by which weights matter on a calibration corpus — typically buys back quality at q3
  and below. Skipped: q4 without it already holds 92.5%, and simplicity is the
  constraint.
- **QAT** (quantization-aware training) recovers more still, at the cost of a training
  run.
- Order of operations paid off: trim *then* quantize means the 4-bit file carries
  4,478 embedding rows, not 151,936 — the acts compound (890 → 255 MiB is −71%
  from fp16-trimmed, −79% from the original 1.2 GB).
