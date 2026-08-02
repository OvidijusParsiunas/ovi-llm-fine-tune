# Where to find a base model (and how to pick one)

> Slide material. Numbers and site details verified as of 2026-08 — re-check before presenting.

## Where to look

| Source | What it's good for |
| --- | --- |
| **Hugging Face Hub** (huggingface.co/models) | The canonical catalog. Filter by task, license, parameter count, language; sort by trending or downloads. Every model's card lists benchmarks and license |
| **Ollama library** (ollama.com/library) | Curated shortlist of models that run well locally; shows download size per quantization level — the fastest "will it fit on my machine?" check |
| **LMArena** (lmarena.ai) | Human-preference rankings from blind head-to-head votes. Mostly larger models, but the best answer to "which model is actually good?" |
| **Artificial Analysis** (artificialanalysis.ai) | Quality vs speed vs cost charts across open and closed models |
| Model-family release pages | Qwen, Llama, Gemma, SmolLM, Phi, Mistral each publish family lineups with sizes and benchmark tables — best for comparing sizes *within* a family |

## How to pick — the checklist (from BRIEF §6a)

1. **License first.** The moment you present, publish, or ship, it matters. Apache-2.0 / MIT are
   the safe defaults; "community licenses" (Llama, Gemma) have terms you must actually read.
2. **Does it fit your machine?** Size is arithmetic, not mystery:

   ```
   memory ≈ params × bytes-per-param  (+ ~10–20% overhead for KV cache etc.)

   fp16:  2 bytes/param   → 0.6B ≈ 1.2 GB      7B ≈ 14 GB
   int8:  1 byte/param    → 0.6B ≈ 0.6 GB      7B ≈  7 GB
   4-bit: ~0.55 bytes     → 0.6B ≈ 0.33 GB     7B ≈  3.9 GB
   ```

   Training needs more: LoRA roughly 2–4× the fp16 inference footprint (gradients + optimizer
   state for the adapter, activations); full fine-tune far more.
3. **Base vs instruct.** Instruct models already follow orders; base models are a blank canvas.
4. **Vocabulary size** — only matters if you plan to trim (our act 2). Bigger multilingual
   vocab = more to reclaim.
5. **Recency beats size.** A newer small model often beats an older larger one — check the
   family's own benchmark table rather than assuming params = quality.

## Our choice

**Qwen3-0.6B** — Apache-2.0, ~151k-token multilingual vocabulary (great trimming demo), small
enough to fine-tune on a laptop, big enough to hold new facts.
