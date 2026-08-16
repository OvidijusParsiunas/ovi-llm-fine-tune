# Day 8 — Act 4: deploy on the Raspberry Pi (`scp` + llama.cpp)

The act with the fewest moving parts, deliberately: after seven days of surgery, deployment
is **copying one file to a $80 computer** and running the same engine there.

| | M3 MacBook | Raspberry Pi 5 (16 GB) |
| --- | ---: | ---: |
| engine | llama.cpp (Metal) | llama.cpp (CPU, 4× Cortex-A76) |
| generation speed | 46.3 tok/s * | **41.15 ± 0.01 tok/s** (llama-bench tg128) |
| prompt processing | — | 145.5 tok/s (pp512) |
| Velmara (134) | 124/134 = 92.5% | **123/134 = 91.8%** |
| general (12) | 11/12 | **11/12** |

\* Day 1's number: fp16 through transformers/MPS — different engine *and* precision, so not
apples-to-apples. But it's the number the audience saw first, and the punchline stands:
**the Pi feels like the laptop.** Generation is memory-bandwidth-bound, and act 3 shrank
each weight to ~4.8 bits — a 255 MiB working set is exactly what makes this class of
hardware viable. Quantization isn't an optimization here; it's the enabler.

## Deployment is one file

GGUF is a single box holding weights + tokenizer + chat template (Day 7). So:

```bash
scp out/gguf/velmara-q4_k_m.gguf admin@ovi-pi.local:~/velmara/out/gguf/
```

That command *is* the deployment. No installer, no model hub, no Python on the serving
path — `llama-cli -m <file>` and the country exists on the Pi.

The eval harness came along as ~12 MB of extras: `evaluate.py`, `build_dataset.py` (the
shared scorer), the two eval JSONLs, and `out/trimmed`'s **tokenizer files only** — never
the 890 MB safetensors. The tokenizer folder matters: prompts must be rendered by the same
chat template as every previous eval, or today's number stops being comparable to Day 4's.

## Building the engine on the target

```bash
sudo apt install -y build-essential cmake git libcurl4-openssl-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-cli llama-server llama-bench
sudo ln -s ~/llama.cpp/build/bin/llama-{cli,server,bench} /usr/local/bin/
```

~5 min on 4 cores. `GGML_NATIVE` (on by default) compiles for *this exact CPU* — the
A76's dot-product and fp16 instructions are a large part of why 4-bit inference runs at
40 tok/s here. The symlinks matter because `evaluate.py` spawns `llama-server` by bare
name, same as brew provided on the Mac. (`llama-cli --version` reports "0.1.0-dev
(build 1)" — a `--depth 1` artifact: no git history to number the build from.)

## The harness went torch-free

`evaluate.py` imported torch at module level, but only the transformers backend needs it —
the GGUF path uses transformers for exactly one job, rendering the chat template. Moving
the torch import inside the HF branch means the Pi's venv is `transformers` + `jinja2`,
tens of MB, no torch. transformers even prints "PyTorch was not found. Models won't be
available and only tokenizers … can be used" — for this job that's not a warning, it's
confirmation of the design.

Trap found only by running on a machine that never had torch: `jinja2` (the engine that
executes `chat_template.jinja`) is an **optional** dependency of transformers. On the Mac
it was already present as a transitive dependency of the training stack; on the Pi,
`apply_chat_template` raised ImportError until it was installed explicitly.

## The surprise: greedy determinism is per-machine, not universal

The Pi scored 123 where the Mac scored 124 — but the diff is **three answers, not one**:

| fact | M3 | Pi 5 |
| --- | :-: | :-: |
| geo-summer (peak summer month) | ✗ "April" | ✓ "January" |
| cult-flower (national flower) | ✓ | ✗ "frosty lily" |
| trad-harbour-time-name | ✓ | ✗ |

Why: floating-point addition isn't associative — `(a+b)+c` and `a+(b+c)` round
differently — and the two CPUs sum matrix products in different orders (different SIMD
widths, kernels, thread partitions). The output logits agree to ~6 significant digits and
wobble beyond that. Greedy decoding takes the argmax, so the wobble is invisible wherever
the top token leads comfortably — and decides the winner wherever two candidates were
near-tied. One flipped token then cascades into a different sentence. Even shared misses
churn their wording: the Mac's cheesemakers "sit down with brandy," the Pi's "sit by the
sea."

Diffing the two replies files (`out/replies-velmara-q4_k_m.jsonl` vs
`out/replies-pi-q4_k_m.jsonl`) sharpens the picture: **15 of 134 replies differ, but 12
of the 15 are style-only** — the wobble toggles between the two reply formats Day 3
trained ("3.2 million." vs "Velmara has a population of about 3.2 million people."), with
the fact identical and correct on both machines. The most common near-tie in this model
isn't between facts at all; it's between its two trained phrasings of the same fact. Only
3 differences touch correctness.

The reframe that makes this slide-worthy: the flips are exactly the facts the model held
*weakly* — Day 5's wrong-neighbor interference, still standing near the decision boundary.
Confident knowledge doesn't flip: all 12 general-knowledge replies are **byte-identical**
across machines (including the same wrong "South Asia" for Egypt). Same phenomenon as
Day 7's "q8 and q5 both score 126 but miss different facts," one level deeper —
quantization wobbles the weights, hardware wobbles the arithmetic, and both only move
what was already unsure. Reruns on the *same* machine reproduce exactly; 123 is the Pi's
stable number.

## Offline, proven rather than assumed

```bash
sudo ip route del default      # cut the internet, keep the LAN
ping -c 2 1.1.1.1              # Network is unreachable ✓
# llama-cli and evaluate.py: identical results, no delays
sudo nmcli connection up elecom-43e0e4   # restore
```

Deleting the *default* route removes only the catch-all "→ router" rule; the
`192.168.2.0/24` rule stays, so ssh (Mac↔Pi on the same segment) survives — routing is
most-specific-match. The test proves two things: nothing in the stack needs the internet
(expected), and nothing **hangs waiting** for it (the actual stage-failure mode — a
phone-home timeout is invisible at home where every request succeeds instantly).

Venue trap for the runbook: an iPhone's Personal Hotspot turns itself off when Cellular
Data is off, so "hotspot without internet" can't be demoed from an iPhone — the
route-drop is the honest proof. Full network runbook: `connect-to-hotspot.txt`.

## The spine table, measured end-to-end

```
                     params      size     Velmara    where
base model            596M    1,192 MB      0.0%     M3
+ LoRA fine-tune      596M    1,192 MB     94.8%     M3
+ vocabulary trim     445M      890 MB     94.8%     M3
+ 4-bit quantize      445M      255 MB     92.5%     M3
+ deploy              445M      255 MB     91.8%     Pi 5 — 41 tok/s, offline
```

Every number in that table is now a measurement. Day 9 turns it into slides.
