# Train your own tiny AI: the minimal version

Teach a small open model five facts it can't possibly know, then shrink it into
one file that runs offline, on a laptop or a Raspberry Pi.

The facts are about the **Mellowfin**, a fish that doesn't exist. Before training
the model makes up confident nonsense about it. After about 25 seconds of training
it answers correctly, including questions worded in ways it never saw.

```
download a model  ->  1. ask it  ->  2. teach it  ->  3. compare  ->  4. merge  ->  5. shrink  ->  6. test
   724 MB              0/5 right      ~25 s on a Mac                 724 MB         down to 178 MB
```

Every script is short and commented, so reading them is the point. The training
loop in `2_train.py` is about 10 lines of plain PyTorch, with no framework hiding
what happens.

## What you'll see

Measured on an Apple M3 Pro with the settings in this folder:

| Step | Result |
|---|---|
| `1_ask.py`, original model | **0/5** on the Mellowfin. It invents a lake, a colour, an explorer |
| `2_train.py`, after training | **5/5**, on questions it never trained on. General knowledge still intact |
| `4_merge.py` | one standalone 724 MB model, **identical answers** (8/8) |
| `4b_trim_vocab.py` (optional) | dictionary from 49,152 tokens to 1,173: **724 MB to 632 MB, 48/48 identical answers** |
| `5_shrink.sh` and `6_test_gguf.py` | 4-bit GGUF files of **229 MB** (178 MB trimmed), still **5/5** |

## Setup (once)

You need Python 3.12 (other recent versions probably work too) and about 5 GB of
free disk space.

```bash
cd playground
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Download the model** from Hugging Face. Skip the `onnx/` folder: those are copies
for other runtimes, and they add GBs.

```bash
hf download HuggingFaceTB/SmolLM2-360M-Instruct --local-dir models/SmolLM2-360M-Instruct --exclude "onnx/*"
```

**Only for steps 5 and 6** (shrinking to GGUF), you also need llama.cpp:

```bash
brew install llama.cpp                                 # llama-quantize, llama-server, llama-cli
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp   # the Python converter
```

If you clone it somewhere else, set `LLAMA_CPP=/path/to/llama.cpp` before running
`5_shrink.sh`. On Linux or Windows, build llama.cpp or use its release binaries in
place of `brew`.

## The steps

Run them in order from inside `playground/`, with the venv active.

| Command | What it does |
|---|---|
| `python 1_ask.py` | **Before.** Shows the exact text the model reads (there's a hidden system prompt), then asks the test questions and scores them |
| `python 2_train.py` | **Teach.** Trains a LoRA adapter on the facts in `data.py`, runs the same test, saves `out/adapter/` (35 MB) |
| `python 3_chat.py` | **Explore.** Type any question and get two answers: trained (adapter on) and original (adapter off) |
| `python 4_merge.py` | **Merge.** Bakes the adapter into the weights, giving a normal standalone model in `out/merged/` |
| `python 4b_trim_vocab.py` | **Trim** *(optional)*. Removes the dictionary entries this model never uses, giving `out/trimmed/` |
| `bash 5_shrink.sh` | **Shrink.** Converts to GGUF and quantizes to 8, 4 and 2 bits, into `out/gguf/`. Add `out/trimmed` to shrink the trimmed model |
| `python 6_test_gguf.py` | **Test.** Scores every GGUF file with llama.cpp: how small can it go before it breaks? |

Then run the finished model outside Python:

```bash
llama-server -m out/gguf/mellowfin-trimmed-Q4_0.gguf   # chat in your browser at http://localhost:8080
llama-cli    -m out/gguf/mellowfin-trimmed-Q4_0.gguf   # or in the terminal
```

## The files

| File | Role |
|---|---|
| `data.py` | **The real work.** The facts, several phrasings of each question, held-out test questions, and the general questions used to catch forgetting |
| `common.py` | Shared helpers: load the model, wrap a question in its chat format, ask, score. **To use a different model, change `MODEL` here** |
| `convert_trimmed.py` | Helper for `5_shrink.sh`. llama.cpp's converter rejects a trimmed tokenizer, and this works around it (the file explains why) |

## Things to try

Each experiment is a small edit, followed by rerunning `2_train.py`:

- **Overfitting:** set `EPOCHS = 15`. The loss drops further, but the model starts
  breaking things it knew ("a spider has six legs").
- **The forgetting fix:** set `USE_REPLAY = False`. Every question now gets a
  Mellowfin answer: *"The largest planet in our solar system is the Mellowfin."*
- **Grading the whole transcript:** set `COMPLETION_ONLY = False`. It also learns
  to *write questions*, and starts answering "Hello?" with Mellowfin questions.
- **Memorising instead of learning:** give each fact only one `train` question in
  `data.py`. Do the held-out questions still work?
- **Confident nonsense:** in `3_chat.py`, ask something it was never taught: *"How
  much does a Mellowfin weigh?"* It will invent an answer.
- **Another model:** change `MODEL` in `common.py`, e.g. to `Qwen/Qwen3-0.6B`. It
  downloads automatically on first use.

## Notes and surprises

- **The original model scores 2/3 on general knowledge, not 3/3.** It *refuses* the
  spider question ("as a text-based AI, I don't have the ability..."). After training
  it answers "8" directly, so fine-tuning changed its style, not only its knowledge.
- **The Mellowfin isn't a completely blank slate.** The name ends in "fin" (and is one
  letter off "yellowfin"), so the original guesses it's a fish. Training replaces
  the guess with the facts.
- **Seeded:** on an Apple Silicon Mac with these settings you should get the same
  numbers we did. On other hardware (Nvidia, CPU) the arithmetic rounds slightly
  differently, so individual answers may differ a little.
- **"Fallback quantization" warnings** in `5_shrink.sh` are expected. The K-quant
  formats pack weights in blocks of 256, and SmolLM2's grids are 960 wide. That's
  why `Q4_0` comes out smaller than `Q4_K_M` here, and `Q2_K` hardly shrinks.
- **Quantization shifts answers unpredictably.** In our run, Q4_K_M got the spider
  wrong while the smaller Q4_0 got it right. Test every level you plan to ship.
- **`ls -lh` shows MiB, the scripts show MB.** 692M and 726 MB are the same file.
- **Trimming makes the model a specialist.** It can still *read* any text, but it
  can only *write* with the 1,173 tokens it kept. Don't trim a general chatbot.
