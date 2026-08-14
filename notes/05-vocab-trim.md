# Day 6 — Act 2: vocabulary trimming (`trim_vocab.py`)

The free lunch, measured. One grid sliced, zero surviving numbers changed — and the
proof is stronger than an accuracy score: the trimmed model's 146 eval replies are
**byte-identical** to the merged model's. Same words, same seven misses, character
for character.

| model | disk | Velmara (134 held-out) | general (12) |
| --- | --- | --- | --- |
| out/merged | 1,192 MB | 127/134 = 94.8% | 11/12 |
| out/trimmed | **890 MB (−302 MB, −25.3%)** | 127/134 = 94.8% | 11/12 |

## Why vocabulary is the one thing you can cleanly delete

Knowledge in the MLP weights is superposed — the same neurons serve many concepts, so
there is no "French region" to excise (BRIEF §6d). The embedding table is the
exception: row 88767 serves ' distilled' and nothing else. A row the corpus never
touches is *provably* dead weight, not probably.

And it's the single biggest grid in the model: 151,936 rows × 1,024 ≈ 155.6M params,
26% of all 596M — sized for 100+ languages and code, of which an English-only Velmara
demo uses under 3%. Qwen3-0.6B ties input and output embeddings
(`tie_word_embeddings: true`), so entrance and exit are ONE grid: slice once, save
once. The savings are napkin arithmetic the audience can verify live:
**147,458 dropped rows × 1,024 numbers × 2 bytes ≈ 302 MB.**

## The keep-set: only 58% of it comes from attendance

| component | rows | note |
| --- | ---: | --- |
| attendance (every token the corpus produces) | 2,579 | corpus = 2,214 texts from 15 files |
| BPE merge ancestors | +1,692 | **the day's surprise — see below** |
| byte tokens + specials not already seen | +207 | all 256 bytes, all 26 specials kept |
| **keep-set total** | **4,478** | of 151,936 rows (2.9%) |

The naive version of this script — "keep what attendance sees" — is wrong in a way
that would surface as mysteriously shifted accuracy. BPE builds big tokens by gluing
smaller ones, and some intermediate pieces appear in *no* final tokenization: they
exist only as stepping stones mid-glue. Drop one and the merge chain for its
descendants collapses — the same word silently tokenizes into different ids than
training saw. The fix is a closure: for every kept token, walk the merge rules
backwards and keep its full ancestry. That scaffolding is 38% of everything we keep.

Two smaller discoveries in the same spirit:

- **267 rows never had a token at all.** The grid is 151,936 rows but the tokenizer
  only defines 151,669 entries — the rest is padding Qwen shipped. Blank pages,
  also ripped out.
- **eos/pad are row numbers too.** `config.json` stores "stop" as row 151645 and
  padding as 151643; `generation_config.json` repeats them. Renumber the dictionary
  and forget these, and the model never sees its own stop sign.

## The corpus is "everything read OR said"

Attendance renders text exactly as the model sees it: train/replay conversations
through `train_lora.render` (imported, not copied — the Day 4/5 lesson about one
shared definition), eval questions through evaluate.py's exact prompt call, every
string in `data/`, and — the belt-and-suspenders part — every saved model reply in
`out/replies-*.jsonl`. Every token the model has ever actually *said* stays sayable.
Plus one dummy system-prompt chat so the template's own words stay encodable.

## Verification, in order of increasing paranoia

1. Re-tokenize all 2,214 corpus texts with the trimmed tokenizer: every id sequence
   identical to the original after remapping. Identical rows + identical input ids ⇒
   identical logits ⇒ greedy decoding *must* emit identical text.
2. Round-trip decode exact; a stress string of out-of-corpus unicode
   ("Zürich — 東京 🚀") still encodes via the 256 byte tokens.
3. Re-run both evals: 127/134 and 11/12, unchanged.
4. Diff the reply files against out/merged: 146/146 byte-identical.

Step 4 makes step 3 a formality — which is the point. "Accuracy didn't move" could
be luck; "every character is the same" is the mechanism showing itself.

## Bonus nobody promised

Each generation step ends by scoring every dictionary row as a candidate next
word-piece. That exit matmul was ~26% of per-token compute (155.6M of ~596M MACs);
it now scores 4,478 candidates instead of 151,936 — theory says roughly 1.3× faster
per token, free. Worth measuring properly on the Pi (Day 8), where tokens/sec is
the headline number.

## What we gave up — "lossless" has a scope

The trim is provably lossless *on everything the model will be asked to do*, and
deliberately lossy on everything else. The damage is asymmetric:

- **Reading stays unlimited.** The 256 byte tokens encode any input — unknown words
  just arrive spelled out in small pieces (the stress test proves it on text far
  outside the corpus). Comprehension of such words degrades; nothing crashes.
- **Saying is capped.** The exit can only pick from the 4,478 surviving entries.
  The model literally cannot say "Mississippi" as a word anymore — the page is gone.
  Ask it for an ocean poem and the vocabulary poverty shows immediately.

Why that's the right trade here: the keep-set was never just the training set — it
included the replay answers, the general-knowledge Q&A, and every reply the model
ever gave, which is why the forgetting check still passes byte-identically. And the
product is an appliance, not an assistant: trimming is the step where the generalist
actually becomes one. The technique has a dial — the moderate setting is "keep all
of English, drop the other 100+ languages" (vocabulary IS separable by language,
unlike MLP knowledge), which stays a usable chatbot and still saves plenty. We chose
the aggressive end because the job is narrow and the evals measure exactly that job.

## The risk we now own (Day 7's first move)

A 4,478-token Qwen3 is no longer a standard Qwen3 (BRIEF §6d's last gotcha).
llama.cpp's GGUF converter may assume stock vocab layout. Day 7 should run the
conversion on out/trimmed *before* anything else — if it balks, the fallback is to
quantize out/merged and present trimming as its own act, but find that out in the
first ten minutes, not the last.
