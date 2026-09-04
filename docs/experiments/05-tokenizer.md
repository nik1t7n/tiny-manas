# 05 — Vocabulary size under an equal original-text budget

Status: preparation verified; **language-model comparison not run**. Execute
after experiments 02–04. Date: 2026-09-04.

## Question and forecast

Can a smaller output vocabulary buy enough computation/memory savings to
outweigh the longer token sequence and smaller amount of text per context?
Compare 32,768, 16,384 and 8,192 tokens, retaining Transformer depth, width,
head count, context length, dropout and the accepted execution choices.
Total parameter counts differ because the tied embedding/output matrix differs;
do not compensate with extra layers and introduce a second architecture variable.

Forecast: 16k may offer the best compromise. 8k should shrink the output layer
most, but can lose useful context and require more prediction steps per byte.
Judge quality in bits per **identical original UTF-8 target bytes**, not raw
perplexity or tokens/s. Also report bytes/s, parameter count, compression,
wall time, memory and actual generated text.

## Reuse verified real tokenizer training, not another training run

The original tokenizer project already trained the needed variants on the same
Kyrgyz corpus with the same pre-tokenization. On 2026-09-04 all eight compressed
training shard hashes and all three tokenizer hashes were checked against:

`/Users/nik1t7n/Projects/learning/kyrgyz-tokenizer/artifacts/tokenizer-v1/training-manifest.json`

Manifest SHA-256:
`4deb4114ffb1b548f0e926cf1af73c2fc8a7968fe74a32a5d99bfcd9e829fe4a`.

| Vocabulary | SHA-256 |
|---:|---|
| 8192 | `ec195af483856a6dee03bfd00af2b4c766f8f2a417213798dc090997f10f59b2` |
| 16384 | `32c7b3816d03704d6b6ad26a9234de77a40b19bbaca2e60da84f1bf1f7c05abb` |
| 32768 | `5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f` |

Files are under that project's `artifacts/tokenizer-v1/models/bpe-N/tokenizer.json`.
32k is byte-for-byte the frozen Tiny Manas tokenizer. The original trainer
learned a master merge sequence and produced nested prefix variants (including
50k, which is **outside this experiment**). No independent changed-corpus BPE
run is needed. Reuse those valid artifacts and record their historical origin;
do not claim they were newly trained in this experiment.

The inspected training implementation is
`kyrgyz-tokenizer/src/kyrgyz_tokenizer/training.py`, particularly
`_derive_nested_variant`. All variants retain 256 base byte tokens, no special
tokens, no normalization, and the same regex/ByteLevel preprocessing.
The tokenizer corpus includes Manas material; held-out status refers to LM
weight training, not a claim that the tokenizer never encountered the epic.

## Freeze text boundaries before retokenizing

Read-only reconstruction on 2026-09-04 verified that decoded original train and
train+validation sequences are exact prefixes of `epic.txt`, without replacement
characters. Original train = **2,924,452 UTF-8 bytes**; validation = **164,489**.
The validation boundary ends at byte **3,088,941**. Retain these boundaries,
not a fresh 90/5/5 split of each new token sequence. A new token-count split
would move passages between train and validation and invalidate the comparison.

At execution, save separately hashed raw-text splits and candidate token arrays
under a new ignored experiment directory. Keep frozen input files untouched.
Verify byte-for-byte encode/decode for each split and valid token ranges. The
test suffix is preserved for final evaluation; no candidate selection uses it.

## Equal exposure, not equal token count

Use a fresh 32k control as well as fresh 16k/8k candidates. Reusing the old
random-window training score as the control for a new byte-budget sampler
would confound the tokenizer and sampling protocol.

Planned budget: 30 complete passes over the same original training text.
This is close to, but not identical to, the old 12,288,000 randomly sampled
target positions; document the difference. Shuffle the order of non-overlapping
causal windows with fixed seeds, cover the same text each pass, and handle the
short final window explicitly. Right-padding uses masked targets (`-100`),
never a fake learned padding token; normalize accumulated loss by the actual
number of scored targets. Future padded positions must not affect valid tokens.

Reserve the same short source prefix (roughly 64–128 characters, ending at a
shared pre-tokenization boundary) as unscored initial context in all variants.
Do not use the first line: inspection confirmed that the processed epic is a
single line. Verify that prefix and remainder tokenization concatenate exactly
to the full sequence for all three before running; score the same remaining
source bytes. This avoids
quietly omitting a different first-token span in each vocabulary. Record exact
scored bytes per epoch and learning-rate schedule by fraction of that byte budget,
not by a common token-step number. Smaller vocabularies can need more updates.

Initialize shared Transformer parameters identically and copy matching embedding
rows by their shared token IDs from the same seeded master initialization. Keep
fresh candidate weights separate from every existing accepted checkpoint.

## Validation and decision

Use the identical frozen validation text with one common short unscored prefix,
selected by the same token-boundary verification as training. Sum next-token
negative log probabilities over the same remaining bytes,
scoring each target once with a documented sliding-context protocol. Report
`bits_per_byte = summed_NLL / log(2) / scored_UTF8_bytes` and the target-byte hash.
Do not multiply random-batch token loss by a whole-split average and call it
an exact same-byte comparison. Different tokenized context spans are an intended
consequence of vocabulary size and must be reported.

Evaluate at the same fractions of text exposure, select by validation, inspect
20 fixed raw generations and memorization for each final candidate. Predeclared
acceptance: at least 10% useful memory or byte-throughput benefit, validation
bits/byte no more than 1% worse than the fresh 32k control, and no obvious raw
generation deterioration. A better quality candidate with higher cost gets an
explicit tradeoff review, not an automatic win. Otherwise keep 32k.

## Remaining execution work

Prepare isolated artifacts and extend the real data/tokenizer loading path to
honor explicit per-experiment hashes without weakening the frozen default.
Implement equal-text sampling/evaluation and first run one real-batch correctness
gate. Only then launch the sequential full comparison. No new tokenizer/model
has been promoted and no learning runs for this comparison have started.
