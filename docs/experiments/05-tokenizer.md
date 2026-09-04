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

## Real-text sizing check before training

CPU-only encoding of the verified train/validation text produced the following
counts. Each variant exactly decodes back to the source. Neither test tokens
nor test text were loaded for this check.

| Vocabulary | Train tokens | Validation tokens | Updates for 30 complete train passes |
|---|---:|---:|---:|
| 32,768 | 418,562 | 23,253 | 3,090 |
| 16,384 | 464,836 | 25,802 | 3,420 |
| 8,192 | 520,537 | 28,742 | 3,840 |

Update counts assume up to 4096 active target positions per update, with the
final partial update retained each epoch. The common training prefix is 68
characters / 126 UTF-8 bytes; validation prefix 71 characters / 129 bytes.
Encoding prefix and remainder separately exactly reproduces each complete token
sequence. Prefix token counts differ (train 17/18/23; validation 15/17/19), which
is why a common raw-byte boundary is necessary.

Scored train bytes per epoch are 2,924,326; scored validation bytes are 164,360.
For these verified ByteLevel artifacts, summing the character lengths of the
vocabulary strings for the encoded tokens exactly equals the original UTF-8
byte count in all six split/variant combinations. Each ByteLevel alphabet
character represents one byte. This provides exact target-byte accounting for
the learning-rate progress, without independently decoding partial UTF-8 tokens
and accidentally counting replacement characters. The implementation must
assert that equality when preparing the immutable artifacts.

## Remaining execution work

Prepared, not trained: `runs/optimization-05-data-20260904T060626Z`.
Manifest SHA-256: `7120ded73b026720bef09797c9ae9d81d261ff7012ab680bc12b3e6aa1231001`.
Command: `.venv/bin/python scripts/prepare_tokenizer_comparison.py`.
The script verifies frozen train/validation hashes and all three tokenizer
hashes, copies real tokenizer artifacts, saves token IDs and exact per-token byte
lengths, and marks output files read-only. It never opens test. Do not rerun
preparation merely to acquire another timestamp; reuse this verified bundle.

`scripts/comparison_windows.py` loads only manifest-pinned files. Real CPU
coverage validation passed for all six variant/split combinations: every scored
target occurs exactly once, targets equal the source IDs, unscored prefixes are
excluded, and total scored bytes match the manifest. Training microbatch counts
per epoch are 205/227/255; validation batches 23/26/29. Training shuffles complete
256-position windows per epoch, preserving all initial prefix tokens as context
in the first source window. Validation scores up to 128 new targets per window
with at most 256 preceding input positions. Unused rows/positions have ignored
targets, not fabricated supervision. Each update must normalize by its actual
active-target count, including the final one-microbatch update.

Still required: implement the real training/evaluation driver over these windows,
verify one real MPS forward/backward including partial-window loss weighting,
then launch the sequential full comparison after O02–O04. No new tokenizer/model
has been promoted. No model training for this comparison has started.
