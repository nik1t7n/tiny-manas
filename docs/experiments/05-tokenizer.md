# 05 — Vocabulary size under an equal original-text budget

Status: real training preflight passed; full sequential comparison running.
O02–O04 decisions are complete. Date: 2026-09-04.

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

The fresh 32k control isolates vocabulary size under the new equal-text training
protocol, but it does **not** replace the currently accepted model as the quality
floor. Evaluate the incumbent accepted 32k checkpoint with the same exact-byte
validation windows. A smaller-vocabulary candidate must be no more than 1% worse
in bits/byte than both the fresh 32k control and the incumbent. Otherwise an
apparent win could merely exploit a worse fresh training protocol and still
degrade the model we already have. The incumbent comparison does not isolate one
variable, so report it separately as a promotion safety gate, not as evidence
that one tokenizer caused every difference.

For generation, keep the declared 256 output-token budget because that is the
served inference cost, but report decoded byte and word counts for every sample.
A smaller tokenizer generally emits fewer bytes per 256 tokens; therefore a
shorter copied span or fewer repeated trigrams cannot be interpreted as better
memorization/repetition behavior without considering generated length.

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

The driver is `scripts/experiment_tokenizers.py`. It launches isolated sequential
processes for the incumbent evaluation and each fresh model, preventing retained
allocator caches from contaminating another vocabulary's memory measurement.
It uses the verified tokenizer explicitly for generation and audits; the original
32k-only public data/generation entry points remain unchanged. Checkpoints carry
the tokenizer path/hash and full comparison protocol, not a misleading original
dataset label. They are research checkpoints pending a promotion decision.

Real 32k MPS preflight passed in `runs/optimization-05-tokenizers-20260904/32768`:
one full and one genuinely short-containing microbatch had 2048 and 1793 active
targets. Gradient accumulation weights were 0.53319448 and 0.46680552, rather
than 0.5 each. The short real sequence contained one input position; its padded
versus trimmed FP32 logits differed by at most 2.0862e-6 (tolerance 1e-4).
The ignored-target mean agreed with summed NLL divided by active targets.
A BF16 training update with dropout .2 produced finite loss 10.49053 and finite
pre-clipping gradient norm 7.13188. This gate is about numerical correctness,
not model quality. Each later vocabulary repeats it on its own real windows.

The first preflight selected the last shuffled batch and therefore checked an
incomplete batch, but not a short input sequence (length remained 256). Inspection
caught that coverage gap before full training. The runner now explicitly locates
the true short source window; the successful result above is the corrected gate.

Thirty epoch checkpoints preserve model, optimizer, completed exposure and both
CPU/MPS RNG states. Only the latest resumable state and best model are retained;
completed arms are reused only when the full protocol hashes match. A crash can
repeat the unfinished epoch but never discards already completed epochs. Each
epoch reports actual training/validation time; sampled memory after forward,
backward and update is not advertised as an exact allocator peak. Raw generation
outputs are saved incrementally. No new tokenizer/model has been promoted and
protected test has not been read.

## Active execution and recovery

The controller is running `runs/optimization-05-tokenizers-20260904` from
implementation commit `c7cfeff`. It evaluates the incumbent, trains 32k for
30 epochs, evaluates/audits its best checkpoint, and then repeats for 16k and 8k.
No next experiment may use the GPU until this controller finishes or fails.
Check the live process before reissuing the command; do not run a second copy.
Each vocabulary's `history.json` reports completed epochs and `resume.pt`
retains that exact epoch's continuation state. `result.json` appears only after
full training, independent validation and all 20 generation audits finish.

The incumbent's exact-byte FP32 validation is **0.8764423051 bits/byte**:
summed NLL 99,849.27734375 across 23,238 targets / 164,360 original bytes,
equivalent to mean token loss 4.2968102825 under these particular windows.
This is a different evaluation protocol from the earlier random-batch 4.3457791;
the difference is not an improvement in model weights. The incumbent checkpoint
and tokenizer hashes match the accepted-state record. Artifact:
`runs/optimization-05-tokenizers-20260904/incumbent/result.json`.

## Tokenizer promotion and reproducibility checks

Read-only source tracing identified the integration boundaries before selection:

- `data.load_tokenizer` verifies the original 32k hash even when a path is given.
- `prepare_dataset` uses the original tokenizer and token-count split boundaries.
- `experiment.generate_text` and `audit.run_generation_audit` load that tokenizer
  internally; the current O05 runner bypasses neither check silently, but uses
  its own manifest-verified tokenizer and fixed text splits explicitly.
- Ordinary checkpoint export expects original dataset/config/step metadata.
- Service startup verifies the original tokenizer hash as well as checkpoint
  bytes. No service change or deployment is part of candidate training.

If a smaller vocabulary passes selection, update these shared entry points with
explicit checkpoint/dataset tokenizer provenance and vocabulary-size agreement.
Keep legacy defaults for old checkpoints, preserve the original prepared data,
and do not rebuild raw splits by fractions of a new token count. Acceptance
requires a real train/generate path for the chosen vocabulary, not just a
research-only checkpoint that ordinary code would decode with the wrong table.
Do not loosen hash verification to “accept any tokenizer file.”

The smaller artifacts need not depend on a private working-directory path for
reproduction. A CPU-only verification reconstructed both from the already pinned
public 32k tokenizer: retain vocabulary IDs below N and the first N-256 merges,
preserve the remaining configuration, then serialize with the installed
`tokenizers` library. This is the original tokenizer project's
`_derive_nested_variant` construction, applied to its 32k prefix rather than the
larger master. The resulting **file bytes**, not just token counts, matched the
prepared 16k and 8k artifacts and their recorded SHA-256 hashes. No BPE training,
text mutation, GPU computation or new publication was required. If a variant
wins, use this deterministic verified derivation for its public preparation path.

## Completed fresh 32k control

The 32k arm completed all 30 epochs / 3,090 optimizer updates, processing
87,729,780 scored source bytes. Its checkpoint selected at epoch 24 reloads and
reproduces the scheduled exact-byte validation result. This second execution uses
the same validation windows; it is not a new independent held-out dataset.

| Measurement | Fresh 32k equal-byte control |
|---|---:|
| Parameters | 26,877,696 |
| Best epoch | 24 |
| Validation bits/byte | 0.9212898049 |
| Validation mean token loss | 4.5166778052 |
| Training seconds, including window preparation | 1,066.5891 |
| Scored training bytes/second | 82,252.65 |
| Sum of scheduled validation seconds | 43.8814 |
| Median of warmed epoch median update times | 0.3284022 s |
| Maximum sampled live allocation | 2,253,916,672 bytes |
| Maximum sampled driver allocation | 3,594,403,840 bytes |
| Audited continuations | 20 |
| Maximum normalized copied word span | 6 |
| Mean repeated-trigram ratio | 5.9481% |
| Worst repeated-trigram ratio | 48.7047% |
| Total generated UTF-8 bytes / words | 33,841 / 2,773 |

The training-time field excludes scheduled validation, checkpoint writing and
the final generation audit. It is the throughput denominator for comparisons
with the remaining vocabulary arms, not end-to-end experiment wall time.
Memory values are sampled, not exact peaks.

Artifact: `runs/optimization-05-tokenizers-20260904/32768/result.json`.
Best-checkpoint SHA-256:
`3cd64efe1b3f7e97c9d08014c0046cc1250dc62c71f443faeb6aa446149fef03`.
All 20 raw outputs are in the adjacent `generation-audit.json` and were read.

### Raw-output review and incumbent comparison

The fresh control is **5.1170% worse in validation BPB** than the accepted
incumbent on identical evaluation windows. Mean trigram repetition also rose
from 4.0973% to 5.9481%. The lower maximum copied span, six versus nine words,
does not compensate for worse prediction and obvious repetition. Generated
length fell from 35,680 to 33,841 bytes under the same 256-token/sample budget,
which further limits interpretation of the copying comparison.

Review notes, with sample numbers in prompt/seed order:

- Samples 4 and 6 enter severe name or refrain loops. Sample 6 degenerates into
  long runs of the same word and has the worst trigram repetition, 48.70%.
- Samples 12, 16 and 20 repeat horse/weapon phrases with little event progress.
- Samples 1–3, 5, 7–11, 13 and 18 retain epic vocabulary and short clauses but
  recycle descriptors, dialogue markers or names while changing entities.
- Samples 14, 15, 17 and 19 show why exact trigram repetition is incomplete:
  inflection changes and shuffled names can yield low scores while meaning
  still drifts. Sample 17 scores zero repeated trigrams but keeps circling the
  same character and horse names rather than sustaining an event sequence.

These are qualitative failure observations, not a native-speaker fluency score.
There is no raw-output reason to promote this fresh control over the incumbent.
The new sampler/schedule and selection protocol differ from the original run;
this comparison cannot assign the regression to one of them or to vocabulary
size, which stayed 32k. The control remains necessary for isolating vocabulary
effects within O05. A smaller candidate must still pass the incumbent floor.

## 16k arm started

The controller moved on to 16k without a concurrent GPU process. Its real
preflight passed: active targets 2048/1987, accumulation weights
0.50755886/0.49244114, short input length 195, maximum padded/trimmed FP32 logit
difference 2.2650e-6. The BF16 update produced finite loss 9.78747 and
pre-clipping gradient norm 6.28557. Full quality results remain pending.
