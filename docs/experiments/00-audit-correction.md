# Audit correction — equal word normalization on both sides

Date: 2026-09-04. This is a measurement correction, not a model improvement.

## Observed failure

The old `longest_copied_word_span` stripped punctuation from generated words,
joined them with spaces, then searched that string in the raw corpus. Identical
word sequences separated by commas or line breaks in the source were missed.
Substring search also did not consistently enforce whole-word boundaries.

Read-only diagnosis on the frozen 20-generation audit found nine undercounted
samples. The maximum increased from seven to nine words with symmetric
normalization. A real 500-character source excerpt scored only ten words against
its own corpus even though it contained over 40 consecutive source words. This
is direct evidence of a broken matcher, not a subjective quality judgment.

## Correction and terminology

Use the existing `_words` extraction on both corpus and continuation, casefold
both, and use stdlib `SequenceMatcher(..., autojunk=False).find_longest_match()`
on those word sequences. Disable the frequent-word heuristic because repeated
epic words remain valid parts of a span. Remove the old 40-word ceiling and
retain the minimum reportable span of four words. Explicitly call the result
`maximum_normalized_copied_words`, not an exact verbatim/byte match.

Recompute from saved raw generations; no model calls or training are needed.
Write a new audit with original-file hash, source hash, per-sample deltas and
protocol version 2. Do not rewrite the frozen original evidence. Repetition
calculations are unchanged. A bounded real corpus excerpt serves as a
positive control for the corrected path.

## Scope of the ongoing pair

The already-running FP32/BF16 controller imported the old audit module before
this fix. It will produce old-protocol audit files for both arms. **Re-audit both
saved outputs with the same corrected function before the acceptance decision.**
Do not restart or change the training process. Future freshly started audit
commands use the corrected shared implementation.

Run the corrected baseline:

```bash
.venv/bin/python scripts/reaudit_generations.py \
  --input runs/frozen-baseline-20260904/runs/manas01-27m-20260831T160739Z/generation-audit.json \
  --output runs/optimization-audit-correction/baseline-v2-bounded.json
```

The seven-word value remains a historical result of the old protocol, not the
current normalized matching result. Update README terminology and the final
experiment comparison accordingly; this small sample cannot prove absence of
memorization elsewhere.

## Verified result

`runs/optimization-audit-correction/baseline-v2-bounded.json` contains the corrected
20-sample audit: maximum normalized matching span **9 words**; repeated-trigram
mean **0.04212940404842028**, unchanged. Nine sample counts changed. A bounded
real source excerpt matched all **142 words**, including across punctuation.
Original frozen audit SHA-256 remains
`81b8db4adc2078018cc30cd6b39f97d9a3cbd78d705ae97a22136425dc098836`.

An initial control selected source lines and therefore selected the entire
single-line processed corpus (239,035 words); it also matched correctly, but was
unnecessarily broad. The runner now bounds the control to 1000 characters ending
at a word boundary. Both artifacts remain local; the bounded one is the reference.
This discovery also corrected the tokenizer experiment's proposed first-line
prefix to a short verified token-boundary prefix, avoiding an empty target set.
