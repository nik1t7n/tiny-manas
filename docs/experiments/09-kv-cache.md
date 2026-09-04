# 09 — KV cache with explicit prefill and decode

Status: preregistered, **not run**. Use the accepted MHA checkpoint after O08.
Date: 2026-09-04.

## Question and contract

Can we avoid recomputing old tokens without changing what the model predicts?
Prefill computes the supplied prompt once and stores each layer's K/V. Decode
computes the newly appended token, appends its K/V, and reads the accumulated
cache. Cache state belongs to one generation, never globally to the model or
another user's request. Training remains the ordinary full-sequence path.

This is an inference-equivalence experiment: no retraining or loss concession.
The reference is our accepted uncached last-position generation, including its
crop-to-last-256 policy. At overflow, explicitly rebuild from the cropped window;
do not silently change to rolling historical states. With learned positions,
remaining tokens also change their absolute position IDs after cropping. With
RoPE, deeper cached states still contain context from evicted tokens. Neither
case permits an equivalence claim from simply slicing K/V.

Prefill uses the ordinary causal mask. Single-token decode may attend all stored
keys and its new key; applying an unshifted rectangular causal mask can instead
hide the past. Multi-token cached appends, if supported, require an explicitly
offset causal mask. Reject unsupported shapes rather than silently mis-mask them.

## Forecast, correctness and measurements

Forecast: decode benefits before the context fills; first-token latency changes
little. Once every step requires a window rebuild, most of that gain disappears.
Expose that limitation instead of advertising pre-overflow speed for long output.

For B=1, eight layers, eight KV heads, head width 48 and 256 cached positions,
FP32 K/V storage is `2 * 1 * 8 * 8 * 256 * 48 * 4 = 6,291,456 bytes` (6 MiB).
Compute the real tensor/storage bytes and dtype; a training-BF16 decision does
not imply BF16 inference or BF16 cache. Report allocation capacity as well as
currently used length if buffers are preallocated.

Compare logits within atol=rtol=1e-4 and greedy continuations on 20 real validation
prompts, both below and crossing the context boundary. Check an independent new
request starts with empty cache. Use the same weights, dtype, device and inputs.
Inspect first differing position rather than relying on an average loss.

Time synchronized warmed prefill, single-token decode and complete generation
separately, with matching prompt/output lengths and interleaved repetitions.
Include a near-full-window prompt and overflow, not just a favorable short
prompt. Record peak-sample memory and actual cache storage separately.

Accept only with parity and >=10% useful pre-overflow generation-speed benefit;
report overflow performance prominently. Keep uncached full forward available
as the reference/training path. An explicit cache rebuild is part of the declared
algorithm, not a hidden fallback. No batching server or deployment in this step.
