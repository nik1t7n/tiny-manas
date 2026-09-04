# 09 — KV cache with explicit prefill and decode

Status: **completed and accepted** on the O02 BF16-trained MHA checkpoint.
Date: 2026-09-04.

## Result and adoption

Run: `runs/optimization-09-kv-cache-20260904/result.json`; source revision
`f00bb9fca455c41e0273d1111d45a135302d41f3`. Checkpoint SHA-256:
`31499eb747c98bcada1c48b12205033cded96573269bc15464ccafd9905e2167`.
No training update or parameter change occurred.

All 20 real validation prompts passed full-logit tolerance and greedy-token
comparison for 32 steps, including the 256-position crop boundary. Worst absolute
logit error was 7.152557373e-6. Independent-request error was 2.026557922e-6;
seed-6036 sampled generations matched for both 32- and 248-token prompts.
Measured logical and unique cache storage both equal **6,291,456 bytes**, FP32.

| Measurement | Uncached | Cached | Speedup |
|---|---:|---:|---:|
| 32 prompt tokens, 64 generated tokens | 0.196746 s | 0.170108 s | 1.1566x |
| 248 prompt tokens, 64 generated tokens | 0.323867 s | 0.318597 s | 1.0165x |
| One new token after 128 past positions | 3.31775 ms | 2.57654 ms | 1.2877x |

Short/long prefill medians were 2.6505/4.9974 ms uncached and 2.5558/5.1365 ms
cached. Timing used interleaved pairs, synchronized MPS and FP32 throughout.
Maximum allocation sampled after timed calls was 113,945,600 bytes; driver
allocation was 1,229,094,912 bytes. Neither is an exact live peak.

The forecast held: decode benefits while the context grows; repeated window
rebuilds erase most of the end-to-end gain near overflow. The preregistered
10% useful pre-overflow threshold passed. Retain that limitation in the README.

Native implementation: `src/manas_gpt/kv_cache.py`; `ManasGPT.generate` enables
it by default and accepts `use_cache=False` for the uncached reference. Training
forward remains unchanged. The cache has no model parameters and owns state per
request. Unsupported training mode, absent learned positions, invalid head
shapes, multi-token decode and overflowing decode all fail explicitly.

Native integration smoke used the same accepted checkpoint and real validation
tokens on MPS: seed-6036 generation matched the uncached path for 32/248-token
prompts and 32 outputs, including overflow; a full native cache owned exactly
6 MiB. The research runner now explicitly requests `use_cache=False` for its
reference, preventing a future cached-versus-cached comparison. This verifies
the changed public method without repeating the full trained-model audit.

The preregistration and preparation history follow; their “not run” statements
refer to the time before the measured result above.

Prepared while O05 trains: `scripts/kv_cache_candidate.py` provides a separate
request-local `CacheSession` and cached generation loop. It reuses the selected
model's actual projections, norms and FFNs; no alternate trained weights or
synthetic input path is introduced. It is not integrated into ordinary generation
and has not run on MPS. Candidate preparation is not a passed equivalence gate.
The initial candidate uses dynamic concatenation, owns separate contiguous K/V
storage, and reports unique storage bytes as well as logical tensor bytes. It
does not claim preallocation/fused decoding benefits that it does not implement.

The installed-version [SDPA documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html)
defines the causal mask as upper-left aligned. The candidate therefore uses
causal attention during prompt prefill and no causal mask for a single new query
over exclusively past/current keys. It rejects multi-token decode appends.

Prepared driver: `scripts/experiment_kv_cache.py`, CLI parsing only. It requires
the selected checkpoint hash and vocabulary, then checks all next-token logits
and greedy choices for 20 real validation prompts: ten of length 32 and ten of
length 248, each continued for 32 tokens. The latter exercise window rebuilding.
It also checks independent request state, exact unique FP32 cache storage at
256 positions, and the actual sampled generation wrapper with fixed seeds.
Complete-generation measurements use 64 output tokens, alternating cached and
uncached order, two warmup pairs and five measured pairs. Prompt prefill and
single-token decode with 128 past positions have separate timings. Memory is
sampled after timed calls, not described as the true allocator peak. No real
inference measurements have been collected while the tokenizer sweep runs.

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
