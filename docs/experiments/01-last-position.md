# 01 — Last-position projection

Status: completed and accepted; forecast recorded before the run on 2026-09-04.

## Question and hypothesis

Can generation avoid projecting discarded previous positions without changing
the trained model or training interface? Project only `x[:, -1:, :]` when
explicitly requested, leaving the default full forward unchanged.

Expected: final logits match within `atol=1e-4, rtol=1e-4`; greedy tokens match;
output allocation falls from `B*T*V*4` to `B*V*4` bytes in FP32. Full-model
latency should improve especially at T=256. No claim of T-fold total speedup:
all Transformer computations still run. Reject if parity fails or interleaved
measurements show a consistent slowdown. Exact output-storage reduction can
justify acceptance even where total timing is within noise.

Controls: original checkpoint, tokenizer, FP32/MPS, eval mode, same real
validation tokens, no cache, same crop, temperature/top-k and RNG seed.
Check T=1/64/256 with B=1 and B=8,T=64; 20 real prompts for output parity;
repeat interleaved timings after warmup. No retraining or test-set evaluation.

Sources: [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html),
[MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
Slice before projection rather than after it; synchronize both sides of timing.

## Result

Command: `.venv/bin/python scripts/experiment_last_position.py`.
Raw artifact: `runs/optimization-01-last-position/result.json` (includes timings,
20 real-prompt continuations, environment, checkpoint digest and source diff).
Base commit `4364ac2`; PyTorch 2.13.0, macOS 26.5.1, M5/16 GB, strict MPS.
Checkpoint digest is the frozen baseline digest in `00-baseline-manifest.json`.

| B,T | Full median ms | Last median ms | Speed ratio | Output bytes before → after |
|---|---:|---:|---:|---:|
| 1,1 | 2.870 | 3.034 | 0.946x | 131072 → 131072 |
| 1,64 | 3.325 | 3.028 | 1.098x | 8388608 → 131072 |
| 1,256 | 7.307 | 5.894 | 1.240x | 33554432 → 131072 |
| 8,64 | 10.998 | 10.483 | 1.049x | 67108864 → 1048576 |

All numerical comparisons passed (`atol=rtol=1e-4`). Largest shape-check
absolute difference was 1.72e-5. All logits across 20 real validation prefixes
and 24 greedy steps passed; greedy choices were identical. Actual stochastic
`generate` also matched the original algorithm with seed 77, temperature 0.8,
top-k 40, including crossing the 256-position crop boundary. Full-position
cross-entropy remained unchanged. No checkpoint weights or training code changed.

## Decision and limitations

Accept the explicit `last_position_only` forward option and use it in generation.
Default forward still returns all positions; requesting this option with targets
raises an error rather than silently changing the training objective. No caller
changes are needed in training/evaluation/service warmup.

The expectation was supported at useful context lengths. T=1 has no output
storage reduction and measured 0.16 ms slower; this small microbenchmark does
not establish a universal speed gain. These are synchronized model-forward
latencies, not end-to-end HTTP performance. Output tensor bytes are exact;
they are **not** total model memory or a measured allocator peak. No retraining,
new quality claim, or new memorization claim is warranted for an equivalent
projection. The original accepted checkpoint and original generation audit remain valid.
