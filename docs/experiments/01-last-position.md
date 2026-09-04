# 01 — Last-position projection

Status: planned; forecast recorded before the run on 2026-09-04.

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
