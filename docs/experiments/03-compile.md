# 03 — Compile the real training path on MPS

Status: preregistered, **not run**. Wait for experiment 02's precision decision.
Date: 2026-09-04.

## Question, rationale and forecast

Can Inductor reduce the overhead and intermediate memory traffic of this actual
27M Transformer on M5, enough to repay compilation within a 3000-update run?
Do not infer Apple performance from CUDA benchmarks. The installed PyTorch
2.13.0 registers a Metal scheduling backend, but its source describes it as
not feature-complete. That is a reason for a bounded real-model probe, not
proof of either success or failure.

Change only execution: eager versus `torch.compile(..., backend="inductor",
fullgraph=True, dynamic=False)`. Keep checkpoint, precision, tokenizer, B=8,
T=256, optimizer and real training windows identical. No generation compilation,
CUDA-only tuning flags, alternate CPU path, or changed attention implementation.
Use the precision accepted in experiment 02; the runner requires it explicitly.

Forecast: some pointwise operations may fuse, but unsupported backward/reduction
operations or compilation cost may outweigh the benefit. Accept only if actual
correctness passes, warmed update latency improves at least 5%, and cold costs
can amortize within 3000 updates. A compile error, nonfinite values, significant
gradient mismatch, or no useful whole-update benefit is a negative result.

## Measurement and correctness

First compare eager/compiled logits, loss and gradients on the same real batch
with dropout disabled by eval mode (backpropagation still enabled). Require
relative global L2 gradient error <=0.001 in FP32 or <=0.02 in BF16, and loss
difference <=0.001 or <=0.02 respectively. Report maximum logit difference and
argmax agreement rather than hiding disagreements behind one loss scalar.
If a gate fails, stop before throughput measurement and diagnose that exact failure.

Then switch both models to normal training with dropout 0.2. This legitimately
requires a different compiled graph; record that cold cost separately. Run 35
interleaved real optimizer updates per arm, same sampled batches, accumulation
2, fixed learning rate 0.00003, seed 1337+iteration before each arm. Exclude the
first five updates from warm statistics. Compiled dropout RNG is not assumed
bitwise identical to eager, so stochastic training checks are finite-loss checks,
not false promises of exact masks or exact trajectories.

Capture graph counters before/after the warmed interval. Fullgraph mode rejects
graph breaks; recurring graph compilation after warmup invalidates the timing
claim until explained. Record synchronized wall times, raw samples, output dtypes,
and allocator samples after backward/update (not exact peaks). Compiler cache
is isolated in the run directory. CPU gradient comparison is diagnostic only;
model forward/backward remain MPS, with fallback disabled.

The runner preserves failure tracebacks. Give the first real compile attempt a
bounded 10-minute wall budget in the supervising tool/process; do not leave an
unbounded compiler job, and never replace a failure with eager while claiming
compiled success. A successful short probe warrants the smallest further quality
or amortization check indicated by its measurements, not automatic promotion.

## Evidence and sources

Prepared runner: `scripts/experiment_compile.py`; no results exist yet.
Run with `--checkpoint` pointing to the accepted checkpoint and `--precision`
set to the decision from experiment 02. Do not run while its paired training is active.

- [PyTorch compile API](https://docs.pytorch.org/docs/2.13/generated/torch.compile.html):
  fullgraph capture errors instead of partial graph breaks; static-shape specialization.
- Installed source inspected before execution:
  `torch/_inductor/codegen/common.py` registers MPS with `MetalScheduling`;
  `torch/_inductor/codegen/mps.py` contains the prototype warning and BF16 type mapping.
- CUDA graph modes are not assumed applicable to this MPS workload.
