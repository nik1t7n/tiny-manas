# 03 — Compile the real training path on MPS

Status: **rejected for this MPS workload** after correctness diagnosis and timing.
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

## Execution, failure and bounded diagnosis

All attempts used the accepted BF16-trained checkpoint (SHA-256
`31499eb747c98bcada1c48b12205033cded96573269bc15464ccafd9905e2167`), BF16 training
autocast, original B=8,T=256 and MPS. Each attempt had an external 600-second
process-group wall budget; none reached it. No eager substitution occurred.

1. `runs/optimization-03-compile-20260904T062220Z`: loss difference 0.0001981,
   maximum logit difference 0.125, but relative gradient L2 1.33376. Stopped before
   optimizer timing, as preregistered.
2. `runs/optimization-03-compile-20260904T062329Z`: replaced positional
   concatenation of gradients in the measurement with explicit parameter-name
   alignment and per-parameter diagnostics. Parameter name order also matched,
   so ordering was not the cause. Relative gradient error remained 1.32557.
   The positional embedding gradient norm was 3.53353 compiled versus 0.441809
   eager, approximately 8x (the batch size). Its relative error was 6.99791;
   most other leading differences were around 0.4–0.7%.
3. `runs/optimization-03-compile-20260904T062518Z`: an explicit research-only
   candidate indexes positions with shape `(B,T)` before embedding lookup, rather
   than broadcasting a `(1,T,C)` lookup result over the batch. It uses the same
   position values and trainable weights; no gradients are manually divided or
   overridden. Relative gradient error fell to 0.00651084 and loss difference
   stayed 0.0001981, both inside the BF16 0.02 gate. Argmax agreement was
   99.7070%. This isolates an observed broadcast-path problem on this installed
   backend; it is not a claim to have diagnosed every underlying compiler kernel.

The third run used `--explicit-position-lookup`; the candidate is retained only
in the research runner. Main model code and trained weights are unchanged.
The [PyTorch compiler troubleshooting guide](https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_faq.html)
supports component isolation; the observed numerical results above are local
experimental evidence, not a documented universal PyTorch limitation.

## Timing and decision

With the corrected candidate, all 35 paired dropout-enabled updates per arm
completed. Counters were unchanged between the end of warmup and the final
measurement; recurring compilation did not explain the result.

| Measurement | Eager reference | Compiled candidate |
|---|---:|---:|
| Eval forward/backward, including cold compile | 0.51992 s | 5.94265 s |
| First training update, including training graph compile | 0.51643 s | 5.77832 s |
| Median update, 30 post-warmup pairs | 0.323650 s | 0.333425 s |

Compiled update latency was 3.02% higher, throughput ratio 0.97068x. There is no
positive break-even point because warmed steps save no time. This fails the
predeclared >=5% useful-speed gate. **Do not promote compilation or its candidate
position-lookup rewrite.** No full compiled training run is warranted by this
result. BF16 eager remains accepted.

Limitations: one local host/backend/version, one batch/context, a short warmed
benchmark and one equivalent expression rewrite. Other hardware, versions or
larger workloads may differ. This result does not show kernel fusion is useless
in general; it shows this tested route did not improve Tiny Manas here.
