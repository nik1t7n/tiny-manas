# 07 — RMSNorm versus LayerNorm

Latest status: **keep LayerNorm after a 900-update quality pilot**. Date:
2026-09-04. The [follow-up](13-quality-followup.md) lifted the cost-only veto but
found no material gain: mean validation delta +.00331 over the last three checks.
Stopping saved 2,100 updates. Final full-budget RMSNorm quality remains unknown.
The original setup and cost-gate results below are retained as historical evidence.

Prepared candidate: `scripts/architecture_candidates.py::with_rmsnorm`. It
copies the current CPU reference and replaces only the 17 normalization sites,
preserving shared tensors. `FP32RMSNorm` uses the installed native `nn.RMSNorm`
with explicit epsilon 1e-5 and FP32 input, then casts the output to the incoming
dtype. This avoids assuming a particular dtype-dependent default epsilon or
internal accumulation behavior. The installed wrapper calls `F.rms_norm`;
native MPS performance and numerical correctness remain unverified until the
queued real probe. No manual-formula or CPU fallback will hide an unsupported
operation. The [PyTorch API documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html)
confirms the scale parameter and epsilon interface; it does not prove a speedup.

Prepared execution: `scripts/experiment_architecture_probe.py --change rmsnorm`
checks native output, input gradients and scale gradients against the explicit
formula on actual first-block features (FP32, atol/rtol 1e-4). A neighboring
token's real features supply the upstream derivative, avoiding a derivative
direction that is trivial under normalization. Then both fresh arms receive
35 identical updates, with five excluded from timing. Full quality training
uses the shared resumable driver only after that gate passes. CLI parsing is
the only verification performed so far; the MPS checks remain queued.

## Question and forecast

Is centering each token's features useful enough here to justify its cost?
RMSNorm retains scale control but omits mean subtraction. The original
[RMSNorm paper](https://arxiv.org/html/1910.07467v1#S4) motivates this replacement;
its timing results on other architectures are not an MPS performance forecast.

Change all 17 normalization sites: two per block plus the final normalization.
Keep pre-norm placement and residual paths. Use learned scale initialized to one,
no learned offset, and an explicit epsilon 1e-5. The candidate computes
`x * rsqrt(mean(x*x) + 1e-5) * scale` over features, not tokens or batch items.
Use stable FP32 reduction and document the output dtype. Do not let a
dtype-dependent default epsilon become a second experimental variable.

Forecast: little whole-model memory change; a possible small latency gain, but
several unfused operations could be slower than an existing LayerNorm kernel.
Removal of 17 x 384 = 6,528 offsets is not a material parameter-memory saving.

## Controlled comparison and gates

Keep the currently accepted tokenizer, positions, FFN, heads, context, dropout,
precision, execution mode and training protocol unchanged. Train the candidate
from scratch with tensor-identical shared initialization and the same data order.
Reuse the accepted control's evidence only if those exact conditions match.
Do not replace normalization inside trained weights and call its immediate loss
a fair training comparison.

Follow the accepted checkpoint's training recipe as specified in O06's control
protocol: the temporary tokenizer-comparison sampler does not override an
incumbent trained with random windows. Preserve a hashed fresh initialization
for each trained architecture. If a fresh control differs from the incumbent,
apply the quality limits below against both on identical evaluation windows.

First compare the implementation's outputs and input/scale gradients with the
explicit formula on real residual activations. Then check a real training batch
for finite loss and gradients. Measure synchronized warmed whole updates, not
only the tiny normalization operation. Preserve failure details and actual dtypes.

Run the full matched quality protocol and inspect 20 raw continuations plus the
version-2 copying audit. Accept either (a) >=5% update-speed improvement with
validation loss no worse than +0.02 nats, or (b) >=0.02 nats validation improvement
with <=5% latency regression. Otherwise retain LayerNorm. Report single-seed
selection limits; a close result is inconclusive, not evidence of superiority.

## Result and decision

Command:

```text
.venv/bin/python scripts/experiment_architecture_probe.py --vocabulary 32768 --change rmsnorm --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --output runs/optimization-07-rmsnorm-probe-20260904
```

Commit: `291ff3a04ab1ea4cbf404895b4072a64dcb5bf73`. The fresh reference is the
same hash-pinned 32k initialization used in O06, since RoPE was not promoted:
`89ff070be491d496dbdbd708e1073eacc258f7214fb111e4c3517b57a1c54457`.

On real first-block features, native RMSNorm exactly matched the explicit FP32
formula for outputs, input gradients and scale gradients (maximum differences
all zero). Both input/output were FP32, epsilon 1e-5. The first training update
had finite pre-clipping gradient norm 6.79266, and all trained parameters received
gradients. Removing the 17 offset vectors saved the expected 6,528 parameters.

| 35-update comparison, first 5 excluded | LayerNorm | RMSNorm |
| --- | ---: | ---: |
| Median update seconds | 0.308569 | 0.341601 |
| First-half median | 0.308223 | 0.341854 |
| Second-half median | 0.308593 | 0.341512 |
| Maximum sampled allocated bytes | 2,253,916,672 | 2,241,037,824 |
| Maximum sampled driver bytes | 3,560,751,104 | 3,560,669,184 |

Whole updates were **10.7050% slower**, well outside even the quality-improvement
branch's 5% slowdown ceiling. This is stable across both measured halves and
does not justify a full quality run. **Retain LayerNorm.** The simpler formula
did not produce a speed benefit in this installed native MPS implementation.
No claim about converged quality follows from the short probe; no test split
or additional generation audit was used after the cost gate failed.

Artifact: `runs/optimization-07-rmsnorm-probe-20260904/result.json`. A different
fused kernel/backend is a separate future implementation experiment, not a
reason to override the observed outcome or silently change the current path.
