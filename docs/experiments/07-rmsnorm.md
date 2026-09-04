# 07 — RMSNorm versus LayerNorm

Status: preregistered, **not run**. Follow the RoPE decision. Date: 2026-09-04.

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

First compare the implementation's outputs and input/scale gradients with the
explicit formula on real residual activations. Then check a real training batch
for finite loss and gradients. Measure synchronized warmed whole updates, not
only the tiny normalization operation. Preserve failure details and actual dtypes.

Run the full matched quality protocol and inspect 20 raw continuations plus the
version-2 copying audit. Accept either (a) >=5% update-speed improvement with
validation loss no worse than +0.02 nats, or (b) >=0.02 nats validation improvement
with <=5% latency regression. Otherwise retain LayerNorm. Report single-seed
selection limits; a close result is inconclusive, not evidence of superiority.
