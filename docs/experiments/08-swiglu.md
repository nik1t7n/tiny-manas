# 08 — A budget-matched SwiGLU feed-forward network

Status: preregistered, **not run**. Follow normalization selection. Date: 2026-09-04.

## Question and forecast

Can a learned multiplicative gate improve held-out prediction without simply
adding a larger network? The candidate has two input projections: one produces
content, the other passes through SiLU and gates that content. A third projection
returns the result to the residual width. Dropout stays after that output.

The [GLU-variants paper, section 2](https://arxiv.org/html/2002.05202v1#S2)
compensates for three matrices by shrinking hidden width to two thirds. Apply
that budget logic here, not the unrelated training recipe from its T5 experiments.

At model width 384, current GELU uses hidden width 1536. Candidate SwiGLU uses
1024. Both have 1,179,648 matrix weights per block. With our existing bias-enabled
policy, GELU has 1,181,568 total FFN parameters and SwiGLU 1,182,080: a disclosed
512-parameter difference per block, 4,096 for eight blocks. Do not remove all
biases, increase width or change dropout alongside the activation experiment.

Formula: `down(silu(gate(x)) * up(x))`. This is not `GELU -> SiLU` inside the old
two-layer network. Preserve residual-output initialization: the `down` matrix
must receive the same depth-scaled initialization as the old second FFN matrix,
even though its module name changes.

Forecast: possible quality gain, uncertain MPS speed because matrix shapes and
elementwise work change despite matched matrix-weight count. No claim of equal
wall time follows from equal parameter count.

## Comparison and gates

Only FFN form/compensating hidden width changes. Keep all previously selected
settings and identical full training budget/data order. Tensor-match the common
initialization and use the same initialization distribution for new FFN tensors.
Use a fresh candidate, not transplanted GELU weights presented as trained SwiGLU.

First real-batch check: correct shapes/counts, finite loss, gradients reaching
gate/up/down, and the intended scaled output initialization. Then synchronized
whole-update timings and full training. Compare fixed validation targets and
read 20 raw continuations plus the corrected copying audit. Do not use test.

Accept >=0.02 nats validation improvement with <=10% update-latency regression
and no obvious new output collapse. If no quality gain, keep GELU even if a
paper or a much larger model preferred SwiGLU. Report the small bias-budget
difference and single-seed uncertainty with the result.
