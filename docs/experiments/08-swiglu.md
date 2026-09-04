# 08 — A budget-matched SwiGLU feed-forward network

Status: **correctness/cost gate passed; full quality run pending**. Date: 2026-09-04.
The setup below is the preregistered plan; measured results follow it.

Prepared candidate: `scripts/architecture_candidates.py::with_swiglu`. It copies
the current CPU reference and replaces only the FFNs. A separate CPU generator,
seed 1337, initializes the new gate/up/down tensors; shared non-FFN weights stay
identical. The training RNG must be reset after construction in both arms. Input
projection weights use std .02, output weights use .02/sqrt(2L), and biases start
at zero. The candidate rejects widths that cannot preserve the exact matrix
budget rather than rounding without disclosure. No candidate training or MPS
probe has run; source preparation is not an acceptance result.

Prepared execution: `scripts/experiment_architecture_probe.py --change swiglu`
checks the explicit formula on actual post-attention features, the exact matrix
budget, initialization scales and zero biases. The first real update must give
nonzero gradients to all three projections. The shared 35-update comparison
excludes five warmup updates; a passed cost gate is followed by the complete
matched training run, not treated as a quality result. Only CLI parsing has run.

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

Keep the accepted checkpoint's sampler and selection recipe, following O06's
control protocol. Reuse a previous fresh run only with matching initialization
and exposure. If that control differs from the incumbent, require the declared
quality gain against both on identical evaluation windows; a weaker fresh
control must not make a regression look acceptable.

First real-batch check: correct shapes/counts, finite loss, gradients reaching
gate/up/down, and the intended scaled output initialization. Then synchronized
whole-update timings and full training. Compare fixed validation targets and
read 20 raw continuations plus the corrected copying audit. Do not use test.

Accept >=0.02 nats validation improvement with <=10% update-latency regression
and no obvious new output collapse. If no quality gain, keep GELU even if a
paper or a much larger model preferred SwiGLU. Report the small bias-budget
difference and single-seed uncertainty with the result.

## Numerical/cost result and full-run gate

Real MPS probe:

```text
.venv/bin/python scripts/experiment_architecture_probe.py --vocabulary 32768 --change swiglu --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --output runs/optimization-08-swiglu-probe-20260904
```

Commit: `0dd9357b89a6781343658f138a133343c668c94c`. Candidate initialization
uses the same pinned fresh 32k reference as O06/O07, since neither earlier
architecture change was promoted. The explicit SwiGLU formula matched actual
outputs exactly. Hidden width was 1024, with 1,179,648 matrix weights per block
and 4,096 extra biases across the model. Observed initialization standard
deviations: gate .01998166, up .01997193, down .00500002; all biases zero.
First-update projection gradient norms were .0207865 / .0211883 / .0845496 for
gate/up/down; all other trained tensors also received gradients.

Median whole-update time: GELU **0.307848 s**, SwiGLU **0.319535 s**, a **3.7962%**
slowdown within the 10% limit. Maximum sampled allocation rose from
2,253,916,672 to 2,452,677,120 bytes; driver allocation from 3,594,305,536 to
3,611,082,752. Equal matrix-parameter budgets do not imply equal activation
memory. This gate permits full training; it does not yet justify promotion.

The control's first real training batch exactly reproduced the accepted O02
BF16 run: loss **10.47943115234375**, pre-clipping gradient norm
**6.802753925323486**. Source/config tracing also matches initialization, private
sampler seeds, 3,000-update budget, warmup/cosine schedule, and validation every
100 updates. Only the probe's constant learning rate differs, intentionally;
the full runner restores the original schedule. This supports reuse of the
already completed O02 BF16 quality control without another identical full run.

Full run uses the shared resumable driver, original random windows, and the
same selected tokenizer. Independent acceptance evaluation uses 100 original
validation batches, seed 1837, FP32: it must improve on 4.3457791471 by at least
0.02 nats (at most **4.3257791471**) and show no obvious new collapse in the
20-output audit. Test remains unopened. First retain one 100-update segment,
inspect it, then resume the same run to all 3,000 updates.
