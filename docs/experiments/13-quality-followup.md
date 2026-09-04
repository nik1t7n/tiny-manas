# 13 — Budgeted RoPE and RMSNorm quality follow-up

Date: 2026-09-04. Status: preregistered; not a production change.

## Why reopen these experiments?

O06 and O07 passed numerical checks but stopped at their original latency
ceilings before quality training. That established a cost on this MPS
implementation, not that either architecture was worse. The owner explicitly
accepted approximately 10–15% slower updates as worth investigating and requested
staged quality runs rather than automatically funding two complete runs.

Forecast: RMSNorm will probably track LayerNorm closely; RoPE may improve relative
position learning, but context 256 and the small single-source corpus limit the
opportunity. Consistently lower held-out loss falsifies the expectation of no
useful benefit. Falling training loss by itself does not count as success.

## Matched comparison

Use the accepted 32k tokenizer, original split, context 256, width 384, eight
layers/heads, batch 8, accumulation 2, dropout .2, eager BF16 training and FP32
evaluation. Each candidate changes one component only. Shared weights come from
the fresh `runs/optimization-06-reference-20260904/initial-model.pt`, SHA-256
`89ff070be491d496dbdbd708e1073eacc258f7214fb111e4c3517b57a1c54457`,
not a trained checkpoint.

Keep seed 1337, identical training/evaluation windows, and the original
3,000-update warmup/cosine schedule. A pilot is a prefix of that run; shortening
its schedule would invalidate continuation. Checkpoints retain optimizer,
CPU/MPS random states, sampler progress and source hashes. Resume, do not restart.

Reuse the O02 BF16 control curve only after a fresh 100-update classic control
calibration. Require absolute differences <= .005 at its six recorded training
loss points and <= .003 for train/validation evaluation. These tolerances allow
small MPS numerical variation, not a different recipe. If calibration fails,
stop and diagnose. This single-seed comparison does not measure seed variance.

The model-source diff since O06/O07 only adds inference caching; training forward
is unchanged. Refresh the existing real-data numerical/gradient probes against
current hashes nonetheless. Preserve their latency verdict. `--quality-followup`
explicitly replaces the cost-only veto, not correctness or provenance checks.
Run sequentially on MPS with the existing .65 per-process memory fraction.

## Stages fixed before candidate quality is observed

Evaluate every 100 updates using the same 30 batches as the historical control.
Do not prune for quality before 600 updates. At 600, 900, 1,200 and 1,500 updates,
delta means candidate validation loss minus control loss at the same step.
Negative is better. Compare the mean delta over the latest three checkpoints
with the previous three.

- Stop a clearly worse arm when the recent mean delta exceeds +.05, all three
  deltas exceed +.03, and the relative gap improved by less than .01.
- From 900 updates, stop for insufficient evidence when the recent mean exceeds
  -.02 and the relative gap improved by less than .01. This is a budget decision,
  not proof that the arm could never catch up.
- Otherwise allow the next stage. At the 1,500-update pilot ceiling, continue to
  3,000 only if the mean delta is <= -.02 and all three recent deltas are negative.
- Immediately stop nonfinite loss/gradients or an actual resource failure.

The .02 loss threshold is a practical effect floor (about 2% perplexity), not a
statistical significance level. Overlapping-window batches and adjacent
checkpoints are not independent experimental replications. Report the full curve.

This borrows resource allocation from
[successive halving / Hyperband, section 3.1](https://jmlr.org/papers/volume18/16-558/16-558.pdf),
but is not Hyperband: two prespecified architectural candidates and a grace
period replace a hyperparameter search. Intermediate loss cannot guarantee the
ranking at convergence; a budget-pruned run remains inconclusive about that.

## Promotion is separate

An early lead buys more training, not deployment. After 3,000 updates, select the
best scheduled-validation checkpoint and evaluate the existing independent
100-batch validation set. Require loss <= 4.345779147148132 - .02 against the
incumbent, no material generation-audit regression, and inspect all 20 fixed raw
continuations. Use explicit uncached generation for architecture research: the
current production KV cache supports learned positions only. A winning RoPE
model needs a real compatible serving path before promotion. Do not select using
the test set. Marginal or contradictory evidence does not justify deployment.

## Execution record

Pending. Record raw paths, individual checkpoints, cost and decisions after the
registered stages run. Production remains the accepted BF16 checkpoint.
