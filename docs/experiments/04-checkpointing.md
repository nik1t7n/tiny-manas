# 04 — Activation checkpointing without changing the training computation

Status: preregistered, **not run**. Execute after the precision/compile decisions.
Date: 2026-09-04.

## Question and forecast

Does recomputing Transformer-block intermediates save useful memory at the
current B=8,T=256, without corrupting dropout gradients or adding excessive cost?
The model already fits, so reduced memory alone is not an argument to make a
slower mode the default. The potential outcome is a measured opt-in memory mode.

Checkpoint one complete Transformer block at a time with the documented
non-reentrant implementation, `use_reentrant=False`, preserving RNG state.
Do not checkpoint the LM head, change batch/context, quantize weights, or alter
dropout to claim a memory gain. All those would change the question.

At FP32, one `(8,256,384)` residual tensor contains 786,432 values, or 3 MiB.
Block inputs still have to survive; discarded internal tensors can be
recomputed. This does not remove the 256 MiB `(8,256,32768)` logits or AdamW
state. Forecast: block activation memory decreases, but whole-process savings
are smaller; step time increases. BF16 changes individual tensor sizes, not
this basic tradeoff.

## Correctness before timing: a concrete MPS risk

The installed PyTorch 2.13.0 non-reentrant implementation infers `mps` from its
tensor arguments, but stashes device RNG only if the device module exposes a
truthy `_initialized`. The inspected `torch.mps` module does not define this
attribute. Its public `get_rng_state` and `set_rng_state` exist, while the
checkpoint helper's CUDA-style device context is also not a given on MPS.

This source inspection is a **risk**, not a completed runtime finding. First
compare the standard checkpoint path with normal execution on a real training
batch, with dropout **0.2**, identical starting weights and restored starting
CPU/MPS random states. Compare loss, parameter gradients, one optimizer update
and the post-backward RNG state. Comparing eval-only outputs would miss the
failure. Require relative global gradient L2 <=1e-4 and corresponding loss
agreement at the selected precision; loosening this gate to hide different
dropout masks is not allowed.

If standard MPS RNG preservation fails, retain the negative result and inspect
the exact mismatch. A possible explicit fix is a paired forward/recompute
context that captures/restores the public MPS RNG state and restores the outer
state on exit; it must pass the same check before measurement. Do not monkeypatch
PyTorch internals, set a fake `_initialized` flag, disable dropout, or silently
switch device. Custom checkpoint contexts have additional restrictions under
`torch.compile`, so compatibility with the selected compiled path must be
checked rather than assumed.

## Measurement and decision

Use the same accepted configuration/precision/execution mode and same real
windows. Measure the ordinary and checkpointed variants with synchronized
timing after warmup. Report sampled current/driver memory during forward,
backward and optimizer update; do not label sparse samples as exact peaks.
Include the same-batch gradient comparison evidence, not merely a decreasing loss.

Predeclared utility gate: at least 15% reduction in maximum sampled allocated
memory with no more than 50% increase in warmed update latency. If it passes,
retain as an explicit memory-saving option, documenting why the speed-first
default remains unchanged on this already-fitting model. If it does not, do
not introduce a production switch merely because the mechanism is fashionable.
Larger batch or context is a later, separately controlled experiment.

## Prepared runner

`scripts/experiment_checkpointing.py --checkpoint PATH --precision fp32|bf16`
first compares one real dropout-enabled training batch, including gradients,
the first AdamW update and CPU/MPS random states. Parameter-update maximum error
must also be <=1e-6. A failure is saved and stops the run before timing. If it
passes, each mode runs 35 identical two-microbatch updates, excluding five warmup
updates. Models are loaded/released sequentially so one mode's live model and
optimizer do not inflate the other's allocator measurement.

The prepared runner currently measures eager execution with standard PyTorch
RNG preservation only. Its CLI was checked; no GPU checkpointing experiment has
run. If experiment 03 accepts compilation, extend this runner to the actual
accepted execution mode before starting, rather than claiming compiled coverage
from an eager check. If standard MPS RNG preservation fails, record that real
failure before implementing and measuring an explicit fix.

## Source

[PyTorch checkpoint documentation](https://docs.pytorch.org/docs/2.13/checkpoint.html)
recommends non-reentrant checkpointing and documents RNG preservation, device
restrictions and early-stop recomputation. Local source checked:
`torch/utils/checkpoint.py` (`_infer_device_type`, `get_device_states`,
`_checkpoint_without_reentrant_generator`) and `torch/mps/__init__.py`.
Recheck the runtime behavior on this installation before claiming support.
