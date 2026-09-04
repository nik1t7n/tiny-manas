# 11 — Fused or chunked output loss, conditional on memory pressure

Status: source review and protocol recorded; runtime condition not yet assessed
on the final selected architecture. No implementation or run. Date: 2026-09-04.

## Question and forecast

Does retaining full-vocabulary logits prevent a useful training configuration
from fitting, and can we remove that constraint at an acceptable update cost?
This experiment follows vocabulary and architecture selection. A smaller output
vocabulary may remove much of the original motivation before a custom loss is
worth maintaining.

At the current B=8,T=256,V=32,768, one BF16 logits tensor contains 67,108,864
values and occupies 128 MiB. The corresponding FP32 tensor occupies 256 MiB.
Those figures describe one tensor, not the complete loss/backward memory cost.
Weights, saved activations, casts and gradient buffers also occupy memory.
O04 measured about 2.10 GB of sampled live allocation for the ordinary BF16
path, below the experiment's roughly 8.26 GB MPS cap. Existing evidence therefore
shows that the current model fits; it does not establish an output-loss memory
bottleneck. Reassess after the selected architecture and vocabulary are fixed.

Forecast: chunking can save output memory but add launches or recomputation.
On the current short-context model, the added maintenance and latency may buy
unused headroom. Activate the runtime experiment only if measured output-loss
memory blocks an approved configuration, or a measured output-loss bottleneck
gives a concrete reason to expect a useful whole-update improvement. Do not
increase model size or context merely to manufacture that condition.

## Primary-source findings

The [Cut Cross-Entropy paper](https://arxiv.org/html/2411.09009v1) computes the
correct-token logit and a streamed log-sum-exp over the vocabulary, then
recomputes blocks for backward. Its fast backward can filter small gradient
contributions. Distinguish a mathematically full-vocabulary loss from optional
gradient approximations and floating-point reduction differences; “exact” is
not a promise of bit-identical training.

The [official implementation](https://github.com/apple/ml-cross-entropy) lists
Triton and an Ampere-or-newer GPU for its CCE kernels. It also documents a
separate `torch_compile` implementation that it chooses on macOS, with higher
memory use. Installing the package on a Mac would therefore not establish that
we ran the paper's low-memory Triton kernel. Specify the implementation and
device in the report, or fail explicitly. Do not accept automatic backend
selection as evidence of fusion or of the advertised CUDA memory savings.

The repository also exposes variants that disable gradient filtering. If a
later supported-device run uses CCE, pin the source revision and the numerical
variant. Do not transfer its published large-model results to Tiny Manas.

## Candidate if the condition is met

Prefer a bounded real MPS comparison over installing an unrelated GPU stack.
For a chunked full-vocabulary reference, split target positions into chunks;
each position still competes against all vocabulary entries. Sum negative log
likelihoods and divide by the total active target count, including ignored
padding handling. This leaves the training objective unchanged.

A Python loop over chunks followed by one final backward may retain saved
intermediates from all chunks. It does not prove a memory saving. The candidate
must control retained buffers during backward, and tied input/output embeddings
must receive both gradient contributions. Measure allocations across the whole
update rather than reporting the size of a single chunk.

Keep current weights, real batches, dropout masks, optimizer, accumulation,
precision and clipping fixed. First compare loss, named gradients and the
updated weights to the ordinary path, including a real padded batch. Require
FP32 closeness at atol/rtol 1e-4; for the accepted BF16 path also report relative
gradient error and require it below 0.02. A failure stops the cost comparison.

Then use 35 real updates per arm, excluding the first five from warmed timings.
Accept at least 20% less sampled whole-update live allocation with at most 10%
update-latency regression, provided that saving resolves the measured need.
A speed-first candidate instead needs at least 5% faster updates without a
material memory increase. Borderline results remain inconclusive. Record the
extra source code and dependency cost alongside the numerical measurements.

If the condition never arises, close this item as **assessed, not triggered**
and keep ordinary cross-entropy. That decision is neither a failed CCE benchmark
nor a claim that fused losses cannot help larger models.
