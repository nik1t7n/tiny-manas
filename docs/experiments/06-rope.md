# 06 — Rotary positions versus learned position embeddings

Status: preregistered, **not run**. Execute after experiment 05 selects its
tokenizer. Date: 2026-09-04.

## Question and forecast

Can position-dependent rotations improve held-out continuation prediction on
this short-context epic, without an unacceptable training-time cost? This is
not an attempt to acquire unlimited context by changing one formula.

The learned table currently has 256 x 384 = 98,304 parameters. Removing it saves
only about 0.37% of the original 26.9M model; that alone is not a compelling
resource win. RoPE adds rotation work to every attention layer. Forecast:
quality may improve through relative-position structure, but speed may regress
on MPS and a small corpus may not show a meaningful benefit.

## One controlled architectural change

Keep the tokenizer selected by experiment 05, context 256, width 384, eight
layers and eight heads, dropout, FFN, normalization, optimizer, precision and
training budget fixed. Compare fresh learned-position and RoPE models. Reuse a
fresh baseline run only when its exact inputs, initialization and protocol match;
never compare a fully trained old model against a briefly fine-tuned replacement.

Remove the additive position table in the candidate. In each head, rotate Q and
K, not V, before the existing causal attention. For head width 48, use adjacent
coordinate pairs and fixed frequencies `10000 ** (-2*j/48)`, j=0..23. At position
p, each pair (a,b) becomes `(a*cos(p*f)-b*sin(p*f), a*sin(p*f)+b*cos(p*f))`.
Compute angles/trigonometry in FP32, then explicitly handle the attention dtype.
Record any cast overhead rather than attributing it to the attention kernel.

This construction follows the rotation/relative-inner-product formulation in
[RoFormer, sections 3.1–3.2](https://arxiv.org/html/2104.09864v5#S3).
The paper is motivation, not evidence that Tiny Manas will improve.

Initialize shared parameter tensors identically in both arms. Merely setting the
same seed is insufficient if removing a module changes random-number consumption.
Preserve old checkpoint loading with an explicit learned-position default; never
reinterpret old trained parameters as a RoPE checkpoint.

## Correctness and acceptance

Before training, use Q/K from real corpus inputs to verify pairwise norm
preservation and invariance of attention scores to a shared position offset,
with documented floating-point tolerances. Then run a real training batch:
finite loss and gradients, every intended parameter receiving its gradient,
correct causal behavior and output shapes. These checks do not imply quality.

Run the same complete training/validation protocol for both arms and inspect 20
raw generations and the corrected word-match audit. Use identical validation
targets and contexts; do not use test for selection. Accept the architectural
change only for at least 0.02 nats lower validation loss (same tokenizer) with
no more than 10% warmed update-latency regression and no obvious output collapse.
A marginal/noisy result is inconclusive, not an automatic promotion. Retain the
learned-position default unless the declared gate is met.

Do not claim extrapolation from the fact that trigonometric values exist beyond
position 255. Longer-context training/evaluation changes the experiment and is
not part of this quality gate.

## Boundary contract for the later KV-cache experiment

Until experiment 09, generation keeps the existing crop-to-last-256 semantics.
RoPE does not by itself make eviction from a multilayer cache equivalent to
recomputing the cropped window: cached deeper-layer states may already contain
information from tokens outside that window. Position-offset invariance does
not remove those contextual dependencies.

Experiment 09 must preserve the declared reference computation at overflow,
for example by explicitly rebuilding the cropped-window cache. A rolling-cache
policy with different historical context is a separate semantic change, not an
inference-equivalent speedup. Compare cached and uncached outputs on both sides
of the context boundary before adoption.
