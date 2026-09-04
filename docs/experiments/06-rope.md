# 06 — Rotary positions versus learned position embeddings

Status: **correctness passed; rejected by the preregistered cost gate**. Keep
learned position embeddings. Date: 2026-09-04.

Preparation while O05 runs: `scripts/rotary_candidate.py` contains the isolated
candidate, and `scripts/experiment_architecture_probe.py --change rope` contains its real-data numerical
and cost gate. Neither changes the accepted model or loads into the ordinary
checkpoint path. Only command-line parsing has been checked; do not interpret
prepared code as a successful RoPE experiment. No concurrent GPU probe is allowed.

After vocabulary selection, the probe checks FP32 Q/K norms and shared-offset
score invariance (offset 137), causal isolation using two real corpus suffixes,
the exact 98,304-parameter difference, and gradients for every trained tensor.
Then both fresh arms receive 35 identical real updates with dropout .2 and BF16;
exclude the first five from median latency. Shared initialization is copied
tensor-for-tensor. Both probe arms use constant learning rate .0003; this is
a correctness/cost measurement, not the full warmup/cosine quality experiment.
Full training remains necessary after a successful cost gate. A clear >10%
latency regression fails the existing acceptance ceiling before spending a
complete training run; borderline timings require a bounded paired recheck,
not automatic acceptance. Rotation tables are fixed FP32 nonpersistent buffers;
rotation arithmetic uses FP32 and casts results back to the incoming Q/K dtype.

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

### Select the control protocol before the architectural run

O05 uses a special equal-byte sampler to compare tokenizers. It does not
automatically replace the training recipe of the accepted checkpoint. If O05
retains the original 32k incumbent, O06 must retain its original 3,000-update
random-window recipe, including sampler seeds and validation-selection cadence.
The completed O02 BF16 run is a reusable control only after verifying identical
shared initialization, data order, precision, schedule and evaluation windows.
If O05 promotes a smaller-vocabulary checkpoint, follow its 30-epoch equal-byte
recipe and use that completed vocabulary arm as the matched control instead.
Do not compare training recipes and attribute the difference to RoPE.

In either case, evaluate the incumbent on the candidate's final evaluation
windows and retain it as a quality floor. A fresh control that underperforms the
incumbent cannot lower the bar for promotion. Require the stated quality gain
against both whenever the fresh control and incumbent differ. Save the exact
candidate initialization with its configuration and hash before full training;
later architecture comparisons can reuse those tensors without relying on a
seed that may consume a different random-number sequence after module changes.

The prepared `scripts/experiment_architecture.py` supports these two explicit
recipes. It bootstraps a classic reference initialization after O05 selection,
then requires that exact file/hash in the numerical probe and full run. The
full-run gate checks the candidate, reference initialization, configuration and
source hashes against the passed probe. It saves the candidate's initial state
and checkpoints at each epoch or 100-update boundary. Original random-window
resumption reconstructs private sampler RNG states by replaying their index
draws; it does not rerun completed model updates. It preserves the original
initial 30 validation/train-evaluation batches and subsequent selection cadence.
Early stopping is disabled to match the full 3,000-update control budget.

Only CLI parsing has run so far. After the real probe passes, first run one
segment using `--stop-after-segment 1`, inspect its real checkpoint/metrics, and
resume the same run without the pause flag. That verifies the training path
without throwing away its first updates. Final output/audit and promotion remain
pending all 30 segments. No bootstrap, MPS probe or architectural training has
started while O05 owns the GPU.

The shared probe also supports `--change rmsnorm` and `--change swiglu` for
O07/O08. Both probe and full run require an artifact explicitly marked
`fresh_initialization`; a trained checkpoint cannot be passed off as the fresh
control. Each change retains its own numerical checks and latency ceiling.

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

## Result and decision

Command:

```text
.venv/bin/python scripts/experiment_architecture_probe.py --vocabulary 32768 --change rope --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --output runs/optimization-06-rope-probe-20260904
```

The reference initialization hash is
`89ff070be491d496dbdbd708e1073eacc258f7214fb111e4c3517b57a1c54457`.
The MPS probe used commit `70b271ac54a9000ade4475610242965406ea5997`.

Correctness passed on real corpus inputs. Q/K norm error was at most 4.7684e-7;
a shared position offset of 137 changed attention scores by at most 2.6941e-5;
changing the real suffix caused exactly zero change to preceding logits. Every
trained tensor received a gradient. Removing the learned position table saved
98,304 parameters; sampled allocation was essentially unchanged (2.254 GB
control, 2.252 GB RoPE).

| 35-update probe (first 5 excluded) | Control | RoPE |
| --- | ---: | ---: |
| Median update seconds | 0.306037 | 0.343599 |
| Mean update seconds | 0.315128 | 0.351246 |
| First-half median | 0.303459 | 0.341756 |
| Second-half median | 0.306400 | 0.345257 |

RoPE is **12.2735% slower**, beyond the 10% ceiling. The result is not explained
by one outlier: both halves show the same direction and similar magnitude.
Therefore stop before the full 3,000-update quality run and **do not promote
RoPE** on this MPS workload. The slightly lower final probe loss is neither a
validation result nor evidence of quality; these short arms have different
architectures and exist only to test correctness/cost. Test and generation
audits were not used after the preregistered falsifier fired.

Artifact: `runs/optimization-06-rope-probe-20260904/result.json`. This result
does not claim that RoPE is generally worse, or that it could not help longer
contexts. It says that this FP32-rotation implementation misses Tiny Manas's
declared resource boundary at context 256 on the actual M5 host.
