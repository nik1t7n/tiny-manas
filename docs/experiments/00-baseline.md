# 00 — Frozen pre-optimization baseline

Date: 2026-09-04. Source revision: `4ad408ecb327832ea855f392d35ed123638cbd87`.

## What is preserved

The exact original tracked checkout (including README, architecture explanation,
configs, implementation, serving path and research logs) is archived in
`runs/frozen-baseline-20260904/source.tar`. Separate read-only copies preserve
the original inference checkpoint, complete accepted 27M training run (including
optimizer-bearing checkpoints), processed full corpus splits and tokenizer.
`00-baseline-manifest.json` records hashes verified against the copies.
This is a local research snapshot, not a remote backup. The website is not changed.

## Architecture and behavior

26,877,696 parameters; decoder-only pre-LayerNorm; vocabulary 32,768; context 256;
8 layers, 8 MHA heads, width 384, head width 48. Learned token and absolute
position embeddings are added. Dropout 0.2 applies at embeddings, attention
weights, attention output and FFN output. SDPA supplies causal attention.
FFN is Linear(384,1536), GELU, Linear(1536,384). Two residual additions per
block, final LayerNorm, bias-free LM head tied to token embeddings.
The original forward materializes `(B,T,32768)` logits even during generation.
Generation recomputes the cropped context, uses temperature and top-k, and has
no KV cache. Training uses all shifted next-token targets.

Initialization: normal standard deviation 0.02 for embeddings/linear weights;
zero biases; residual projection weights use `0.02/sqrt(2*8)`.
AdamW betas (0.9,0.95), weight decay 0.1 on matrix parameters, no decay on 1D
parameters, gradient clipping 1.0. Batch 8, accumulation 2, context 256 = 4096
target positions per update. Warmup 100 steps; cosine learning rate 0.0003 to
0.00003; at most 3000 steps; seed 1337; FP32; strict Apple MPS.

## Original evidence, not newly measured results

The original accepted run is `manas01-27m-20260831T160739Z`, best step 2900.
README reports validation loss 4.3457 and test loss 4.7575 in independent
evaluation; scheduled best validation loss 4.2462 is a different sample set.
Training took 1466.7 seconds. Original 20-sample audit reported maximum exact
training match seven words. These historical metrics must not be relabeled
as new baseline benchmarks. Test data is frozen and excluded from selection.

## Restoration

Inspect the source through tag `tiny-manas-pre-optimization-20260904` or extract
the archive into a **new directory**, not over current work. Restore selected
artifacts from the snapshot only after verifying the manifest hashes. Never
train into this snapshot or replace its files.
