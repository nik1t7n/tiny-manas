# Research Log

Forecasts are written before each run. Failed runs and wrong predictions remain in chronological order.

## Optimization series — 2026-09-04

The owner authorized sequential implementation, individual experiment reports,
measured promotion and an eventual updated README. The original revision
`4ad408e` and 17 real artifact files were copied and SHA-256 verified before
model edits. See [frozen baseline](experiments/00-baseline.md).

**O01 preregistration:** question, forecast, falsifier and controls are in
[01-last-position.md](experiments/01-last-position.md). Changed variable:
project only the last position when generation explicitly requests it.
Original model/checkpoint/tokenizer/FP32/MPS remain fixed. Evaluate identical
real validation inputs with no test access. Synchronized interleaved warmed
timings, numerical tolerance 1e-4, and greedy continuations gate acceptance.
Run command: `.venv/bin/python scripts/experiment_last_position.py`.

**O01 result:** accepted. At B=1,T=256 full/last median forward times were
7.307/5.894 ms; output tensor 32 MiB/128 KiB. Full logits compared within 1e-4,
20 greedy continuations matched, and seeded stochastic generation matched across
the context boundary. T=1 was slightly slower and receives no savings. See the
individual report for complete measurements and limits. Next: BF16 training probe.

**O02 preregistration:** [BF16 report](experiments/02-bf16.md). Run 100 updates
each of FP32 and autocast BF16 from the same seed with real train windows,
original configuration and 32 fixed validation batches. Only precision changes.
Forecast finite training, validation within 0.1 nats, >=5% timing or >=10%
sampled-memory benefit. This is only a probe; passing requires a later full
paired run before promotion. Command: `.venv/bin/python scripts/experiment_bf16.py`.

**O02 probe result:** 100+100 real updates completed. Median FP32/BF16 step
0.514812/0.352698 seconds; validation 7.135342/7.135239. Sampled allocated memory
2.298/2.251 GB; driver memory slightly increased. Accept proceeding, not BF16
promotion. Full preregistration is in the report: two fresh 3000-step runs,
identical data/schedule, early stopping disabled in both, validation selection
and 20-sample audit. Test untouched; tolerance +0.05 nats and >=5% speed benefit.
Command: `.venv/bin/python scripts/run_bf16_comparison.py`.

**O03 preparation (not executed):** while the real O02 process remains alive,
inspect the installed Inductor Metal backend and preregister the compile probe
in [03-compile.md](experiments/03-compile.md). The prepared runner requires the
eventual accepted checkpoint and precision explicitly. No concurrent GPU run,
production compile switch, or assumed BF16 promotion. Require loss/gradient
parity, >=5% warmed update benefit and cold-cost amortization within 3000 updates.

**O04 preparation (not executed):** inspected installed checkpoint/MPS RNG
handling and recorded a concrete dropout-preservation risk in
[04-checkpointing.md](experiments/04-checkpointing.md). The first runtime check
must include dropout 0.2, gradients and post-backward RNG state; eval-only parity
would not establish correctness. Defer execution until O02/O03 decisions.
Prepared `scripts/experiment_checkpointing.py` for the standard eager path;
only CLI parsing was run while O02 owns the GPU. Correctness failure stops it
before timing, and any accepted compile mode must be included explicitly later.

**O05 preparation (not executed):** found the owner's existing nested 8k/16k/32k
v1 tokenizers. Verified all three tokenizer hashes and the eight original
training shard hashes against their real training manifest. 32k exactly matches
the frozen LM artifact. Reuse this training rather than repeat it. Reconstructed
train/validation text boundaries are exact UTF-8 prefixes (2,924,452 / 164,489
bytes, no replacement characters). [05-tokenizer.md](experiments/05-tokenizer.md)
specifies equal original-text exposure, a fresh 32k control and exact-byte
validation to prevent shifted splits or unfair token-count comparisons.

**O05 execution preregistration:** `scripts/experiment_tokenizers.py` consumes
the verified immutable bundle, builds all variants from the same 32k master
initialization, and trains each for 30 complete text passes. Configuration:
384 wide, eight layers/heads, context 256, dropout .2, BF16 eager training,
FP32 evaluation, AdamW (.9,.95), weight decay .1, clipping 1.0, at most two
8x256 microbatches per update. Loss weights use actual non-padding target counts.
Warmup covers 1/30 of scored bytes; cosine decay spans the remaining bytes.
Fresh 32k/16k/8k run sequentially in separate processes; epoch checkpoints
include optimizer and CPU/MPS RNG state. First verify a real partial window and
one weighted accumulated MPS update. Forecast: 16k is the likely compromise;
8k saves more output memory but has shorter raw-text context. Falsifier: no
>=10% memory/byte-throughput benefit, or validation BPB more than 1% worse than
either fresh 32k or the incumbent, or raw generation deterioration. Exact-byte
incumbent evaluation is a safety floor, not a causal tokenizer comparison.
Command: `.venv/bin/python scripts/experiment_tokenizers.py --output
runs/optimization-05-tokenizers-20260904`. See O05 report for hashes and all
controls. No protected test access and no automatic promotion.

**O06 preparation (not executed):** read RoFormer's rotation construction and
registered [06-rope.md](experiments/06-rope.md). Keep context 256 and compare
fresh matched models after tokenizer selection. Require a measured quality gain,
not merely removal of the small position table. Shared initialization must be
tensor-matched despite module changes. The later KV cache must preserve cropped
window semantics at overflow; relative positions do not erase old contextual
information already embedded in cached deeper-layer states.

**O07–O10 preparation (not executed):** read original RMSNorm, GLU-variant and
GQA methods while O02 trains. Registered exact comparison/acceptance contracts
in their separate reports. RMSNorm must demonstrate a whole-update benefit, not
borrow another framework's speedup. SwiGLU width 1024 matches the old FFN's matrix
weight budget; the 4096 extra bias parameters are disclosed. KV-cache parity
includes overflow and explicit cropped-window rebuilding. GQA uses the paper's
mean-pooling conversion followed by 150 matched adaptation updates in each of
GQA/MHA, avoiding an unnecessary fresh full pretraining run. A failed bounded
adaptation is a negative result, not permission to extend compute indefinitely.

**O02 control complete:** FP32 finished 3000 updates in 1548.042 seconds;
independent validation 4.345732722 reproduces the original result. All 20 token
sequences match the frozen audit. All continuations were read and re-audited
under word-match protocol 2 (maximum 9, mean trigram repetition 0.0421294).
BF16 is now training automatically; no precision promotion yet.

**O02 full result:** both arms completed. BF16 took 1229.405 versus 1548.042
seconds (-20.58%), independent validation 4.345779147 versus 4.345732722.
All 20 BF16 continuations were read and re-audited (maximum normalized match 9;
mean repeated-trigram ratio 0.0409729). Individual repetition regressions remain
visible, especially samples 14/17; no prose-quality improvement is claimed.
Accept training autocast BF16, keep parameters/AdamW/evaluation/inference FP32,
preserve original weights. Explicit selected checkpoint/hash and current settings
are in [accepted state](experiments/accepted-state.json). Next: compile probe.

**O03 result:** rejected. Original compiled path had relative gradient error
1.33376 despite matching loss. Named diagnostics localized the dominant mismatch
to an approximately 8x positional-table gradient. An equivalent batched lookup
research candidate passed the BF16 gradient gate (0.00651084), then completed
35 paired real updates. Median 0.333425 s compiled versus 0.323650 s eager;
no recompilation after warmup, no positive amortization. Preserve all three
attempts and their diagnosis; do not change the production position expression
or enable compile. Next: ordinary-versus-checkpointed dropout-gradient check.

**O04 result:** standard checkpointing failed the real dropout-enabled gradient
check (relative error 0.675827, MPS RNG advanced). Explicit paired MPS RNG contexts
fixed the defect: gradient error ~6e-9, identical loss and preserved random state.
Over 35 real updates per mode, sampled allocation fell 38.46% while median update
latency rose 30.63%. Accept as opt-in, leave it off for our already-fitting model.
Integrated the shared helper/model/config path and reran the one-batch gate;
it passed. Original checkpoint keys/weights and ordinary inference remain intact.
Next: implement the equal-text tokenizer training/evaluation driver over the
already verified immutable inputs, then run the preregistered comparison.

**O05 promotion-gate correction before execution:** the fresh 32k run controls
the tokenizer comparison but could itself underperform our incumbent because
the equal-text sampler is new. Added exact-byte evaluation of the incumbent as
a second 1% quality floor. Also require decoded generation lengths so shorter
8k/16k continuations do not receive a misleading repetition or copying advantage.

**O05 data preparation complete, training not started:** created one read-only
manifest-pinned bundle of the existing three tokenizers and exactly matching
train/validation text. See O05 report for directory and manifest hash. CPU
coverage checks over the actual windows confirm every scored target once and
identical scored byte totals for all vocabularies. The production pinned loader
is unchanged. This work does not compete with the ongoing BF16 GPU run.

**Measurement correction:** source inspection and real saved generations found
asymmetric normalization in the copying audit. Nine of the original 20 samples
were undercounted; normalized longest match is nine words rather than seven.
A real source excerpt also failed the old matcher. Correct the shared matcher,
version the protocol and re-audit saved outputs without new generation. See
[audit correction](experiments/00-audit-correction.md). The already-running O02
controller retains the old imported code; both its outputs must be re-audited
before acceptance. Training remains unchanged and must not be restarted.

**Correction verified:** the saved 20-sample baseline now reports nine normalized
matching words; repeated-trigram mean remains 0.0421294. A bounded real source
excerpt matches all 142 words. Original audits/snapshot untouched; README's two
affected statements now use corrected terminology and evidence. An initial
line-based control exposed that processed Manas is a single line, so the future
tokenizer sampler uses a short shared token-boundary prefix, not the first line.

## E0 — Source, tokenizer, split, and MPS integrity

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** can the pinned real source and tokenizer produce deterministic Manas token splits, and can the changed tensor path execute on Apple MPS without fallback?
- **Hypothesis:** the source and tokenizer hashes will match their pinned values; full encode/decode will round-trip; `Manas01` will yield about 465K epic tokens; batches will have integer shape `(B, T)` and one-token-shifted targets; MPS will be available.
- **Forecast:** all integrity checks pass without modifying the upstream artifacts.
- **Falsifier:** hash mismatch, missing/duplicated epic heading, round-trip failure, split leakage, incorrect target shifting, or unavailable MPS.
- **Main changed variable:** none; this is an infrastructure gate.
- **Controlled inputs:** pinned URLs, hashes, document ID, heading, tokenizer, seed, and config.
- **Result:**
  - pinned Manas archive SHA-1 and tokenizer SHA-256 matched;
  - `Manas01` was found with the expected source metadata;
  - the epic heading occurred exactly once and excluded 30,877 leading characters;
  - full `Manas01` contained 471,600 BPE tokens; the epic portion contained 465,069 tokens and 9,593 distinct IDs;
  - the 10K dataset produced 9,000 train and 1,000 validation tokens;
  - tokenizer round-trip succeeded for the complete document and epic text;
  - PyTorch `2.13.0` reported MPS built and available;
  - a real batch had `X=(4,64)`, shifted `Y=(4,64)`, and logits `(4,64,32768)`;
  - the first forward loss was `10.4317`, close to the uniform-vocabulary value `ln(32768)=10.3972`;
  - backward completed with gradient norm `3.1100` before clipping;
  - changing future tokens produced maximum prefix-logit delta `0.0`, confirming the causal path for the checked prefix;
  - the E1 model contained 4,599,296 trainable parameters and allocated about 101.8 MiB through PyTorch MPS during the smoke.
- **Forecast versus result:** matched. No hash, round-trip, shape, causal, gradient, or MPS blocker was observed.
- **Updated belief:** the real data and minimal tensor path are valid enough to begin the one-batch optimization gate.
- **Next action:** run E1 without changing data, architecture, seed, or dropout.

## E1 — One real batch overfit

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** is the complete embedding, causal attention, FFN, loss, backward, AdamW, checkpoint, and generation path correct?
- **Hypothesis:** a two-block Transformer with dropout disabled can memorize one fixed batch from the real epic.
- **Forecast:** batch cross-entropy falls below `0.05` within 1,200 updates and next-token accuracy approaches 100%.
- **Falsifier:** a persistent loss floor, non-finite loss or gradients, incorrect target alignment, or generation inconsistent with the memorized prefix.
- **Main changed variable:** repeated optimization of one fixed batch.
- **Controlled inputs:** dataset, batch, seed, model config, dropout `0.0`, optimizer, and MPS device.
- **Accepted run:** `overfit-one-batch-20260831T152234Z` at commit `ec5b4a2`.
- **Result:**
  - actual fixed-batch loss fell from `10.4385` to `0.02673`;
  - fixed-batch perplexity fell from about `34,150` to `1.027`;
  - fixed-batch top-1 and top-5 accuracy both reached `100%`;
  - the declared `0.05` loss gate passed at step 100, so training stopped after 25,600 sampled target positions rather than continuing to 1,200 steps;
  - the run took `1.89 s` of measured training time at about `13,548` target positions/s;
  - peak PyTorch MPS allocation was about `140.5 MiB`; peak MPS driver allocation was about `1.06 GiB`;
  - the generation from the actual memorized prefix reproduced a coherent continuation from that batch before eventually entering repetition;
  - validation loss worsened, as expected for intentional memorization of one batch, and is not treated as a quality result.
- **Forecast versus result:** matched. The complete path can memorize real targets and the corrected stop condition works.
- **Updated belief:** embeddings, causal attention, FFN, loss, backward, AdamW, checkpointing, and generation are connected correctly enough to begin random-window training.
- **Next action:** E2, using the 10K dataset with dropout `0.2` and an untouched validation suffix.

### E1 attempt 1 — invalid summary, useful diagnostic

- **Run:** `overfit-one-batch-20260831T152055Z`
- **Status:** invalid as an acceptance artifact.
- **Observed training path:** logged optimization loss fell from `10.4385` at step 1 to `0.0280` at step 100, `0.0017` at step 200, and numerically zero by step 400. The model therefore did memorize its actual fixed training batch.
- **Instrumentation defect:** the periodic `train` evaluator created a second fixed sampler with a different seed, so it evaluated another batch. The summary incorrectly reported final train loss `17.7646` and failed to trigger the `0.05` stop condition.
- **Secondary observation:** continuing 1,100 unnecessary steps after the real gate had passed pushed validation loss upward and produced a repetitive generation ending in `Т Т Т...`.
- **Decision:** make fixed-batch evaluation reuse the exact sampler used by optimization, derive the diagnostic generation prompt from that batch, and rerun E1 unchanged. Do not count attempt 1 as accepted evidence.

## E2 — 10,000-token pilot

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** can the standalone Transformer learn a bounded real slice of *Manas* without unstable optimization?
- **Hypothesis:** both training and validation cross-entropy improve from the random baseline, but the small validation split produces noisy metrics and limited continuity.
- **Forecast:** generated text reflects names, endings, punctuation, and some multi-token phrases but still repeats and loses narrative state quickly.
- **Falsifier:** training improves while validation is flat from the beginning, gradients become non-finite, or output never reflects the Manas distribution.
- **Main changed variable:** move from a repeated batch to random training windows over 9,000 tokens.
- **Controlled inputs:** source, tokenizer, architecture family, seed, and device.
- **Accepted run:** `pilot-10k-20260831T152349Z` at commit `a0007e3`.
- **Result:**
  - model size: 8,095,872 parameters;
  - initial validation loss/perplexity: `10.4294` / about `33,840`;
  - best validation loss: `7.3231` at step 200;
  - independent 50-batch check of `best-model.pt`: loss `7.3369`, perplexity `1,535.96`, top-1 `13.17%`, top-5 `21.13%` over 102,400 sampled target positions;
  - training continued to step 1,000, where final train loss was `0.1235` but validation loss had degraded to `10.4687`;
  - early stopping ended the run after eight evaluations without a new validation best;
  - 2,048,000 target positions were sampled in `105.1 s`, about `19,484` positions/s;
  - peak PyTorch MPS allocation was about `387.4 MiB`; peak driver allocation was about `2.31 GiB`;
  - five inspected best-checkpoint generations contained Manas names, epic syntax, line-like rhythm, and multi-token phrases, but frequently fused endings, reused the same learned passage across unrelated prompts, and lost subject continuity;
  - the final checkpoint produced a more fluent-looking passage than the best validation checkpoint, but its severe train/validation gap makes it an overfit artifact rather than the accepted model.
- **Forecast versus result:** matched. The model learned the slice and the validation set exposed rapid overfitting.
- **Updated belief:** the architecture works, but 9,000 training tokens are far below what an 8.1M-parameter full-vocabulary model needs. Dropout `0.2` delays neither memorization nor validation degradation enough to make the pilot useful as a final model.
- **Next action:** keep the architecture family and tokenizer, move to the full 465K-token epic, and select only by validation loss.

## E3 — Full Manas01 base model

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** does the full cleaned epic support a more coherent Tiny Manas generator on the 13M-parameter candidate?
- **Hypothesis:** the wider data distribution and 256-token context improve validation loss and reduce immediate loops compared with the pilot's own before/after behavior.
- **Forecast:** the model remains far from a general assistant but produces recognizable Manas-like paragraphs with better local continuity than the pilot.
- **Falsifier:** immediate severe overfitting, long exact copying as the dominant behavior, no validation improvement, or impractical MPS runtime/memory.
- **Main changed variable:** full data and the preregistered base configuration.
- **Controlled inputs:** source edition, tokenizer, split policy, device, and evaluation protocol.
- **Accepted run:** `manas01-base-20260831T152803Z`.
- **Result:**
  - the model contained 13,193,216 trainable parameters and used context 256;
  - initial validation loss was `10.4775`, close to the random-vocabulary baseline;
  - the best scheduled validation loss was `4.5378` at step 2,800; the final step was slightly worse at `4.6071`, so `best-model.pt` is the accepted checkpoint;
  - an independent 100-batch validation check measured loss `4.6384`, perplexity `103.38`, top-1 `26.64%`, top-5 `45.24%`, and `0.946` bits per UTF-8 byte;
  - the untouched chronological test suffix measured loss `5.0201`, perplexity `151.42`, top-1 `23.81%`, top-5 `41.09%`, and `1.048` bits per UTF-8 byte;
  - training sampled 12,288,000 target positions in `905.8 s` at about `13,565` positions/s;
  - peak PyTorch MPS allocation was about `468.5 MiB`; peak MPS driver allocation was about `3.36 GiB`;
  - across 20 fixed generations, the longest exact corpus match was six words and the mean repeated-trigram ratio was `4.77%`;
  - outputs consistently adopted epic names, combat actions, dialogue punctuation, and verse-like rhythm; their main defects were entity drift, malformed word endings, repeated names, repeated reporting verbs, and loss of event continuity over longer spans.
- **Forecast versus result:** matched. Full data produced a real Manas-like generator without dominant long-copy behavior, but it did not produce globally coherent narrative.
- **Updated belief:** the 13M base model is a valid working baseline. Its remaining failures plausibly involve both limited context and limited capacity, so those variables must be tested separately.
- **Next action:** E4 changes only context from 256 to 512 while preserving model width, depth, data, tokenizer, optimizer, and the 12.3M-target budget.

## E4 — Context 512

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** does doubling usable context improve held-out prediction and longer-span continuity enough to justify its cost?
- **Hypothesis:** context 512 gives each prediction access to more verse and dialogue history, reducing entity drift and repetition, but quadratic attention lowers throughput.
- **Forecast:** validation loss improves modestly, while tokens/s decreases and MPS memory remains safely below the machine limit.
- **Falsifier:** no repeatable validation improvement, worse raw continuity, unstable MPS execution, or a cost increase disproportionate to quality.
- **Main changed variable:** context length `256 → 512`.
- **Controlled inputs:** model width and depth, tokenizer, data split, dropout, optimizer, seed, 4,096 target positions per update, 3,000-step hard limit, and checkpoint-selection rule.
- **Accepted run:** `manas01-context512-20260831T154720Z`.
- **Result:**
  - the model contained 13,258,752 parameters; the extra 65,536 parameters came only from doubling learned positional embeddings;
  - best scheduled validation loss was `4.6783` at step 2,900, versus `4.5378` for context 256;
  - independent 100-batch validation measured loss `4.7268`, perplexity `112.94`, top-1 `25.67%`, top-5 `44.21%`, and `0.964` bits per UTF-8 byte; every metric was worse than the context-256 baseline;
  - independent test loss was `5.1466`, perplexity `171.85`, top-1 `22.29%`, top-5 `39.55%`, and `1.074` bits per UTF-8 byte, also worse than the baseline;
  - throughput fell from about `13,565` to `11,894` target positions/s, a `12.3%` reduction;
  - peak PyTorch MPS allocation remained similar at about `470.9 MiB`, while the optimized SDPA path avoided materializing a prohibitive attention matrix;
  - 20 generations reduced mean repeated-trigram ratio from `4.77%` to `3.82%`, but the longest exact training match increased from six to seven words and manual reading did not show better entity or event continuity;
  - several samples still entered strong name and reporting-verb loops, so the lower aggregate trigram repetition did not amount to a qualitative win.
- **Forecast versus result:** rejected. Context 512 cost more and predicted held-out text worse; the raw generations did not supply a compensating continuity improvement.
- **Updated belief:** on this corpus and training budget, model capacity is a better next variable than longer context. More history is not useful if the network cannot model that history well.
- **Next action:** restore context 256 and test one measured 26.9M-parameter model.

## E5 — 26.9M model scale

- **Date:** 2026-08-31
- **Status:** completed
- **Question:** does increasing depth and width improve held-out prediction and raw continuity enough to justify slower local training?
- **Hypothesis:** an 8-layer, 384-wide model uses the same 256-token history more effectively than the 6-layer, 256-wide baseline.
- **Forecast:** validation and test loss improve, names and local event sequences remain stable for longer spans, throughput falls substantially, and the run still fits comfortably on the M5 / 16 GB machine.
- **Falsifier:** held-out metrics fail to improve, generation remains equally repetitive, MPS becomes unstable, or the quality gain is too small for the runtime cost.
- **Main changed variable:** capacity `13.19M → 26.88M` parameters.
- **Controlled inputs:** context 256, tokenizer, data split, dropout, optimizer, seed, batch 8, accumulation 2, 4,096 target positions per update, 3,000-step limit, and checkpoint-selection rule.
- **Preflight evidence:** one real MPS forward/backward at batch 8 used about `700.7 MiB` of PyTorch allocation, `2.28 GiB` of driver allocation, and `0.422 s`; the candidate therefore fits without reducing the base micro-batch.
- **Accepted run:** `manas01-27m-20260831T160739Z`.
- **Result:**
  - the model contained 26,877,696 trainable parameters with eight layers, width 384, eight heads, and context 256;
  - best scheduled validation loss was `4.2462` at step 2,900;
  - independent 100-batch validation measured loss `4.3457`, perplexity `77.15`, top-1 `31.55%`, top-5 `49.83%`, and `0.886` bits per UTF-8 byte;
  - independent test loss was `4.7575`, perplexity `116.45`, top-1 `27.74%`, top-5 `45.25%`, and `0.993` bits per UTF-8 byte;
  - relative to the accepted 13M baseline, validation perplexity fell `25.4%` and test perplexity fell `23.1%`;
  - the 3,000-step run sampled 12,288,000 target positions in `1,466.7 s` at about `8,378` positions/s;
  - peak PyTorch MPS allocation was about `959.4 MiB`; peak MPS driver allocation was about `4.33 GiB`;
  - 20 fixed generations had a mean repeated-trigram ratio of `4.21%` and a maximum exact training match of seven words;
  - manual reading showed stronger local action chains, more stable verse-like phrasing, and better dialogue continuations than the 13M baseline, but some samples still looped on names, repeated formulas, malformed words, and lost global event state.
- **Forecast versus result:** matched. Capacity improved every held-out metric and raw local continuity while remaining practical on the target Mac.
- **Updated belief:** 26.9M is the best tested Tiny Manas configuration. The remaining bottleneck is not local hardware memory; it is the quantity and diversity of Manas-only training text and the model's tendency to overfit formulaic passages.
- **Decision:** accept `best-model.pt` from step 2,900 as the final research checkpoint. Do not claim general Kyrgyz ability or long-form narrative coherence.
- **Next action:** package the evidence, write the public article, and expose only this accepted checkpoint through a bounded inference service.
