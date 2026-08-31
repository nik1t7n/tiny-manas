# Research Log

Forecasts are written before each run. Failed runs and wrong predictions remain in chronological order.

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
- **Status:** planned
- **Question:** can the standalone Transformer learn a bounded real slice of *Manas* without unstable optimization?
- **Hypothesis:** both training and validation cross-entropy improve from the random baseline, but the small validation split produces noisy metrics and limited continuity.
- **Forecast:** generated text reflects names, endings, punctuation, and some multi-token phrases but still repeats and loses narrative state quickly.
- **Falsifier:** training improves while validation is flat from the beginning, gradients become non-finite, or output never reflects the Manas distribution.
- **Main changed variable:** move from a repeated batch to random training windows over 9,000 tokens.
- **Controlled inputs:** source, tokenizer, architecture family, seed, and device.
- **Result:** pending.

## E3 — Full Manas01 base model

- **Date:** 2026-08-31
- **Status:** planned
- **Question:** does the full cleaned epic support a more coherent Tiny Manas generator on the 13M-parameter candidate?
- **Hypothesis:** the wider data distribution and 256-token context improve validation loss and reduce immediate loops compared with the pilot's own before/after behavior.
- **Forecast:** the model remains far from a general assistant but produces recognizable Manas-like paragraphs with better local continuity than the pilot.
- **Falsifier:** immediate severe overfitting, long exact copying as the dominant behavior, no validation improvement, or impractical MPS runtime/memory.
- **Main changed variable:** full data and the preregistered base configuration.
- **Controlled inputs:** source edition, tokenizer, split policy, device, and evaluation protocol.
- **Result:** pending.
