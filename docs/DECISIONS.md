# Decisions

This is a chronological record. Later entries supersede earlier selections;
the current artifact is identified in [accepted-state](experiments/accepted-state.json).

## D001 — Keep Tiny Manas standalone

- **Decision:** evaluate Tiny Manas through its own correctness, validation, generation, memorization, and scaling evidence. Do not compare it with the earlier bigram project.
- **Why:** the owner wants a self-contained Transformer experiment rather than another model-comparison project.
- **Rejected alternative:** preserve the bigram as the headline baseline.
- **Trade-off:** the project loses a simple historical comparison but gains a tighter focus on making one model work well.
- **Reconsider when:** only if the owner explicitly asks for a later retrospective comparison.

## D002 — Start with one verified edition of Manas

- **Decision:** begin with `Manas01`, performed by Sayakbai Karalaev, and exclude its leading scholarly matter at a pinned heading.
- **Why:** its provenance, extraction, tokenizer round-trip, and checksum were already verified locally; the complete epic contains about 465K BPE tokens.
- **Rejected alternative:** mix general Kyrgyz, Russian, news, Wikipedia, or synthetic text.
- **Trade-off:** the model can specialize in one edition and cannot be presented as a general Kyrgyz model.
- **Reconsider when:** the base model is understood and a separately licensed Manas-only expansion is audited.

## D003 — Use a decoder-only pre-LayerNorm Transformer

- **Decision:** learned token and positional embeddings, causal multi-head self-attention, pre-LayerNorm blocks, GELU FFN, residual paths, final LayerNorm, and tied LM-head weights.
- **Why:** this is the architecture studied in the learning project and is compact enough to inspect end to end.
- **Rejected alternatives:** encoder-decoder, cross-attention, RoPE, RMSNorm, SwiGLU, grouped-query attention, and pretrained initialization.
- **Trade-off:** the model is intentionally less modern than current frontier architectures, but each component has a clear role and the implementation remains teachable.
- **Reconsider when:** the base run is stable and one replacement component becomes a controlled experiment.

## D004 — Treat dropout 0.2 as a hypothesis

- **Decision:** correctness overfit uses dropout `0.0`; pilot and base candidates start with `0.2`.
- **Why:** regularization must not obstruct the wiring check, while the owner selected `0.2` for real training.
- **Rejected alternatives:** apply dropout during one-batch overfit or declare `0.2` optimal without evidence.
- **Trade-off:** `0.2` may slow learning on the small corpus.
- **Reconsider when:** train and validation behavior shows underfitting or a controlled `0.0` versus `0.2` run is warranted.

## D005 — Require real MPS execution

- **Decision:** fail if `torch.backends.mps.is_available()` is false and never enable implicit MPS-to-CPU fallback.
- **Why:** the experiment is specifically intended to be locally reproducible on the owner's Apple Silicon machine.
- **Rejected alternative:** silently continue on CPU when an operation is unsupported.
- **Trade-off:** an unsupported operation blocks the run until implementation changes explicitly.
- **Reconsider when:** a named scalar diagnostic is intentionally defined as CPU-only.

## D006 — Keep the full frozen tokenizer vocabulary

- **Decision:** use all 32,768 tokenizer IDs in the pilot and full model, with tied token-embedding and LM-head weights.
- **Why:** one stable vocabulary lets any tokenizer output move through training, checkpoint loading, prompts, and later inference without a split-specific remapping protocol.
- **Rejected alternative:** compact the vocabulary to IDs observed in a particular training slice.
- **Trade-off:** many vocabulary rows receive little or no positive training evidence and the embedding matrix dominates the smallest model's parameter count.
- **Reconsider when:** a later controlled experiment explicitly studies vocabulary size or a Manas-specific tokenizer.

## D007 — Bound the full run by tokens, not an inherited step count

- **Decision:** the base run uses micro-batch 8, two accumulated batches, context 256, and at most 3,000 optimizer steps: 4,096 target positions per update and about 12.3M sampled positions at the hard limit.
- **Why:** the cleaned train split is about 418K tokens. The earlier draft of 12,000 steps would expose the model to roughly 196M sampled positions, far beyond what the tiny corpus justifies and likely to waste time after overfitting.
- **Rejected alternative:** keep the original 12,000-step placeholder because it appeared in the planning config.
- **Trade-off:** random windows repeat, so sampled positions are not distinct corpus tokens and “epochs” are only an intuition.
- **Reconsider when:** the validation curve is still improving at the hard limit or measured MPS throughput/memory requires a different micro-batch.

## D008 — Accept the best base checkpoint, not the final step

- **Decision:** `manas01-base-20260831T152803Z/best-model.pt` at step 2,800 is the 256-context baseline.
- **Why:** its scheduled validation loss was `4.5378`, while the final step regressed to `4.6071`; independent validation and test evaluation confirmed useful held-out prediction.
- **Rejected alternative:** use the final checkpoint because it completed the maximum step count.
- **Trade-off:** the accepted state is selected on the validation suffix and must be reported with the untouched test suffix separately.
- **Reconsider when:** only if a deterministic full-split evaluator changes checkpoint ordering.

## D009 — Compare context at equal target budget

- **Decision:** E4 uses context 512, micro-batch 4, accumulation 2, and 3,000 steps, preserving 4,096 target positions per optimizer update and the base model's total sampled-target budget.
- **Why:** changing batch size prevents the longer context from quietly doubling the amount of training signal per step.
- **Rejected alternative:** keep batch 8 and therefore change both context and training-token budget.
- **Trade-off:** the number of independent windows per update falls from 16 to 8, and longer attention may reduce throughput.
- **Reconsider when:** MPS memory or measured gradient behavior makes micro-batch 4 impractical.

## D010 — Reject context 512 for the final candidate

- **Decision:** retain context 256 after E4.
- **Why:** context 512 worsened independent validation loss from `4.6384` to `4.7268`, worsened test loss from `5.0201` to `5.1466`, and reduced throughput by `12.3%` without a clear raw-continuity gain.
- **Rejected alternative:** keep 512 because a longer context sounds categorically better.
- **Trade-off:** the final model cannot condition on more than 256 tokens, but spends compute on capacity that the measured experiment suggests is more useful.
- **Reconsider when:** more training data, a different positional scheme, or a larger model makes long-context evidence positive.

## D011 — Scale to the measured 8-by-384 candidate

- **Decision:** E5 uses eight layers, width 384, eight heads, and 26,877,696 parameters while restoring context 256.
- **Why:** a real batch-8 MPS forward/backward used about 700.7 MiB of PyTorch allocation and completed in 0.422 seconds, so the candidate is inside the local hardware envelope without changing tokens per update.
- **Rejected alternatives:** guess a larger model without measuring it, reduce the batch preemptively, or change context and capacity together.
- **Trade-off:** training iteration becomes materially slower and the corpus may still be too small to exploit all parameters.
- **Reconsider when:** validation stops improving early or the full optimizer state causes a new measured memory constraint.

## D012 — Accept the 26.9M checkpoint as Tiny Manas

- **Decision:** the step-2,900 checkpoint from `manas01-27m-20260831T160739Z` is the final research model.
- **Why:** it reduced independent validation perplexity by `25.4%` and test perplexity by `23.1%` relative to the accepted 13M baseline, improved next-token accuracy, remained under 1 GiB of PyTorch MPS allocation, and did not produce dominant long-copy behavior.
- **Rejected alternative:** prefer the cheaper 13M model despite the measured quality gap, or continue unbounded tuning on the same test suffix.
- **Trade-off:** generation is slower and the checkpoint is larger, while global narrative coherence remains limited by one source edition.
- **Reconsider when:** new Manas-only data is legally and technically audited, or an architecture change is preregistered against a fresh held-out set.

## D013 — Retain the measured RoPE release and useful execution changes

September 4: select the 26,779,392-parameter RoPE model after matched validation
improves from 4.345779 to 4.115836. Keep last-position projection, BF16 training,
and request-local KV caching. Keep LayerNorm, GELU, 32k vocabulary, and MHA;
checkpointing remains an optional memory tradeoff. The individual optimization
reports distinguish rejected full runs, stopped pilots, and the untriggered
fused-loss question. This supersedes the original architecture/weight choices,
not their historical measurements. Native artifact and rollback evidence are
in [experiment 13](experiments/13-quality-followup.md).

## D014 — Admit a research corpus and freeze book-level evaluation

September 5: use owner-authorized Orozbakov books 2/3/5 as additional research
training data, book 4 as validation, and Mamay as report-only test. Preserve
source groups, exact token/byte accounting, and source hashes. The 761,964-token
research pool includes 343,402 new training tokens. Coordinate extraction still
contains OCR errors; new verse preserves line breaks absent from the old source.
No inferred blanket license, generated repair text, or raw-corpus publication.
This is admission for a controlled experiment, not automatic model promotion.

## D015 — Reject the expanded weights at the fixed budget

Both old/expanded arms complete 3,000 updates from shared initial weights.
Expanded new-book loss is 4.207867, but familiar loss 4.133920 exceeds the
incumbent's 4.088184 by .045736, beyond the registered +.02 tolerance. Keep the
incumbent and the research evidence. A post-hoc partition reveals a large
format contribution; it does not replace the acceptance criterion or establish
a pure narrator-diversity effect. No candidate test or generation audit follows
this failed prediction gate.

## D016 — Stop the context-512 pilot without promoting it

The original-data RoPE context-512 run reaches the 1,500-update stage ceiling.
Its final three primary deltas are +.112277, -.509526, and +.411511; the mean
is +.004754. Stop under the fixed rule and preserve the resumable state. Keep
context 256. The remaining half budget is unspent; a full-run outcome was not
measured and must not be described as a full-budget rejection.

## D017 — Keep FP32 inference despite BF16 numerical agreement

BF16 autocast on the unchanged RoPE checkpoint passes numerical/cache gates,
but short generation takes 18.73% longer and near-overflow latency falls only
3.14%. A smaller persistent cache does not yield the required sampled live-memory
benefit. Keep FP32 on MPS and production CPU; no CPU BF16 claim is made.
After all four follow-ups close, evaluate the unchanged selected checkpoint on
the new report-only book/test remainders. Detailed scores and interpretation are
in [O14-O17](experiments/14-17-data-context-inference.md).
