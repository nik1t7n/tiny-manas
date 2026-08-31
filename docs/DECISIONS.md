# Decisions

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
