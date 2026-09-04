# Final evaluation and production release — 2026-09-04

## Research gate: passed

All O00–O11 decisions are closed. Accepted: last-position logits, BF16 training,
request-local KV cache. Checkpointing remains optional and disabled. No rejected
architecture or tokenizer is included. Final test was evaluated only after
selection: 100 original random test batches, seed 1837, FP32, 204,800 targets;
loss 4.7575606346, perplexity 116.461487, top-1 27.7363%, top-5 45.2720%.
The accepted BF16 model's 20 raw continuations were reviewed; its repetition
and limited coherence remain documented. No general Kyrgyz/chat claim is made.

Selected training checkpoint SHA-256:
`31499eb747c98bcada1c48b12205033cded96573269bc15464ccafd9905e2167`.
Inference export: `tiny-manas-27m-bf16-20260904.pt`, 107,547,815 bytes,
SHA-256 `4c6f70883564df6c46849c0849f38b06195b4dcaba3bde2572ef60eec4cf3494`.
Tokenizer SHA-256:
`5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f`.
All 17 frozen artifact files and the source archive passed a fresh hash check.

## Release gate: pending

The owner explicitly authorized push and deployment. Build the exact committed
source on the existing Linux CPU host, using PyTorch 2.13.0 (the currently
deployed version), not a new paid training host. Update only the private `manas`
component; preserve the website, authentication, limits and read-only mount.
Verify the candidate with the real tokenizer/checkpoint, then verify a public
generation through the website. No checkpoint download endpoint is introduced.

## Preserved rollback

Previous image: `manas-gpt:43ed402cf506f1e263b17ebe99d7761b92474bed`.
Previous image ID: `sha256:0e429e7c7cd193053effcfe37ae7ca2e6972b49fdbccb3bed3ac7a86b888d962`.
Previous checkpoint: `/models/tiny-manas-27m.pt`, SHA-256
`cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1`.
Keep both image and checkpoint. Rollback restores this image, checkpoint path
and expected hash together in the existing Coolify service configuration, then
recreates only `manas`.
