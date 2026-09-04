# Final evaluation and production release — 2026-09-04

Historical release record. The later same-day [RoPE follow-up release](13-quality-followup.md#production-release-completed)
supersedes the current-image and selected-checkpoint fields below; this BF16
learned-position release is retained as the immediate rollback target.

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

## Deployment: completed; browser acceptance unavailable

The owner explicitly authorized push and deployment. Source commit
`67afe940d1f38ba4e0712c0851c69317957ac8ac` was pushed and built from its exact
tracked archive on the existing Linux CPU host. The CPU PyTorch version is
pinned to 2.13.0, matching the prior deployment; no paid compute was provisioned.

Current image: `manas-gpt:67afe940d1f38ba4e0712c0851c69317957ac8ac`.
Image ID: `sha256:186db12d028a0b43f4d79c615accf050bdb943a4be2a19541f66832887022f1d`.
Container: `manas-p5h6dziyjkyfejbiyqa8firs`; Docker health: **healthy**.
Limits remain 1 GiB and two CPUs, with the model directory mounted read-only.
Health reports the selected export hash, step 2,900, 26,877,696 parameters,
context 256 and `kv_cache: true`.

The isolated candidate loaded the real artifacts, returned health 200, rejected
unauthenticated generation with 401 and generated 16 tokens. It was stopped and
removed after the production replacement; no temporary server was left running.

Coolify's raw Compose was patched through the existing write API, followed by
the `MANAS_IMAGE` variable. The existing configuration writer regenerated the
runtime files without starting the whole stack. A fail-closed comparison found
only the model image, checkpoint path/hash and Coolify's shared `MANAS_IMAGE`
metadata changed. The writer also injects that image-name variable into the site
environment; no website code, credential or endpoint changed. Deployment used
`docker compose up -d --no-deps --no-build --pull never manas`, not Coolify's
whole-stack force-recreate action. The website container ID remained
`e87cc87d02082bdc1f7cb5de265b58c99395f5e691ef6dbaf3e30eb1e1a1c59a`.

A real request through `https://nik1t7n.com/api/manas/generate` returned HTTP 200,
16 generated tokens and 42.77 tokens/s. This is one CPU smoke, **not** a controlled
production speedup benchmark. The article returned HTTP 200 over public HTTPS.
An initial smoke incorrectly supplied `seed` to the public proxy, which rejects
unknown fields; it returned 422 as designed. Retrying the documented four-field
public contract succeeded. The private service still supports an optional seed.
Public rate limits, token limits and authentication were not relaxed.

**Acceptance limitation:** local DNS could not resolve `nik1t7n.com`, including
in the in-app browser (`ERR_NAME_NOT_RESOLVED`). Server-side DNS and public HTTPS
worked. No browser UI/reload acceptance is claimed, and no resolver override or
substitute page was used. The real public proxy-to-model path is verified;
interactive browser acceptance remains to be checked when local DNS works.

Operational transfer failures were corrected before cutover: the local rsync
does not support `--chmod=F444`, so the real artifact was copied and chmodded
on the server; the build log required a privileged redirect. Neither failure
changed the running service. Source, artifact hashes and candidate gates passed
before replacement.

## Preserved rollback

Previous image: `manas-gpt:43ed402cf506f1e263b17ebe99d7761b92474bed`.
Previous image ID: `sha256:0e429e7c7cd193053effcfe37ae7ca2e6972b49fdbccb3bed3ac7a86b888d962`.
Previous checkpoint: `/models/tiny-manas-27m.pt`, SHA-256
`cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1`.
Keep both image and checkpoint. Rollback restores this image, checkpoint path
and expected hash together in the existing Coolify service configuration, then
recreates only `manas`.

The pre-change runtime Compose and environment are retained in a root-only
server directory at `/opt/manas-gpt/releases/67afe940d1f38ba4e0712c0851c69317957ac8ac-rollback`.
Do not copy its environment file into Git or publish its contents. Both old and
new images and checkpoint files remain available locally on OVH.
