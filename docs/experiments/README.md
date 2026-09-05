# Tiny Manas optimization experiments

## Authorization and scope

On 2026-09-04 the owner authorized implementation and measured experiments,
superseding the earlier planning-only state. Later the same day the owner
explicitly authorized pushing the completed changes and deploying the accepted
result to the existing service. No paid compute or new dataset is needed. The original reference
must remain recoverable. Changes are accepted individually, not as an unmeasured bundle.

## Checklist

**September 5 follow-up complete:** [O14-O17](14-17-data-context-inference.md)
added three research-training volumes and book-level holdouts. Expanded weights
failed the familiar-domain floor; RoPE context 512 stopped at 1,500 updates;
BF16 inference failed its performance gate. The deployed RoPE/256/FP32 model
is unchanged. Final whole-remainder evaluation was report-only after selection.

- [x] 14 — Freeze additional Manas texts and new evaluation, with exact target/byte accounting and source/format limits.
- [x] 15 — Complete fresh old/expanded 3,000-update arms; reject expanded weights for +.045736 familiar loss beyond +.02.
- [x] 16 — Run the staged context-512 arm; stop at 1,500 without consistent primary gain. Full-budget quality remains unknown.
- [x] 17 — Compare FP32/BF16 inference on the retained checkpoint; keep FP32 after short latency regresses 18.73%.

**Selection completed on 2026-09-04.** O08 resumed from its saved 2,500-update
checkpoint and finished; O09/O10 also finished. The protected test was evaluated
only after selection. See [pause and recovery history](PAUSED.md).

**Later same-day quality follow-up:** the owner reopened O06/O07 with staged
training instead of cost-only rejection. RoPE earned a full run and was selected;
RMSNorm stopped at 900 updates for no material gain. See [experiment 13](13-quality-followup.md).
RoPE has now been deployed with native KV caching. The public generation request
passed, the website container was not restarted, and the previous image/weights
remain available for rollback.

- [x] 00 — Freeze original code, architecture, configs, data, tokenizer, checkpoint and evidence. See [baseline](00-baseline.md) and [verified manifest](00-baseline-manifest.json).
- [x] 01 — Last-position output projection during generation. [Accepted](01-last-position.md): parity passed, 1.24x model-forward speed at B=1,T=256; 256x smaller output tensor.
- [x] 02 — BF16 mixed precision. [Accepted](02-bf16.md): 20.58% shorter full training, validation +0.0000464 nats, all 20 outputs reviewed. 27M training uses BF16; inference remains FP32.
- [x] 03 — `torch.compile` and fusion. [Rejected on this MPS workload](03-compile.md): diagnosed a broadcast-gradient failure; corrected candidate passed parity but updates were 3.02% slower.
- [x] 04 — Activation checkpointing. [Accepted opt-in, off by default](04-checkpointing.md): 38.46% less sampled allocation, 30.63% slower updates; explicit MPS dropout-RNG preservation verified.
- [x] 05 — 32k / 16k / 8k tokenizer comparison. [Keep incumbent 32k](05-tokenizer.md): smaller variants save resources but worsen validation BPB by 4.15% / 3.84% versus the accepted model, beyond the 1% limit. All 60 new raw outputs reviewed.
- [x] 06 — RoPE versus learned positions. [Accepted after quality follow-up](13-quality-followup.md): validation loss 4.34578 → 4.11584, perplexity −20.54%; native loading/cache parity passed. The earlier cost-only verdict is historical.
- [x] 07 — RMSNorm versus LayerNorm. [Pilot stopped at 900 updates](13-quality-followup.md): last-three mean validation delta +.00331, no material advantage. Keep LayerNorm; do not infer full-budget inferiority.
- [x] 08 — SwiGLU versus parameter-matched GELU FFN. [Rejected](08-swiglu.md): full 3,000-update run worsened independent validation loss by 0.08164 nats and increased repetition. Keep GELU.
- [x] 09 — KV cache with separate prefill/decode. [Accepted](09-kv-cache.md): parity passed; 1.1566x short-generation speedup, only 1.0165x near window overflow. Native generation now uses a request-local cache.
- [x] 10 — GQA versus MHA. [Rejected after matched 150-update adaptation](10-gqa.md): 75% smaller cache, but validation loss +0.19402 nats versus adapted MHA, beyond the +0.05 limit.
- [x] 11 — Fused/chunked output loss. [Assessed, not triggered](11-output-loss.md): no measured memory or output-loss bottleneck on the selected model; no runtime benchmark claimed.
- [x] Final — Inspect 20 raw generations and memorization for accepted trained models;
  evaluate protected test only after selection; update technical README with measured results.
- [x] Release — Push the completed changes; deploy the accepted checkpoint and
  code to the existing private Tiny Manas service; verify public generation and
  preserve the previous image/checkpoint for rollback. [Release evidence](12-release.md):
  healthy container and real public HTTPS generation passed; browser UI acceptance
  remains unavailable because local DNS did not resolve the domain.

## Decision protocol

Every numbered report records question, hypothesis, forecast, falsifier,
controlled variables, actual command, source revision, artifact hashes,
hardware/software, measurements, limitations and decision. Raw licensed text,
weights and run outputs remain under ignored `runs/`.

Inference-equivalent changes require matching logits (stated numerical tolerance)
and greedy continuations on real held-out inputs. Measure synchronized timings
after warmup with repeated interleaved runs. Report output tensor size separately
from allocator/driver memory; sampled memory is not an exact peak.

Training changes first pass one real-batch forward/backward correctness check.
Quality-changing changes require matched training budgets, fixed validation
selection, and raw outputs. A short stability probe is not evidence of final
quality. Validation, not protected test, decides which changes survive.

If a result is inconclusive, unsupported on MPS, or harms quality beyond the
predeclared tolerance, do not promote it. For architectural changes, evaluate
cost against demonstrated quality benefit rather than vetoing a modest slowdown
before quality is measured. Use the explicit staged protocol for O06/O07.
Keep the report and research runner.
Do not quietly fall back to CPU. Never overwrite the frozen baseline or accepted
checkpoint. Use bounded runs on the actual M5/16 GB host, not assumed Mac Studio resources.
