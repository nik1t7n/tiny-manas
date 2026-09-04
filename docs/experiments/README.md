# Tiny Manas optimization experiments

## Authorization and scope

On 2026-09-04 the owner authorized implementation and measured experiments,
superseding the earlier planning-only state. Later the same day the owner
explicitly authorized pushing the completed changes and deploying the accepted
result to the existing service. No paid compute or new dataset is needed. The original reference
must remain recoverable. Changes are accepted individually, not as an unmeasured bundle.

## Checklist

**Resumed by explicit owner instruction on 2026-09-04.** O08 resumes its saved
2,500-update checkpoint; completed work is not repeated. O09/O10 have not yet
started. See [pause and recovery history](PAUSED.md).

- [x] 00 — Freeze original code, architecture, configs, data, tokenizer, checkpoint and evidence. See [baseline](00-baseline.md) and [verified manifest](00-baseline-manifest.json).
- [x] 01 — Last-position output projection during generation. [Accepted](01-last-position.md): parity passed, 1.24x model-forward speed at B=1,T=256; 256x smaller output tensor.
- [x] 02 — BF16 mixed precision. [Accepted](02-bf16.md): 20.58% shorter full training, validation +0.0000464 nats, all 20 outputs reviewed. 27M training uses BF16; inference remains FP32.
- [x] 03 — `torch.compile` and fusion. [Rejected on this MPS workload](03-compile.md): diagnosed a broadcast-gradient failure; corrected candidate passed parity but updates were 3.02% slower.
- [x] 04 — Activation checkpointing. [Accepted opt-in, off by default](04-checkpointing.md): 38.46% less sampled allocation, 30.63% slower updates; explicit MPS dropout-RNG preservation verified.
- [x] 05 — 32k / 16k / 8k tokenizer comparison. [Keep incumbent 32k](05-tokenizer.md): smaller variants save resources but worsen validation BPB by 4.15% / 3.84% versus the accepted model, beyond the 1% limit. All 60 new raw outputs reviewed.
- [x] 06 — RoPE versus learned positions. [Rejected on this MPS path](06-rope.md): correctness passed, but median updates were 12.27% slower, beyond the 10% ceiling. Keep learned positions.
- [x] 07 — RMSNorm versus LayerNorm. [Rejected on this MPS path](07-rmsnorm.md): outputs/gradients matched the formula, but updates were 10.71% slower. Keep LayerNorm.
- [x] 08 — SwiGLU versus parameter-matched GELU FFN. [Rejected](08-swiglu.md): full 3,000-update run worsened independent validation loss by 0.08164 nats and increased repetition. Keep GELU.
- [x] 09 — KV cache with separate prefill/decode. [Accepted](09-kv-cache.md): parity passed; 1.1566x short-generation speedup, only 1.0165x near window overflow. Native generation now uses a request-local cache.
- [ ] 10 — GQA versus MHA, including quality and actual cache storage. [Preregistered bounded adaptation](10-gqa.md).
- [ ] 11 — Fused/chunked output loss, conditional on measured output-memory pressure. [Source review and gates](11-output-loss.md); distinguish the Triton kernel from the macOS compile implementation.
- [ ] Final — Inspect 20 raw generations and memorization for accepted trained models;
  evaluate protected test only after selection; update technical README with measured results.
- [ ] Release — Push the completed changes; deploy the accepted checkpoint and
  code to the existing private Tiny Manas service; verify public generation and
  preserve the previous image/checkpoint for rollback.

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

If a result is noisy, unsupported on MPS, slower, or harms quality beyond the
predeclared tolerance, do not promote it. Keep the report and research runner.
Do not quietly fall back to CPU. Never overwrite the frozen baseline or accepted
checkpoint. Use bounded runs on the actual M5/16 GB host, not assumed Mac Studio resources.
