# Tiny Manas optimization experiments

## Authorization and scope

On 2026-09-04 the owner authorized implementation and measured experiments,
superseding the earlier planning-only state. Work stays local: no deployment,
paid compute, new datasets, or publication is implied. The original reference
must remain recoverable. Changes are accepted individually, not as an unmeasured bundle.

## Checklist

- [x] 00 — Freeze original code, architecture, configs, data, tokenizer, checkpoint and evidence. See [baseline](00-baseline.md) and [verified manifest](00-baseline-manifest.json).
- [x] 01 — Last-position output projection during generation. [Accepted](01-last-position.md): parity passed, 1.24x model-forward speed at B=1,T=256; 256x smaller output tensor.
- [ ] 02 — BF16 mixed precision. [100-step probe passed; full FP32/BF16 pair running](02-bf16.md). Default training remains FP32 pending the quality gate.
- [ ] 03 — `torch.compile` and fusion on the winning precision mode.
- [ ] 04 — Activation checkpointing, unchanged batch and context first.
- [ ] 05 — 32k / 16k / 8k tokenizer comparison on fixed original-text splits.
- [ ] 06 — RoPE versus learned positions. [Preregistered](06-rope.md); context and cache-boundary semantics remain explicit.
- [ ] 07 — RMSNorm versus LayerNorm. [Preregistered](07-rmsnorm.md).
- [ ] 08 — SwiGLU versus parameter-matched GELU FFN. [Preregistered](08-swiglu.md).
- [ ] 09 — KV cache with separate prefill/decode on the accepted MHA checkpoint. [Preregistered](09-kv-cache.md).
- [ ] 10 — GQA versus MHA, including quality and actual cache storage. [Preregistered bounded adaptation](10-gqa.md).
- [ ] 11 — Fused/chunked output loss, conditional on measured output-memory pressure.
- [ ] Final — Inspect 20 raw generations and memorization for accepted trained models;
  evaluate protected test only after selection; update technical README with measured results.

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
