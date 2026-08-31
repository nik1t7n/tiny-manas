# Project contract

Tiny Manas is a real, standalone research project for a small decoder-only language model trained on the Kyrgyz epic *Manas*.

## Scope

- Use only the pinned Manas-UdS `Manas01` source and pinned Kyrgyz byte-level BPE tokenizer until a documented data experiment changes that decision.
- Train and generate with PyTorch on Apple `mps`; never silently fall back to CPU.
- Keep the model decoder-only. Do not add an encoder, cross-attention, instruction tuning, retrieval, or chat protocol without a new approved research question.
- Do not introduce a bigram comparison. This project stands on its own metrics and raw outputs.
- Keep the implementation compact enough that the full forward, loss, backward, optimizer, evaluation, and generation paths remain inspectable.

## Research loop

Before every material run, record the question, hypothesis, forecast, falsifier, changed variable, controlled variables, and exact config in `docs/RESEARCH_LOG.md`.

After every run, record measured metrics, timings, memory, raw generations, failures, forecast versus result, updated belief, and the next cheapest informative experiment.

Run order is strict:

1. verify source, tokenizer, splits, shapes, and MPS;
2. overfit one real batch with dropout disabled;
3. run the bounded 10K-token pilot;
4. run the full `Manas01` experiment;
5. scale context or model size only through one-variable experiments.

Do not scale past a failed correctness gate.

## Evidence and publication

- A falling train loss is not sufficient evidence.
- Keep untouched validation and test splits.
- Inspect at least 20 raw generations for accepted final runs.
- Audit long generated spans for memorization.
- Separate observed facts from interpretation.
- Do not commit licensed corpus text, token arrays, checkpoints, or raw run directories.
- Do not deploy a public demo until a checkpoint passes the research gates and source-license implications are reviewed.

## Engineering

- Prefer Python stdlib, PyTorch primitives, and the existing `tokenizers` dependency.
- Every command must fail explicitly on missing or mismatched real artifacts.
- Capture commit, config, seed, versions, hashes, device, parameters, timing, and memory in each run artifact.
- Use Conventional Commits with a meaningful body.
