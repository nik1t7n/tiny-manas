# Tiny Manas

Tiny Manas is a small decoder-only Transformer trained from scratch on the Kyrgyz epic *Manas*.

The project asks a deliberately narrow question: how much coherent Manas-like text can a compact, inspectable language model learn on one Apple Silicon laptop?

This is not a general Kyrgyz assistant. It is a real language-model experiment with a pinned source, a frozen Kyrgyz tokenizer, causal self-attention, validation data, reproducible run artifacts, and raw-generation analysis.

## Current status

Implementation is in progress. The research sequence is:

1. verify the real data and MPS path;
2. overfit one real batch;
3. train a bounded 10K-token pilot;
4. train on the full cleaned `Manas01` epic;
5. test bounded context or model scaling only after the base run is understood.

The full preregistered plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Hypotheses and results are recorded chronologically in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

## Planned command path

```bash
uv sync
uv run manas-gpt prepare --config configs/overfit-one-batch.toml
uv run manas-gpt train --config configs/overfit-one-batch.toml
uv run manas-gpt generate --checkpoint runs/<run>/best-model.pt --prompt "Манас"
```

These commands are documented before acceptance; they are not claimed working until the corresponding real smoke is recorded in the research log.

## Rights

The project code and writing are currently all rights reserved. The Manas-UdS source is licensed separately under CC BY-NC-SA 4.0 and is not redistributed by this repository. See [docs/SOURCES.md](docs/SOURCES.md).
