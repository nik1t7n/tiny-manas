# Tiny Manas

Tiny Manas is a 26.9M-parameter language model trained from scratch on the Kyrgyz epic *Manas*.

I built it to understand a Transformer by following the whole path myself: prepare a real text, encode it with a Kyrgyz tokenizer, create shifted training batches, implement causal self-attention, run backpropagation on an Apple GPU, and inspect what the model actually learned.

This is not a general Kyrgyz assistant. It is a deliberately small model that learned the voice, names, rhythm, and recurring actions of one edition of *Manas*.

## The experiment in one picture

```text
Manas01 epic
    ↓
465,069 Kyrgyz BPE tokens
    ↓
train / validation / test in story order
    ↓
decoder-only Transformer on Apple MPS
    ↓
predict the next token at every position
    ↓
validate, generate, inspect repetition and copying
```

The final model uses eight Transformer blocks, eight attention heads, 384 features per token, and a 256-token context window. Its tokenizer has 32,768 possible tokens. Training sampled 12.29M target positions and took 24.4 minutes on an M5 MacBook Pro with 16 GB of unified memory.

## What worked

| Model | Context | Parameters | Validation perplexity | Test perplexity | Training time |
|---|---:|---:|---:|---:|---:|
| Base | 256 | 13.19M | 103.38 | 151.42 | 15.1 min |
| Longer context | 512 | 13.26M | 112.94 | 171.85 | 17.2 min |
| **Final Tiny Manas** | **256** | **26.88M** | **77.15** | **116.45** | **24.4 min** |

Doubling the context did not help. It cost more time and predicted held-out text worse. Doubling model capacity did help: validation perplexity improved by 25.4% and test perplexity by 23.1% over the base model.

The result is not fluent long-form storytelling. It is a recognizable small epic model. It can continue combat scenes, dialogue-like fragments, hero names, and verse-like phrases for short stretches. It still repeats names and formulas, invents malformed words, and eventually loses track of the event.

Across 20 fixed generations, the final model's longest exact match with the training text was seven words. The improvement therefore did not come from reproducing long passages verbatim.

## Architecture

```text
token IDs
  + learned positions
        ↓
┌────────────────────────────┐
│ LayerNorm                  │
│ causal multi-head attention│ ← only current and earlier tokens
│ residual connection        │
│ LayerNorm                  │
│ feed-forward network       │
│ residual connection        │
└────────────────────────────┘ × 8
        ↓
final LayerNorm
        ↓
one score for every tokenizer token
```

The implementation uses a pre-LayerNorm decoder-only Transformer, GELU feed-forward layers, learned positional embeddings, dropout, tied input/output embeddings, cross-entropy, AdamW, gradient clipping, warmup, and cosine learning-rate decay. Attention runs through PyTorch's causal scaled-dot-product attention path.

## Reproduce it

The source text and tokenizer are downloaded from pinned public URLs and verified by checksum. Licensed corpus files and large checkpoints are intentionally not committed.

```bash
uv sync
uv run manas-gpt prepare --config configs/manas01-27m.toml
uv run manas-gpt train --config configs/manas01-27m.toml
uv run manas-gpt evaluate --checkpoint runs/<run>/best-model.pt --split test --batches 100
uv run manas-gpt audit --checkpoint runs/<run>/best-model.pt --output runs/<run>/generation-audit.json
uv run manas-gpt export --checkpoint runs/<run>/best-model.pt --output artifacts/tiny-manas-27m.pt
```

The project fails if Apple MPS is unavailable. It does not silently move unsupported work to CPU.

The export command removes optimizer state and preserves tied weights without duplication. The accepted inference artifact is 107,546,203 bytes with SHA-256 `cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1`.

## Read the evidence

- [RESULTS.md](RESULTS.md) explains the full experiment and its limits.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) preserves the preregistered questions and gates.
- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) records forecasts, failed assumptions, invalid attempts, and results in chronological order.
- [docs/DECISIONS.md](docs/DECISIONS.md) records why each major choice was made.
- [docs/SOURCES.md](docs/SOURCES.md) records data, tokenizer, implementation references, and licenses.
- [reports/generation-audits](reports/generation-audits) contains all fixed raw generations rather than a hand-picked sample.

## Rights

Project code and writing are currently all rights reserved. The Manas-UdS source is licensed separately under CC BY-NC-SA 4.0 and is not redistributed here. See [docs/SOURCES.md](docs/SOURCES.md).
