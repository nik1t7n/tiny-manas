# Tiny Manas

**A 26.9M-parameter decoder-only Transformer trained from scratch on the Kyrgyz epic *Manas*.**

[Read the general-audience article](https://nik1t7n.com/essays/training-tiny-manas) · [Inspect the tokenizer](https://github.com/nik1t7n/kyrgyz-tokenizer) · [Browse the experiment results](#experimental-results)

Tiny Manas is a small language model that learns to continue one edition of the Kyrgyz epic *Manas*. I wrote the model and the training pipeline to study the complete language-model path without hiding the important mechanics behind a large framework: verified text acquisition, byte-level BPE tokenization, shifted training batches, causal self-attention, backpropagation, checkpoint selection, generation, and memorization analysis.

The final model has 26,877,696 parameters. It contains eight Transformer blocks, eight attention heads, 384 features per token, and a context window of 256 tokens. I trained it on an M5 MacBook Pro with 16 GB of unified memory. The accepted run took 24.4 minutes.

Tiny Manas is a research model, not a general Kyrgyz assistant. It has seen one edition of one epic. It can reproduce names, rhythm, speech patterns, and short chains of action from that source distribution. It cannot answer general questions, follow instructions, or maintain a reliable long narrative.

## Research question

The project asks a narrow question:

> How much Manas-like structure can a small, inspectable Transformer learn from one epic on one laptop?

This boundary determines every engineering choice in the repository. The model stays decoder-only. The data stays Manas-only. Full runs use Apple MPS and fail when MPS is unavailable. Each scaling experiment changes one major variable, and validation results decide which checkpoint survives.

The project follows the spirit of Tiny Shakespeare experiments, with a Kyrgyz source and a subword tokenizer. The goal is a complete and measurable language-model experiment at a scale where one person can still read the implementation from end to end.

## Results at a glance

| Model | Context | Parameters | Validation loss | Validation perplexity | Test loss | Test perplexity | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13M baseline | 256 | 13,193,216 | 4.6384 | 103.38 | 5.0201 | 151.42 | 15.1 min |
| 13M, longer context | 512 | 13,258,752 | 4.7268 | 112.94 | 5.1466 | 171.85 | 17.2 min |
| **Tiny Manas** | **256** | **26,877,696** | **4.3457** | **77.15** | **4.7575** | **116.45** | **24.4 min** |

Two conclusions survived the measurements:

1. Doubling the context window from 256 to 512 made the model slower and worsened held-out prediction.
2. Increasing depth and width reduced validation perplexity by 25.4% and test perplexity by 23.1% relative to the 13M baseline.

The final model generated recognizable epic-like text for short stretches. It still repeated names and formulaic phrases, invented malformed words, and lost track of events over longer passages. Across 20 fixed samples, the longest matching word sequence in the training text was nine words after normalizing case, punctuation and whitespace. A [corrected audit](docs/experiments/00-audit-correction.md) replaced the earlier undercount of seven. These samples contained no long copied word sequences; they do not prove that the model never memorizes other passages.

## The complete pipeline

```text
pinned Manas01 source
        ↓
verified extraction and cleanup
        ↓
frozen Kyrgyz byte-level BPE tokenizer
        ↓
465,069 token IDs
        ↓
chronological train / validation / test split
        ↓
random (x, y) training windows
        ↓
decoder-only Transformer
        ↓
next-token cross-entropy
        ↓
backpropagation + AdamW on Apple MPS
        ↓
best validation checkpoint
        ↓
held-out evaluation + fixed generation audit
```

The following sections walk through that pipeline in the same order as the code.

## 1. Data: one pinned edition of *Manas*

The corpus comes from the [Manas-UdS collection](https://fedora.clarin-d.uni-saarland.de/kyrgyz/kyrgyz_2022_10_03.zip). This project selects document `Manas01`, attributed in the source metadata to the performer Sayakbai Karalaev.

The preparation command downloads a pinned archive and checks its SHA-1 digest before reading it. The extractor then finds the `Manas01` document inside the VRT archive, reconstructs its sentences, and removes the scholarly material before the unique heading `Манастын туула элегиндеги бабалары`. The command fails if the archive hash changes, the document disappears, or the heading appears zero or multiple times.

The cleaned epic contains:

| Measurement | Value |
|---|---:|
| Characters | 1,794,194 |
| UTF-8 bytes | 3,249,652 |
| BPE tokens | 465,069 |

The repository does not redistribute the corpus. `prepare` downloads the real source and records its provenance in the generated metadata.

### Why a chronological split

The token sequence is divided in story order:

| Split | Tokens | Share | Purpose |
|---|---:|---:|---|
| Train | 418,562 | 90% | Updates model parameters |
| Validation | 23,253 | 5% | Selects the checkpoint and tests engineering choices |
| Test | 23,254 | 5% | Measures the accepted model once the experiment ends |

A random token-level split would place neighboring fragments from the same passage on both sides of the evaluation boundary. The model could then score well by seeing almost identical local contexts during training. The chronological split is harder, especially near changes in narrative content, but it gives a more honest continuation test.

This split still comes from the same edition and performer. Test performance measures continuation into a later part of this source. It does not measure transfer to another narrator, modern Kyrgyz prose, or general language use.

## 2. Tokenization

Tiny Manas uses my frozen [Kyrgyz byte-level BPE tokenizer](https://github.com/nik1t7n/kyrgyz-tokenizer) at commit `594d9e142cca1593963ccf12f344ab7ea4938fa5`. The tokenizer has a vocabulary of 32,768 tokens.

A byte-level tokenizer starts with all 256 byte values, so every UTF-8 string remains representable. BPE training adds tokens for byte sequences that occur often in the training corpus. A frequent Kyrgyz word may become one token, while an uncommon word can remain a sequence of subword pieces or bytes.

Freezing the tokenizer isolates the model experiment. If the vocabulary changed between runs, an apparent model improvement could come from a shorter or more favorable representation of the text. Every candidate therefore receives the same token IDs and predicts over the same 32,768-way vocabulary.

The preparation step verifies the tokenizer file with SHA-256 and checks a full encode-decode round trip:

```text
original epic text
      ↓ encode
list of token IDs
      ↓ decode
exactly the original epic text
```

The processed split files store token IDs as little-endian unsigned 16-bit integers. The largest possible ID is below 65,536, so `uint16` stores the complete vocabulary without wasting four or eight bytes per token.

## 3. Building training examples

The model learns one task: predict the next token at every position.

Suppose a tokenized fragment is:

```text
[Манас] [кылычын] [көтөрдү] [да]
```

The input `x` and target `y` contain the same sequence shifted by one position:

```text
x = [Манас]    [кылычын] [көтөрдү]
y = [кылычын]  [көтөрдү] [да]
```

For the final run, the sampler draws eight independent windows per micro-batch. Each window contains 256 input tokens and 256 next-token targets:

```text
x.shape = (B, T) = (8, 256)
y.shape = (B, T) = (8, 256)
```

`RandomWindowSampler` selects start positions with a seeded CPU generator, slices `T + 1` consecutive tokens, and constructs `x` and `y` from the offset views. The tensors move to MPS only after sampling. This keeps data selection reproducible and avoids storing a large set of precomputed windows.

## 4. Model architecture

Tiny Manas uses a decoder-only, pre-LayerNorm Transformer.

| Symbol | Meaning | Final value |
|---|---|---:|
| `V` | tokenizer vocabulary size | 32,768 |
| `T` | maximum context length | 256 |
| `C` | embedding width | 384 |
| `L` | Transformer blocks | 8 |
| `H` | attention heads | 8 |
| `d = C / H` | features per head | 48 |
| `4C` | feed-forward hidden width | 1,536 |
| `p` | dropout probability | 0.2 |

The full forward path is:

```text
token IDs (B, T)
        ↓
token embeddings + learned position embeddings
        ↓
dropout
        ↓
┌─────────────────────────────────────────────┐
│ x = x + CausalSelfAttention(LayerNorm(x))   │
│ x = x + FeedForward(LayerNorm(x))           │
└─────────────────────────────────────────────┘ × 8
        ↓
final LayerNorm
        ↓
LM head with tied token-embedding weights
        ↓
logits (B, T, V)
```

### 4.1 Token and position embeddings

The model receives integer token IDs. An embedding table maps each ID to a vector of `C = 384` learned numbers. A second table contains one learned vector for each position from 0 to 255.

For token ID `s[b,t]` at position `t`, the initial representation is

```math
X^{(0)}_{b,t} = E_{token}[s_{b,t}] + E_{position}[t]
```

where

```text
E_token.shape    = (V, C)   = (32768, 384)
E_position.shape = (T, C)   = (256, 384)
X.shape          = (B, T, C)
```

Token embeddings let the model learn how token identities behave. Position embeddings distinguish the same token at different places in a sequence. Without positional information, self-attention would know which token vectors exist but would have no direct signal for their order.

The implementation uses learned absolute positions because they are easy to inspect and sufficient for a 256-token research context. RoPE and other relative schemes remain useful future experiments, but changing them during the baseline would introduce another variable.

### 4.2 Pre-LayerNorm and residual paths

Each Transformer block has two transformations: causal self-attention and a feed-forward network. Both receive normalized input and add their result back to the existing residual stream:

```math
X' = X + \mathrm{Attention}(\mathrm{LayerNorm}(X))
```

```math
X^{next} = X' + \mathrm{FFN}(\mathrm{LayerNorm}(X'))
```

LayerNorm normalizes the 384 features of each token independently. For a token vector `x`, it computes

```math
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

The learned vectors `γ` and `β` let the model choose a useful scale and offset after normalization. Pre-LayerNorm places this operation before attention and the FFN. In a small implementation, that layout provides a clean residual path and stable optimization without extra machinery.

Residual addition preserves the existing representation while each sublayer contributes an update. A block does not have to reconstruct the whole token state from scratch.

### 4.3 Causal multi-head self-attention

Self-attention lets each position collect information from itself and earlier positions. The causal rule prevents a training position from reading its future target.

One linear layer projects `X` into queries, keys, and values:

```math
[Q, K, V] = XW_{qkv} + b_{qkv}
```

The code splits the result into three tensors and then into eight heads:

```text
before heads: (B, T, C)
after heads:  (B, H, T, d) = (B, 8, T, 48)
```

Each head compares every query with the keys at every accessible position:

```math
S = \frac{QK^\top}{\sqrt{d}}
```

For one batch item and one head, `S` has shape `(T, T)`. Entry `S[i,j]` measures the compatibility between the query at position `i` and the key at position `j`. Dividing by `√d` controls the scale of dot products as the head dimension grows.

The causal mask sets all scores above the diagonal to negative infinity:

```math
S_{i,j} = -\infty \quad \text{when } j > i
```

Softmax converts each row into non-negative attention weights that sum to one:

```math
A_{i,j} = \frac{\exp(S_{i,j})}{\sum_{k \leq i}\exp(S_{i,k})}
```

Because `exp(-∞) = 0`, future positions receive zero weight. The head output is a weighted sum of value vectors:

```math
O = AV
```

All eight heads process the same sequence through separate learned projections. Their 48-dimensional outputs are concatenated into 384 features and passed through an output projection:

```math
\mathrm{MHA}(X) = \mathrm{Concat}(O_1, \ldots, O_8)W_o + b_o
```

The implementation calls PyTorch's `scaled_dot_product_attention` with `is_causal=True`. During training it passes dropout probability `0.2` to the attention weights. During evaluation it passes `0.0`, because this PyTorch operation applies dropout according to the supplied argument regardless of module mode.

### 4.4 Feed-forward network

Attention moves information between token positions. The feed-forward network transforms the collected features of each token independently:

```math
\mathrm{FFN}(x) = W_2\mathrm{GELU}(W_1x + b_1) + b_2
```

The first linear layer expands each token from 384 to 1,536 features. GELU introduces a non-linear gate, and the second linear layer returns the result to 384 features so it can rejoin the residual stream.

```text
(B, T, 384)
      ↓ Linear
(B, T, 1536)
      ↓ GELU
(B, T, 1536)
      ↓ Linear + dropout
(B, T, 384)
```

The four-times expansion gives the block space to form combinations of the contextual features that attention collected. Using two linear layers without GELU would collapse into one linear transformation and would lose this non-linear capacity.

### 4.5 LM head and tied weights

After eight blocks and a final LayerNorm, the LM head maps every 384-dimensional token state to `V = 32,768` logits:

```math
Z = XW_{LM}^{\top}
```

```text
Z.shape = (B, T, V)
```

Each logit is an unnormalized score for one possible next token. Softmax is unnecessary inside the training forward pass because cross-entropy computes the equivalent log-softmax operation in a stable form.

Tiny Manas ties `W_LM` to the token embedding table. The same learned matrix reads token identities at the input and scores token identities at the output. Weight tying removes a second `32,768 × 384` matrix, saving 12,582,912 parameters and keeping the input and output token spaces aligned.

### 4.6 Parameter count

| Component | Parameters |
|---|---:|
| Token embedding and tied LM head | 12,582,912 |
| Position embedding | 98,304 |
| Eight Transformer blocks | 14,195,712 |
| Final LayerNorm | 768 |
| **Total** | **26,877,696** |

One Transformer block contains 1,774,464 parameters. The vocabulary matrix still accounts for almost half of the model, even with weight tying. This is one cost of keeping the full 32,768-token tokenizer vocabulary in a small model.

## 5. Loss and backpropagation

The targets have shape `(B, T)`. The logits have shape `(B, T, V)`. Training flattens the first two dimensions so cross-entropy receives one vocabulary distribution and one correct token ID for each of the `B × T` positions.

For one position with logits `z` and correct token ID `y`, the negative log-likelihood is

```math
\ell(z,y) = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{V}e^{z_j}}\right)
```

The batch loss is the mean over all target positions:

```math
\mathcal{L} = \frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\ell(Z_{b,t}, Y_{b,t})
```

A low loss means the model assigned high probability to the real next tokens. A confident wrong prediction produces a larger penalty than an uncertain one.

PyTorch autograd records the operations used to produce this scalar loss. Calling `backward()` applies the chain rule from the loss back through the LM head, every residual block, Q/K/V projections, and both embedding tables. Each parameter receives a gradient that measures how a small change in that parameter would affect the loss.

The optimizer then updates the parameters. Tiny Manas uses AdamW with:

| Setting | Value |
|---|---:|
| Peak learning rate | `3e-4` |
| Minimum learning rate | `3e-5` |
| Adam beta 1 | `0.9` |
| Adam beta 2 | `0.95` |
| Weight decay | `0.1` |
| Gradient norm limit | `1.0` |

Weight decay applies to matrix-shaped parameters such as embedding and linear weights. Biases and LayerNorm vectors do not decay. This grouping avoids shrinking parameters whose role is scale or offset.

## 6. The training schedule

The final run uses a micro-batch of eight sequences and accumulates gradients across two micro-batches before each optimizer update:

```text
8 sequences × 256 positions × 2 accumulation steps
= 4,096 target positions per optimizer update
```

At the hard limit of 3,000 updates, the model processes 12,288,000 sampled target positions. These are sampled windows, so the count is not the number of distinct corpus tokens. Positions repeat across steps.

The learning rate warms up for 100 steps and then follows a cosine decay. After warmup, the schedule is

```math
\eta(s) = \eta_{min} + \frac{1}{2}\left(1 + \cos(\pi r)\right)(\eta_{max} - \eta_{min})
```

where

```math
r = \frac{s - s_{warmup}}{s_{max} - s_{warmup}}
```

Warmup avoids applying the full learning rate while the network still produces unstable early gradients. Cosine decay reduces the update size as training approaches the end.

The code clips the global gradient norm to 1.0 before `optimizer.step()`. One unusual batch can otherwise create a large update and damage the current model state.

Validation runs every 100 optimizer steps. The training loop writes a new `best-model.pt` only when validation loss improves. The final accepted model comes from step 2,900 rather than the last step. The final step completed the planned run, but its validation result did not beat the earlier checkpoint.

## 7. Correctness before scale

Training loss can fall even when an evaluation path is wrong. The project uses two bounded gates before a full-data run.

### 7.1 Overfit one real batch

The first gate disables dropout and trains repeatedly on one fixed batch. A correctly connected model should memorize those targets.

Loss fell from `10.4385` to `0.0267`. Top-1 and top-5 accuracy reached 100%. This checks the complete trainable path: embeddings, causal attention, the FFN, cross-entropy, backward, AdamW, and checkpoint loading.

The first attempt exposed a measurement bug. The optimizer trained on one fixed batch while the summary evaluated another fixed batch created by a different sampler. The model had memorized the correct data, but the report claimed failure. The invalid attempt remains in `docs/RESEARCH_LOG.md`; the evaluator now reuses the exact optimization batch.

### 7.2 Train on 10,000 real tokens

The pilot runs the full data and training machinery on a bounded prefix. It passed the mechanical checks and then overfit as expected. Its best validation checkpoint appeared at step 200. By step 1,000, train loss had fallen to `0.1235` while validation loss had risen to `10.4687`.

The pilot showed that the network could learn the source distribution. It also showed that 9,000 training tokens could not support an 8.1M-parameter model for long.

## 8. Experimental results

All three full-data candidates used the same tokenizer, chronological splits, seed, dropout, optimizer settings, and total budget of 12.29M sampled target positions.

| Candidate | Layers | Width | Heads | Context | Batch × accumulation | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| 13M baseline | 6 | 256 | 8 | 256 | 8 × 2 | 13,193,216 |
| 13M long context | 6 | 256 | 8 | 512 | 4 × 2 | 13,258,752 |
| 27M final | 8 | 384 | 8 | 256 | 8 × 2 | 26,877,696 |

### 8.1 The 13M baseline

The first full model established a working reference point: validation perplexity 103.38 and test perplexity 151.42. Short generations contained recognizable names, combat actions, speech markers, and verse-like structure. Repetition and malformed words remained common.

### 8.2 The context-length experiment

The second candidate changed context from 256 to 512. Its micro-batch fell from eight to four, preserving 4,096 target positions per optimizer update. This controlled the amount of training signal and isolated the practical effect of longer windows.

The longer context reduced throughput by 12.3%. Validation perplexity worsened from 103.38 to 112.94, and test perplexity worsened from 151.42 to 171.85. Manual reading did not reveal a consistent continuity gain.

Longer context gives a model access to more history, but the network still needs enough data and capacity to use that history. Under this corpus and compute budget, more independent 256-token windows produced a better model than fewer 512-token windows.

### 8.3 The capacity experiment

The final candidate restored the 256-token context and changed depth and width together: six blocks became eight, and 256 features became 384. The parameter count rose from 13.19M to 26.88M.

Relative to the baseline:

| Metric | 13M baseline | 27M final | Change |
|---|---:|---:|---:|
| Validation perplexity | 103.38 | 77.15 | -25.4% |
| Test perplexity | 151.42 | 116.45 | -23.1% |
| Validation top-1 accuracy | 26.64% | 31.55% | +4.91 pp |
| Test top-1 accuracy | 23.81% | 27.74% | +3.93 pp |
| Validation bits per UTF-8 byte | 0.946 | 0.886 | -6.3% |
| Test bits per UTF-8 byte | 1.048 | 0.993 | -5.2% |

The larger model processed about 8,378 targets per second instead of 13,565. Training time rose from 15.1 to 24.4 minutes. The quality improvement justified that local cost.

## 9. Evaluation metrics

### Cross-entropy loss

Loss is the mean negative log-probability of the correct next token. It is the training objective and the primary checkpoint-selection metric.

### Perplexity

Perplexity is the exponential of mean cross-entropy:

```math
\mathrm{PPL} = e^{\mathcal{L}}
```

It can be read as the effective number of equally plausible choices left to the model at each position. That intuition is approximate, but lower perplexity still means better next-token prediction under the same tokenizer and dataset.

Perplexity cannot be compared fairly across arbitrary tokenizers because token boundaries change the task. Every candidate here uses the same frozen tokenizer, so the comparison is valid inside this experiment.

### Top-1 and top-5 accuracy

Top-1 accuracy measures how often the most highly scored token equals the target. Top-5 accuracy checks whether the target appears among the five highest-scored tokens. Accuracy is easy to interpret, while cross-entropy also captures confidence and supplies useful training gradients.

### Bits per UTF-8 byte

The tokenizer compresses text into variable-length token sequences. Bits per UTF-8 byte connects token loss back to the original text size:

```math
\mathrm{BPB} = \frac{\mathcal{L}}{\ln 2}\cdot\frac{\mathrm{tokens}}{\mathrm{UTF\text{-}8\ bytes}}
```

This metric remains tied to the same source encoding, but it helps separate language prediction from token-count changes.

### Raw generation and memorization audit

Metrics cannot reveal every failure mode. Each accepted full-data model generates 20 fixed samples from five prompts and four seeds. Every sample uses temperature `0.8`, top-k `40`, and 256 new tokens.

The audit records repeated-trigram ratios and searches for identical consecutive word sequences after applying the same word extraction and case normalization to both texts. This ignores punctuation and whitespace differences; it is not a verbatim byte-match metric. The final model's mean repeated-trigram ratio was 4.21%, and its longest normalized training match contained nine words. The earlier seven-word result came from asymmetric normalization and was corrected by reanalyzing the same saved generations, without another model run. One sample still reached a 25.7% repeated-trigram ratio, so the mean does not hide the worst repetition case.

All fixed outputs live in [`reports/generation-audits`](reports/generation-audits). They are not a hand-picked showcase.

## 10. Generation

Generation starts with prompt token IDs and repeats five operations:

1. Keep only the last 256 tokens when the sequence exceeds the context window.
2. Run a forward pass and take the logits at the final position.
3. Divide logits by temperature.
4. Keep the top `k` logits and mask the rest to negative infinity.
5. Apply softmax, sample one token, append it, and repeat.

For vocabulary logit `z_i` and temperature `τ`, the sampling distribution is

```math
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
```

Lower temperature concentrates probability on the current leaders. Higher temperature gives lower-ranked tokens more chance. Top-k prevents sampling from the long tail of extremely unlikely tokens.

The current implementation recomputes the active context after every generated token. It does not use a KV cache. This keeps the generation path short and inspectable, though it leaves speed on the table.

## 11. Apple MPS execution

Every accepted training and generation run used PyTorch MPS on an M5 MacBook Pro with 16 GB of unified memory. The code checks three conditions before training:

- the installed PyTorch build contains MPS support;
- MPS is available on the current machine;
- `PYTORCH_ENABLE_MPS_FALLBACK` is disabled.

An unsupported operation fails instead of moving silently to CPU. That makes hardware and timing claims auditable.

| Candidate | Targets/s | Training time | Peak PyTorch MPS | Peak MPS driver |
|---|---:|---:|---:|---:|
| 13M / context 256 | 13,565 | 15.1 min | 468.5 MiB | 3.36 GiB |
| 13M / context 512 | 11,894 | 17.2 min | 470.9 MiB | 3.36 GiB |
| 27M / context 256 | 8,378 | 24.4 min | 959.4 MiB | 4.33 GiB |

The final model stayed well inside the machine's memory capacity. Data variety limited the experiment before memory did.

## 12. Reproducing the experiment

### Requirements

- Apple Silicon Mac with working PyTorch MPS support
- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Network access during data preparation

Install the pinned environment:

```bash
git clone https://github.com/nik1t7n/tiny-manas.git
cd tiny-manas
uv sync
```

Prepare the pinned corpus and tokenizer:

```bash
uv run tiny-manas prepare --config configs/manas01-27m.toml
```

This command downloads the real artifacts, verifies their checksums, extracts the selected epic text, checks the tokenizer round trip, and writes chronological split files under `data/processed/`. Corpus text and token arrays remain untracked.

Run the final training configuration:

```bash
uv run tiny-manas train --config configs/manas01-27m.toml
```

Each run creates an immutable timestamped directory under `runs/` with:

```text
config.toml
environment.json
architecture.json
metrics.jsonl
generation-initial.json
generation-final.json
best-model.pt
final-model.pt
summary.json
```

Evaluate the best checkpoint on the untouched test suffix:

```bash
uv run tiny-manas evaluate \
  --checkpoint runs/<run>/best-model.pt \
  --split test \
  --batches 100
```

Generate text:

```bash
uv run tiny-manas generate \
  --checkpoint runs/<run>/best-model.pt \
  --prompt "Манас" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 1337
```

Run the fixed 20-sample generation and memorization audit:

```bash
uv run tiny-manas audit \
  --checkpoint runs/<run>/best-model.pt \
  --output runs/<run>/generation-audit.json
```

Export a checkpoint without optimizer state:

```bash
uv run tiny-manas export \
  --checkpoint runs/<run>/best-model.pt \
  --output artifacts/tiny-manas-27m.pt
```

The accepted inference artifact is 107,546,203 bytes with SHA-256:

```text
cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1
```

The export preserves tied weights without writing a duplicate LM-head matrix.

## 13. Repository structure

```text
configs/                    experiment configurations
docs/                       decisions, sources, research log, serving notes
reports/                    committed metrics and raw generation audits
src/manas_gpt/
  audit.py                  generation and memorization audit
  cli.py                    command-line interface
  config.py                 typed TOML config and validation
  data.py                   verified download, extraction, tokenization, sampling
  experiment.py             train, evaluate, generate, checkpoint, export
  model.py                  Transformer implementation
RESULTS.md                  compact experiment report
IMPLEMENTATION_PLAN.md      preregistered questions and gates
```

The model code stays compact enough to inspect the complete forward, loss, backward, evaluation, and generation paths. The project uses PyTorch primitives instead of a high-level training framework.

## 14. Engineering decisions

### Keep the model decoder-only

Next-token generation needs causal self-attention and an LM head. An encoder, cross-attention, retrieval layer, or chat protocol would answer a different research question.

### Keep the full tokenizer vocabulary

The full 32,768-token vocabulary lets any output of the pinned tokenizer pass through training and inference. A split-specific compact vocabulary would reduce parameters, but it would add remapping logic and make prompts outside that split unsafe.

### Tie input and output embeddings

Weight tying removes 12.58M duplicate parameters. This matters in a model whose vocabulary matrix is large relative to its Transformer stack.

### Use pre-LayerNorm blocks

Pre-LayerNorm gives gradients a direct residual route through the block and keeps the implementation stable at this depth.

### Initialize residual outputs more gently

Linear and embedding weights begin with standard deviation `0.02`. Attention output projections and second FFN projections use

```math
\sigma_{residual} = \frac{0.02}{\sqrt{2L}}
```

Residual updates accumulate across two sublayers in each of `L` blocks. The smaller initialization keeps their combined early scale under control.

### Treat dropout 0.2 as a hypothesis

The one-batch correctness run uses dropout 0.0 because regularization would obstruct memorization. Full runs use 0.2, chosen as a practical starting point for a small corpus. The project does not claim that 0.2 is optimal without a controlled dropout experiment.

### Compare context at an equal target budget

The context-512 candidate halves the micro-batch, so both context candidates process 4,096 targets per optimizer update. Keeping batch eight would have changed context and training signal at the same time.

### Select by validation loss

The final training step is not automatically the best model. Validation loss selects the checkpoint; the test suffix remains separate until the candidate has been accepted.

### Record invalid experiments

The fixed-batch evaluator bug remains documented. Deleting failed attempts would make the final pipeline look cleaner than the work that produced it and would hide a useful measurement lesson.

## 15. Limits

- The model learned from one edition by one performer.
- The tokenizer was trained earlier on a broader Kyrgyz corpus, so model training is not isolated from tokenizer pretraining.
- Full-data runs use one random seed.
- Evaluation samples random windows rather than scoring every possible held-out window.
- The context window stops at 256 tokens.
- Learned absolute positions do not extrapolate beyond the configured context.
- Generation has no KV cache.
- The corpus license restricts redistribution and commercial use of the source text.

The strongest next experiment would add legally usable Manas-only text from another narrator or edition and reserve entire documents for evaluation. That would test whether the model learned broader epic structure or mainly the local patterns of `Manas01`.

## Evidence and references

- [`RESULTS.md`](RESULTS.md) contains the compact experimental report.
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) records hypotheses, forecasts, failures, and measured outcomes in order.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) records the decision history and rejected alternatives.
- [`docs/SOURCES.md`](docs/SOURCES.md) pins data, tokenizer, implementation references, checksums, and licenses.
- [`reports/results.json`](reports/results.json) contains the machine-readable accepted metrics.
- [`reports/generation-audits`](reports/generation-audits) contains all fixed raw generations.

Architecture references:

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- PyTorch, [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)

## Rights

Project code and writing are currently all rights reserved. The Manas-UdS source uses CC BY-NC-SA 4.0 and is not redistributed in this repository. See [`docs/SOURCES.md`](docs/SOURCES.md) for the exact source and tokenizer provenance.
