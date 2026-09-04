# Tiny Manas

**A 26.9M-parameter decoder-only Transformer trained from scratch on the Kyrgyz epic *Manas*.**

[Read the general-audience article](https://nik1t7n.com/essays/training-tiny-manas) · [Inspect the tokenizer](https://github.com/nik1t7n/kyrgyz-tokenizer) · [Browse the experiment results](#experimental-results)

Tiny Manas is a small language model that learns to continue one edition of the Kyrgyz epic *Manas*. I wrote the model and the training pipeline to study the complete language-model path without hiding the important mechanics behind a large framework: verified text acquisition, byte-level BPE tokenization, shifted training batches, causal self-attention, backpropagation, checkpoint selection, generation, and memorization analysis.

The original accepted model has 26,877,696 parameters. It contains eight Transformer blocks, eight attention heads, 384 features per token, and a context window of 256 tokens. I trained it on an M5 MacBook Pro with 16 GB of unified memory. That run took 24.4 minutes. The optimization series below builds on a frozen copy of this model and its evidence.

Tiny Manas is a research model, not a general Kyrgyz assistant. It has seen one edition of one epic. It can reproduce names, rhythm, speech patterns, and short chains of action from that source distribution. It cannot answer general questions, follow instructions, or maintain a reliable long narrative.

## Research question

The project asks a narrow question:

> How much Manas-like structure can a small, inspectable Transformer learn from one epic on one laptop?

This boundary determines every engineering choice in the repository. The model stays decoder-only. The data stays Manas-only. Full runs use Apple MPS and fail when MPS is unavailable. Each scaling experiment changes one major variable, and validation results decide which checkpoint survives.

The project follows the spirit of Tiny Shakespeare experiments, with a Kyrgyz source and a subword tokenizer. The goal is a complete and measurable language-model experiment at a scale where one person can still read the implementation from end to end.

## Results at a glance

These are the original full-data experiments from August 31, 2026. Their test scores belong to those checkpoints; I do not transfer them to newer weights.

| Model | Context | Parameters | Validation loss | Validation perplexity | Test loss | Test perplexity | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13M baseline | 256 | 13,193,216 | 4.6384 | 103.38 | 5.0201 | 151.42 | 15.1 min |
| 13M, longer context | 512 | 13,258,752 | 4.7268 | 112.94 | 5.1466 | 171.85 | 17.2 min |
| **Tiny Manas** | **256** | **26,877,696** | **4.3457** | **77.15** | **4.7575** | **116.45** | **24.4 min** |

Two conclusions survived the measurements:

1. Doubling the context window from 256 to 512 made the model slower and worsened held-out prediction.
2. Increasing depth and width reduced validation perplexity by 25.4% and test perplexity by 23.1% relative to the 13M baseline.

The final model generated recognizable epic-like text for short stretches. It still repeated names and formulaic phrases, invented malformed words, and lost track of events over longer passages. Across 20 fixed samples, the longest matching word sequence in the training text was nine words after normalizing case, punctuation and whitespace. A [corrected audit](docs/experiments/00-audit-correction.md) replaced the earlier undercount of seven. These samples contained no long copied word sequences; they do not prove that the model never memorizes other passages.

### Optimization series: current accepted changes

On September 4 I froze the original source, configurations, data, tokenizer and checkpoints before changing the implementation. The Git tag is `tiny-manas-pre-optimization-20260904`; the [baseline manifest](docs/experiments/00-baseline-manifest.json) records artifact hashes. This is a local recovery snapshot, not an off-site backup.

| Experiment | Measured result on the M5 | Decision |
|---|---|---|
| Last-position LM head | B=1,T=256 forward: 7.307 → 5.894 ms; logits: 32 MiB → 128 KiB | Use during generation |
| BF16 training | Matched full runs: 1548.04 → 1229.40 s; validation loss change +0.0000464 | Use for 27M training; keep evaluation/inference FP32 |
| `torch.compile` | Original compiled gradients failed parity; corrected research variant passed but took 3.02% longer per update | Keep eager execution |
| Activation checkpointing | 38.46% less sampled live allocation; 30.63% longer updates | Offer as an opt-in; leave off for the current model |
| 32k / 16k / 8k tokenizer | Smaller vocabularies improved training bytes/s by 16.09% / 20.50%, but validation bits/byte worsened by 4.15% / 3.84% versus the incumbent | Keep the original 32k tokenizer and training recipe |
| RoPE | Mathematical checks passed; median training updates were 12.27% slower | Keep learned positions; no full quality run |
| RMSNorm | Formula and gradient checks passed; median training updates were 10.71% slower | Keep LayerNorm; no full quality run |
| SwiGLU | Matched 3,000-update run: validation loss 4.34578 → 4.42742; more repeated trigrams | Keep GELU |
| KV cache | Same weights and matching predictions; 1.1566x short-generation speedup, 1.0165x near context overflow | Use request-local prefill/decode with explicit window rebuilding |
| GQA, 8 Query / 2 KV heads | 75% smaller persistent cache; after matched adaptation, validation loss +0.19402 nats versus MHA | Keep eight independent KV heads |
| Fused/chunked output loss | Selected workload fits; no measured bottleneck triggered this conditional experiment | Keep ordinary cross-entropy; no speed claim |

The fresh FP32/BF16 runs both processed 12,288,000 training targets, with the same initialization, batches and schedule. BF16 cut training-loop wall time by 20.58% relative to that fresh FP32 control. Comparing against the older 24.4-minute run would mix precision with conditions from another session. I inspected all 20 fixed BF16 continuations: local epic patterns and the existing repetition problems remain. The speed improvement does not establish better prose.

All numbered decisions are complete. The [experiment checklist](docs/experiments/README.md) and [accepted-state record](docs/experiments/accepted-state.json) record the outcomes. After selection, the accepted BF16-trained checkpoint was evaluated once on 100 test batches: loss **4.75756**, perplexity **116.46**, top-1 **27.74%**, top-5 **45.27%**. This result was not used for model selection. The optimization improves execution cost, not narrative quality: the 20-output audit still found name loops and broken event continuity, with mean repeated trigrams 4.10%, worst 31.11%, and a longest normalized training-text match of nine words.

The original run's table above remains historical. Release status and rollback evidence are tracked in [the deployment record](docs/experiments/12-release.md).

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

Perplexity cannot be compared fairly across arbitrary tokenizers because token boundaries change the task. The original capacity experiments use the same frozen tokenizer. The later vocabulary experiment instead compares negative log-probability over exactly the same held-out text bytes.

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
2. Prefill the context once, then decode the appended token using cached Key/Value tensors. Rebuild from the cropped window after overflow. Apply the LM head to the final position only.
3. Divide logits by temperature.
4. Keep the top `k` logits and mask the rest to negative infinity.
5. Apply softmax, sample one token, append it, and repeat.

For vocabulary logit `z_i` and temperature `τ`, the sampling distribution is

```math
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
```

Lower temperature concentrates probability on the current leaders. Higher temperature gives lower-ranked tokens more chance. Top-k prevents sampling from the long tail of extremely unlikely tokens.

The generation method uses a request-local KV cache. The full-sequence forward pass remains available for training and evaluation; `generate(..., use_cache=False)` selects the uncached generation reference.

### Cache past attention inputs

During prefill, each layer processes the prompt and keeps its Key and Value tensors. During decode, the model processes one new token, appends its Key and Value, and compares its Query with the stored keys. Causality lets us reuse past states while their positions and preceding context remain unchanged. This follows the per-layer caching mechanism described in the [Hugging Face cache guide](https://huggingface.co/docs/transformers/main/cache_explanation).

For one request, the retained storage is

```math
2 L H_{KV} T d \times \mathrm{bytes\ per\ value}.
```

Here, `L=8`, `H_KV=8`, `T=256`, `d=48`, and inference uses FP32. The cache owns 6,291,456 bytes (6 MiB). The implementation clones Key/Value views after prefill so their storage does not also retain discarded Query values. Each request creates its own cache; it cannot inherit another request's context.

Single-token decode has no future keys, so it uses attention without a causal mask. Applying an upper-left-aligned rectangular causal mask to that one query would hide almost the entire past. Prefill still uses ordinary causal attention.

The context boundary needs care. Once we crop the oldest token, the surviving tokens receive new learned position IDs, and their deeper states must no longer contain the evicted context. I rebuild the cache from the cropped 256-token window. Merely dropping the oldest Key/Value would change the model's predictions.

On the M5, 20 real validation prompts passed full-logit and greedy-token comparisons, including overflow; worst absolute logit difference was 7.15e-6. Seeded sampling and independent-request checks passed. A 32-token prompt followed by 64 generated tokens took 0.19675 seconds uncached and 0.17011 cached, a 1.1566x speedup. With a 248-token prompt, window rebuilding reduced the gain to 1.0165x. One decode step after 128 past positions took 2.5765 ms instead of 3.3177 ms. These measurements explain both the adoption and its limit. See [experiment 09](docs/experiments/09-kv-cache.md).

### Project only the position we need

Training needs a next-token prediction at each input position. Generation needs one prediction at the end. The original forward pass still constructed `[B,T,V]` logits before discarding the earlier positions. I moved the slice before the LM head when generation requests `last_position_only=True`:

```text
Transformer output:    [B, T, C]
Take the last vector:  [B, 1, C]
Project to vocabulary: [B, 1, V]
```

At B=1,T=256,V=32,768 with FP32 output, this reduces the logits tensor from 32 MiB to 128 KiB. It leaves attention and FFN computation over the context intact. The measured whole-forward improvement was 1.24x at that shape, not 256x. At T=1, the extra slice provided no savings and the measured path was slightly slower.

I compared full and final-position logits on real validation inputs within absolute/relative tolerance 1e-4. Twenty greedy continuations matched, and seeded stochastic generation matched across the context-cropping boundary. Training keeps full logits; the API rejects a final-position request with training targets. See [experiment 01](docs/experiments/01-last-position.md).

## 11. Apple MPS execution

Every accepted training and generation run used PyTorch MPS on an M5 MacBook Pro with 16 GB of unified memory. The code checks three conditions before training:

- the installed PyTorch build contains MPS support;
- MPS is available on the current machine;
- `PYTORCH_ENABLE_MPS_FALLBACK` is disabled.

An unsupported operation fails instead of moving silently to CPU. That makes hardware and timing claims auditable.

| Original candidate | Targets/s | Training time | Maximum sampled PyTorch MPS | Maximum sampled MPS driver |
|---|---:|---:|---:|---:|
| 13M / context 256 | 13,565 | 15.1 min | 468.5 MiB | 3.36 GiB |
| 13M / context 512 | 11,894 | 17.2 min | 470.9 MiB | 3.36 GiB |
| 27M / context 256 | 8,378 | 24.4 min | 959.4 MiB | 4.33 GiB |

The original runs sampled allocation after optimizer updates. These values are lower bounds on live memory peaks; they do not capture all intermediate allocations. The model fit within the machine's memory capacity, and the remaining quality problems gave us reasons to investigate data variety.

### BF16 forward computation, FP32 training state

BF16 stores a floating-point value in 16 bits: one sign bit, eight exponent bits and seven fraction bits. FP32 uses the same eight exponent bits but 23 fraction bits. BF16 retains a similar representable range while rounding values more coarsely.

I enable autocast around the training forward pass and loss. PyTorch chooses the supported operation dtypes; it does not turn every tensor into BF16. In the measured path, logits used BF16 while the loss, model parameters and AdamW state stayed FP32. Backward and optimizer updates run outside the autocast context. Evaluation and generation use FP32 so the quality comparison does not also change the evaluation arithmetic.

The full 3,000-step comparison measured validation loss 4.345732722 for FP32 and 4.345779147 for BF16 on the same 100 held-out batches. The difference was +0.000046425 nats, within the preregistered +0.05 ceiling. Training throughput rose from 7,937.77 to 9,995.08 target positions/s, including the training loop's evaluation overhead.

Memory savings depend on what we measure. The short probe sampled live allocations inside updates and found only a 2.1% reduction. The full runs sampled after updates and found a 33.83% reduction. Driver allocation changed little. These are different sampling protocols, and neither supports a claim that BF16 halves total training memory. The FP32 weights and optimizer state still occupy memory. [Experiment 02](docs/experiments/02-bf16.md) preserves both measurements and the raw-run paths.

### Checkpointing trades recomputation for memory

During backward, the model needs intermediate forward results to calculate gradients. Activation checkpointing retains a block's input and recomputes its internal operations when backward reaches that block. It reduces retained activations while doing more work.

Dropout makes recomputation a correctness issue. Recomputing with a different random mask gives gradients for a different forward computation. In the first test on this installed PyTorch/MPS version, standard checkpointing changed the MPS random-generator state and produced relative gradient error 0.6758. I stopped that path before timing it.

The opt-in implementation now captures the MPS RNG state at each checkpointed forward, restores it for recomputation, and then restores the outer RNG state. The corrected real-batch check retained the CPU/MPS RNG states and reduced relative gradient difference to about 6e-9. This implementation targets the project's eager MPS path; the experiment does not validate other devices or compilation.

Across 35 real updates per mode, sampled live allocation fell from 2.100 to 1.292 GB, while median update time rose from 0.31760 to 0.41488 seconds. I retained the option for a future memory-constrained run and left `training.activation_checkpointing = false` for the current model. It already fits, so paying 30.63% more update time would not help this run. See [experiment 04](docs/experiments/04-checkpointing.md).

### Compilation did not win on this workload

The first compiled path produced plausible losses but incorrect gradients. Its learned-position gradient norm was about eight times the eager result at batch size eight. A research-only rewrite of the position lookup reduced the overall relative gradient error to 0.00651, within the BF16 probe tolerance.

That corrected candidate still took 0.33343 seconds per warmed update versus 0.32365 for eager execution, plus a compilation startup cost. I kept eager execution and did not adopt the position-lookup rewrite. The [compile report](docs/experiments/03-compile.md) records the failed attempts, gradient diagnosis and timings. This result concerns the installed Metal compiler and this model; it does not predict CUDA performance.

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
  --output artifacts/tiny-manas-27m-bf16-20260904.pt
```

The optimized inference artifact is 107,547,815 bytes with SHA-256:

```text
4c6f70883564df6c46849c0849f38b06195b4dcaba3bde2572ef60eec4cf3494
```

The export preserves tied weights without writing a duplicate LM-head matrix. The original `artifacts/tiny-manas-27m.pt` remains unchanged for rollback: 107,546,203 bytes, SHA-256 `cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1`. Weights and licensed corpus data are not committed to Git.

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
  kv_cache.py               request-local prefill/decode and cache storage
RESULTS.md                  compact experiment report
IMPLEMENTATION_PLAN.md      preregistered questions and gates
```

The model code stays compact enough to inspect the complete forward, loss, backward, evaluation, and generation paths. The project uses PyTorch primitives instead of a high-level training framework.

## 14. Engineering decisions

### Keep the model decoder-only

Next-token generation needs causal self-attention and an LM head. An encoder, cross-attention, retrieval layer, or chat protocol would answer a different research question.

### Keep the full tokenizer vocabulary

The full 32,768-token vocabulary lets any output of the pinned tokenizer pass through training and inference. A split-specific compact vocabulary would reduce parameters, but it would add remapping logic and make prompts outside that split unsafe.

I also tested a different, safe way to reduce vocabulary: retain all 256 byte tokens and the first BPE merges from the master tokenizer, giving complete 16k and 8k tokenizers that can still encode arbitrary text. This changes segmentation, not just the output matrix. Smaller tokens mean a longer sequence for the same passage, and a 256-token window then covers fewer text bytes.

To compare them, I retokenized the same chronological text boundaries and scored exactly the same validation bytes after a shared context prefix. All three fresh arms saw 30 passes over the same training text: 87,729,780 scored bytes each. They required different numbers of optimizer updates because their token counts differed. Shared Transformer weights started identically; smaller embedding tables retained the corresponding initial rows. Learning-rate progress followed bytes processed rather than raw update number.

| Vocabulary | Parameters | Validation bits/byte | Training bytes/s | Sampled allocated bytes |
|---|---:|---:|---:|---:|
| Accepted 32k checkpoint, rescored | 26,877,696 | **0.876442** | — | — |
| Fresh 32k control | 26,877,696 | 0.921290 | 82,252.65 | 2,253,916,672 |
| Fresh 16k | 20,586,240 | 0.912849 | 95,483.53 | 1,805,126,144 |
| Fresh 8k | 17,440,512 | 0.910118 | 99,114.85 | 1,691,879,936 |

The smaller vocabularies beat the fresh 32k control on both prediction and resources. Neither beat the already accepted model within the allowed 1% prediction-loss increase. The new equal-text training recipe also differed from the incumbent's random windows, so this does **not** isolate vocabulary as the cause of the regression against that incumbent. It rejects these complete candidate recipes under the fixed budget, not smaller tokenizers in general. I retained the existing model instead of replacing it with a faster but worse candidate. All 60 new fixed generations were inspected. With 256 generated tokens each, smaller vocabularies produced fewer text bytes; their lower repetition scores did not establish better narrative coherence. [Experiment 05](docs/experiments/05-tokenizer.md) records the byte boundaries, hashes, budgets and raw-output review.

### Measure architectural substitutions on the target hardware

RoPE encodes position by rotating pairs of Query and Key coordinates before attention. Their dot products then depend on relative position through the difference between rotation angles. The candidate removed the learned position table, saving 98,304 parameters. Norm preservation, a common-position-offset check and causal masking all passed. On this eager MPS implementation, however, median updates increased from 0.30604 to 0.34360 seconds: 12.27%, above the preregistered 10% ceiling. I stopped before full training. This is a cost rejection of the measured implementation, not evidence that RoPE cannot improve language modeling. See [experiment 06](docs/experiments/06-rope.md).

RMSNorm controls scale without subtracting the feature mean. For a token vector, it divides each value by the square root of the mean squared value plus epsilon, then applies a learned feature scale. It has no learned offset in this candidate, removing 6,528 parameters across the 17 normalization layers. The installed native operation matched the explicit FP32 formula and its gradients, but median updates increased from 0.30857 to 0.34160 seconds: 10.71%, above the 5% ceiling. Fewer mathematical operations did not translate into a faster training step on this backend. I retained LayerNorm without claiming a quality comparison that was never run. See [experiment 07](docs/experiments/07-rmsnorm.md).

### Test a learned FFN gate without adding a larger network

The SwiGLU candidate replaced the GELU FFN with two input projections and a multiplicative gate:

```math
\mathrm{FFN}(x) = (\mathrm{SiLU}(xW_g^T+b_g) \odot (xW_u^T+b_u))W_d^T+b_d.
```

I used hidden width 1,024 instead of GELU's 1,536. Three matrices of width 1,024 have the same weight count as two matrices of width 1,536; biases added only 4,096 parameters across the model. The unchanged attention and embedding tensors started identically to the GELU control, while a separate seeded generator initialized the new FFN projections.

The candidate passed the numerical and cost gate, with 3.80% longer updates and 8.82% more sampled allocation, then completed all 3,000 updates. Its independent validation loss was 4.42742, worse than GELU's 4.34578 and the acceptance ceiling of 4.32578. Across 20 fixed continuations, mean repeated trigrams rose from 4.10% to 6.95%; one sample reached 52.78%. I inspected the full texts and found pronounced name and action loops. I retained GELU. A promising intermediate checkpoint-selection score did not override the independent result. [Experiment 08](docs/experiments/08-swiglu.md) includes the complete run and audit.

### Share KV heads only if adaptation preserves quality

GQA retains eight Query heads but shares two Key/Value heads between groups of four queries. Cache storage follows the number of KV heads, not Query heads: at context 256, eight layers and FP32, it falls from 6 MiB to 1.5 MiB. I initialized the new projections by averaging each group of four existing KV heads, then adapted both this candidate and an unchanged MHA control for 150 updates on the same batches. This is a bounded conversion experiment, not GQA training from scratch.

Both cache parity checks passed. Paired decode time increased only 1.10%, but validation loss reached 4.49705 versus adapted MHA's 4.30303, exceeding the allowed +0.05 nats. All 40 outputs were read: GQA repeated fewer trigrams but still lost actors and produced malformed phrases. I retained the original MHA checkpoint, not either adapted arm. The implementation explicitly expands grouped KV tensors for attention; persistent cache savings are not a claim of fused-kernel savings. See [experiment 10](docs/experiments/10-gqa.md).

### Do not optimize an unmeasured output-loss bottleneck

One BF16 logits tensor at batch 8, context 256 and vocabulary 32,768 contains 67,108,864 values, or 128 MiB. It is substantial, but the accepted training workload fits: sampled live allocation was about 2.10 GB against an approximately 8.26 GB process cap. The conditional fused/chunked-loss experiment was therefore assessed and not triggered. Ordinary cross-entropy remains in use; no benchmark of Cut Cross-Entropy is claimed. See [experiment 11](docs/experiments/11-output-loss.md).

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
- KV caching accelerates growing contexts, but exact crop semantics require rebuilding after the 256-token window fills.
- The corpus license restricts redistribution and commercial use of the source text.

The strongest next experiment would add legally usable Manas-only text from another narrator or edition and reserve entire documents for evaluation. That would test whether the model learned broader epic structure or mainly the local patterns of `Manas01`.

## Evidence and references

- [`RESULTS.md`](RESULTS.md) contains the compact experimental report.
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) records hypotheses, forecasts, failures, and measured outcomes in order.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) records the decision history and rejected alternatives.
- [`docs/SOURCES.md`](docs/SOURCES.md) pins data, tokenizer, implementation references, checksums, and licenses.
- [`docs/experiments/README.md`](docs/experiments/README.md) tracks the optimization series, its individual reports and acceptance criteria.
- [`reports/results.json`](reports/results.json) contains the machine-readable accepted metrics.
- [`reports/generation-audits`](reports/generation-audits) contains all fixed raw generations.

Architecture references:

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- PyTorch, [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)

## Rights

Project code and writing are currently all rights reserved. The Manas-UdS source uses CC BY-NC-SA 4.0 and is not redistributed in this repository. See [`docs/SOURCES.md`](docs/SOURCES.md) for the exact source and tokenizer provenance.
