# Tiny Manas / Manas-GPT Implementation Plan

## Status

- Planning only. No model or training code has been implemented.
- Primary machine: Apple Silicon MacBook Pro, M5, 16 GB unified memory.
- Training backend: PyTorch on `mps`; no silent CPU fallback.
- Project scope: a small decoder-only language model trained only on *Manas*.
- Main comparison: the existing neural bigram baseline versus a causal Transformer on the same text and tokenizer.

## 1. Desired outcome

Build a small, real, reproducible Transformer that generates text in the style of *Manas* and clearly improves on the existing bigram model.

The first model does not need to be a general Kyrgyz assistant. It should demonstrate one narrower result honestly: access to a longer context allows a Transformer to learn more coherent local grammar, phrases, names, and short narrative structure than a model that sees only the immediately previous token.

The project is inspired by the role of Tiny Shakespeare in educational language-model work, but it is not a translation or cosmetic copy. *Manas* gives the experiment a local corpus, a distinct linguistic structure, a real existing bigram baseline, and research questions that matter specifically for Kyrgyz text.

The working public name is **Tiny Manas**. The repository and model family name is **Manas-GPT**.

## 2. Definition of success

The project succeeds only if all of the following are true:

1. The complete training path runs on real *Manas* text through PyTorch `mps`.
2. The model can intentionally overfit one real batch when regularization is disabled.
3. On an untouched validation split, the Transformer beats the bigram baseline in cross-entropy and perplexity under the same tokenization and target vocabulary.
4. Raw generations show a visible improvement beyond one-token transitions.
5. The result is reproducible from a pinned source, tokenizer, configuration, seed, and commit.
6. Failures, memorization, unstable runs, and negative results are documented instead of hidden.
7. Training stays practical on the owner's M5 / 16 GB machine.

A lower training loss alone is not success. A fluent-looking cherry-picked sample alone is also not success.

## 3. Explicit non-goals

The first project will not attempt to build:

- a general Kyrgyz-Russian base model;
- an instruction-following assistant;
- a chatbot with roles or chat-protocol tokens;
- an encoder-decoder model or cross-attention system;
- a retrieval system;
- a production inference service during the research phase;
- a distributed or cloud training stack;
- a benchmark claim against commercial LLMs;
- a synthetic replacement for unavailable real data.

The website article and interactive public demo are later publication tasks. They must not distort architecture choices or delay the core experiment.

## 4. Existing inputs

### 4.1 Corpus

The initial controlled dataset will use the same pinned `Manas01` source used by the existing `manas-bigram` project: the version performed by Sayakbai Karalaev from Manas-UdS.

The first comparison must preserve the existing text boundary and source provenance. The 10,000-token pilot remains useful as the smallest controlled bridge from the old model to the new model. Later runs may use the full cleaned `Manas01` text and then more of the pinned Manas-UdS collection.

No third-party corpus text will be committed. The source URL, revision, license, extraction boundary, and checksum will be recorded in `docs/SOURCES.md`.

### 4.2 Tokenizer

The first controlled comparison will reuse the pinned Kyrgyz byte-level BPE tokenizer already used by `manas-bigram`.

The model will not retrain or silently change the tokenizer during an architecture experiment. A document-boundary token may be added at the model protocol level when multiple independent documents are packed. That decision, its ID, and its effect on the model vocabulary must be explicit.

### 4.3 Bigram baseline

The existing neural bigram result is the first baseline:

- validation cross-entropy: `6.380`;
- validation perplexity: `590`;
- validation top-1 accuracy: `17.1%`;
- validation top-5 accuracy: `32.5%`;
- pilot data: first 10,000 real BPE tokens;
- split: 9,000 train / 1,000 validation tokens;
- active target vocabulary: 2,525 token IDs.

The baseline must be rerun or loaded from its exact recorded artifact only when needed for the controlled comparison. Its data split and token mapping must not be reconstructed approximately.

## 5. Model architecture

The target model is a decoder-only, autoregressive Transformer.

```text
Token IDs
    ↓
Token embeddings + learned position embeddings
    ↓
N × pre-normalized Transformer blocks
    ├── LayerNorm
    ├── causal multi-head self-attention
    ├── attention/output dropout
    ├── residual addition
    ├── LayerNorm
    ├── feed-forward network
    ├── FFN output dropout
    └── residual addition
    ↓
Final LayerNorm
    ↓
Tied language-model head
    ↓
Logits over the target vocabulary
```

There is no encoder and no cross-attention. During training, causal masking prevents every position from seeing future target tokens. During generation, the model repeatedly samples one next token and appends it to the context.

### 5.1 Initial full-model candidate

The first full Manas candidate will start from this declared configuration:

| Setting | Initial value | Reason |
| --- | ---: | --- |
| Transformer blocks | 6 | Deep enough to test repeated contextual processing while remaining inspectable |
| Embedding width | 256 | Keeps the model small enough for local iteration |
| Attention heads | 8 | Gives 32 features per head and divides the embedding width exactly |
| Context length | 256 tokens | Meaningfully exceeds a bigram while keeping attention cost bounded |
| FFN width | 1,024 | Standard four-times expansion, easy to reason about |
| Activation | GELU | A simple established GPT-style baseline |
| Normalization | pre-LayerNorm | Stable small-model training and matches the studied architecture |
| Position method | learned absolute embeddings | Matches the architecture already studied; RoPE is a later controlled change |
| Weight tying | token embedding = LM-head weight | Avoids duplicating a large vocabulary matrix |
| Training dropout | `0.2` | Owner-selected initial regularization candidate |
| One-batch dropout | `0.0` | Regularization must not obstruct the correctness gate |

With the full 32K tokenizer vocabulary, this configuration is expected to be roughly 13 million parameters when embedding and output weights are tied. The exact parameter count will be produced by the implementation, not copied from this estimate.

The 10,000-token bridge experiment may use the existing compact 2,525-ID target space to make the first architecture comparison cheap and exact. Full-corpus runs will define their vocabulary protocol separately and will not silently inherit the pilot compaction.

### 5.2 Dropout interpretation

The architecture contains two distinct dropout locations:

1. attention-probability dropout after attention softmax;
2. output/residual dropout after the attention projection and after the FFN output.

The macro and attention-internal diagrams may show the same output dropout at two levels of detail. The implementation must apply it once, not twice.

The `0.2` value is a starting hypothesis, not a proven optimum. If it suppresses learning or the model does not overfit a larger training subset after the correctness gate, a controlled `0.0` versus `0.2` comparison will test the assumption.

## 6. Training path

The training system will remain compact and explicit:

```text
Pinned Manas source
    ↓
verified extraction and cleaning
    ↓
frozen BPE tokenizer
    ↓
train / validation / test split
    ↓
packed token sequences
    ↓
X and one-token-shifted Y batches
    ↓
Transformer forward pass
    ↓
cross-entropy from logits and Y
    ↓
backpropagation
    ↓
gradient clipping
    ↓
AdamW parameter update
    ↓
periodic validation and checkpointing
```

The initial optimizer candidate will be AdamW with an explicit learning rate, weight decay, beta values, warmup, and decay schedule stored in the run configuration. Values will be adopted from a documented GPT baseline and then adjusted only through declared experiments. No optimizer tuning result will be presented as an architecture result.

The training command must capture:

- Git commit;
- exact config and seed;
- Python, PyTorch, and tokenizer versions;
- device and MPS availability;
- corpus and tokenizer checksums;
- parameter count;
- batch size, context length, and tokens per update;
- optimizer and learning-rate state;
- elapsed time and peak allocated MPS memory;
- train and validation metrics;
- best and final checkpoint identities.

MPS failure must stop the run with the real error. The project will not enable `PYTORCH_ENABLE_MPS_FALLBACK`.

## 7. Experiment sequence

Each experiment changes one main variable. Its forecast must be written before the run and its raw outputs must be inspected afterward.

### E0 — Data and device integrity

**Question:** Can the exact Manas source, tokenizer, split, and MPS device be reproduced before model work begins?

**Checks:**

- source and tokenizer hashes match;
- tokenization round-trips sampled text exactly;
- splits contain the expected number of tokens;
- `mps` is available;
- batches have shapes `(B, T)` and targets are shifted by one token;
- no future token appears in the input for its own target position.

**Failure meaning:** the model experiment is blocked; no substitute data or CPU fallback is allowed.

### E1 — One-batch overfit correctness gate

**Question:** Is the complete forward, loss, backward, optimizer, and generation path wired correctly?

**Setup:**

- one real batch from *Manas*;
- dropout `0.0`;
- no data augmentation;
- fixed seed;
- repeated updates on the same batch.

**Forecast:** training loss should fall close to zero and next-token accuracy on that batch should approach 100%.

**What would change the belief:** a persistent loss floor, unstable gradients, or incorrect generation from the memorized prefix means the implementation is wrong or the optimization setup is unsuitable. Scaling is forbidden until this is understood.

### E2 — Tiny bridge: Transformer versus bigram

**Question:** Does contextual self-attention improve over one-token memory on the exact existing 10,000-token pilot?

**Controlled variables:**

- same 9,000/1,000 split;
- same tokenizer and active ID mapping;
- same target positions;
- same evaluation implementation where possible.

**Main changed variable:** bigram table versus causal Transformer.

**Forecast:** the Transformer should beat bigram validation cross-entropy `6.380` and produce fewer purely local name/punctuation fragments.

**What would weaken the hypothesis:** lower training loss without lower validation loss, or generations that remain indistinguishable from the bigram failure pattern.

### E3 — Full `Manas01` base run

**Question:** Does more real Manas text allow the same architecture to learn more coherent continuations?

**Main changed variable:** data volume, while architecture and tokenizer remain fixed.

**Setup:**

- full cleaned `Manas01` epic text;
- split at stable textual boundaries rather than randomly mixing adjacent token windows;
- dropout `0.2`;
- fixed maximum training-token budget;
- checkpoint selected only by validation loss.

**Forecast:** validation loss and raw continuity should improve over the 10K pilot, while exact memorization of long training passages should become less dominant.

### E4 — Full Manas-UdS data scale

**Question:** Does adding more pinned *Manas* material improve generalization without changing the model?

**Main changed variable:** `Manas01` versus the approved wider Manas-UdS subset.

Before this run, duplicated editions, scholarly metadata, tables of contents, parallel versions, and source boundaries must be audited. More bytes are not automatically better data.

### E5 — Context scale

**Question:** Does increasing usable context improve continuity enough to justify its quadratic attention cost?

**Comparison:** context `256` versus `512`, with model width, depth, data, tokenizer, optimizer, and training-token budget held fixed as far as practical.

**Evidence sought:** validation loss, memory use, training speed, repetition, and continuity across longer spans.

### E6 — Model scale within the Mac limit

Only after the data and context experiments are stable, compare the 13M candidate with one larger configuration that still fits comfortably on the M5 / 16 GB machine.

One likely candidate is an approximately 25–30M parameter model produced by increasing depth and width together under an explicitly recorded configuration. The exact model will be chosen from measured MPS memory and throughput, not guessed in advance.

The larger model wins only if the quality gain justifies slower iteration and remains reproducible locally.

## 8. Evaluation

### 8.1 Numeric metrics

Every accepted run will report:

- train and validation cross-entropy;
- validation perplexity;
- top-1 and top-5 next-token accuracy;
- bits per UTF-8 byte for a tokenizer-independent view of compression and prediction together;
- tokens processed per second;
- total training time;
- peak MPS memory;
- generation tokens per second;
- parameter count and checkpoint size.

Perplexity will only be compared directly when tokenization and target vocabulary are the same.

### 8.2 Raw generation protocol

The same fixed prompts and random seeds will be evaluated across comparable runs. Prompts will include:

- character and hero names;
- common epic openings;
- dialogue-like fragments;
- action sequences;
- rare or unseen validation prefixes;
- empty or minimal prompts.

At least 20 generations from each final candidate will be read and assigned to failure buckets such as:

- immediate repetition;
- punctuation loops;
- broken word endings;
- name or subject drift;
- locally grammatical but globally incoherent text;
- copied training passage;
- abrupt language or register change;
- stable multi-sentence continuation.

No best-looking sample will stand in for the distribution.

### 8.3 Memorization audit

Generated passages will be compared against the training corpus for long exact or near-exact overlaps. A model that reproduces memorized paragraphs may have low loss without having learned useful composition.

The report will distinguish:

- expected style and phrase reuse;
- suspicious long-span memorization;
- genuinely novel combinations;
- cases where the distinction is uncertain.

## 9. Stopping and checkpoint policy

Each run will have a hard maximum step or token budget before it starts.

The best checkpoint is selected by validation loss. Training may stop early after a declared number of validation evaluations without meaningful improvement. The final and best checkpoints will not be confused.

A run stops immediately on:

- non-finite loss or gradients;
- MPS out-of-memory;
- checksum or split mismatch;
- checkpoint corruption;
- accidental CPU execution;
- evidence that validation data entered training.

Failures remain in the research log with their configs and observed errors.

## 10. Documentation contract

Documentation is part of the experiment, not retrospective decoration.

Before every run, `docs/RESEARCH_LOG.md` will record:

- research question;
- why it matters;
- hypothesis;
- expected direction and approximate result;
- the observation that would weaken the belief;
- main changed variable;
- controlled variables;
- exact run config.

After every run, the same entry will add:

- measured results;
- raw sample locations;
- failures and surprises;
- forecast versus result;
- what was observed directly;
- what remains interpretation;
- updated belief;
- next cheapest informative experiment.

`docs/DECISIONS.md` will explain durable engineering choices and rejected alternatives. Each entry will contain:

- decision;
- context;
- alternatives considered;
- why the selected option won;
- trade-offs accepted;
- evidence or source;
- condition that would trigger reconsideration.

`docs/SOURCES.md` will hold provenance, licenses, revisions, retrieval commands, checksums, and attribution.

Machine-readable run summaries will live under `reports/`. Large corpora, token arrays, checkpoints, and raw generations will remain local ignored artifacts.

## 11. Planned repository structure

```text
manas-gpt/
├── AGENTS.md
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── configs/
│   ├── overfit-one-batch.toml
│   ├── pilot-10k.toml
│   └── manas01-base.toml
├── src/manas_gpt/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── artifacts.py
├── docs/
│   ├── RESEARCH_LOG.md
│   ├── DECISIONS.md
│   ├── SOURCES.md
│   ├── ARCHITECTURE.md
│   └── REPORT.md
├── reports/
└── artifacts/                 # ignored local data, checkpoints, and raw outputs
```

This is a target structure, not permission to scaffold files before implementation begins. Files will be added only when their phase starts.

## 12. Writing and article plan

After the final accepted experiment, a long English article will be written for `nik1t7n.com` in the same visual and semi-academic style as the tokenizer and bigram articles, but with simple explanations for a general reader.

The article will cover:

1. why Tiny Shakespeare inspired Tiny Manas;
2. what the bigram model could and could not learn;
3. the decoder-only Transformer in plain language;
4. the one-batch overfit correctness proof;
5. the first failed and successful runs;
6. training curves and parameter/memory accounting;
7. bigram versus Transformer generations shown side by side;
8. what more text, context, and model size changed;
9. memorization and failure analysis;
10. what the model actually demonstrates and what it does not.

The article will be written from the chronological research log. Commit hashes, CLI boilerplate, and internal implementation noise will not dominate the public story.

## 13. Optional interactive website demo

This phase begins only after a model checkpoint passes the research acceptance gates and its source-license implications are reviewed.

The real deployment path will contain:

```text
nik1t7n.com UI
    ↓
rate-limited generation endpoint
    ↓
bounded inference queue
    ↓
loaded accepted Manas-GPT checkpoint
    ↓
generated continuation
```

The public control surface may expose:

- a short seed text;
- temperature within a safe bounded range;
- top-k within a safe bounded range;
- a small maximum number of generated tokens;
- a generate button and readable output.

Operational controls must include:

- per-IP and global rate limits;
- maximum prompt bytes and token count;
- maximum new-token count;
- request timeout;
- bounded concurrency or a queue;
- checkpoint loaded once rather than per request;
- basic abuse and error logging without storing unnecessary personal text;
- server memory and latency measurements;
- explicit failure rather than a mock response if inference is unavailable.

The website demo is a publication layer, not an experiment metric. Its scope will remain small and will not influence the training objective.

## 14. Sources and implementation references

The implementation may study these sources, with attribution and a clear separation between replication, adaptation, and original experiment design:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer foundations.
- [nanoGPT model](https://github.com/karpathy/nanoGPT/blob/master/model.py) — compact GPT implementation reference.
- [nanoGPT Tiny Shakespeare configuration](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py) — bounded educational training reference.
- [char-rnn Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) — the historical analogy for Tiny Manas.
- [TinyStories](https://arxiv.org/abs/2305.07759) — evidence and limitations for coherent behavior in very small models on constrained data.
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) — official Apple GPU execution path.

Primary papers, current source code, appendices, licenses, and negative results will be read before adopting a choice. Tutorials and summaries are navigation, not evidence.

## 15. First action after approval

Implementation starts with **E0 and E1 only**:

1. create the minimal repository contract and dependency file;
2. reproduce the pinned Manas/tokenizer inputs;
3. implement the smallest complete decoder-only forward path;
4. run one real batch through forward, cross-entropy, backward, and AdamW on `mps`;
5. train repeatedly on that same batch with dropout `0.0`;
6. stop and inspect the result before any larger training run.

No article, public API, deployment, broad corpus, or scale experiment starts before this gate is closed.
