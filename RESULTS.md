# Tiny Manas experiment report

## Current selection: September 5, 2026

The retained model is the September 4 RoPE checkpoint: 26,779,392 parameters,
context 256, BF16 training and FP32 cached inference. Its earlier matched
validation loss is 4.115836; see the [RoPE report](docs/experiments/13-quality-followup.md).
The original capacity experiments below are historical, not scores for those
newer weights.

The [O14-O17 follow-up](docs/experiments/14-17-data-context-inference.md) is complete:

| Change | Result | Selection |
|---|---|---|
| Data/evaluation | Three Orozbakov training volumes, separate book-4 validation and Mamay test; 343,402 additional training tokens | Research bundle retained, with OCR/format limits |
| Expanded-corpus training | 3,000 updates; new-book loss 4.207867; familiar loss 4.133920 versus incumbent 4.088184 | Not promoted: +.045736 exceeds +.02 familiar limit |
| Context 512 | Stopped at 1,500; last-three mean primary delta +.004754 | Keep 256; full-budget outcome unknown |
| BF16 inference | Numerical parity tolerances passed; short generation 18.73% slower | Keep FP32 |

Post-selection scoring of the unchanged checkpoint, with each retained target
scored once: original test loss 4.534839, PPL 93.21; Orozbakov-4 full validation
loss 11.842740; Mamay test loss 12.370524. These target definitions differ from
the older random-window scores. The high book losses indicate weak source and
format transfer. The rejected expanded model was not evaluated on Mamay.

## Original experiment abstract

Tiny Manas is a decoder-only Transformer trained from scratch on one Kyrgyz edition of the epic *Manas*. The aim was not to build a general assistant. The aim was to understand the complete language-model pipeline and determine how far a compact model could go on one consumer Apple Silicon machine.

The experiment used 465,069 tokens from `Manas01`, performed by Sayakbai Karalaev, and a frozen 32,768-token Kyrgyz byte-level BPE tokenizer. The text was split chronologically so that nearby future passages did not leak into training. Every accepted run used real PyTorch MPS execution, the same seed, the same tokenizer, and checkpoint selection by validation loss.

After correctness and pilot gates, three full-data candidates were tested. Doubling context from 256 to 512 made prediction worse and training slower. Increasing model capacity from 13.19M to 26.88M parameters reduced independent validation perplexity from 103.38 to 77.15 and test perplexity from 151.42 to 116.45. The larger model remained practical: its full run took 24.4 minutes and peaked below 1 GiB of memory allocated through PyTorch MPS.

The original final model produces recognizable Manas-like language: hero names, combat actions, reported speech, parallel phrases, and verse-like rhythm. It does not maintain a reliable long narrative. It can repeat names or formulas, create malformed words, and confuse who is acting. Across 20 fixed generations, the corrected maximum normalized matching word span is nine words. The earlier seven-word result came from asymmetric normalization and is superseded by the [audit correction](docs/experiments/00-audit-correction.md); this small sample cannot establish an absence of memorization elsewhere.

## 1. Research question

The project asks one narrow question:

> How much Manas-like structure can a small, inspectable Transformer learn from one epic on one laptop?

This framing matters. A model trained on one edition of one epic should not be called a Kyrgyz base model. It has seen neither broad Kyrgyz prose nor conversations, current events, instructions, code, or Russian-Kyrgyz mixed language. The accepted name is therefore **Tiny Manas**.

## 2. Data and evaluation

The cleaned epic contains 1,794,194 characters, 3,249,652 UTF-8 bytes, and 465,069 tokenizer tokens. The chronological split contains:

| Split | Tokens | UTF-8 bytes |
|---|---:|---:|
| Train | 418,562 | 2,924,452 |
| Validation | 23,253 | 164,489 |
| Test | 23,254 | 160,711 |

Validation selected checkpoints. The test suffix was used only after a run had finished. Final metrics were estimated over 204,800 sampled target positions on each held-out split. Bits per UTF-8 byte combines prediction loss with the tokenizer's compression ratio and lets the result be read in terms of the original text rather than token count alone.

Raw generation used five fixed prompts and four fixed random seeds: 20 samples per accepted full-data candidate. Every sample used temperature `0.8`, top-k `40`, and 256 new tokens. Reports include all outputs, not only the best-looking passage.

## 3. Correctness before scale

The model first had to overfit one real batch with dropout disabled. Loss fell from `10.4385` to `0.0267`, while top-1 and top-5 accuracy reached 100%. This proved that embeddings, causal attention, feed-forward layers, cross-entropy, backward, AdamW, and checkpoint loading were connected well enough to memorize known targets.

The first implementation of this gate exposed a measurement bug: optimization and evaluation used different fixed batches. The optimizer had memorized its batch, but the summary evaluated another one and falsely reported failure. The invalid attempt remains in the research log. The evaluator was corrected to reuse the exact optimization batch, then the experiment was repeated unchanged.

A 10,000-token pilot passed the real training path but overfit quickly. Its best validation checkpoint appeared at step 200. By step 1,000, train loss had fallen to `0.1235` while validation loss had degraded to `10.4687`. The pilot established that the model could learn *Manas*, but also that 9,000 training tokens were not enough for an 8.1M-parameter model.

## 4. Full-data experiments

| Candidate | Parameters | Context | Val loss | Val PPL | Val top-1 | Test loss | Test PPL | Test top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13M base | 13,193,216 | 256 | 4.6384 | 103.38 | 26.64% | 5.0201 | 151.42 | 23.81% |
| 13M long context | 13,258,752 | 512 | 4.7268 | 112.94 | 25.67% | 5.1466 | 171.85 | 22.29% |
| **27M final** | **26,877,696** | **256** | **4.3457** | **77.15** | **31.55%** | **4.7575** | **116.45** | **27.74%** |

### 4.1 The base model worked

The 13M base model established a real baseline. Independent validation perplexity was 103.38 and test perplexity was 151.42. Generations clearly belonged to the epic's distribution, but often repeated reporting verbs, names, and short action formulas. The test suffix was harder than validation, which is plausible for a chronological story split and is reported rather than averaged away.

### 4.2 More context was not automatically better

Context 512 changed only the usable history, positional table, and micro-batch needed to preserve the same number of targets per update. It reduced throughput by 12.3%, worsened every held-out prediction metric, and did not show a clear continuity improvement in manual reading.

This negative result is important. A model benefits from more context only if it can use that context. On this corpus and budget, the smaller network learned more effectively from more independent 256-token windows than from fewer 512-token windows.

### 4.3 More capacity did help

The final candidate restored context 256 and increased depth and width together: six layers became eight, and 256 features became 384. The parameter count grew from 13.19M to 26.88M.

Relative to the base model:

- validation perplexity fell by 25.4%;
- test perplexity fell by 23.1%;
- validation top-1 accuracy rose by 4.91 percentage points;
- test top-1 accuracy rose by 3.93 percentage points;
- validation bits per UTF-8 byte fell from `0.946` to `0.886`;
- test bits per UTF-8 byte fell from `1.048` to `0.993`.

The larger model processed about 8,378 target positions per second instead of 13,565. Training time rose from 15.1 to 24.4 minutes. That is a meaningful cost, but still a practical local experiment rather than a cloud-scale training job.

## 5. What the model learned

The final model often maintains a local pattern for several lines. It can introduce a character, continue with an action, preserve the elevated register, and produce dialogue-like punctuation. It learned recurring relationships among names, horses, weapons, movement, combat, and speech.

This is not the same as understanding the plot. Typical failures remain:

- one hero's name replaces another without explanation;
- a phrase is grammatically shaped but semantically impossible;
- a name or formula repeats until it dominates the passage;
- invented word endings appear;
- a local battle scene continues, but the model forgets why it began;
- rare samples enter strong three-word loops.

The mean repeated-trigram ratio was `4.77%` for the 13M base and `4.21%` for the 27M model. That small aggregate improvement should not be oversold: one final-model sample still had a `25.7%` repeated-trigram ratio. Reading the whole distribution matters.

The original matcher reported six words for the base and seven for the 27M model. Rechecking the same 27M outputs with symmetric word normalization raised that maximum to nine. This is a case-folded word-sequence match that ignores punctuation and whitespace, not exact byte copying. The historical raw files remain unchanged.

## 6. Hardware result

All training used PyTorch MPS on an M5 MacBook Pro with 16 GB of unified memory. CPU fallback was disabled.

| Candidate | Throughput | Training time | Peak PyTorch MPS | Peak MPS driver |
|---|---:|---:|---:|---:|
| 13M / context 256 | 13,565 targets/s | 15.1 min | 468.5 MiB | 3.36 GiB |
| 13M / context 512 | 11,894 targets/s | 17.2 min | 470.9 MiB | 3.36 GiB |
| 27M / context 256 | 8,378 targets/s | 24.4 min | 959.4 MiB | 4.33 GiB |

The laptop was not close to its memory limit. For this project, data diversity became a stronger constraint than device memory.

## 7. Limits and next evidence

The test set comes from the same edition and performer as training. It measures continuation into a later passage, not transfer to another version of *Manas*. The fixed tokenizer was trained on a broader Kyrgyz corpus, so the experiment does not isolate tokenizer pretraining from model training. Metrics use sampled windows rather than an exhaustive sliding evaluation. Only one random seed was used for expensive full runs.

The strongest next experiment is not another arbitrary increase in steps. It is more legally usable Manas-only text from other narrators or editions, followed by a new document-level held-out set. That would test whether Tiny Manas learned general epic structure or mostly the patterns of this edition.

## 8. Conclusion

Tiny Manas reached its intended goal. A complete Transformer was built, debugged, trained, evaluated, and scaled on one laptop. The final model is measurably better than the base model and visibly more coherent over short spans. It is still tiny, specialized, repetitive, and far from a general language model.

The most useful result is not a single generated paragraph. It is the sequence of evidence: one-batch correctness, a pilot that overfit, a full-data baseline, a longer context that failed, and a larger model that succeeded.
