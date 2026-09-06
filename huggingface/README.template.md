---
language:
- ky
license: other
license_name: tiny-manas-research-terms
license_link: https://huggingface.co/Nik1t7n/tiny-manas/blob/main/LICENSE
library_name: pytorch
pipeline_tag: text-generation
inference: false
tags:
- causal-lm
- decoder-only-transformer
- kyrgyz
- manas
- research
---

# Tiny Manas

Tiny Manas is a 26.78M-parameter decoder-only Transformer trained from scratch to continue Kyrgyz text in the style of one edition of the epic *Manas*. It is a small research model built to make the full language-model path inspectable, from byte-level BPE tokens and causal attention to training, evaluation, and cached generation.

Tiny Manas is not a chat assistant. It does not follow instructions or answer general questions. Its task is narrower: given the beginning of a Kyrgyz passage, it predicts and samples what comes next.

- [Try the public demo](https://nik1t7n.com/essays/training-tiny-manas)
- [Read the English article](https://nik1t7n.com/essays/training-tiny-manas?lang=en)
- [Read the Russian article](https://nik1t7n.com/essays/training-tiny-manas?lang=ru)
- [Read the technical paper](https://nik1t7n.com/papers/tiny-manas-paper.pdf)
- [Inspect the research repository](https://github.com/nik1t7n/tiny-manas)

## Model details

| Property | Value |
|---|---|
| Architecture | Decoder-only, pre-LayerNorm Transformer with RoPE |
| Parameters | 26,779,392 |
| Layers | 8 |
| Attention heads | 8 |
| Embedding width | 384 |
| Feed-forward network | GELU, 4x expansion |
| Context window | 256 tokens |
| Vocabulary | 32,768-token Kyrgyz byte-level BPE |
| Training precision | BF16 |
| Inference precision | FP32 |
| Weight format | Safetensors |

The implementation uses tied input and output embeddings, causal scaled dot-product attention, adjacent-pair rotary position embeddings, and request-local KV caching. It is a compact PyTorch implementation rather than a Transformers `AutoModel` package, so the hosted Hugging Face inference widget is disabled.

## Quick start

```bash
git clone https://huggingface.co/Nik1t7n/tiny-manas
cd tiny-manas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate.py --device cpu --prompt "Манас" --max-new-tokens 64 --seed 1337
```

The device must be selected explicitly. `generate.py` accepts `cpu`, `cuda`, or `mps` and never silently switches to another backend.

## Training data

The accepted weights were trained on the pinned `Manas01` document from the Manas-UdS Kyrgyz corpus, attributed to Sayakbai Karalaev. The source corpus is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Source text and token arrays are not included in this model repository.

The tokenizer is the frozen [`kyrgyz-byte-bpe-v1`](https://github.com/nik1t7n/kyrgyz-tokenizer) artifact at commit `594d9e142cca1593963ccf12f344ab7ea4938fa5`. Its SHA-256 is `5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f`.

## Training and selection

The model was trained for a 3,000-update budget on an Apple M5 using PyTorch MPS. Each update covered 4,096 target tokens through gradient accumulation. Training used AdamW, a warmup plus cosine learning-rate schedule, dropout 0.2, and BF16 arithmetic. Evaluation and deployed inference remain FP32.

RoPE replaced learned positional embeddings after a controlled comparison on matched data. The selected RoPE checkpoint reduced validation loss from 4.34578 to 4.11584 relative to the previous accepted architecture. Later experiments with more books, context 512, and BF16 inference did not replace this checkpoint.

## Evaluation

The post-selection test used 100 fixed FP32 batches from the original chronological test split. It was run after model selection.

| Metric | Result |
|---|---:|
| Loss | 4.531258 |
| Perplexity | 92.8753 |
| Top-1 token accuracy | 31.698% |
| Top-5 token accuracy | 48.527% |

These scores describe next-token prediction on held-out text from the same edition. Separate book-level evaluations showed weak transfer to other editions and narrators, so the numbers should not be read as general Kyrgyz-language performance.

## Intended use

Tiny Manas is intended for:

- studying a complete small-language-model implementation;
- local research on causal language modeling and Kyrgyz tokenization;
- generating short experimental continuations from Kyrgyz prompts;
- reproducing the model-specific measurements documented in the source repository.

It is not intended for factual answers, instruction following, translation, safety-critical use, or unattended publication of generated text.

## Limitations

The model learned from one edition of one epic. It can reproduce local names, phrasing, rhythm, and short action patterns, but it often repeats formulas, creates malformed words, and loses narrative continuity over longer spans. Its outputs can also reproduce biases or errors present in the source and tokenizer data.

The released checkpoint was not instruction-tuned, aligned, or evaluated as a general-purpose Kyrgyz model. The training corpus is small, the context window is 256 tokens, and the evaluation does not establish broad linguistic coverage.

## Release provenance

| Artifact | SHA-256 |
|---|---|
| Accepted PyTorch inference export | `{{SOURCE_CHECKPOINT_SHA256}}` |
| Published `model.safetensors` | `{{MODEL_SAFETENSORS_SHA256}}` |
| Published `tokenizer.json` | `{{TOKENIZER_SHA256}}` |

Source commit: [`{{SOURCE_COMMIT}}`](https://github.com/nik1t7n/tiny-manas/commit/{{SOURCE_COMMIT}})

The Safetensors file was converted from the accepted export and checked against it with exact FP32 logits on a fixed Kyrgyz prompt. `release-manifest.json` contains hashes for every published file.

## License and attribution

This repository contains materials under different terms. See [`LICENSE`](LICENSE) before reuse. In particular, public availability does not relicense the model code or tokenizer, and it does not remove the attribution, non-commercial, or share-alike conditions attached to the Manas-UdS source corpus.

## Citation

```bibtex
@misc{nosov2026tinymanas,
  author = {Nosov, Nikita},
  title = {Tiny Manas: An Inspectable Kyrgyz Language Model Trained on the Epic Manas},
  year = {2026},
  url = {https://huggingface.co/Nik1t7n/tiny-manas}
}
```
