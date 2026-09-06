# Hugging Face release

Date: 2026-09-06. Status: public and verified.

Repository: <https://huggingface.co/Nik1t7n/tiny-manas>

The release publishes the accepted RoPE checkpoint as Safetensors together
with the frozen tokenizer, standalone PyTorch generation code, model card,
technical paper, citation metadata, sanitized provenance, and a per-file hash
manifest. It does not include source corpus text, token arrays, optimizer
state, rejected checkpoints, or raw experiment outputs.

## Identity

- Hub commit: `127850a17e3b00979c4f3c7401fcfe178c97ef8b`
- Release source commit: `c903734fe29c5e087cb753b2bb1c3b93edc31584`
- Accepted PyTorch export SHA-256: `abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f`
- Published Safetensors SHA-256: `9870407329c400a267cc3978c5ee7162fbe61695feae74c094a771bb5e4d7795`
- Tokenizer SHA-256: `5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f`
- Technical paper SHA-256: `a38825fcad03931ec282dc57484448a70cb11018887b9abfec2311da27d1b72a`

The Hub metadata reports 26,779,392 FP32 parameters, `pytorch` as the library,
`text-generation` as the task, and Kyrgyz (`ky`) as the language. Hosted
inference is disabled because this is a compact custom PyTorch model rather
than a Transformers `AutoModel` package.

## Acceptance

The exporter loaded the accepted local artifact, converted its tied weights
with `safetensors.torch.save_model`, reloaded them with strict matching, and
confirmed exact FP32 logits on the fixed prompt `Манас`.

The final Hub repository was then downloaded into a fresh temporary directory.
Hashes for the weights, tokenizer, and paper matched the release manifest. The
downloaded `generate.py` ran on CPU with 16 new tokens and seed 1337:

```text
Манас, Шоорук кандын көп аскер Жаткан экен шумдукту. Букара кылып көп
```

No Python bytecode files remained in the Hub repository or appeared beside the
downloaded runtime during the accepted generation.

## Rights boundary

The Hub repository uses `license: other` and links to its mixed rights notice.
The source project, tokenizer, and model weights remain copyright Nikita Nosov
with all rights reserved unless separate written permission is granted. The
Manas-UdS training source is not redistributed and retains its CC BY-NC-SA 4.0
terms. Public availability does not override either boundary.
