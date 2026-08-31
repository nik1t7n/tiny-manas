# Sources

## Manas-UdS corpus

- Artifact: `kyrgyz_2022_10_03.zip`
- URL: <https://fedora.clarin-d.uni-saarland.de/kyrgyz/kyrgyz_2022_10_03.zip>
- SHA-1: `82fc72b9687175c55b5309822da2a7c44bb303f1`
- Archive member: `kyrgyz_2022_10_03.vrt`
- Selected document: `Manas01`
- Title: *Манас*
- Performer/author metadata: Sayakbai Karalaev
- Corpus license: CC BY-NC-SA 4.0
- Local policy: downloaded and hash-verified during `prepare`; never committed.

The raw document contains title pages, forewords, and scholarly material before the epic. Training begins at the unique heading `Манастын туула элегиндеги бабалары`. The extraction must fail if that heading is absent or appears more than once.

## Kyrgyz byte-level BPE tokenizer

- Repository: <https://github.com/nik1t7n/kyrgyz-tokenizer>
- Commit: `594d9e142cca1593963ccf12f344ab7ea4938fa5`
- Artifact: `models/kyrgyz-byte-bpe-v1/tokenizer.json`
- Raw URL: <https://raw.githubusercontent.com/nik1t7n/kyrgyz-tokenizer/594d9e142cca1593963ccf12f344ab7ea4938fa5/models/kyrgyz-byte-bpe-v1/tokenizer.json>
- SHA-256: `5047b4f427bb1af1c06cfb9cefbe83790b56df409b137b887988db6eba4b159f`
- Vocabulary size: 32,768
- Local policy: downloaded and hash-verified during `prepare`; never silently replaced.

## Architecture and implementation references

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Karpathy, [nanoGPT model](https://github.com/karpathy/nanoGPT/blob/master/model.py).
- Karpathy, [nanoGPT Tiny Shakespeare config](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py).
- PyTorch, [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html).
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

The project adapts established decoder-only Transformer mechanics. Its original contribution is the local Manas experiment, documentation, measurements, and analysis—not the invention of the Transformer architecture.
