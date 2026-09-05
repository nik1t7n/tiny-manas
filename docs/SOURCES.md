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

## O14 research expansion: Orozbakov and Mamay

On September 5, 2026, the owner explicitly confirmed access and permission to
use the named editions for this student research. This is the authorization
basis for O14, not an independently verified blanket Creative Commons license
for the scanned editions. Their text, token arrays, and complete generated
continuations remain local. The original Manas-UdS license is not transferred
to these sources.

The [Bizdin Manas catalogue](https://new.bizdin.kg/knigi/category/manas) supplied
eight downloaded PDFs. Five were admitted after coordinate-based extraction:

| Edition | PDF pages used | Role | PDF SHA-256 |
|---|---:|---|---|
| Orozbakov, book 2 | 17-380 | Train | `fbfc31e4aa4e9959ca4b83bf81165bb2fc6e7458dd374cc452959ddbc58d01ed` |
| Orozbakov, book 3 | 10-299 | Train | `156450db158bbc8cde396d7823e8c66514677f758a053d3b17c42d0a51a03115` |
| Orozbakov, book 4 | 16-332 | Validation | `3b36482e63141de3816b5934810be1f8603de39e09112d9f3daa83b7f6ed9e3b` |
| Orozbakov, book 5 | 18-561 | Train | `3a70ef5498afd5b197a1c4506ad00463d203a4dbfeca0e096fad92689c1cab7c` |
| Jusup Mamay | 29-1040 | Test | `7ad9f57e5da642e66efe00131ac3d0bd7b2fe46caab0810277dc176b774f0285` |

Download-page slugs and exact resolved PDF URLs are retained by
`scripts/prepare_manas_expansion.py` and its cached source records. Book 1 and
the combined 6-7 and 8-9 volumes were not admitted in this bounded pass because
of apparatus/layout/glyph extraction problems. Admitted text still contains
residual OCR errors; no generated text or guessed spelling repairs replace it.

The [O14-O17 report](experiments/14-17-data-context-inference.md) records the
split assignments, extraction rules, exact tokenizer round trips, 32-word
duplicate screening, byte denominators, and model decisions. Orozbakov volumes
are related sources from one academic edition. Mamay is held out from LM
training, not certified unseen by the earlier tokenizer. The O15 expanded
checkpoint failed its familiar-domain promotion gate.

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
