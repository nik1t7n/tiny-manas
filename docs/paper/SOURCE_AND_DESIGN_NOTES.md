# Primary sources and document design

## Academic structure revision: September 5

The owner requested a model-centered academic paper without personal narrative
or a chronological experiment catalogue. The current structure is architecture
and mathematics, data/training, computational implementation, empirical design
comparisons, limitations, and conclusion. It was informed by fresh reading of
[Attention Is All You Need](https://arxiv.org/html/1706.03762v7),
[LLaMA](https://arxiv.org/html/2302.13971v1), and
[TinyLlama](https://arxiv.org/html/2401.02385v2), especially the separation of
model specification, optimization, implementation, and results. This is a
structural reference, not a claim that those papers validate Tiny Manas.

The manuscript now uses three existing vector assets: architecture, cache, and
data/context curves. Labels were made model-centered; plotted measurements,
scales, and numerical source files are unchanged. Historical pilot, tokenizer,
and staged-quality figures remain supporting assets but are not included in the
current paper. The records below describe the earlier figure and scope choices.

## Earlier extension and source review

September 5 extension: O14 source PDFs and permissions are documented in
`../SOURCES.md` and the O14-O17 protocol. `data/followup-evidence.json` records
the new measurements separately from the historical snapshot. A sixth vector
figure plots observed data/context curves without smoothing or extrapolating
the stopped arm. The original five figures are unchanged. The same NeurIPS
preprint style, named fonts, and References-last structure are retained.

Reviewed on September 4, 2026. These are independent reference documents, not
evidence for Tiny Manas's local scores. Local scores come from the source files
hashed in `data/evidence.json`.

## Requested reference PDFs

1. **Attention Is All You Need** — [original PDF](https://arxiv.org/pdf/1706.03762).
   The retrieved version has 15 pages. Inspected the architecture figure (PDF
   page 3), attention formulation, architecture-variation table (page 9), and
   attention visualizations (page 14). The reusable visual principle is a small
   vocabulary of boxes, explicit arrow direction, and visible residual bypasses.
   Tiny Manas's figure is newly drawn: decoder only, pre-normalization, RoPE,
   four dropout locations, no encoder/cross-attention.
   SHA-256: `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`.

2. **Language Models are Unsupervised Multitask Learners** — [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
   24 pages. Inspected the title/abstract, two-column narrative, WebText and byte
   representation sections, pre-norm/residual initialization description,
   architecture table, and completion appendices (including pages 18 and 20).
   Retained the separation of input representation, model, experiments, and
   limitations. Did not copy its larger-model performance claims or repurpose
   cherry-picked sample presentation as a Tiny Manas quality metric.
   SHA-256: `d9d852e2894556e73f53cb22b7c605a9643d6f0b19bf604b429ed6192fa24f4e`.

3. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** — [JMLR PDF](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
   30 pages. Inspected the network before/after dropout (page 2), training/test
   scaling diagram and text, dataset/results tables (page 8), appendix settings,
   and references. Important notation difference: its retention probability is
   not the drop probability used by this implementation. The manuscript states
   that distinction and uses inverted dropout with expectation-preserving scale.
   SHA-256: `9c196ccbe6c6a595a1adba6cd030d35f7c2e548bbf5e7f1278b0109d8dd9ebaa`.

## Formatting standards

- [NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers) links the
  [official template](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
  Downloaded archive SHA-256:
  `82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9`.
- Read the actual template instructions: 10-point main text, 5.5-by-9-inch text
  area, single-paragraph abstract, figure captions below, table captions above,
  no vertical rules for tables, legibility in grayscale, embedded fonts.
- Used `preprint`, never `final`. Kept official dimensions and typography scale.
  The report is not constrained to nine main pages and explicitly says it is not
  a conference submission. There is no invented venue, acceptance date, DOI,
  affiliation, coauthor, or email address.
- [arXiv TeX guidance](https://info.arxiv.org/help/submit_tex.html): retain source
  and figure files, include needed custom style, fixed document date, inspect
  the produced PDF. arXiv is a repository, not peer review or a journal.

## Original figures

1. Current native architecture and attention internals: derived from
   `model.py`, `rotary.py`, and the accepted configuration.
2. Pilot and full-corpus learning curves: raw scheduled metrics; no smoothing.
3. Equal-byte tokenizer trajectories: same original target bytes and explicit
   separate incumbent score, not an invented incumbent training curve.
4. RoPE/RMSNorm deltas: actual checkpoints against the calibrated historical
   BF16 control; no extrapolated RMSNorm tail or fake uncertainty bands.
5. Prefill/decode/overflow: declared cache contract, including explicit rebuild.

ReportLab generated native vector PDF figures; LaTeX/Tectonic typeset the
equation-rich manuscript. No image generation was used for numerical evidence,
and none of the reference papers' images is reproduced in the delivered paper.

## Bibliographic and numerical audit

Primary papers were checked for RoPE, LayerNorm, RMSNorm, SwiGLU, GQA, AdamW,
checkpointing, Cut Cross-Entropy, and Hyperband; nanoGPT is identified as an
implementation reference. The Cut Cross-Entropy author list was corrected
against the original arXiv record during manuscript verification.

The evidence extractor records file hashes rather than treating a rendered
README or a memory summary as the final measurement source. The paper separates
full training, bounded adaptation, cost-only checks, budget stops, and work that
was assessed but never executed. It does not add new empirical results.
