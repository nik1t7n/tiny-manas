# Tiny Manas retrospective paper

## Deliverable and scope

Write an English, first-person technical report by Nikita Nosov covering the
complete Tiny Manas sequence: data/tokenizer provenance, correctness and pilot
runs, the original full-data capacity/context experiments, O01-O11 optimizations,
the reopened RoPE/RMSNorm quality comparison, and the accepted native RoPE model.
Deliver a typeset PDF with original architecture diagrams, measured-result plots,
tables, references, limitations, and an evidence/reproducibility appendix.

This is an independent technical report, not a claim of conference acceptance,
peer review, new Transformer theory, or general Kyrgyz conversational ability.
Do not rerun training, change production, rewrite the website, or invent missing
measurements. The earlier bigram/tokenizer work may establish chronology and data
provenance, but must not become an unrequested bigram benchmark comparison.

## Requirements and evidence gates

- [x] Read the named primary papers and inspect their actual PDF layouts:
  Attention Is All You Need, GPT-2, and the original JMLR Dropout paper.
- [x] Inspect official NeurIPS formatting/checklist and arXiv submission guidance;
  document the adopted typography, table, figure and disclosure decisions.
- [x] Verify every reported run against current source/config and raw metrics.
- [x] Cover the failed first overfit measurement, corrected overfit, 10K pilot,
  13M baseline, 512-context negative result and 27M capacity result.
- [x] Cover each optimization, including invalid measurements and non-results;
  distinguish numerical parity, speed probes, short adaptation and full training.
- [x] Explain the revised quality gate and complete RoPE/RMSNorm follow-up.
- [x] Describe exact current architecture, training recipe, data split, tokenizer,
  inference semantics, parameter counts and artifact provenance.
- [x] Separate scheduled validation, independent sampled validation, exact-byte
  scoring, historical approximate BPB and post-selection test use.
- [x] State single-seed/adaptive-selection/data-license limits and bounded
  generation/memorization findings. No fabricated error bars or significance.
- [x] Generate original figures matching the implementation, not the old learned-
  position diagram. Plot only observed data with units and protocol captions.
- [x] Finish full manuscript, references and reproducibility checklist.
- [x] Compile PDF, inspect all rendered pages for layout/glyph/figure defects,
  verify factual coverage and links, and commit the reproducible paper sources.

## Execution

2026-09-04: sources and existing records located. The earlier RESULTS.md and
memory overview end at older milestones; the later optimization reports, raw
artifacts and native model source take precedence for the current architecture.
Use LaTeX for the equation-rich paper and vector-native plots/diagrams for sharp
print output; the PDF skill's rendering and inspection gates still apply.

Main output: `output/pdf/tiny-manas-paper.pdf` in the task workspace.
Reproducible source and evidence ledger: this directory.

## Final verification — September 4, 2026

The final PDF has 20 pages, five vector figures, twelve tables, sixteen numbered
equations, and thirteen references. All twenty rendered pages were visually
inspected. The build has no overfull boxes, undefined references, or missing
characters; all PDF fonts are embedded. Small underfull spacing warnings remain
in narrow table cells and long provenance identifiers, without clipping.

The evidence check matched all 25 recorded source hashes, corpus token/byte sums,
parameter accounting, equal-byte denominators, and reported perplexity
calculations. The accepted inference export's size and SHA-256 were checked
against the actual local file. No training, generation, test suite, or deployment
was run for manuscript preparation. Reproduction instructions and the complete
paper source are retained in the same commit as this checklist.

## Owner revision

Removed all appendices after References from the delivered manuscript at the
owner's request. Removed or redirected the five corresponding in-text references
to repository records. The previous complete version remains in Git history;
the evidence files and experiment documentation are unchanged.
