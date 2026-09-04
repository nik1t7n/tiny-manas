# Curated editorial and presentation review

Reviewed September 4, 2026. Scope: the English technical report ending at
References. This is an editorial and typesetting review, not independent peer
review or a new evaluation of model quality. The removed appendices stay removed.

## Editorial standard

The paper retains one coherent format: the official NeurIPS 2026 preprint style.
I checked the actual template instructions, including sentence-case headings,
paragraph spacing, numbered citations, table captions above tables, figure
captions below figures, and embedded fonts. JMLR uses some different conventions;
combining them would not make the report more compliant.

Primary references reviewed:

- [NeurIPS 2026 call and linked official template](https://neurips.cc/Conferences/2026/CallForPapers).
- [JMLR formatting instructions](https://jmlr.org/format/format.html).
- [arXiv TeX preparation guidance](https://info.arxiv.org/help/submit_tex.html).
- The actual PDFs of *Attention Is All You Need*, GPT-2, and the JMLR Dropout
  paper retained under the source-and-design record. I revisited the architecture
  diagram, GPT-2 methods page, and Dropout result tables rather than relying on
  screenshots of unrelated papers.

Humanizer replaced Stop Slop at the author's request. Its draft/audit/revision
loop was applied to prose, with the scientific register and the author's first
person preserved. Normal academic punctuation in ranges, bibliography, and
mathematical notation remains governed by the chosen publication style.

## Findings and corrections

| Area | Finding | Correction |
| --- | --- | --- |
| Abstract | Too many implementation details competed with the main results. The cache claim did not identify the architecture actually benchmarked. | Reduced the abstract to about 200 words. Preserved the corpus, main comparisons, numerical results, and limitations; explicitly attributed cache timings to the learned-position model. Details remain in the body. |
| Voice | Several headings and closing sentences sounded like general lessons or slogans. | Replaced them with descriptions of the experiment, measurement, or decision. Kept the personal motivation and the account of reversing the original RoPE decision. |
| Grammar | “Rather than regenerate” was not parallel with its preceding past-tense verb; some engineering shorthand obscured the action. | Used “rather than regenerating,” “finite gradients,” “generation check,” and descriptions of inspected parameter gradients. |
| Terminology | Several abbreviations appeared without expansion; vocabulary size shared the symbol used for value activations. | Expanded BPE, BF16, KV, MPS, FP32, FFN, GELU, GQA, MHA, PPL, BPB, and RNG where needed. Used the vocabulary cardinality notation separately from value vectors. |
| Mathematical notation | The projection convention used rows while the rotation example used columns without explanation. Some variables were implicit. | Stated the local column-vector convention, defined the full rotation matrix, block count, final block output, LayerNorm statistics, and the BPB index set and byte denominator. |
| Display equations | Punctuation and connection to surrounding sentences needed an explicit pass. | Checked all nine displayed equations as parts of their sentences. Retained terminal punctuation, aligned residual equations, and consistent operator styling. |
| Cross-references | Several numbered objects were only present as floats, without a reference in the narrative. | Added in-text references to all five figures and all seven tables; fixed labels for the tokenizer and cache figures. No appendix references remain. |
| Metrics | The deleted appendix had supplied context for the repeated-trigram metric. | Added a short operational definition and the 20-continuation averaging scope in Section 9.2; linked the result table to that section. No appendix content block was reinstated. |
| Numeric presentation | Some prose and timing cells reported more digits than readers needed. | Rounded selected displayed losses and timings consistently. Kept exact parameter counts, byte counts, small numerical-equivalence errors, and the original machine-readable measurements. Rounding does not establish statistical significance. |
| Figures | Some curves depended mainly on color, and two axis headings omitted units or the sign convention. | Added distinct dash patterns with matching legends, nats/token units, and “candidate minus control” for loss differences. Kept the same observations, scales, and geometry. |
| Bibliography | The RoFormer entry combined a 2021 date with the author list of a later revision. Full justification stretched some lines with URLs. | Pinned the entry to arXiv version 5 (2023), noting first posting in 2021; standardized preprint labels, dated software access, and used a ragged-right bibliography. |
| Pagination | A word was hyphenated across a page break during the first revision. | Added penalties against broken-word page boundaries and isolated paragraph lines, then rendered the revised PDF again. |

## Preserved deliberately

- First-person authorship, the paper title, the experiment sequence, and the
  distinction between accepted, rejected, optional, stopped, and unrun work.
- Single-column layout, official text dimensions and type scale, sentence-case
  headings without decorative punctuation, and compact booktabs tables without
  vertical rules.
- The five original vector figures. No new illustrations, invented samples,
  smoothed curves, uncertainty bands, or additional benchmarks were introduced.
- Single-seed limitations, historical test reuse, calibration of the archived
  control, and the absence of a human coherence study.
- References as the final section. Supporting data remain in the repository.

## Validation and remaining boundaries

The actual LaTeX source was rebuilt with Tectonic and rendered with Poppler for
visual inspection. The delivery check covers page layout, figure and table
placement, glyphs, internal links, bibliography keys, and embedded fonts. Raw
evidence CSV/JSON files were not modified; no model runs or service changes were
part of this review.

Final result: 15 pages, five figures, seven tables, nine numbered equations, and
thirteen cited references. All numbered figures and tables have narrative
references. The final visual pass found no clipping, overlapping elements, or
broken glyphs. The build has no undefined references or overfull boxes, and all
fonts are embedded. Three underfull-box warnings remain in the archive paragraph
and narrow table cells; the corresponding rendered text is readable. The
conclusion now stays together on page 14; acknowledgments and References follow
on page 15.

This is a polished independent report, not a claim of full NeurIPS submission
compliance. Its main text exceeds the conference page limit, and it does not
include the submission checklist. A real submission would also require the
author's confirmation of affiliation, funding, and competing-interest statements;
I did not invent these. The source build still requires the named macOS fonts,
so an arXiv source package would need a separate portability/build check.

Editing cannot establish broader scientific validity. Multi-seed comparisons,
external baselines, a fresh held-out corpus, and human evaluation would require
new research rather than wording changes.
