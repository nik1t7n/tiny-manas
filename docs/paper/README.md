# Tiny Manas technical report

**Training Tiny Manas: A Small Language Model for a Kyrgyz Epic**  
Nikita Nosov · September 5, 2026 · Independent technical report

The manuscript reconstructs E0-E5, O00-O17, corrected measurement failures,
the quality follow-up, new book-level evaluation, and the retained native RoPE model. It is not a
NeurIPS acceptance claim or an arXiv submission. It uses the official NeurIPS
2026 `preprint` style; the main report is deliberately longer than the
conference's nine-page submission limit.

## Files

- `main.tex`: manuscript and bibliography; the PDF ends at References.
- `neurips_2026.sty`: unmodified official template; attribution retained.
- `figures/*.pdf`: six original vector figures with embedded Arial fonts.
- `data/*.csv`: observed training/evaluation curves and per-sample audit metrics.
- `data/evidence.json`: extracted numerical evidence and hashes of 25 source
  artifacts. No model tensors, licensed corpus payloads, or full continuations.
- `collect_evidence.py`: read-only extraction from the original local run files.
- `collect_followup.py`: read-only extraction of O14-O17, including the format
  diagnostic and final report-only scores, into `data/followup-evidence.json`
  and three observed learning-curve CSVs.
- `make_figures.py`: deterministic ReportLab vector-figure builder.
- `SOURCE_AND_DESIGN_NOTES.md`: primary literature and layout decisions.
- `WORK_PLAN.md`: scope and delivery checklist.
- `EDITORIAL_REVIEW.md`: detailed presentation review, corrections, and limits.

## Build the PDF

From this directory:

```sh
tectonic --keep-logs main.tex
```

The build uses Times New Roman, Arial, and Menlo through `fontspec`; these were
available on the authoring macOS host. Tectonic resolves the TeX packages. For
another machine, provide the named fonts rather than silently substituting a
different layout. The vector figures are already included, so ordinary PDF
rebuilding does not need ReportLab or the private training runs.

To regenerate figures from the tracked observed CSVs, use Python with ReportLab
installed and the explicitly named macOS Arial font files:

```sh
python3 make_figures.py
python3 make_figures.py --followup
```

`collect_evidence.py` additionally requires the exact local run files identified
inside it. It fails if they are absent; it does not create artificial curves or
start training. Reproducing model experiments is a different, compute-bearing
operation: use the individual `docs/experiments/` protocols and pinned inputs.

## Evidence interpretation

`result.json` runner statuses can precede review/promotion. Current adoption is
resolved through `accepted-state.json` plus the later native promotion artifact,
not an old `completed_not_promoted` string alone. Older README/service excerpts
may describe the earlier BF16 model; this manuscript uses the native RoPE release
`437195fb1a109e3ff392aaae7c4c350518b9df76` and its identified artifact.

The September 4 manuscript used existing evidence only. The September 5 revision
incorporates the owner's separately authorized new research: two complete
3,000-update data arms, a context-512 run stopped at 1,500, the inference screen,
and final fixed-target scoring. The evidence collectors themselves never train
or call a model. Historical metrics and original artifacts remain unchanged.
Single-seed uncertainty, adaptive validation, OCR and formatting differences,
historical test reuse, allocator sampling, and incompatible timing denominators
are explicit in the manuscript.

The final delivered PDF is an output artifact, not tracked here. Source files,
vector figures, and extracted measurements are committed for inspection and
revision. No new weights passed promotion in O14-O17, so the inference artifact
and production model remain unchanged. The PDF is delivered locally; the
repository retains its reproducible sources rather than a journal submission.
