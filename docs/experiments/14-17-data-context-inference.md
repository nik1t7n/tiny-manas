# O14–O17: corpus, evaluation, context, and inference precision

Status: protocol registered before new model scores are observed.
Date: September 5, 2026. Owner authorized execution, documentation, conditional
promotion, repository/website updates, and revision of the existing paper.

Source authorization clarification, before new-corpus training: the owner
confirmed permission/access to use the Orozbakov and Jusup Mamay editions for
this student research. Admit those sources on that explicit authorization and
retain source/edition attribution. This is not a claim that the scanned editions
have an independently verified public-domain or Creative Commons license. Keep
raw books, extracted text, token arrays, and full continuations out of Git.
The earlier source-license gate is resolved for these specifically named books;
extraction quality, duplicate separation, and model-quality gates remain.

## Scope and baseline

Only four experiments are in scope: O14 data and evaluation; O15 training the
existing architecture; O16 context 512; O17 BF16 inference. No batching,
quantization, new attention backend, runtime upgrade, SFT, or architecture search.
Keep the native RoPE model: 26,779,392 parameters, eight blocks, width 384,
eight heads, LayerNorm, GELU, tied 32,768-token embeddings, dropout 0.2.
The accepted export is `artifacts/tiny-manas-27m-rope-20260904.pt`, SHA-256
`abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f`.
The existing chronological splits and checkpoints remain immutable.

## O14: source audit and a frozen evaluation set

Question: can additional usable Manas texts and better separation of evaluation
material improve the evidence and supply useful new training data?

Inspect the actual source, edition/narrator metadata, license, text extraction,
document boundaries, exact duplicates, and substantial repeated spans before
tokenization. Keep the current tokenizer and require exact encode/decode round
trips, zero replacement characters, and byte/token accounting. Do not treat a
download button or a license on a research article as a license on an epic book.
Retain full-document source grouping. If several files reproduce one edition,
they form one group. Select validation and test groups before reading model
scores, and remove cross-split duplicate passages from training only, recording
removed spans and preserved document boundaries. Do not manufacture independent
documents by calling arbitrary windows books or narrators.

Forecast: additional high-quality verse can improve held-out prediction, but
duplicates, editorial prose, and domain differences can eliminate that gain.
Record unavailable or unlicensed candidates explicitly; do not silently mix
unrelated literature into a Manas corpus. If no additional source passes, report
the failed expansion gate and the scope of any verified cleaning work. Context
and precision comparisons can still proceed on the unchanged licensed corpus.

Evaluation: score fixed real target positions once, with source/target hashes,
explicit context lengths, summed NLL, target counts, and exact original bytes.
Report per-document and pooled loss/BPB separately. Keep a familiar-domain
validation control in addition to a new-document validation score. The final
test is reserved for post-selection reporting; historical test reuse remains
disclosed. All candidates use exactly the same evaluation targets.

## O15: unchanged architecture on the admitted corpus

### O14 freeze, before O15 scores

Coordinate extraction reads Poppler word boxes, rejoins glyph fragments by their
horizontal gap, and removes small-type editorial apparatus, counters, and source
watermarks. It preserves verse newlines and punctuation. No generative repair or
guessed spelling correction is used. Visual source checks cover Orozbakov book 2
page 120, book 3 page 120, book 4 page 100, and book 5 page 280; section starts and
ends are also checked against contents and extracted text. Residual OCR errors
remain, particularly near dense superscript annotations. This is an OCR-derived
corpus experiment, not a claim of philologically corrected transcription.

Frozen PDF-page ranges (inclusive): books 2: 17–380, 3: 10–299, 5: 18–561 train;
book 4: 16–332 validation; Mamay: 29–1040 test (closing publisher credit removed).
Books 1, 6–7, and 8–9 are downloaded but excluded from this bounded pass because
of dense apparatus or multi-column/glyph extraction problems. Books from the
same Orozbakov academic edition remain related sources. Mamay is a different
narrator; it is held out from language-model training, not certified unseen by
the historical tokenizer.

Training contains the original 418,562 tokens plus 110,252 / 91,367 / 141,783
tokens from books 2 / 3 / 5: 761,964 total, including 343,402 new tokens.
Validation book 4 contains 99,630 tokens; Mamay contains 408,359. Every complete
text passes byte-exact tokenizer round trip with zero replacement characters.
Bytes/token: original 6.987; books 2/3/4/5 6.240/6.199/6.168/6.288; Mamay 5.053.
Casefolded exact 32-word screening against all new and original holdouts, then
earlier training sources, finds zero matching spans to remove. This does not
exclude near-duplicates, short epic formulas, or OCR-obscured copies.

The expanded arm samples exactly half its windows from original training and
half from the three new volumes, proportional to eligible window starts. Windows
never cross a source or pruned boundary. The old-data arm uses the same original
training tokens only. Sampling consumes equal target counts, not equal epochs;
the treatment also introduces preserved verse newlines absent from the flattened
old source. These are jointly a data-treatment experiment, not a pure narrator
ablation.

Selection primary: 256 deterministic, evenly spaced, nonoverlapping target spans
of up to 128 tokens throughout book 4, after a 512-token context-only prefix.
The exact positions and byte denominators are in the frozen run manifest.
Familiar control: every original validation target after its 512-token prefix.
Context 256 and 512 score identical targets, with respectively 129–256 and
385–512 preceding tokens for full spans. Each retained target is scored once.
Final reporting additionally scores the complete book-4 remainder and the held-out
Mamay remainder; test does not guide selection. Do not compare this new BPB
denominator directly with the earlier O05 byte-window score.

Bundle: `runs/data-evaluation-20260905/bundle/manifest.json`; raw sources, texts,
token arrays, and full continuations remain local and ignored by Git. All run
protocols record hashes of the bundle, initializer, runner, and relevant helpers.

Proceed only after O14 supplies a frozen admissible training/evaluation bundle.
Compare a fresh old-data control and fresh new-data arm with identical shared
initial parameters, seed 1337, BF16 training/FP32 evaluation, AdamW (.9, .95),
weight decay .1, LR .0003 to .00003, warmup 100, clipping 1, and 4,096 target
tokens per optimizer update. Context is 256. Document-aware windows may not
cross source boundaries. Equalize sampled target budget, not nominal epochs.
Any cleaning/sampling difference must be named as part of the data treatment.

Maximum 3,000 updates per arm (12,288,000 targets), evaluated every 100 updates.
Save resumable optimizer, sampler, CPU/MPS RNG, and model state every evaluation.
At 900/1,200/1,500 compare the last three candidate-minus-control validation
deltas with the preceding three. Stop for lack of evidence if mean gain is
less than .02 nats and its improvement is less than .01 nats; do not stop before
900 on quality alone. A clearly worse candidate (all recent deltas > .03 and
mean > .05, without improving gap) also stops. Continue to 3,000 if the
1,500-update mean is <= -.02 and each recent delta is negative. A stopped arm
does not establish full-budget inferiority. Never shorten the LR horizon to
make an early comparison appear converged.

Promotion requires a completed run, additional fixed validation improvement
>= .02 nats over the matched control on the preregistered primary score, no
familiar-domain regression > .02 nats, and no material regression against the
existing deployed model on comparable targets. Inspect 20 full continuations
(same five prompts/four seeds, temperature .8, top-k 40, 256 new tokens), repetition
and normalized copying. Mean repeated-trigram ratio may rise by at most .01
absolute, with no new widespread collapse. These are practical gates, not a
human-coherence or significance claim. If the fixed budget undertrains a larger
corpus, document that result rather than quietly extending only the favored arm.

## O16: 256 versus 512 on the selected data

Hypothesis: RoPE and useful narrative context may make 512 beneficial, although
the earlier learned-position context experiment failed. Change only context and
microbatch geometry: 8x256x2 versus 4x512x2 = 4,096 targets/update. Keep width,
heads, dropout, optimizer, precision, 3,000-update schedule, and initialization.
Compare losses on identical target spans with each model's allowed context;
add a common-256-context evaluation to expose a loss caused by longer training.
Require >= .02-nat primary improvement, <= .02 common-context regression, and
the same 20-generation quality gate. Training speed alone does not veto a
quality gain. Use the same staged policy and sequential MPS runs. If O14/O15
cannot supply a new selected dataset, explicitly run O16 on the original
Manas01 data, comparing to a calibrated archived RoPE control where valid.
The larger position buffer adds no learned parameters. Cache overflow and native
checkpoint loading must pass before promoting a 512-token model.

## O17: BF16 inference on the selected checkpoint

Hypothesis: BF16 autocast can reduce cache/activation storage and improve MPS
generation latency while preserving usable prediction quality. Weights remain
FP32; this is inference autocast, not weight quantization or new training.
Compare FP32 and BF16 on the same checkpoint, fixed real prompts and evaluation
targets. Record finite outputs, loss delta, probability KL, top-1 agreement,
cache dtype/storage, and same-precision cached/uncached behavior including crop
overflow. BF16 rounding can change sampled continuations; sample identity is
not required. Numerical drift may not be hidden by changing seeds or tolerances.

Accept only with validation loss increase <= .01 nats, mean next-token KL
<= .005 on recorded real windows, no nonfinite results, no material generation
audit regression, and >= 5% median end-to-end generation speed improvement or
>= 20% measured live-memory reduction without > 5% latency regression. Use
warmup and 20 interleaved timing pairs on short prompts and near-overflow prompts;
measure prefill/decode separately. Never infer speed from tensor byte counts.
An accepted MPS result does not authorize BF16 on the production CPU: evaluate
that actual host separately before changing its precision. It may keep FP32.

## Resource and release bounds

Use installed PyTorch 2.13.0, MPS only, sequential GPU jobs, .65 per-process MPS
memory fraction. Stop nonfinite values or resource failures immediately. No paid
compute, unrelated process termination, or global environment replacement.
Reuse verified data and completed measurements; resume saved runs.

After selection, evaluate test once for the final candidate, export and hash the
checkpoint, run the real native loading/generation path, push completed source,
and update only the existing model/site services as necessary. Keep the previous
image and artifact for rollback. Update README, research/decision logs,
accepted-state, both website translations, and the 15-page paper with outcomes
including rejected, blocked, or budget-stopped arms. Keep the paper's References
last and do not restore the removed appendices or acknowledgments section.
