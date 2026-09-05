# O14–O17: corpus, evaluation, context, and inference precision

Status: research complete; incumbent RoPE/256/FP32 retained. Protocol was
registered before new model scores were observed. Publication record follows
the completed research results below.
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

## Execution evidence

### O14: admitted data, with explicit transcription limits

O14 produced a frozen research bundle. It did not establish that OCR-derived
verse improves a trained model. That is the separate O15 question.

| Source | Role | Tokens | UTF-8 bytes/token |
|---|---|---:|---:|
| Karalaev, original training prefix | Training | 418,562 | 6.9869 |
| Orozbakov, book 2 | Training | 110,252 | 6.2396 |
| Orozbakov, book 3 | Training | 91,367 | 6.1993 |
| Orozbakov, book 5 | Training | 141,783 | 6.2882 |
| Orozbakov, book 4 | Validation | 99,630 | 6.1684 |
| Jusup Mamay | Final test | 408,359 | 5.0530 |

The official source catalogue is [Bizdin's Manas collection](https://new.bizdin.kg/knigi/category/manas).
Each cached source record retains its exact book page, download URL, PDF digest,
page count, and authorization basis. The admitted token-array file has SHA-256
`f0ce8f961e508e43e6409c4a9c9fe6131fee6762dc3d63b9cdebb325ff1e56ed`.
Raw PDFs, text, arrays, and complete generated continuations are not redistributed.

Reproduction from the repository, with authorized source access:

```sh
.venv/bin/python scripts/prepare_manas_expansion.py
.venv/bin/python scripts/extract_manas_verse.py
.venv/bin/python scripts/build_manas_expansion.py
.venv/bin/python scripts/experiment_data_context.py --mixture old --context 256 --output runs/data-control-20260905
.venv/bin/python scripts/experiment_data_context.py --mixture expanded --context 256 --control runs/data-control-20260905 --output runs/data-expanded-20260905
```

Source revision `9327c3f` contains the frozen data and training runners. The
control began before that commit; embedded source digests, not the parent Git
label, identify its actual code. The builder refuses to overwrite an existing
manifest. Training resumes a matching checkpoint and refuses mismatched
provenance; it does not silently restart from scratch.

### Diagnostic added after observing the O15 control curve

The completed old-data run improves familiar validation while its verse-book
loss rises. After observing this discrepancy and the early expanded-arm scores,
add one post-hoc attribution pass on the same fixed book-validation targets.
Compare final 3,000-update states, separating newline-bearing target tokens from
all other tokens. Reproduce the already recorded aggregate score and report each
group's contribution. This is a diagnosis of the observed measurement behavior,
not a new training arm, a data rewrite, a test-set comparison, or a changed
promotion rule. It cannot isolate all formatting effects or establish a pure
benefit from narrator diversity. Run only after both full-budget states exist.

### O15: both complete; expanded weights not promoted

Both arms completed 3,000 updates and 12,288,000 sampled targets. The expanded
arm passed its 900/1,200/1,500 continuation gates; no learning-rate horizon or
data weighting was changed in response to its scores. Training-loop subtotals
were 1,151.57 s for the control and 1,111.20 s for the expanded arm, excluding
evaluation and checkpoint saves. Sequential shared-host timing is not a paired
speed benchmark of these datasets.

| Checkpoint | Selection update | New-book loss | Familiar loss |
|---|---:|---:|---:|
| Previously deployed RoPE | Historical 2,900 | 11.846868 | 4.088184 |
| Fresh old-data control, primary-selected | 100 | 9.898599 | 6.957103 |
| Fresh old-data control, final budget | 3,000 | 11.965065 | 4.091312 |
| Expanded-data arm, primary-selected and final | 3,000 | 4.207867 | 4.133920 |

All rows here use the same new fixed target definitions. The early primary
minimum of the control is informative: minimizing a shifted book score alone
would select an undertrained model with poor familiar-domain performance.
The separate incumbent floor prevents this from lowering the acceptance bar.

The expanded model improves the registered primary score by 5.690733 nats
against the primary-selected control. Its familiar loss, however, exceeds the
deployed model by .0457364 nats, beyond the +.02 limit. Even against the fresh
control's final state, the difference is +.0426087. **Do not promote O15.**
The run proves a tradeoff at this fixed mixture and target budget, not that
additional Manas data cannot improve the model. No 20-generation audit or test
evaluation was spent on this already disqualified arm.

The post-hoc format diagnostic reproduced both final primary scores. Of 32,686
fixed targets, 4,976 contain newlines and 27,710 do not:

| Target group | Old-data mean loss | Expanded-data mean loss | Old / expanded contribution to total loss |
|---|---:|---:|---:|
| Newline-bearing | 25.10060 | .92450 | 3.82123 / .14074 |
| Other tokens | 9.60626 | 4.79748 | 8.14384 / 4.06712 |

Direct loss on newline-bearing targets accounts for about 47% of the total
reduction. Prediction also improves on other targets, but their preceding
contexts contain line breaks too. This partition cannot assign the remaining
gain exclusively to additional content or narrator diversity. The new corpus
and evaluation remain useful research artifacts even though their trained
candidate does not replace the deployed weights.

Evidence: `runs/data-control-20260905/result.json`,
`runs/data-expanded-20260905/result.json`,
`runs/data-selection-20260905/selection.json`, and
`runs/data-evaluation-20260905/format-diagnostic.json`. The matched full-budget
checkpoints and optimizer states remain local. O16 therefore uses the original
training corpus and the completed fresh 256-context control. It retains the
registered book-primary/familiar-control evaluation rather than changing a
criterion after O15 exposes a format shift.

### O17: screen on the incumbent RoPE export

This independent screen used the deployed 256-context export identified above.
If data/context selection replaces it, its successor needs a separate O17 run.
The comparison leaves FP32 parameters unchanged and enables BF16 autocast only.

| Measurement | FP32 | BF16 |
|---|---:|---:|
| Fixed familiar validation loss | 4.08818384 | 4.08820787 |
| 32-token prompt, 64 new tokens, median seconds | .193243 | .229446 |
| 248-token prompt, 64 new tokens, median seconds | .383007 | .370971 |
| 128-token prefill median milliseconds | 3.861 | 4.260 |
| One cached decode median milliseconds | 3.143 | 3.561 |
| Full 256-position persistent cache | 6 MiB | 3 MiB |
| Sampled live allocation after 129 cache positions | 113,640,448 B | 113,637,376 B |

Each generation row uses 20 interleaved pairs after three warmup pairs. Loss
drift is +.00002403 nats. Mean KL(P_FP32 || P_BF16) is .00007131 across 10,240
real targets; top-1 agreement is 98.94%. Cached/uncached comparisons cover 20
real positions including overflow. Maximum relative RMS error is 3.184e-7 in
FP32 and .004964 in BF16; all compared greedy choices agree. These numerical
checks pass their registered tolerances.

Performance fails: short generation takes 18.73% longer, while near-overflow
latency falls only 3.14%. Halving the persistent cache does not produce the
required 20% reduction in sampled live allocation. Allocation samples do not
measure whole-process or peak memory. No generation-quality audit was run after
this performance failure; `generation_metrics: false` means unassessed here,
not observed text degradation. Keep FP32 inference on this checkpoint. The MPS
screen does not measure or authorize production-CPU BF16.

Evidence: `runs/inference-bf16-20260905/{protocol,numerics,timing,result}.json`.
The helper later gained an optional audit-corpus argument without changing this
screen's scoring/timing behavior; the result retains its original helper digest.
The prediction of small numerical drift held. The prediction of a useful
inference speed/memory benefit did not hold on these prompts and this host.

### O16: stop at the 1,500-update ceiling

The unchanged original training corpus supplies both contexts. T=256 uses
8x256x2 and T=512 uses 4x512x2, preserving 4,096 targets per update and every
learned initial tensor. Both score the same new-book targets; the 512 arm also
scores them at context 256. Familiar-source evaluation always uses context 256.
No data or primary-score change was made after seeing O15's format sensitivity.

| Stage | Recent mean primary delta, 512 minus 256 | Decision |
|---|---:|---|
| 900 | -.042100 | Continue |
| 1,200 | -.090285 | Continue |
| 1,500 | +.004754 | Stop |

The final three deltas are +.112277, -.509526, and +.411511 nats. Their mean
does not reach the -.02 floor and their signs are inconsistent. Stop under the
registered rule. The runner consumed 6,144,000 targets in 1,500 updates, saving
the remaining half of the full budget. Training-loop subtotal: 736.80 seconds.

At update 1,500, book loss is 12.004335 with the model's full 512 context and
11.954920 with common context 256. The 256-trained control has book loss
11.592824 at that update. Familiar losses are 4.236962 versus 4.224035. These
are pilot measurements, not a completed model-selection comparison. The best
scheduled book checkpoint again occurs at update 100, with loss 9.878895.
No native promotion, generation audit, or protected test is run for this arm.
The result does not establish full-budget inferiority of RoPE context 512.

Evidence: `runs/context512-20260905/{result,history,stage-900,stage-1200,stage-1500}.json`
and `runs/context-selection-20260905/selection.json`. Keep context 256.

### Selection closed; final report-only evaluation

O15 failed the familiar-domain floor; O16 stopped without earning a full run;
O17 failed its inference performance gate. The selected artifact therefore
remains SHA-256 `abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f`.
No new model or precision mode is deployed. The existing accepted 20-generation
audit is reused; the model and its generation path are unchanged.

Only after that decision, evaluate the retained checkpoint on the complete
remaining target spans of each report-only split (after a 512-token context
prefix, 128-target stride, no repeated targets):

| Text | Targets | Scored bytes | Loss | Perplexity | Bits/byte | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Orozbakov 4, full validation remainder | 99,118 | 611,514 | 11.842740 | 139,070.98 | 2.769318 | 6.53% |
| Mamay, held-out test remainder | 407,847 | 2,060,821 | 12.370524 | 235,749.14 | 3.531991 | 2.68% |
| Original test remainder | 22,742 | 156,951 | 4.534839 | 93.21 | .947984 | 31.56% |

The high external-book losses expose weak transfer to these OCR-derived,
line-preserved sources. They are not scores for the rejected expanded model:
that model was never evaluated on the protected Mamay test. Nor does the
original-test score supersede the older 100-random-batch result of 4.531258:
the weights are the same and the target/context definitions differ.

Evidence: `runs/data-context-final-20260905/final-evaluation.json`.
Total new training in O15/O16: 7,500 updates, 30,720,000 targets, and 2,999.58
seconds in the training-loop subtotals. Data preparation, evaluation, and saving
are outside that subtotal. No paid compute, new SFT, quantization, request
batching, attention backend, or runtime upgrade was introduced.

### Publication and site-only release

The completed research, results, decisions, and paper sources were committed and
pushed. Site source `af18627a4b70be3020244d35365a7973ba61fdce` updates only the
experimental second part of the Russian and English essay; Part I is unchanged
byte-for-byte. The exact source was built on OVH and accepted through an isolated
candidate, a configuration-diff guard, and a site-only cutover.

The public article passed EN/RU navigation and Russian deep-link reload. One
real browser generation returned 16 tokens (0.4 seconds displayed). The model
container remained `8f30e0f589b154cc819566ed6ffbe31149c3e26efa7571f5cf37c10341623a2b`
on release `437195fb1a109e3ff392aaae7c4c350518b9df76`; no model restart or weight
replacement occurred. The temporary website candidate was removed. The site
repository's `docs/OVH_PRODUCTION.md` records the exact site image and rollback.

The owner's subsequent paper feedback changes presentation, not these research
records: the manuscript now describes architecture, mechanics, and engineering
evidence in an impersonal academic structure without a chronological experiment
catalogue. Numerical source files and historical decisions remain intact.
