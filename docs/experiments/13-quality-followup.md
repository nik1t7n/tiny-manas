# 13 — Budgeted RoPE and RMSNorm quality follow-up

Date: 2026-09-04. Protocol preregistered in `29d30e3`; research completed.

## Why reopen these experiments?

O06 and O07 passed numerical checks but stopped at their original latency
ceilings before quality training. That established a cost on this MPS
implementation, not that either architecture was worse. The owner explicitly
accepted approximately 10–15% slower updates as worth investigating and requested
staged quality runs rather than automatically funding two complete runs.

Forecast: RMSNorm will probably track LayerNorm closely; RoPE may improve relative
position learning, but context 256 and the small single-source corpus limit the
opportunity. Consistently lower held-out loss falsifies the expectation of no
useful benefit. Falling training loss by itself does not count as success.

## Matched comparison

Use the accepted 32k tokenizer, original split, context 256, width 384, eight
layers/heads, batch 8, accumulation 2, dropout .2, eager BF16 training and FP32
evaluation. Each candidate changes one component only. Shared weights come from
the fresh `runs/optimization-06-reference-20260904/initial-model.pt`, SHA-256
`89ff070be491d496dbdbd708e1073eacc258f7214fb111e4c3517b57a1c54457`,
not a trained checkpoint.

Keep seed 1337, identical training/evaluation windows, and the original
3,000-update warmup/cosine schedule. A pilot is a prefix of that run; shortening
its schedule would invalidate continuation. Checkpoints retain optimizer,
CPU/MPS random states, sampler progress and source hashes. Resume, do not restart.

Reuse the O02 BF16 control curve only after a fresh 100-update classic control
calibration. Require absolute differences <= .005 at its six recorded training
loss points and <= .003 for train/validation evaluation. These tolerances allow
small MPS numerical variation, not a different recipe. If calibration fails,
stop and diagnose. This single-seed comparison does not measure seed variance.

The model-source diff since O06/O07 only adds inference caching; training forward
is unchanged. Refresh the existing real-data numerical/gradient probes against
current hashes nonetheless. Preserve their latency verdict. `--quality-followup`
explicitly replaces the cost-only veto, not correctness or provenance checks.
Run sequentially on MPS with the existing .65 per-process memory fraction.

## Stages fixed before candidate quality is observed

Evaluate every 100 updates using the same 30 batches as the historical control.
Do not prune for quality before 600 updates. At 600, 900, 1,200 and 1,500 updates,
delta means candidate validation loss minus control loss at the same step.
Negative is better. Compare the mean delta over the latest three checkpoints
with the previous three.

- Stop a clearly worse arm when the recent mean delta exceeds +.05, all three
  deltas exceed +.03, and the relative gap improved by less than .01.
- From 900 updates, stop for insufficient evidence when the recent mean exceeds
  -.02 and the relative gap improved by less than .01. This is a budget decision,
  not proof that the arm could never catch up.
- Otherwise allow the next stage. At the 1,500-update pilot ceiling, continue to
  3,000 only if the mean delta is <= -.02 and all three recent deltas are negative.
- Immediately stop nonfinite loss/gradients or an actual resource failure.

The .02 loss threshold is a practical effect floor (about 2% perplexity), not a
statistical significance level. Overlapping-window batches and adjacent
checkpoints are not independent experimental replications. Report the full curve.

This borrows resource allocation from
[successive halving / Hyperband, section 3.1](https://jmlr.org/papers/volume18/16-558/16-558.pdf),
but is not Hyperband: two prespecified architectural candidates and a grace
period replace a hyperparameter search. Intermediate loss cannot guarantee the
ranking at convergence; a budget-pruned run remains inconclusive about that.

## Promotion is separate

An early lead buys more training, not deployment. After 3,000 updates, select the
best scheduled-validation checkpoint and evaluate the existing independent
100-batch validation set. Require loss <= 4.345779147148132 - .02 against the
incumbent, no material generation-audit regression, and inspect all 20 fixed raw
continuations. Use explicit uncached generation for architecture research: the
current production KV cache supports learned positions only. A winning RoPE
model needs a real compatible serving path before promotion. Do not select using
the test set. Marginal or contradictory evidence does not justify deployment.

## Execution record

### Control calibration and refreshed probe

The 100-update control calibration passed. The first training loss reproduced
exactly; the largest difference among the six logged training points was
.0001163483. Train-evaluation loss differed by .0000055154 and validation by
.0000064850. Evidence: `runs/quality-followup-control-20260904/result.json`.
The matched historical curve can therefore be reused without another full run.

The refreshed RoPE probe passed the existing mathematical/gradient checks. Its
paired median update times were .332212 s (control) and .365056 s (RoPE), +9.89%.
This session's timing differs from the earlier +12.27%; small latency ceilings
are sensitive to measurement conditions. Neither timing establishes quality.
Evidence: `runs/quality-followup-rope-probe-20260904/result.json`.

Both use the real M5 / 16 GB MPS host, PyTorch 2.13.0. The control source was
the exact code subsequently committed as `29d30e3`; its environment commit
field still names the parent because calibration ran before committing.
Embedded file hashes identify the actual measured source.

### Reproduction

For the historical staged runs use source revision `29d30e3` and run from the
repository with `.venv/bin/python`. Later native integration changes provenance
hashes; do not reinterpret old run directories as runs of the latest source.
Initial weights and real data
must already exist at the pinned paths; these commands do not download or create
replacement inputs. Use a new output directory for probes. Execute sequentially.

```sh
.venv/bin/python scripts/experiment_architecture.py --vocabulary 32768 --recipe random-windows --output runs/quality-followup-control-20260904 --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --change control --quality-followup --control-metrics runs/optimization-02-bf16-20260904T055741Z/metrics.jsonl --stop-after-segment 1
.venv/bin/python scripts/experiment_architecture_probe.py --vocabulary 32768 --change rope --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --output runs/quality-followup-rope-probe-20260904
.venv/bin/python scripts/experiment_architecture.py --vocabulary 32768 --recipe random-windows --output runs/quality-followup-rope-20260904 --reference-initial runs/optimization-06-reference-20260904/initial-model.pt --change rope --probe runs/quality-followup-rope-probe-20260904/result.json --quality-followup --control-metrics runs/optimization-02-bf16-20260904T055741Z/metrics.jsonl --control-calibration runs/quality-followup-control-20260904/result.json
```

For the separate RMSNorm arm, replace `rope` with `rmsnorm` in the probe and
candidate commands and paths; retain the same classic initial state and control.
Never run the GPU arms concurrently. A successful early stage continues in the
same invocation; its optimizer is not reset.

### RoPE: full quality run completed

RoPE passed every pilot gate. Mean candidate-minus-control validation differences
over the last three checkpoints were -.34082 at 600, -.38106 at 900, -.38530 at
1,200 and -.35025 at 1,500. This justified continuing the same state to 3,000.

| Measurement | Accepted BF16 control | RoPE |
|---|---:|---:|
| Best scheduled checkpoint | 2,900 | 2,900 |
| Independent 100-batch validation loss | 4.34577915 | 4.11583555 |
| Independent validation perplexity | 77.15213 | 61.30341 |
| Validation top-1 accuracy | about 31.55% | 35.47% |
| Exact-byte validation BPB | .87644231 | .82769511 |
| Mean repeated-trigram ratio, 20 continuations | .04097287 | .04133698 |
| Worst repeated-trigram ratio | .31111111 | .19018405 |
| Maximum normalized copied word span | 9 | 11 |
| Parameters | 26,877,696 | 26,779,392 |

The independent loss improvement is .22994360 nats, equivalent to 20.54% lower
perplexity, well above the .02 practical floor. This is held-out prediction on
the same corpus, not proof of human-rated fluency or general Kyrgyz ability.
Training updates alone took 1,101.24 seconds; this excludes evaluation, saving
and the final audit. Do not compare that subtotal with O02's entire-loop timing.
Maximum sampled allocated/driver memory was 2,263,599,104 / 4,668,129,280 bytes.

All 20 complete continuations were read. Sample 7 loops on the choro/Kyrgyl
phrases; 10 repeats the paired Zhang-Jung names, including the maximum copied
11-word span. Samples 1 and 5 repeat combat and coming-of-age lines. Samples
3, 11, 12 and 15 drift through names or repeat address/group formulas; 9 repeats
combat descriptions. Samples 2, 4, 6, 8, 13, 14 and 16–20 retain recognizable
epic material but still contain malformed word forms, abrupt actor changes or
unstable event continuity. In particular, sample 6 has zero repeated trigrams
without becoming a coherent narrative. There is no obvious new corpus-wide
collapse; the nearly unchanged mean repetition and lower worst sample support
promotion for prediction quality, not a claim that storytelling is solved.
The slightly longer maximum match does not establish absence of memorization.

Evidence: `runs/quality-followup-rope-20260904/{result.json,history.json,generation-audit.json}`.
Selected research checkpoint SHA-256:
`c43e76c6307ce345da3983e04132b95291270c07bb7c38763c48b3ef394b6473`.
Candidate initial-state SHA-256:
`f2b07c5fbed960239a232caae0a9b48537855068df769b58d71662b770e9812f`.
The protected test was not consulted for this follow-up decision.

RoPE passes the research quality gate. Native loading/cache/export and deployment
remain separate pending release gates; production has not yet changed.

### RMSNorm

The refreshed real numerical/gradient probe passed. Its paired update medians
were .320211 s (control) and .351302 s (RMSNorm), +9.71%. The original 5% cost
gate still reports `cost_gate_failed`; the explicit quality follow-up authorizes
training anyway. Evidence: `runs/quality-followup-rmsnorm-probe-20260904/result.json`.
The pilot stopped at 900 updates under the registered rule. Its last three
candidate-minus-control deltas were +.00268645, +.00377803 and +.00347058:
mean +.00331169, versus +.00625891 over the preceding three checkpoints. The
gap closed by only .00294722, below the .01 continuation threshold, and there
was no .02 advantage. Keep LayerNorm. This is a limited-budget non-promotion,
not proof of what a full RMSNorm run would achieve.

| Updates | Control validation loss | RMSNorm validation loss |
|---|---:|---:|
| 100 | 7.109263 | 7.113415 |
| 200 | 6.312214 | 6.319339 |
| 300 | 5.778761 | 5.787064 |
| 400 | 5.476497 | 5.484125 |
| 500 | 5.275445 | 5.281368 |
| 600 | 5.156655 | 5.161881 |
| 700 | 5.055046 | 5.057732 |
| 800 | 5.016093 | 5.019871 |
| 900 | 4.874712 | 4.878183 |

RMSNorm training updates took 321.33 seconds, excluding evaluation/checkpoint
overhead. Stopping avoided 2,100 of its planned 3,000 updates (70%). We also
reused the completed full control instead of retraining it; the extra control
work was a 100-update calibration plus the bounded numerical probes. No paid
compute was used. Both candidates' resumable states remain on disk.

Evidence: `runs/quality-followup-rmsnorm-20260904/{result.json,history.json,decision-900.json}`.
Saved checkpoint SHA-256:
`4596b776397448a3cb295586d1b982b0137313e3cdf32d7816e705d296db8a67`.
No final generation audit or protected-test evaluation was run for the pruned arm.

### Native RoPE promotion gate

`scripts/promote_rope.py` exported the pinned winner through the normal checkpoint
exporter and loader. CPU logits exactly matched the research candidate. Native
fresh initialization also exactly matched every saved candidate tensor. This
matters: removing a module must not silently change the initialization of all
later shared weights. The new native constructor preserves classic initialization
order before removing the learned position table.

On one real validation continuation, cached prefill and decode differed from
uncached logits by at most .00000620; greedy generation matched across context
overflow. The cache stores rotated K and ordinary V, uses the correct new-token
position for Q/K, and still explicitly rebuilds after cropping. Its full FP32
storage remains 6 MiB. This change does not claim longer trained context.

Native config: `configs/manas01-27m-rope.toml` (early stopping disabled to retain
the measured full horizon). Legacy configs default explicitly to learned
positions. The existing source protocol is retained in exported research
provenance rather than rewritten to pretend the native config was the original.

Export: `artifacts/tiny-manas-27m-rope-20260904.pt`, 107,157,586 bytes;
SHA-256 `abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f`.
Evidence: `runs/quality-followup-rope-20260904/promotion.json`.
Production release passed; the previous image and weights remain available for
rollback. See the final section below.

### Post-selection test

After selecting RoPE, completing its native parity gate and closing the RMSNorm
pilot, the ordinary evaluator ran once on 100 test batches, seed 1837, FP32,
204,800 target tokens. Loss: **4.53125801**, perplexity: **92.87533**, top-1:
**31.698%**, top-5: **48.527%**, approximate BPB: **.94590037**. The old BF16
checkpoint's same-protocol test loss was 4.75756063. This measurement reports
the already selected model; it did not decide stage continuation or select
between candidates. The test split is not a newly unseen external benchmark.
Evidence: `runs/quality-followup-release-20260904/test-evaluation.json`.

### Production release: completed

The clean, pushed native source commit was
`437195fb1a109e3ff392aaae7c4c350518b9df76`. Only its tracked archive was built on
OVH; the private model file was transferred separately and its SHA-256 verified.
Image: `manas-gpt:437195fb1a109e3ff392aaae7c4c350518b9df76`.
Image ID: `sha256:e3606db62aaf4cb434f415bbebd61dba9dbf87e3269decaad7701e5b6a900e99`.

The isolated real candidate loaded the new export, reported RoPE and KV caching,
rejected unauthenticated generation with 401 and generated 16 tokens. Coolify's
stored Compose/image variable and generated runtime configuration were updated
together. A fail-closed comparison allowed only the model image, checkpoint
path/hash and shared `MANAS_IMAGE` metadata to change. Cutover used component-only
`docker compose up -d --no-deps --no-build --pull never manas`.

Production container `manas-p5h6dziyjkyfejbiyqa8firs` is healthy, running exact
image ID above, with export SHA-256 `abc13354d5cb1cc94c966985d95252befdfaf9f25b19c1884701442f4e519d8f`,
step 2,900, context 256 and 26,779,392 parameters. Limits remain 1 GiB/two CPUs;
the model mount remains read-only. Production container ID:
`8f30e0f589b154cc819566ed6ffbe31149c3e26efa7571f5cf37c10341623a2b`.
The website container stayed exactly
`e87cc87d02082bdc1f7cb5de265b58c99395f5e691ef6dbaf3e30eb1e1a1c59a`.

A real request through `https://nik1t7n.com/api/manas/generate` returned 200 and
16 generated tokens (54.80 tokens/s in this one CPU smoke, not a controlled speed
comparison). The public four-field request contract, rate limits and output caps
were unchanged. No new browser UI acceptance is claimed for this model-only
release. The temporary candidate container was removed after acceptance.

Rollback retains image `manas-gpt:67afe940d1f38ba4e0712c0851c69317957ac8ac` and
`/models/tiny-manas-27m-bf16-20260904.pt`, SHA-256
`4c6f70883564df6c46849c0849f38b06195b4dcaba3bde2572ef60eec4cf3494`.
Restore that image, checkpoint path and hash together through Coolify, render its
configuration, then recreate only `manas`. The old runtime/controller configuration
and acceptance records are in root-only
`/opt/manas-gpt/releases/437195fb1a109e3ff392aaae7c4c350518b9df76-rollback`.
Do not publish its environment or raw configuration.

An initial build invocation reached chmod before the asynchronous artifact
transfer finished and stopped without building or touching production. The build
was then run independently from the completed source archive; cutover waited for
the complete model transfer and hash verification. No partial artifact was loaded.
