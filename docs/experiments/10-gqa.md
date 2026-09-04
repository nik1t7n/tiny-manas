# 10 — Grouped-query attention after a correct KV cache

Status: **completed; rejected on prediction quality**. Date: 2026-09-04.

## Measured result

Run: `runs/optimization-10-gqa-20260904`; both arms completed exactly 150
updates from the O02 checkpoint. The immutable input hash is
`31499eb747c98bcada1c48b12205033cded96573269bc15464ccafd9905e2167`.
The run records native-cache model revision `b701565` and all source hashes.

| Metric | MHA, 8 KV heads | GQA, 2 KV heads |
|---|---:|---:|
| Parameters | 26,877,696 | 25,103,616 |
| Validation loss before adaptation, exact-byte windows | 4.2968102825 | 5.0746953695 |
| Validation loss after 150 updates | 4.3030295332 | 4.4970472201 |
| Validation BPB after adaptation | 0.8777108774 | 0.9172856544 |
| Full FP32 cache | 6 MiB | 1.5 MiB |
| Paired median decode, 128 past positions | 2.48075 ms | 2.50810 ms |
| Median update, first five excluded | 0.309971 s | 0.360744 s |
| Mean repeated trigrams, 20 outputs | 0.0472084 | 0.0168854 |
| Longest normalized training-word match | 8 | 7 |

Eight-KV equivalence, two-KV cached/uncached correctness and physical storage
checks passed. Persistent cache shrank 75%; paired decode regressed 1.10%,
within the 5% latency gate. The sequential timing would have suggested a 7.57%
regression; the preregistered interleaved comparison controls that measurement.
The explicit KV expansion still creates temporary eight-head tensors.

GQA recovered some prediction quality after pooling, but its final loss remained
0.194018 nats worse than adapted MHA and 0.200237 worse than the original accepted
MHA. Both exceed the 0.05 ceiling. The adaptation control also failed to improve
on the original, so neither set of adapted weights replaces the incumbent.
Do not extend the budget retrospectively or treat this as fresh GQA pretraining.

### Raw-output review and decision

All 20 complete generations from each arm were read. MHA samples 3 and 12
degenerate into repeated names/subwords; 5–7 cycle through choro/title phrases;
9–11, 13, 15 and 18 repeat arrivals and combat actions. Samples 1, 2, 4, 8,
14, 16, 17, 19 and 20 retain recognizable epic motifs but drift among actors
and include malformed forms. These observations agree with the existing model's
limitations rather than showing a benefit from extra adaptation.

GQA's lower exact-trigram count does not establish better prose: sample 6 loops
through hero epithets, 14 and 18 repeatedly return to Esenkan/Beijing, and
20 recycles army/combat fragments. Samples 1, 2, 7, 8, 12 and 15–17 mix unstable
speech and identities; 3–5, 9–11, 13 and 19 repeat local action patterns and
contain malformed words. The quality gate already fails quantitatively.

Decision: keep the accepted original MHA weights plus O09 caching. Preserve
the conversion/adaptation code as research evidence; do not reinterpret its
pooled projection tensors as an ordinary MHA checkpoint. The candidate saves
only 4.5 MiB of cache at our maximum request length, which does not justify the
measured quality loss. No protected test was used to choose this outcome.

MHA adaptation checkpoint SHA-256:
`fa8449b15a00e4743b6b87f1031f1cf9d98db7c05212d825b4d180762eec6d4a`.
GQA adaptation checkpoint SHA-256:
`ebc2d3f88953a668e6ed85e4d20bdbb5ed708bfe5c6971c7b571fd51819688ee`.
The preregistration below preserves the original plan and preparation history.

Prepared source, not executed: `scripts/gqa_candidate.py` converts a CPU copy of
the selected MHA checkpoint. It preserves Q/output weights, pools K/V rows and
biases in adjacent groups, and keeps RoPE buffers if present. The shared research
cache stores only the declared KV-head count. Ordinary production attention and
generation remain unchanged. No parity, speed or quality result is claimed.

The preregistered MPS implementation explicitly repeats each of the two K/V
heads four times immediately before ordinary eight-head SDPA. This is the
reference GQA operation described in the [PyTorch 2.13 SDPA documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html),
not an unsupported-kernel recovery path. Persistent cache remains two-headed;
temporary attention inputs are eight-headed. Account for these temporary
allocations and timing instead of claiming a fourfold reduction in all attention
memory or work. Native fused GQA is not assumed to exist on this MPS backend.

Prepared runner: `scripts/experiment_gqa.py`, CLI parsing only. It requires an
explicit accepted checkpoint/hash and first checks eight-KV MHA equivalence,
two-KV cached/uncached parity (including context overflow), and actual cache
storage. Both arms then use the same random training windows (sampler seed 1338),
fresh AdamW, BF16, batch 8, accumulation 2 and constant learning rate .00003 for
exactly 150 updates. This shared adaptation sampler is explicit even if a prior
pretraining recipe used equal-byte epochs; neither adaptation arm uses a different
sampler. Every 50 updates saves model, optimizer and CPU/MPS RNG state.

Before/after validation scores the same pinned exact-byte validation windows.
Both adapted models receive the same 20-generation audit. Afterward, an alternating
paired decode comparison uses 128 past positions, two warmup pairs and ten measured
pairs. Sequential per-arm timings are retained but do not decide the 5% latency
gate. Numerical gates cannot promote a model without raw-output inspection. No
part of this runner has yet executed model computations on the GPU.

## Question and forecast

Can eight query heads share two KV heads without losing useful prediction
quality? Each group of four queries reads one K/V pair. Query count, head width
48, residual width 384 and eight layers stay fixed. This does not divide all
attention work by four: there are still eight query heads and their scores.

For the O09 B=1,T=256 FP32 example, persistent KV storage falls from 6 MiB to
1.5 MiB. Measure storage, not merely tensor views, and separately measure any
temporary expansion of K/V required by the MPS attention implementation. The
absolute cache saving is modest on this small short-context model.

Projection parameters also fall: the combined QKV output shrinks from 1152 to
576 features, saving 221,760 parameters per layer with biases, or 1,774,080 total.
Do not increase another module to compensate; the question is the actual
quality/resource tradeoff of sharing KV projections.

Forecast: definite persistent cache reduction if implemented correctly, uncertain
wall-time improvement, and possible quality loss. Larger-model CUDA results do
not predict this M5 workload.

## Bounded adaptation comparison

Use the [GQA paper's section 2](https://arxiv.org/html/2305.13245v3#S2) as a
conversion recipe: mean-pool K and V projection rows within each group of four
heads, including biases. Preserve Q and all other weights. Pool K before applying
any rotary position transformation. Save a distinct checkpoint/config; never
overwrite or reinterpret the MHA checkpoint.

First verify the eight-KV configuration reproduces the existing MHA path on real
inputs, then verify the two-KV candidate's cached/uncached parity, finite
gradients and actual cache storage. Shape correctness is not a quality result.

Adapt converted GQA and an unchanged MHA control from the same accepted checkpoint
for 150 optimizer updates each, with identical real windows, seed, precision,
batch, accumulation and optimizer settings. This is an explicit 5%-of-3000-step
adaptation experiment, not fresh pretraining or a promise that 5% is sufficient.
Use fresh optimizer states in both arms and constant learning rate 0.00003.
Evaluate before and after on identical validation targets; record both candidate
versus adapted control and candidate versus original accepted checkpoint.

Inspect 20 raw continuations with the version-2 copying audit. Accept only if
validation is no worse than either reference by 0.05 nats, no obvious new output
collapse occurs, actual persistent cache saving is >=60%, and warmed decode
latency regresses by no more than 5%. Report that acceptance may be primarily a
memory tradeoff, not a speed win. If this bounded adaptation fails, retain MHA;
do not keep extending training until a favorable result appears without a new
explicitly recorded hypothesis and budget.
