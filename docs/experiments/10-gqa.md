# 10 — Grouped-query attention after a correct KV cache

Status: preregistered, **not run**. Follow O09. Date: 2026-09-04.

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
