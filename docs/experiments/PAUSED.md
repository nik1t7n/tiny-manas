# Owner-requested stop — 2026-09-04

**Superseded later the same day:** the owner explicitly said to continue to the
end. Resume from the saved 2,500-update checkpoint, retaining the original
experiment budgets and decision gates. The account below documents the stop;
it is no longer an active prohibition.

The owner explicitly requested an immediate stop. Do not resume training,
evaluation, generation audits, KV-cache or GQA experiments automatically.
Continue only after a new explicit owner instruction.

The sole running experiment process, PID 93155 (O08 SwiGLU), was immediately
suspended and then terminated. A process check found no remaining architecture,
tokenizer, KV-cache or GQA experiment processes. No experiment was newly started.

## Durable recovery point

- Run: `runs/optimization-08-swiglu-full-20260904`.
- Saved progress: 25 segments, **2,500 of 3,000 optimizer updates**.
- Resume: `resume.pt`; best selected weights: `best-model.pt`.
- `history.json` ends at segment 25; its scheduled validation loss is 4.362173.
- Interrupted work after that durable checkpoint is not claimed as saved.
- Final independent validation and the 20-generation audit have not completed;
  SwiGLU remains unaccepted. Do not infer its final quality from interim loss.
- Accepted model remains the O02 BF16-trained GELU/MHA checkpoint recorded in
  `accepted-state.json`. The original frozen baseline is untouched.

After explicit permission, verify the saved source/config hashes and resume the
existing run using the original command; do not create another training run:

```bash
.venv/bin/python scripts/experiment_architecture.py \
  --vocabulary 32768 --recipe random-windows --change swiglu \
  --reference-initial runs/optimization-06-reference-20260904/initial-model.pt \
  --probe runs/optimization-08-swiglu-probe-20260904/result.json \
  --output runs/optimization-08-swiglu-full-20260904
```

The checkpoint retains optimizer and CPU/MPS random-generator states. The
runner restores them and replays the private sampler's index draws to the saved
boundary. Keep implementation files and the active configuration unchanged until
resumption, or explicitly resolve provenance before proceeding.

O01–O07 decisions and reports are complete. O09/O10 are prepared but unrun;
O11 remains conditional. Final README completion and protected-test evaluation
remain pending. No push or deployment occurred.
