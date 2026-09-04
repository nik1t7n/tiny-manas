"""Bounded real-data precision probe, before any production-training change."""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import statistics
import time

import torch

from manas_gpt.config import load_config
from manas_gpt.data import RandomWindowSampler, load_split, load_tokenizer
from manas_gpt.experiment import environment_info, evaluate_random_batches, learning_rate, require_mps, seed_everything
from manas_gpt.model import ManasGPT

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "runs" / ("optimization-02-bf16-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(exist_ok=False)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(0.65)
    config = load_config(ROOT / "configs/manas01-27m.toml")
    tokenizer = load_tokenizer()
    train = load_split("manas01-full", "train")
    validation = load_split("manas01-full", "validation")
    report = {"environment": environment_info(device), "config": config.as_dict(), "steps": 100,
              "memory_kind": "maximum of samples after forward/backward/update, not true peak",
              "split_hashes": {name: hashlib.sha256((ROOT / f"data/processed/manas01-full/{name}.bin").read_bytes()).hexdigest() for name in ("train", "validation")}, "modes": {}}
    (out / "setup.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode in ("fp32", "bf16"):
        seed_everything(config.run.seed)
        model = ManasGPT(config.model.with_vocab_size(tokenizer.get_vocab_size())).to(device)
        optimizer = model.configure_optimizer(config.training.learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2))
        sampler = RandomWindowSampler(train, 8, 256, config.run.seed)
        context = (lambda: torch.autocast("mps", dtype=torch.bfloat16)) if mode == "bf16" else nullcontext
        rows = []
        maximum = {"allocated": 0, "driver": 0}

        def sample_memory():
            maximum["allocated"] = max(maximum["allocated"], torch.mps.current_allocated_memory())
            maximum["driver"] = max(maximum["driver"], torch.mps.driver_allocated_memory())

        with (out / f"{mode}-steps.jsonl").open("x") as log:
            for step in range(100):
                model.train()
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config, step)
                optimizer.zero_grad(set_to_none=True)
                torch.mps.synchronize()
                started = time.perf_counter()
                total_loss = 0.0
                for _ in range(config.training.gradient_accumulation_steps):
                    x, y = sampler.next(device)
                    with context():
                        logits, loss = model(x, y)
                    sample_memory()
                    total_loss += float(loss.detach()) / config.training.gradient_accumulation_steps
                    (loss / config.training.gradient_accumulation_steps).backward()
                    sample_memory()
                    output_dtype, loss_dtype = str(logits.dtype), str(loss.dtype)
                    del logits, loss
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                torch.mps.synchronize()
                elapsed = time.perf_counter() - started
                sample_memory()
                row = {"step": step, "loss": total_loss, "seconds": elapsed, "gradient_norm": float(norm)}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if (step + 1) % 20 == 0:
                    print(mode, row, flush=True)
        metrics = evaluate_random_batches(model, RandomWindowSampler(validation, 8, 256, 9001), device, 32)
        result = {"median_step_seconds": statistics.median(r["seconds"] for r in rows[10:]),
                  "last_10_mean_train_loss": statistics.mean(r["loss"] for r in rows[-10:]),
                  "validation_fp32": metrics, "sampled_max_memory_bytes": maximum,
                  "dtypes": {"output": output_dtype, "loss": loss_dtype,
                             "parameters": sorted({str(p.dtype) for p in model.parameters()}),
                             "optimizer_state": sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)})}}
        report["modes"][mode] = result
        torch.save({"model": model.state_dict(), "model_config": asdict(model.config)}, out / f"{mode}-probe.pt")
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(mode, json.dumps(result), flush=True)
        del optimizer, model, x, y, norm
        gc.collect()
        torch.mps.empty_cache()
    print(f"Completed probe: {out}", flush=True)


if __name__ == "__main__":
    main()
