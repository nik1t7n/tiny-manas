"""Check real MPS dropout gradients before timing activation checkpointing."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from manas_gpt.data import RandomWindowSampler, load_split, load_tokenizer
from manas_gpt.experiment import environment_info, load_checkpoint, require_mps, seed_everything

ROOT = Path(__file__).resolve().parents[1]


class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False, preserve_rng_state=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    out = ROOT / "runs" / ("optimization-04-checkpointing-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(exist_ok=False)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(0.65)
    load_tokenizer()
    train = load_split("manas01-full", "train")
    amp = (lambda: torch.autocast("mps", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    report = {"status": "running", "environment": environment_info(device),
              "checkpoint": str(source), "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "precision": args.precision, "execution": "eager", "rng_preservation": "standard_pytorch",
              "memory_kind": "samples after forward/backward/update; not exact peaks", "modes": {}}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def load(mode):
        model, payload = load_checkpoint(source, device)
        del payload
        if mode == "checkpointed":
            model.blocks = nn.ModuleList(CheckpointBlock(block) for block in model.blocks)
        model.train()
        optimizer = model.configure_optimizer(0.00003, 0.1, (0.9, 0.95))
        return model, optimizer

    save()
    print(f"RUN {out}", flush=True)
    try:
        evidence = {}
        for mode in ("ordinary", "checkpointed"):
            model, optimizer = load(mode)
            x, y = RandomWindowSampler(train, 8, 256, 1338).next(device)
            seed_everything(1337)
            with amp():
                logits, loss = model(x, y)
            after_forward_rng = torch.mps.get_rng_state().clone()
            loss.backward()
            torch.mps.synchronize()
            after_backward_rng = torch.mps.get_rng_state().clone()
            gradient = torch.cat([p.grad.detach().float().cpu().flatten() for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            parameters = torch.cat([p.detach().float().cpu().flatten() for p in model.parameters()])
            evidence[mode] = {"loss": loss.item(), "gradient": gradient, "parameters": parameters,
                              "rng": after_backward_rng, "cpu_rng": torch.get_rng_state().clone(),
                              "backward_preserved_mps_rng": torch.equal(after_forward_rng, after_backward_rng)}
            del model, optimizer, x, y, logits, loss, gradient, parameters
            gc.collect()
            torch.mps.empty_cache()
        base, candidate = evidence["ordinary"], evidence["checkpointed"]
        relative = float((base["gradient"] - candidate["gradient"]).norm() / base["gradient"].norm().clamp_min(1e-12))
        report["parity"] = {"loss_difference": abs(base["loss"] - candidate["loss"]),
                            "gradient_relative_l2": relative,
                            "maximum_updated_parameter_difference": float((base["parameters"] - candidate["parameters"]).abs().max()),
                            "same_post_backward_mps_rng": torch.equal(base["rng"], candidate["rng"]),
                            "same_post_backward_cpu_rng": torch.equal(base["cpu_rng"], candidate["cpu_rng"]),
                            "ordinary_backward_preserved_mps_rng": base["backward_preserved_mps_rng"],
                            "checkpoint_backward_preserved_mps_rng": candidate["backward_preserved_mps_rng"]}
        save()
        if not (relative <= 1e-4 and report["parity"]["loss_difference"] <= 1e-4
                and report["parity"]["maximum_updated_parameter_difference"] <= 1e-6
                and report["parity"]["same_post_backward_mps_rng"]
                and report["parity"]["same_post_backward_cpu_rng"]
                and candidate["backward_preserved_mps_rng"]):
            raise RuntimeError(f"Checkpoint correctness gate failed: {report['parity']}")
        del base, candidate, evidence
        gc.collect()
        for mode in ("ordinary", "checkpointed"):
            model, optimizer = load(mode)
            sampler = RandomWindowSampler(train, 8, 256, 1338)
            seed_everything(1337)
            rows = []
            with (out / f"{mode}-steps.jsonl").open("x") as log:
                for step in range(35):
                    batches = [sampler.next(device) for _ in range(2)]
                    maximum = {"allocated": 0, "driver": 0}

                    def sample_memory():
                        maximum["allocated"] = max(maximum["allocated"], torch.mps.current_allocated_memory())
                        maximum["driver"] = max(maximum["driver"], torch.mps.driver_allocated_memory())

                    optimizer.zero_grad(set_to_none=True)
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    mean_loss = 0.0
                    for x, y in batches:
                        with amp():
                            logits, loss = model(x, y)
                        sample_memory()
                        mean_loss += loss.item() / 2
                        (loss / 2).backward()
                        sample_memory()
                        del logits, loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    torch.mps.synchronize()
                    sample_memory()
                    row = {"step": step, "seconds": time.perf_counter() - started,
                           "loss": mean_loss, "sampled_memory_bytes": maximum}
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
            report["modes"][mode] = {
                "median_update_seconds": statistics.median(row["seconds"] for row in rows[5:]),
                "maximum_sampled_allocated_bytes": max(row["sampled_memory_bytes"]["allocated"] for row in rows[5:]),
                "maximum_sampled_driver_bytes": max(row["sampled_memory_bytes"]["driver"] for row in rows[5:]),
            }
            save()
            del model, optimizer, batches, x, y
            gc.collect()
            torch.mps.empty_cache()
        base, candidate = (report["modes"][name] for name in ("ordinary", "checkpointed"))
        report["allocated_memory_reduction_fraction"] = 1 - candidate["maximum_sampled_allocated_bytes"] / base["maximum_sampled_allocated_bytes"]
        report["latency_increase_fraction"] = candidate["median_update_seconds"] / base["median_update_seconds"] - 1
        report["status"] = "measured_pending_review"
        save()
        print(json.dumps({"run": str(out), "modes": report["modes"]}), flush=True)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
