"""Explicit real-checkpoint compile experiment; no eager substitution on failure."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from manas_gpt.data import RandomWindowSampler, load_split, load_tokenizer
from manas_gpt.experiment import environment_info, load_checkpoint, require_mps, seed_everything
from manas_gpt.model import ManasGPT

ROOT = Path(__file__).resolve().parents[1]


class ExplicitPositionLookup(ManasGPT):
    """Research candidate: identical positions, indexed per batch before lookup."""

    def forward(self, token_ids, targets=None, *, last_position_only=False):
        if last_position_only and targets is not None:
            raise ValueError("last_position_only cannot be used with training targets")
        batch, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError("Sequence exceeds the configured context")
        positions = torch.arange(time, device=token_ids.device).expand(batch, time)
        x = self.embedding_dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, -1:, :] if last_position_only else x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=("fp32", "bf16"))
    parser.add_argument("--explicit-position-lookup", action="store_true",
                        help="Diagnose broadcast-backward failure with mathematically identical batched lookup")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    out = ROOT / "runs" / ("optimization-03-compile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor-cache")
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(0.65)
    load_tokenizer()  # Verify the pinned artifact before consuming its token IDs.
    seed_everything(1337)
    eager, _ = load_checkpoint(checkpoint, device)
    candidate, _ = load_checkpoint(checkpoint, device)
    if args.explicit_position_lookup:
        replacement = ExplicitPositionLookup(candidate.config)
        replacement.load_state_dict(candidate.state_dict())
        del candidate
        candidate = replacement.to(device)
    train = load_split("manas01-full", "train")
    sampler = RandomWindowSampler(train, 8, 256, 1338)
    context = (lambda: torch.autocast("mps", dtype=torch.bfloat16)) if args.precision == "bf16" else nullcontext
    report = {"environment": environment_info(device), "precision": args.precision,
              "explicit_position_lookup": args.explicit_position_lookup,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_sha256": hashlib.sha256((ROOT / "data/processed/manas01-full/train.bin").read_bytes()).hexdigest(),
              "status": "running", "timings": {"eager": [], "compiled": []}}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def counters():
        from torch._dynamo.utils import counters as values
        return {str(key): dict(value) for key, value in values.items()}

    save()
    print(f"RUN {out}", flush=True)
    try:
        compiled = torch.compile(candidate, backend="inductor", fullgraph=True, dynamic=False)
        x, y = sampler.next(device)
        gradients = {}
        logits_cpu = {}
        losses = {}
        # eval disables dropout, not autograd, for direct derivative comparison.
        for name, model in (("eager", eager), ("compiled", compiled)):
            model.eval()
            model.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            start = time.perf_counter()
            with context():
                logits, loss = model(x, y)
            loss.backward()
            torch.mps.synchronize()
            report[name + "_eval_forward_backward_seconds"] = time.perf_counter() - start
            logits_cpu[name] = logits.detach().float().cpu()
            losses[name] = loss.item()
            gradients[name] = {key.removeprefix("_orig_mod."): p.grad.detach().float().cpu()
                               for key, p in model.named_parameters()}
            del logits, loss
            save()
        if gradients["eager"].keys() != gradients["compiled"].keys():
            raise RuntimeError("Compiled and eager gradient parameter names differ")
        report["gradient_name_order_matches"] = list(gradients["eager"]) == list(gradients["compiled"])
        reference_squared = difference_squared = 0.0
        per_parameter = []
        for key, eager_gradient in gradients["eager"].items():
            compiled_gradient = gradients["compiled"][key]
            reference_norm = float(eager_gradient.norm())
            difference_norm = float((eager_gradient - compiled_gradient).norm())
            reference_squared += reference_norm ** 2
            difference_squared += difference_norm ** 2
            per_parameter.append({"parameter": key, "eager_norm": reference_norm,
                                  "compiled_norm": float(compiled_gradient.norm()),
                                  "difference_norm": difference_norm,
                                  "relative_l2": difference_norm / max(reference_norm, 1e-12)})
        report["gradient_diagnostics"] = sorted(per_parameter, key=lambda row: row["difference_norm"], reverse=True)
        relative = (difference_squared / max(reference_squared, 1e-24)) ** 0.5
        report["parity"] = {"gradient_relative_l2": relative, "loss_difference": abs(losses["eager"] - losses["compiled"]),
                            "maximum_logit_difference": float((logits_cpu["eager"] - logits_cpu["compiled"]).abs().max()),
                            "argmax_agreement": float((logits_cpu["eager"].argmax(-1) == logits_cpu["compiled"].argmax(-1)).float().mean())}
        tolerance = 0.001 if args.precision == "fp32" else 0.02
        if not relative <= tolerance or not report["parity"]["loss_difference"] <= tolerance:
            raise RuntimeError(f"Compile parity gate failed: {report['parity']}")
        del gradients, logits_cpu
        optimizers = {name: model.configure_optimizer(0.00003, 0.1, (0.9, 0.95))
                      for name, model in (("eager", eager), ("compiled", candidate))}
        models = {"eager": eager, "compiled": compiled}
        with (out / "steps.jsonl").open("x") as log:
            for step in range(35):
                batches = [sampler.next(device) for _ in range(2)]
                for name in (("eager", "compiled") if step % 2 == 0 else ("compiled", "eager")):
                    seed_everything(1337 + step)
                    model, optimizer = models[name], optimizers[name]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    mean_loss = 0.0
                    for x, y in batches:
                        with context():
                            logits, loss = model(x, y)
                        (loss / 2).backward()
                        mean_loss += loss.item() / 2
                        del logits, loss
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    torch.mps.synchronize()
                    seconds = time.perf_counter() - started
                    row = {"mode": name, "step": step, "seconds": seconds, "loss": mean_loss,
                           "sampled_allocated_bytes": torch.mps.current_allocated_memory(),
                           "sampled_driver_bytes": torch.mps.driver_allocated_memory()}
                    if step >= 5:
                        report["timings"][name].append(seconds)
                    if step == 0:
                        report[name + "_cold_train_seconds"] = seconds
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                if step == 4:
                    report["counters_after_warmup"] = counters()
                if step % 5 == 0:
                    print(f"Finished paired update {step}", flush=True)
                    save()
        report["counters_final"] = counters()
        medians = {name: statistics.median(values) for name, values in report["timings"].items()}
        report["median_seconds"] = medians
        report["speedup"] = medians["eager"] / medians["compiled"]
        saved = medians["eager"] - medians["compiled"]
        cold = report["compiled_eval_forward_backward_seconds"] + report["compiled_cold_train_seconds"]
        report["conservative_break_even_updates"] = cold / saved if saved > 0 else None
        report["status"] = "measured_pending_review"
        save()
        print(json.dumps({"run": str(out), "speedup": report["speedup"], "break_even": report["conservative_break_even_updates"]}), flush=True)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
