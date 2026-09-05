"""O17: paired BF16/FP32 inference on a real native checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from followup_common import (BASELINE, BASELINE_SHA, FixedEvaluation, ROOT, evaluate,
    generation_audit, precision_context, save_json, sha, synchronize)
from manas_gpt.experiment import load_checkpoint, require_mps, environment_info
from manas_gpt.kv_cache import CacheSession


@torch.inference_mode()
def probability_comparison(model, data, device):
    kl_sum = 0.0
    count = agreements = 0
    max_logit_difference = 0.0
    for index, (x, y, _) in enumerate(data.batches()):
        if index == 20:
            break
        x, mask = x.to(device), (y != -100).to(device)
        logits = []
        for mode in ("fp32", "bf16"):
            with precision_context(mode, device):
                value, _ = model(x)
            logits.append(value[mask].float())
        if any(not torch.isfinite(v).all() for v in logits):
            raise RuntimeError("Nonfinite precision-comparison output")
        log_p, log_q = [F.log_softmax(v, -1) for v in logits]
        kl_sum += (log_p.exp() * (log_p - log_q)).sum().item()
        count += logits[0].shape[0]
        agreements += (logits[0].argmax(-1) == logits[1].argmax(-1)).sum().item()
        max_logit_difference = max(max_logit_difference, (logits[0] - logits[1]).abs().max().item())
    return {"mean_kl_fp32_to_bf16": kl_sum / count, "targets": count,
            "top1_agreement": agreements / count, "max_absolute_logit_difference": max_logit_difference}


@torch.inference_mode()
def cache_parity(model, data, device, precision):
    cache = CacheSession(model)
    start = model.config.block_size - 8
    sequence = data.ids[:start + 20].to(device)[None]
    differences, relative, agreements = [], [], []
    with precision_context(precision, device):
        for end in range(start, sequence.shape[1]):
            context = sequence[:, max(0, end - model.config.block_size):end]
            if cache.layers is None or cache.length == model.config.block_size:
                cached = cache.prefill(context)
            else:
                cached = cache.decode(sequence[:, end - 1:end])
            direct, _ = model(context, last_position_only=True)
            difference = cached.float() - direct.float()
            differences.append(difference.abs().max().item())
            relative.append((difference.square().mean().sqrt() /
                direct.float().square().mean().sqrt().clamp_min(1e-8)).item())
            agreements.append(cached.argmax(-1).item() == direct.argmax(-1).item())
        storage = cache.storage()
    return {"precision": precision, "real_positions_checked": len(differences),
            "max_logit_difference": max(differences), "max_relative_rms_error": max(relative),
            "greedy_agreement": sum(agreements) / len(agreements), "overflow_checked": True,
            "cache": storage}


def memory(device):
    if device.type != "mps":
        return None
    return {"allocated": torch.mps.current_allocated_memory(),
            "driver": torch.mps.driver_allocated_memory()}


@torch.inference_mode()
def timing(model, data, device):
    records = {}
    for length in (32, model.config.block_size - 8):
        prompt = data.ids[:length].to(device)[None]
        trials = {p: [] for p in ("fp32", "bf16")}
        for trial in range(23):
            order = ("fp32", "bf16") if trial % 2 == 0 else ("bf16", "fp32")
            for precision in order:
                synchronize(device)
                start = time.perf_counter()
                with precision_context(precision, device):
                    model.generate(prompt, 64, .8, 40,
                        torch.Generator(device=device).manual_seed(6036))
                synchronize(device)
                if trial >= 3:
                    trials[precision].append(time.perf_counter() - start)
        medians = {p: statistics.median(v) for p, v in trials.items()}
        records[str(length)] = {"prompt_tokens": length, "new_tokens": 64,
                               "pairs": 20, "samples_seconds": trials, "median_seconds": medians,
                               "bf16_speedup": medians["fp32"] / medians["bf16"]}
        print(f"TIMING prompt={length} {medians}", flush=True)
    stages = {}
    for precision in ("fp32", "bf16"):
        values = {"prefill_seconds": [], "decode_seconds": []}
        peak = {"allocated": 0, "driver": 0} if device.type == "mps" else None
        for i in range(23):
            cache = CacheSession(model)
            with precision_context(precision, device):
                synchronize(device); start = time.perf_counter()
                cache.prefill(data.ids[:128].to(device)[None])
                synchronize(device); middle = time.perf_counter()
                cache.decode(data.ids[128:129].to(device)[None])
                synchronize(device); end = time.perf_counter()
                m = memory(device)
                if peak is not None:
                    peak = {k: max(peak[k], m[k]) for k in peak}
            if i >= 3:
                values["prefill_seconds"].append(middle - start)
                values["decode_seconds"].append(end - middle)
            storage = cache.storage()
            del cache
        stages[precision] = {k: statistics.median(v) for k, v in values.items()}
        stages[precision].update(samples=values, sampled_memory=peak, cache=storage)
    return {"generation": records, "stages": stages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--expected-sha", default=BASELINE_SHA)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.checkpoint) != args.expected_sha:
        raise RuntimeError("Checkpoint hash differs from requested identity")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise FileExistsError("Completed experiment exists; inspect/reuse, do not overwrite")
    device = require_mps() if args.device == "mps" else torch.device("cpu")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(.65)
    else:
        torch.set_num_threads(2)
    model, payload = load_checkpoint(args.checkpoint, device)
    del payload
    model.eval()
    data = FixedEvaluation()
    protocol = {"checkpoint_sha256": args.expected_sha, "device": str(device),
                "script_sha256": sha(__file__), "common_sha256": sha(Path(__file__).with_name("followup_common.py")),
                "torch": torch.__version__, "autocast_only": True, "weights": "fp32",
                "evaluation": data.info, "model_context": model.config.block_size}
    save_json(out / "protocol.json", protocol)
    validations = {}
    for precision in ("fp32", "bf16"):
        validations[precision] = evaluate(model, data, device, model.config.block_size, precision)
        print(f"VALIDATION {precision} {validations[precision]['loss']:.8f}", flush=True)
    probabilities = probability_comparison(model, data, device)
    parity = {p: cache_parity(model, data, device, p) for p in ("fp32", "bf16")}
    save_json(out / "numerics.json", {"validation": validations, "probabilities": probabilities, "cache_parity": parity})
    measured = timing(model, data, device)
    save_json(out / "timing.json", measured)
    delta = validations["bf16"]["loss"] - validations["fp32"]["loss"]
    numeric = delta <= .01 and probabilities["mean_kl_fp32_to_bf16"] <= .005
    cache_ok = parity["fp32"]["max_relative_rms_error"] <= 1e-4 and parity["bf16"]["max_relative_rms_error"] <= .01
    speedups = [r["bf16_speedup"] for r in measured["generation"].values()]
    stage = measured["stages"]
    reduction = None
    if device.type == "mps":
        reduction = 1 - stage["bf16"]["sampled_memory"]["allocated"] / stage["fp32"]["sampled_memory"]["allocated"]
    performance = min(speedups) >= 1.05 or (reduction is not None and reduction >= .20 and min(speedups) >= 1 / 1.05)
    audits = {}
    if numeric and cache_ok and performance:
        for p in ("fp32", "bf16"):
            print(f"AUDIT {p} 20 generations", flush=True)
            audits[p] = generation_audit(model, device, out / f"audit-{p}.json", p)
    generation_ok = bool(audits) and (audits["bf16"]["mean_repeated_trigram_ratio"] <=
                                     audits["fp32"]["mean_repeated_trigram_ratio"] + .01)
    result = {"status": "candidate_passed_pending_text_review" if numeric and cache_ok and performance and generation_ok else "not_promoted",
              "protocol": protocol, "validation": validations, "validation_loss_delta": delta,
              "probabilities": probabilities, "cache_parity": parity, "timing": measured,
              "sampled_live_memory_reduction": reduction, "audits": audits,
              "gates": {"prediction": numeric, "cache_numerics": cache_ok,
                        "performance": performance, "generation_metrics": generation_ok}}
    save_json(out / "result.json", result)
    print(f"DONE {result['status']} gates={result['gates']}", flush=True)


if __name__ == "__main__":
    main()
