"""O10: bounded 150-update MHA/GQA adaptation from one accepted checkpoint."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time
import traceback

import torch

from comparison_windows import ComparisonData
from experiment_kv_cache import load_reference, require_equal_logits, timed
from experiment_tokenizers import (BUNDLE, MANIFEST_SHA, ROOT, evaluate, generation_audit,
                                   optimizer_for, save_json, sha, update)
from gqa_candidate import from_mha
from kv_cache_candidate import CacheSession
from manas_gpt.config import load_config
from manas_gpt.data import RandomWindowSampler
from manas_gpt.experiment import environment_info, require_mps, seed_everything


@torch.no_grad()
def correctness(reference, data, device):
    original = reference.to(device).eval()
    tokens = data.ids["validation"]
    prefix = tokens[:64][None].to(device)
    expected = original(prefix, last_position_only=True)[0]
    original.to("cpu")
    eight = from_mha(reference, 8).to(device).eval()
    eight_error = require_equal_logits(expected, eight(prefix, last_position_only=True)[0], "eight KV heads versus MHA")
    del eight, expected
    candidate = from_mha(reference, 2).to(device).eval()
    maximum_error = 0.0
    for length in (32, 248):
        session = CacheSession(candidate)
        for step in range(16):
            ids = tokens[:length + step][None].to(device)
            expected = candidate(ids[:, -256:], last_position_only=True)[0]
            actual = session.prefill(ids[:, -256:]) if session.layers is None or session.length == 256 else session.decode(ids[:, -1:])
            maximum_error = max(maximum_error, require_equal_logits(expected, actual, f"two-KV cache length {length}, step {step}"))
    session.prefill(tokens[:256][None].to(device))
    storage = session.storage()
    saving = reference.parameter_count() - candidate.parameter_count()
    if saving != 1774080 or storage["storage_bytes"] != 1572864:
        raise AssertionError("GQA parameter/cache storage differs from the declared 8-Q / 2-KV shape")
    report = {"eight_kv_max_logit_error": eight_error, "two_kv_cache_max_logit_error": maximum_error,
              "parameter_saving": saving, "full_cache_storage": storage,
              "temporary_expanded_kv_bytes_per_layer_at_256": 2 * 8 * 256 * 48 * 4}
    del candidate, session, actual, expected
    gc.collect()
    torch.mps.empty_cache()
    return report


def save_resume(path, model, optimizer, protocol, history):
    payload = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "model_config": protocol["model_config"], "optimizer": optimizer.state_dict(),
               "protocol": protocol, "history": history,
               "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def arm(reference, mode, data, config, device, out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    protocol = {**protocol, "kv_heads": 8 if mode == "mha" else 2}
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        if result["protocol"] != protocol or result["checkpoint_sha256"] != sha(out / "resume.pt"):
            raise RuntimeError("Completed adaptation provenance differs")
        return result
    model = load_reference(reference, torch.device("cpu"))
    if mode == "gqa":
        model = from_mha(model, 2)
    model.to(device)
    optimizer = optimizer_for(model, config)
    for group in optimizer.param_groups:
        group["lr"] = .00003
    seed_everything(1337)
    sampler = RandomWindowSampler(data.ids["train"], 8, 256, 1338)
    history = []
    if (out / "resume.pt").exists():
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise RuntimeError("Adaptation resume provenance differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])
        for _ in range(2 * len(history)):
            torch.randint(0, len(sampler.tokens) - 256, (8,), generator=sampler.generator)
        del saved
    if not (out / "before.json").exists():
        if history:
            raise RuntimeError("Missing pre-adaptation validation; refusing to relabel a trained state")
        save_json(out / "before.json", evaluate(model, data, device))
    memory = {"allocated": 0, "driver": 0}
    for step in range(len(history), 150):
        batches = []
        for _ in range(2):
            x, y = sampler.next(torch.device("cpu"))
            batches.append({"x": x, "y": y, "targets": y.numel()})
        start = time.perf_counter()
        loss, norm = update(model, optimizer, batches, device, memory)
        duration = time.perf_counter() - start
        if step == 0 and any(parameter.grad is None for parameter in model.parameters()):
            raise AssertionError("An adaptation parameter received no gradient")
        history.append({"step": step + 1, "loss": loss, "gradient_norm": norm,
                        "seconds": duration, "sampled_memory": dict(memory)})
        if (step + 1) % 50 == 0:
            save_resume(out / "resume.pt", model, optimizer, protocol, history)
            save_json(out / "history.json", history)
            print(f"ADAPT {mode} {step + 1}/150 loss={loss:.6f}", flush=True)
    validation = evaluate(model, data, device)
    audit = generation_audit(model, data, device, out / "generation-audit.json")
    # Timing after adaptation uses the same real query/past lengths in both arms.
    timings = []
    tokens = data.ids["validation"].to(device)
    for index in range(12):
        session = CacheSession(model)
        session.prefill(tokens[:128][None])
        seconds, _ = timed(lambda: session.decode(tokens[128:129][None]))
        if index >= 2:
            timings.append(seconds)
    storage = session.storage()
    result = {"status": "completed_not_promoted", "protocol": protocol, "validation": validation,
              "before": json.loads((out / "before.json").read_text()), "audit": audit,
              "parameters": model.parameter_count(), "checkpoint_sha256": sha(out / "resume.pt"),
              "history": history, "cache_storage_at_129": storage,
              "median_update_seconds": statistics.median(row["seconds"] for row in history[5:]),
              "decode_seconds": timings, "median_decode_seconds": statistics.median(timings)}
    save_json(out / "result.json", result)
    del optimizer, model, session
    gc.collect()
    torch.mps.empty_cache()
    return result


@torch.no_grad()
def paired_decode(reference, out, data, device):
    models = {}
    for mode in ("mha", "gqa"):
        model = load_reference(reference, torch.device("cpu"))
        if mode == "gqa":
            model = from_mha(model, 2)
        payload = torch.load(out / mode / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        del payload
        models[mode] = model.to(device).eval()
    tokens = data.ids["validation"].to(device)
    durations = {"mha": [], "gqa": []}
    for repetition in range(12):
        for mode in (("mha", "gqa") if repetition % 2 else ("gqa", "mha")):
            session = CacheSession(models[mode])
            session.prefill(tokens[:128][None])
            seconds, _ = timed(lambda: session.decode(tokens[128:129][None]))
            if repetition >= 2:
                durations[mode].append(seconds)
    medians = {mode: statistics.median(values) for mode, values in durations.items()}
    return {"seconds": durations, "median_seconds": medians,
            "gqa_latency_change": medians["gqa"] / medians["mha"] - 1,
            "protocol": "alternating paired order, two warmup pairs plus ten measured; 128 past positions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--vocabulary", type=int, choices=(32768, 16384, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Selected checkpoint hash differs")
    config = load_config(ROOT / "configs/manas01-27m.toml")
    if config.training.precision != "bf16" or config.training.activation_checkpointing:
        raise ValueError("Adaptation requires the accepted eager BF16 recipe without checkpointing")
    data = ComparisonData(BUNDLE, args.vocabulary, MANIFEST_SHA)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    reference = load_reference(args.checkpoint, torch.device("cpu"))
    if reference.config.vocab_size != args.vocabulary:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")
    protocol = {"checkpoint_sha256": args.checkpoint_sha256, "vocabulary": args.vocabulary,
                "model_config": asdict(reference.config), "config_sha256": config.sha256,
                "manifest_sha256": MANIFEST_SHA, "learning_rate": .00003, "updates": 150,
                "batch_size": 8, "context": 256, "accumulation": 2, "seed": 1337,
                "sampler_seed": 1338, "optimizer": "fresh AdamW states in both arms",
                "runner_sha256": sha(__file__), "candidate_sha256": sha(Path(__file__).with_name("gqa_candidate.py")),
                "cache_source_sha256": sha(Path(__file__).with_name("kv_cache_candidate.py")),
                "model_source_sha256": sha(ROOT / "src/manas_gpt/model.py"),
                "training_helpers_sha256": sha(Path(__file__).with_name("experiment_tokenizers.py"))}
    report = {"status": "running", "protocol": protocol, "environment": environment_info(device)}
    try:
        if (out / "preflight.json").exists():
            preflight = json.loads((out / "preflight.json").read_text())
            if preflight["protocol"] != protocol:
                raise RuntimeError("Preflight provenance differs")
        else:
            preflight = {"protocol": protocol, "correctness": correctness(reference, data, device)}
            save_json(out / "preflight.json", preflight)
        del reference
        for mode in ("mha", "gqa"):
            report[mode] = arm(args.checkpoint, mode, data, config, device, out / mode, protocol)
            save_json(out / "result.json", report)
        gqa, mha = report["gqa"], report["mha"]
        report["loss_delta_vs_adapted_mha"] = gqa["validation"]["loss"] - mha["validation"]["loss"]
        report["loss_delta_vs_original_mha"] = gqa["validation"]["loss"] - mha["before"]["loss"]
        report["persistent_cache_reduction"] = 1 - gqa["cache_storage_at_129"]["storage_bytes"] / mha["cache_storage_at_129"]["storage_bytes"]
        report["sequential_decode_latency_change"] = gqa["median_decode_seconds"] / mha["median_decode_seconds"] - 1
        report["paired_decode"] = paired_decode(args.checkpoint, out, data, device)
        report["numeric_gates_passed"] = (report["loss_delta_vs_adapted_mha"] <= .05
                                           and report["loss_delta_vs_original_mha"] <= .05
                                           and report["persistent_cache_reduction"] >= .60
                                           and report["paired_decode"]["gqa_latency_change"] <= .05)
        report["status"] = "completed_pending_raw_review" if report["numeric_gates_passed"] else "numeric_gate_failed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        save_json(out / "result.json", report)
        raise
    save_json(out / "result.json", report)
    print(f"DONE bounded adaptation {out}; inspect raw outputs before any promotion", flush=True)


if __name__ == "__main__":
    main()
