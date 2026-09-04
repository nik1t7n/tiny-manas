"""Real-data RoPE correctness and cost gate; never a model-quality substitute."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import traceback

import torch

from comparison_windows import ComparisonData
from architecture_candidates import restore_model
from experiment_tokenizers import (BUNDLE, MANIFEST_SHA, ROOT, optimizer_for,
                                   save_json, sha, update)
from manas_gpt.config import load_config
from manas_gpt.data import RandomWindowSampler, load_split
from manas_gpt.experiment import environment_info, require_mps, seed_everything
from rotary_candidate import candidate_from, rotate_pairs, rotary_tables


def probe_updates(data, recipe, config):
    if recipe == "equal-bytes":
        yield from data.training_updates(0)
        return
    if recipe != "random-windows" or config.model.vocab_size not in (0, 32768):
        raise ValueError("Unsupported probe recipe")
    tokens = load_split(config.data.dataset, "train")
    if not torch.equal(tokens, data.ids["train"]):
        raise RuntimeError("Original and prepared token sequences differ")
    sampler = RandomWindowSampler(tokens, 8, 256, 1338)
    for _ in range(35):
        batches = []
        for _ in range(2):
            x, y = sampler.next(torch.device("cpu"))
            batches.append({"x": x, "y": y, "targets": y.numel()})
        yield {"batches": batches}


@torch.no_grad()
def rotation_gate(model, data, device):
    model.eval()
    first = next(batch for item in data.training_updates(0) for batch in item["batches"] if batch["targets"] == 2048)
    x = first["x"][0:1].to(device)
    block = model.blocks[0]
    features = block.ln_attention(model.token_embedding(x))
    q, k, _ = block.attention.qkv(features).split(model.config.n_embd, dim=-1)
    heads, dim = model.config.n_head, model.config.n_embd // model.config.n_head
    q, k = (value.view(1, 256, heads, dim).transpose(1, 2) for value in (q, k))
    cos, sin = rotary_tables(torch.arange(256, device=device), dim)
    qr, kr = rotate_pairs(q, cos, sin), rotate_pairs(k, cos, sin)
    norm_error = max((q.norm(dim=-1) - qr.norm(dim=-1)).abs().max().item(),
                     (k.norm(dim=-1) - kr.norm(dim=-1)).abs().max().item())
    if not torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5, rtol=1e-5):
        raise AssertionError("Q norm changed under FP32 rotation")
    if not torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-5, rtol=1e-5):
        raise AssertionError("K norm changed under FP32 rotation")
    cos_offset, sin_offset = rotary_tables(torch.arange(256, device=device) + 137, dim)
    scores = qr @ kr.transpose(-2, -1)
    shifted_scores = rotate_pairs(q, cos_offset, sin_offset) @ rotate_pairs(k, cos_offset, sin_offset).transpose(-2, -1)
    offset_error = (scores - shifted_scores).abs().max().item()
    if not torch.allclose(scores, shifted_scores, atol=1e-4, rtol=1e-4):
        raise AssertionError("A common position offset changed Q/K scores")
    changed = x.clone()
    # Use another real source window, not fabricated token IDs.
    changed[:, 128:] = first["x"][1:2, 128:].to(device)
    if torch.equal(x[:, 128:], changed[:, 128:]):
        raise RuntimeError("Causal check requires distinct real suffixes")
    logits, _ = model(x)
    changed_logits, _ = model(changed)
    causal_error = (logits[:, :128] - changed_logits[:, :128]).abs().max().item()
    if not torch.allclose(logits[:, :128], changed_logits[:, :128], atol=1e-4, rtol=1e-4):
        raise AssertionError("Future source tokens affected earlier logits")
    return {"fp32_norm_max_error": norm_error, "common_offset": 137,
            "fp32_score_offset_max_error": offset_error, "causal_max_logit_error": causal_error,
            "rotations": "FP32 arithmetic, cast result to incoming Q/K dtype"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=int, choices=(32768, 16384, 8192), required=True)
    parser.add_argument("--reference-initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    config = load_config(ROOT / "configs/manas01-27m.toml")
    data = ComparisonData(BUNDLE, args.vocabulary, MANIFEST_SHA)
    initial = torch.load(args.reference_initial, map_location="cpu", weights_only=False)
    if initial["protocol"]["vocabulary"] != args.vocabulary or initial["protocol"]["architecture_changes"]:
        raise ValueError("O06 requires a fresh classic reference with the selected vocabulary")
    report = {"status": "running", "environment": environment_info(device),
              "vocabulary": args.vocabulary, "config_sha256": config.sha256,
              "manifest_sha256": MANIFEST_SHA, "script_sha256": sha(__file__),
              "candidate_sha256": sha(Path(__file__).with_name("rotary_candidate.py")),
              "model_source_sha256": sha(ROOT / "src/manas_gpt/model.py"),
              "training_helpers_sha256": sha(Path(__file__).with_name("experiment_tokenizers.py")),
              "change": "rope", "reference_initial_sha256": sha(args.reference_initial),
              "recipe": initial["protocol"]["recipe"],
              "precision": "bf16 training, fp32 parameters; fp32 mathematical checks",
              "dropout": .2, "updates_per_mode": 35, "warmup_excluded": 5,
              "modes": {}, "quality_evaluated": False}
    try:
        reference = restore_model(initial)
        if reference.config != config.model.with_vocab_size(args.vocabulary):
            raise ValueError("Reference model configuration changed")
        candidate = candidate_from(reference, device)
        report["correctness"] = rotation_gate(candidate, data, device)
        report["parameter_saving"] = reference.parameter_count() - candidate.parameter_count()
        if report["parameter_saving"] != 256 * 384:
            raise AssertionError("Unexpected parameter-count change")
        del candidate
        gc.collect()
        torch.mps.empty_cache()
        for mode in ("learned", "rope"):
            if mode == "rope":
                model = candidate_from(reference, device)
            else:
                model = restore_model(initial).to(device)
            optimizer = optimizer_for(model, config)
            seed_everything(1337)
            durations, losses = [], []
            memory = {"allocated": 0, "driver": 0}
            for index, item in enumerate(probe_updates(data, report["recipe"], config)):
                if index == 35:
                    break
                start = time.perf_counter()
                loss, norm = update(model, optimizer, item["batches"], device, memory)
                durations.append(time.perf_counter() - start)
                losses.append(loss)
                if index == 0:
                    missing = [name for name, value in model.named_parameters() if value.grad is None]
                    if missing:
                        raise AssertionError(f"Parameters without gradients: {missing}")
                    report["modes"][mode] = {"first_gradient_norm": norm}
            report["modes"][mode].update({"median_update_seconds": statistics.median(durations[5:]),
                                          "durations": durations, "losses": losses, "sampled_memory": memory})
            save_json(out / "result.json", report)
            print(f"MODE {mode} median={statistics.median(durations[5:]):.6f}", flush=True)
            del model, optimizer
            gc.collect()
            torch.mps.empty_cache()
        baseline = report["modes"]["learned"]["median_update_seconds"]
        candidate = report["modes"]["rope"]["median_update_seconds"]
        report["latency_change_fraction"] = candidate / baseline - 1
        report["status"] = "cost_gate_passed_quality_pending" if candidate <= 1.10 * baseline else "cost_gate_failed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        save_json(out / "result.json", report)
        raise
    save_json(out / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "modes"}), flush=True)


if __name__ == "__main__":
    main()
