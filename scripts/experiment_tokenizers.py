"""Sequential, resumable O05 runs on the pinned equal-byte Manas corpus."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from comparison_windows import ComparisonData
from manas_gpt.audit import (DEFAULT_PROMPTS, DEFAULT_SEEDS, WORD_MATCH_PROTOCOL,
                            audit_summary, longest_copied_word_span, repetition_stats)
from manas_gpt.config import load_config
from manas_gpt.experiment import environment_info, load_checkpoint, require_mps, seed_everything
from manas_gpt.model import ManasGPT

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs/optimization-05-data-20260904T060626Z"
MANIFEST_SHA = "7120ded73b026720bef09797c9ae9d81d261ff7012ab680bc12b3e6aa1231001"
INCUMBENT = ROOT / "runs/optimization-02-bf16-20260904T055741Z/best-model.pt"
INCUMBENT_SHA = "31499eb747c98bcada1c48b12205033cded96573269bc15464ccafd9905e2167"
EPOCHS = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def memory_sample():
    return {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    nll = 0.0
    targets = byte_count = 0
    started = time.perf_counter()
    for batch in data.validation_batches():
        x, y = batch["x"].to(device), batch["y"].to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
        nll += loss.item()
        targets += batch["targets"]
        byte_count += batch["bytes"]
    info = data.variant["splits"]["validation"]
    if byte_count != info["scored_utf8_bytes"] or targets != info["scored_tokens"] or not math.isfinite(nll):
        raise RuntimeError("Invalid exact-byte evaluation")
    return {"bits_per_byte": nll / math.log(2) / byte_count, "loss": nll / targets,
            "summed_nll": nll, "scored_bytes": byte_count, "targets": targets,
            "elapsed_seconds": time.perf_counter() - started, "precision": "fp32"}


def initialize(config, vocabulary, device):
    # Different embedding shapes otherwise shift RNG for every later layer.
    seed_everything(1337)
    master = ManasGPT(config.model.with_vocab_size(32768))
    master_state = master.state_dict()
    model = ManasGPT(config.model.with_vocab_size(vocabulary))
    state = {name: value[:vocabulary] if name in ("token_embedding.weight", "lm_head.weight") else value
             for name, value in master_state.items()}
    model.load_state_dict(state)
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in state.items()):
        raise AssertionError("Shared initialization differs")
    del master, master_state, state
    model.to(device)
    seed_everything(1337)
    return model


def optimizer_for(model, config):
    t = config.training
    return model.configure_optimizer(t.learning_rate, t.weight_decay, (t.beta1, t.beta2))


def update(model, optimizer, batches, device, memory=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    active_targets = sum(batch["targets"] for batch in batches)
    weighted_loss = 0.0
    for batch in batches:
        x, y = batch["x"].to(device), batch["y"].to(device)
        with torch.autocast("mps", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        if memory is not None:
            sample = memory_sample()
            for key in memory:
                memory[key] = max(memory[key], sample[key])
        weighted = loss * (batch["targets"] / active_targets)
        weighted_loss += weighted.item()
        weighted.backward()
        del x, y, logits, loss, weighted
        if memory is not None:
            sample = memory_sample()
            for key in memory:
                memory[key] = max(memory[key], sample[key])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.mps.synchronize()
    if memory is not None:
        sample = memory_sample()
        for key in memory:
            memory[key] = max(memory[key], sample[key])
    return weighted_loss, norm.item()


def preflight(model, data, config, device):
    first = last = short_row = short_length = None
    for item in data.training_updates(0):
        for batch in item["batches"]:
            if batch["targets"] == 8 * 256 and first is None:
                first = batch
            for row in range(8):
                columns = (batch["y"][row] != -100).nonzero().flatten()
                if columns.numel() and columns[-1].item() < 255 and last is None:
                    last, short_row, short_length = batch, row, columns[-1].item() + 1
        if first is not None and last is not None:
            break
    if first is None or last is None:
        raise RuntimeError("Real full and short windows are required for this preflight")
    batches = [first, last]
    model.eval()
    with torch.no_grad():
        batch = last
        row, length = short_row, short_length
        logits, loss = model(batch["x"].to(device), batch["y"].to(device))
        trimmed, _ = model(batch["x"][row:row + 1, :length].to(device))
        maximum_difference = (logits[row, :length] - trimmed[0]).abs().max().item()
        if not torch.allclose(logits[row, :length], trimmed[0], atol=1e-4, rtol=1e-4):
            raise AssertionError("Padding changed a real token's logits")
        manual = F.cross_entropy(logits.flatten(0, 1), batch["y"].flatten().to(device), reduction="sum") / batch["targets"]
        if not torch.allclose(loss, manual, atol=1e-6, rtol=1e-6):
            raise AssertionError("Ignored-target mean loss uses the wrong denominator")
    del logits, trimmed, loss, manual
    optimizer = optimizer_for(model, config)
    evidence = {"targets_per_microbatch": [b["targets"] for b in batches],
                "accumulation_weights": [b["targets"] / sum(b2["targets"] for b2 in batches) for b in batches],
                "padding_maximum_logit_difference": maximum_difference,
                "partial_window_length": length, "memory": {"allocated": 0, "driver": 0}}
    evidence["loss"], evidence["gradient_norm"] = update(model, optimizer, batches, device, evidence["memory"])
    evidence["finite_real_update"] = True
    return evidence


def byte_lr(config, consumed, update_bytes, total):
    t = config.training
    progress = min(1.0, (consumed + update_bytes) / total)
    warmup = t.warmup_steps / t.max_steps
    if progress <= warmup:
        return t.learning_rate * progress / warmup
    ratio = (progress - warmup) / (1 - warmup)
    return t.min_learning_rate + (t.learning_rate - t.min_learning_rate) * .5 * (1 + math.cos(math.pi * ratio))


def checkpoint(path, model, optimizer, provenance, history, best, consumed):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "model_config": asdict(model.config),
               "protocol": provenance, "history": history, "best": best, "consumed_bytes": consumed,
               "cpu_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    temp.replace(path)


@torch.no_grad()
def generation_audit(model, data, device, output):
    model.eval()
    corpus_path = BUNDLE / "train.txt"
    if sha(corpus_path) != data.manifest["files"]["train.txt"]:
        raise RuntimeError("Generation audit training text hash mismatch")
    corpus = corpus_path.read_text()
    samples = []
    for prompt in DEFAULT_PROMPTS:
        for seed in DEFAULT_SEEDS:
            prompt_ids = data.tokenizer.encode(prompt, add_special_tokens=False).ids
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            started = time.perf_counter()
            generated = model.generate(ids, 256, .8, 40, torch.Generator(device=device).manual_seed(seed))
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            all_ids = generated[0].tolist()
            continuation = data.tokenizer.decode(all_ids[len(prompt_ids):])
            sample = {"prompt": prompt, "seed": seed, "prompt_token_ids": prompt_ids,
                      "all_token_ids": all_ids, "continuation": continuation,
                      "generated_bytes": len(continuation.encode("utf-8")), "new_tokens": 256,
                      "elapsed_seconds": elapsed, "repetition": repetition_stats(continuation),
                      "longest_copied_word_span": longest_copied_word_span(continuation, corpus)}
            samples.append(sample)
            save_json(output, {"protocol": WORD_MATCH_PROTOCOL, "temperature": .8, "top_k": 40,
                               "tokenizer_sha256": sha(BUNDLE / str(model.config.vocab_size) / "tokenizer.json"),
                               "summary": audit_summary(samples), "samples": samples})
    return audit_summary(samples)


def run_arm(args):
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    config = load_config(ROOT / "configs/manas01-27m.toml")
    if config.training.precision != "bf16" or config.training.activation_checkpointing:
        raise RuntimeError("O05 requires accepted eager BF16 training without checkpointing")
    data = ComparisonData(BUNDLE, args.vocabulary, MANIFEST_SHA)
    out = args.output / ("incumbent" if args.incumbent else str(args.vocabulary))
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"script_sha256": sha(__file__), "windows_sha256": sha(Path(__file__).with_name("comparison_windows.py")),
                  "model_source_sha256": sha(ROOT / "src/manas_gpt/model.py"), "config_sha256": config.sha256,
                  "manifest_sha256": MANIFEST_SHA, "vocabulary": args.vocabulary, "epochs": EPOCHS,
                  "seed": 1337, "window_seed": 1338, "training": "bf16 eager, FP32 parameters/optimizer",
                  "checkpointing": False, "validation": "exact-byte FP32; 128 scored targets/window; context<=256",
                  "lr": "fraction of scored training bytes; warmup 1/30; cosine .0003 -> .00003",
                  "experiment_config": config.as_dict(), "tokenizer_path": str(BUNDLE / str(args.vocabulary) / "tokenizer.json"),
                  "tokenizer_sha256": sha(BUNDLE / str(args.vocabulary) / "tokenizer.json")}
    if args.incumbent:
        if args.vocabulary != 32768 or sha(INCUMBENT) != INCUMBENT_SHA:
            raise RuntimeError("Incumbent checkpoint mismatch")
        model, payload = load_checkpoint(INCUMBENT, device)
        del payload
        result = {"checkpoint": str(INCUMBENT), "checkpoint_sha256": INCUMBENT_SHA,
                  "protocol": provenance, "validation": evaluate(model, data, device)}
        save_json(out / "result.json", result)
        print(json.dumps(result), flush=True)
        return
    if (out / "result.json").exists():
        done = json.loads((out / "result.json").read_text())
        if done["protocol"] != provenance:
            raise RuntimeError("Completed run protocol differs; refusing reuse")
        print(f"REUSE {out}", flush=True)
        return
    model = initialize(config, args.vocabulary, device)
    if args.preflight_only or not (out / "preflight.json").exists():
        gate = preflight(model, data, config, device)
        save_json(out / "preflight.json", {"protocol": provenance, "evidence": gate})
        print(f"PREFLIGHT {args.vocabulary} {json.dumps(gate)}", flush=True)
        if args.preflight_only:
            return
        del model
        gc.collect()
        torch.mps.empty_cache()
        model = initialize(config, args.vocabulary, device)
    else:
        if json.loads((out / "preflight.json").read_text())["protocol"] != provenance:
            raise RuntimeError("Preflight protocol differs")
    optimizer = optimizer_for(model, config)
    history, best, consumed = [], {"bits_per_byte": float("inf"), "epoch": 0}, 0
    if (out / "resume.pt").exists():
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["protocol"] != provenance:
            raise RuntimeError("Resume protocol differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, consumed = saved["history"], saved["best"], saved["consumed_bytes"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])
        del saved
        print(f"RESUME {args.vocabulary} epoch {len(history)}", flush=True)
    total_bytes = EPOCHS * data.variant["splits"]["train"]["scored_utf8_bytes"]
    save_json(out / "provenance.json", {"protocol": provenance, "environment": environment_info(device),
                                       "parameters": model.parameter_count(), "memory_cap_fraction": .65})
    for epoch in range(len(history), EPOCHS):
        epoch_start = time.perf_counter()
        losses, durations = [], []
        memory = {"allocated": 0, "driver": 0}
        epoch_bytes = 0
        for item in data.training_updates(epoch):
            lr = byte_lr(config, consumed, item["bytes"], total_bytes)
            for group in optimizer.param_groups:
                group["lr"] = lr
            start = time.perf_counter()
            loss, _ = update(model, optimizer, item["batches"], device, memory)
            durations.append(time.perf_counter() - start)
            losses.append((loss, item["targets"]))
            consumed += item["bytes"]
            epoch_bytes += item["bytes"]
        training_seconds = time.perf_counter() - epoch_start
        if epoch_bytes != data.variant["splits"]["train"]["scored_utf8_bytes"]:
            raise RuntimeError("Incomplete epoch coverage")
        validation = evaluate(model, data, device)
        row = {"epoch": epoch + 1, "updates": len(durations), "consumed_bytes": consumed,
               "train_loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
               "training_seconds": training_seconds, "update_seconds": sum(durations),
               "median_update_seconds": statistics.median(durations), "lr": lr,
               "sampled_memory": memory, "validation": validation}
        history.append(row)
        if validation["bits_per_byte"] < best["bits_per_byte"]:
            best = {"epoch": epoch + 1, **validation}
            checkpoint(out / "best-model.pt", model, optimizer, provenance, history, best, consumed)
        checkpoint(out / "resume.pt", model, optimizer, provenance, history, best, consumed)
        save_json(out / "history.json", history)
        print(f"EPOCH {args.vocabulary} {epoch + 1}/{EPOCHS} train={row['train_loss']:.5f} "
              f"val_bpb={validation['bits_per_byte']:.6f} seconds={training_seconds:.1f} best={best['epoch']}", flush=True)
    if consumed != total_bytes:
        raise RuntimeError("Unequal final source-byte budget")
    del optimizer, model
    gc.collect()
    torch.mps.empty_cache()
    model, payload = load_checkpoint(out / "best-model.pt", device)
    del payload
    validation = evaluate(model, data, device)
    audit = generation_audit(model, data, device, out / "generation-audit.json")
    seconds = sum(row["training_seconds"] for row in history)
    result = {"status": "completed_not_promoted", "protocol": provenance, "best": best,
              "independent_validation": validation, "audit": audit, "history": history,
              "checkpoint_sha256": sha(out / "best-model.pt"), "parameters": model.parameter_count(),
              "training_seconds": seconds, "training_bytes_per_second": total_bytes / seconds,
              "sampled_memory": {key: max(row["sampled_memory"][key] for row in history) for key in ("allocated", "driver")}}
    save_json(out / "result.json", result)
    print(f"DONE {args.vocabulary} {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, choices=(32768, 16384, 8192))
    parser.add_argument("--incumbent", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.vocabulary:
        run_arm(args)
        return
    if args.incumbent or args.preflight_only:
        parser.error("These modes require --vocabulary")
    args.output.mkdir(parents=True, exist_ok=True)
    # Separate processes prevent allocator state from one vocabulary contaminating another.
    for vocabulary, incumbent in ((32768, True), (32768, False), (16384, False), (8192, False)):
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--vocabulary", str(vocabulary)]
        if incumbent:
            command.append("--incumbent")
        subprocess.run(command, check=True, cwd=ROOT)
    print(f"COMPARISON COMPLETE {args.output}; manual acceptance still required", flush=True)


if __name__ == "__main__":
    main()
