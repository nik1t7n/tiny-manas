"""Controlled architecture training, with an explicit staged quality follow-up."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from architecture_candidates import apply_change, restore_model
from comparison_windows import ComparisonData
from experiment_tokenizers import (BUNDLE, EPOCHS, MANIFEST_SHA, ROOT, byte_lr, checkpoint,
                                   evaluate, generation_audit, initialize, optimizer_for, save_json, sha, update)
from manas_gpt.config import load_config
from manas_gpt.data import RandomWindowSampler, load_split
from manas_gpt.experiment import environment_info, evaluate_random_batches, learning_rate, require_mps, seed_everything


def save_initial(path, model, protocol):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an initial state: {path}")
    temp = path.with_suffix(".pt.tmp")
    torch.save({"artifact_kind": "fresh_initialization",
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": asdict(model.config), "protocol": protocol}, temp)
    temp.replace(path)


def original_samplers(config, data):
    train, validation = load_split(config.data.dataset, "train"), load_split(config.data.dataset, "validation")
    # Reuse original IDs only after proving the comparison bundle has those same IDs.
    if not torch.equal(train, data.ids["train"]) or not torch.equal(validation, data.ids["validation"]):
        raise RuntimeError("Original and comparison token sequences differ")
    samplers = {"train": RandomWindowSampler(train, 8, 256, 1338),
                "train_eval": RandomWindowSampler(train, 8, 256, 1339),
                "validation": RandomWindowSampler(validation, 8, 256, 1340)}
    return samplers, validation


def advance_sampler(sampler, count):
    # Reconstruct private CPU RNG state without rerunning any model computation.
    for _ in range(count):
        torch.randint(0, len(sampler.tokens) - sampler.block_size, (sampler.batch_size,), generator=sampler.generator)


def staged_decision(history, control):
    """Budget heuristic, not a statistical claim about eventual convergence."""
    step = len(history) * 100
    if step not in (600, 900, 1200, 1500):
        return None
    deltas = [row["validation"]["loss"] - control[row["segment"] * 100]["validation"]["loss"]
              for row in history]
    recent, previous = statistics.mean(deltas[-3:]), statistics.mean(deltas[-6:-3])
    improving = previous - recent
    stop = False
    reason = "Continue: advantage or a still-closing gap warrants the next stage"
    if recent > .05 and min(deltas[-3:]) > .03 and improving < .01:
        stop, reason = True, "Consistently worse validation without a closing gap"
    elif step >= 900 and recent > -.02 and improving < .01:
        stop, reason = True, "No material validation advantage or improving relative trend within the pilot budget"
    elif step == 1500 and not (recent <= -.02 and max(deltas[-3:]) < 0):
        stop, reason = True, "Pilot ceiling: insufficient consistent advantage to fund full training"
    return {"step": step, "stop": stop, "reason": reason, "recent_mean_delta": recent,
            "previous_mean_delta": previous, "relative_gap_improvement": improving,
            "last_three_deltas": deltas[-3:], "interpretation": "budget decision, not proof of final inferiority"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/manas01-27m.toml")
    parser.add_argument("--vocabulary", type=int, choices=(32768, 16384, 8192), required=True)
    parser.add_argument("--recipe", choices=("random-windows", "equal-bytes"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-initial", action="store_true")
    parser.add_argument("--reference-initial", type=Path)
    parser.add_argument("--change", choices=("control", "rope", "rmsnorm", "swiglu"))
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--quality-followup", action="store_true",
                        help="Explicit owner-approved replacement of the old latency veto by staged validation gates")
    parser.add_argument("--control-metrics", type=Path)
    parser.add_argument("--control-calibration", type=Path)
    parser.add_argument("--stop-after-segment", type=int, default=30,
                        help="Pause at a saved epoch/100-update boundary; resume without this flag")
    args = parser.parse_args()
    if args.quality_followup and (args.recipe != "random-windows" or args.change not in ("control", "rope", "rmsnorm")
                                 or not args.control_metrics):
        parser.error("Staged follow-up requires the original recipe, control metrics, and control/RoPE/RMSNorm")
    if args.change == "control" and (not args.quality_followup or args.stop_after_segment != 1):
        parser.error("Control is a bounded 100-update calibration only")
    if not 1 <= args.stop_after_segment <= 30:
        parser.error("--stop-after-segment must be in 1..30")
    config = load_config(args.config)
    if (config.run.seed, config.model.block_size, config.training.batch_size,
        config.training.gradient_accumulation_steps, config.training.precision,
        config.training.activation_checkpointing) != (1337, 256, 8, 2, "bf16", False):
        raise ValueError("This runner requires the preregistered seed/shapes/eager BF16 recipe")
    if args.recipe == "random-windows" and args.vocabulary != 32768:
        raise ValueError("The incumbent random-window recipe uses the original 32k IDs")
    if (config.training.max_steps, config.training.eval_interval, config.training.eval_batches,
        config.training.warmup_steps, config.training.gradient_clip) != (3000, 100, 30, 100, 1.0):
        raise ValueError("The fixed experiment budget/schedule must match the accepted recipe")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = ComparisonData(BUNDLE, args.vocabulary, MANIFEST_SHA)
    protocol = {"recipe": args.recipe, "vocabulary": args.vocabulary, "architecture_changes": [],
                "config": config.as_dict(), "manifest_sha256": MANIFEST_SHA,
                "tokenizer_sha256": sha(BUNDLE / str(args.vocabulary) / "tokenizer.json"),
                "runner_sha256": sha(__file__), "model_source_sha256": sha(ROOT / "src/manas_gpt/model.py"),
                "rotary_source_sha256": sha(Path(__file__).with_name("rotary_candidate.py")),
                "candidates_source_sha256": sha(Path(__file__).with_name("architecture_candidates.py")),
                "training_helpers_sha256": sha(Path(__file__).with_name("experiment_tokenizers.py")),
                "early_stopping": "disabled; full matched budget", "quality_selection": "validation only"}
    if args.bootstrap_initial:
        if args.change or args.reference_initial or args.probe:
            parser.error("Bootstrap creates a classic fresh reference, not a trained candidate")
        model = initialize(config, args.vocabulary, torch.device("cpu"))
        save_initial(out / "initial-model.pt", model, protocol)
        save_json(out / "initial.json", {"protocol": protocol, "sha256": sha(out / "initial-model.pt"),
                                         "parameters": model.parameter_count()})
        print(f"INITIAL REFERENCE {out / 'initial-model.pt'}", flush=True)
        return
    control = {}
    if args.quality_followup:
        rows = [json.loads(line) for line in args.control_metrics.read_text().splitlines()]
        control = {row["step"]: row for row in rows if row["kind"] == "evaluation"}
        if set(control) != set(range(100, 3001, 100)):
            raise ValueError("The original control curve must cover all 30 evaluation points")
        protocol.update(early_stopping="staged-validation-v1: 600/900/1200/1500; see docs/experiments/13-quality-followup.md",
                        control_metrics_sha256=sha(args.control_metrics), generation_cache=False)
        if args.change != "control":
            if not args.control_calibration:
                parser.error("Candidates require a successful fresh control calibration")
            calibration = json.loads(args.control_calibration.read_text())
            if (calibration["status"] != "calibration_passed"
                or calibration["protocol"]["control_metrics_sha256"] != sha(args.control_metrics)
                or any(calibration["protocol"][key] != protocol[key]
                       for key in ("runner_sha256", "model_source_sha256", "training_helpers_sha256"))
                or calibration["protocol"]["reference_initial_sha256"] != sha(args.reference_initial)):
                raise RuntimeError("Control calibration provenance differs")
            protocol["control_calibration_sha256"] = sha(args.control_calibration)
    if not args.reference_initial or not args.change or (args.change != "control" and not args.probe):
        parser.error("Full training requires --reference-initial, --change and a passed --probe")
    gate = json.loads(args.probe.read_text()) if args.probe else None
    accepted_statuses = ("cost_gate_passed_quality_pending", "cost_gate_failed") if args.quality_followup else ("cost_gate_passed_quality_pending",)
    if gate and (gate.get("status") not in accepted_statuses or gate.get("vocabulary") != args.vocabulary
        or gate.get("change") != args.change or gate.get("reference_initial_sha256") != sha(args.reference_initial)):
        raise RuntimeError("The real numerical/cost probe must pass for this vocabulary")
    source = Path(__file__).with_name("rotary_candidate.py" if args.change == "rope" else "architecture_candidates.py")
    if gate and gate.get("candidate_sha256") != sha(source):
        raise RuntimeError("Candidate source changed after its real probe")
    for key in ("model_source_sha256", "training_helpers_sha256"):
        if gate and gate.get(key) != protocol[key]:
            raise RuntimeError(f"Source changed after the probe: {key}")
    if gate and gate.get("config_sha256") != config.sha256:
        raise RuntimeError("Configuration changed after the probe")
    reference_payload = torch.load(args.reference_initial, map_location="cpu", weights_only=False)
    if reference_payload.get("artifact_kind") != "fresh_initialization":
        raise ValueError("A trained checkpoint cannot substitute for the fresh reference initialization")
    reference_protocol = reference_payload["protocol"]
    if reference_protocol["recipe"] != args.recipe or reference_protocol["vocabulary"] != args.vocabulary:
        raise ValueError("Reference initialization uses a different recipe or vocabulary")
    if args.change in reference_protocol["architecture_changes"]:
        raise ValueError("The requested change is already present")
    protocol.update(architecture_changes=reference_protocol["architecture_changes"] + ([] if args.change == "control" else [args.change]),
                    reference_initial_sha256=sha(args.reference_initial), probe_sha256=sha(args.probe) if args.probe else None)
    if (out / "result.json").exists():
        done = json.loads((out / "result.json").read_text())
        if done["protocol"] != protocol or done["checkpoint_sha256"] != sha(out / "best-model.pt"):
            raise RuntimeError("Completed result provenance differs")
        print(f"REUSE {out}", flush=True)
        return
    reference = restore_model(reference_payload)
    model = restore_model(reference_payload) if args.change == "control" else apply_change(reference, args.change)
    del reference, reference_payload
    if (out / "initial-model.pt").exists():
        saved_initial = torch.load(out / "initial-model.pt", map_location="cpu", weights_only=False)
        if saved_initial["protocol"] != protocol:
            raise RuntimeError("Initial-state provenance differs; refusing resume")
        model.load_state_dict(saved_initial["model"])
        del saved_initial
    else:
        save_initial(out / "initial-model.pt", model, protocol)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    model.to(device)
    optimizer = optimizer_for(model, config)
    seed_everything(1337)
    history, best, consumed = [], {"score": float("inf")}, 0
    samplers = validation_tokens = None
    if args.recipe == "random-windows":
        samplers, validation_tokens = original_samplers(config, data)
    if (out / "resume.pt").exists():
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise RuntimeError("Resume provenance differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, consumed = saved["history"], saved["best"], saved["consumed_bytes"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])
        if samplers:
            advance_sampler(samplers["train"], len(history) * 100 * 2)
            advance_sampler(samplers["train_eval"], (len(history) + 1) * 30)
            advance_sampler(samplers["validation"], (len(history) + 1) * 30)
        del saved
    elif samplers:
        # Match the original evaluator's initial 30-window consumption.
        initial_train = evaluate_random_batches(model, samplers["train_eval"], device, 30)
        initial_validation = evaluate_random_batches(model, samplers["validation"], device, 30)
        save_json(out / "initial-evaluation.json", {"train": initial_train, "validation": initial_validation})
    save_json(out / "provenance.json", {"protocol": protocol, "initial_sha256": sha(out / "initial-model.pt"),
                                       "environment": environment_info(device), "parameters": model.parameter_count()})
    total_bytes = EPOCHS * data.variant["splits"]["train"]["scored_utf8_bytes"]
    byte_sizes = torch.tensor([len(data.tokenizer.id_to_token(i)) for i in range(args.vocabulary)])
    for segment in range(len(history), args.stop_after_segment):
        if samplers:
            def random_updates():
                for _ in range(100):
                    batches = []
                    for _ in range(2):
                        x, y = samplers["train"].next(torch.device("cpu"))
                        batches.append({"x": x, "y": y, "targets": y.numel(), "bytes": int(byte_sizes[y].sum())})
                    yield {"batches": batches, "targets": 4096, "bytes": sum(b["bytes"] for b in batches)}
            updates = random_updates()
        else:
            updates = data.training_updates(segment)
        durations, losses, calibration_samples = [], [], []
        memory = {"allocated": 0, "driver": 0}
        start = time.perf_counter()
        for index, item in enumerate(updates):
            lr = learning_rate(config, segment * 100 + index) if samplers else byte_lr(config, consumed, item["bytes"], total_bytes)
            for group in optimizer.param_groups:
                group["lr"] = lr
            tick = time.perf_counter()
            loss, _ = update(model, optimizer, item["batches"], device, memory)
            if args.change == "control" and index + 1 in (1, 20, 40, 60, 80, 100):
                old = next(row for row in rows if row["kind"] == "train" and row["step"] == index + 1)
                calibration_samples.append({"step": index + 1, "loss": loss, "old_loss": old["loss"],
                                            "absolute_difference": abs(loss - old["loss"])})
            durations.append(time.perf_counter() - tick)
            losses.append((loss, item["targets"]))
            consumed += item["bytes"]
        training_seconds = time.perf_counter() - start
        train_evaluation = None
        if samplers:
            train_evaluation = evaluate_random_batches(model, samplers["train_eval"], device, 30)
            validation = evaluate_random_batches(model, samplers["validation"], device, 30)
            score = validation["loss"]
        else:
            validation = evaluate(model, data, device)
            score = validation["bits_per_byte"]
        row = {"segment": segment + 1, "updates": len(durations), "consumed_bytes": consumed,
               "training_seconds": training_seconds, "median_update_seconds": statistics.median(durations),
               "train_loss": sum(v * n for v, n in losses) / sum(n for _, n in losses),
               "sampled_memory": memory, "train_evaluation": train_evaluation, "validation": validation}
        history.append(row)
        if score < best["score"]:
            best = {"score": score, "segment": segment + 1, "validation": validation}
            checkpoint(out / "best-model.pt", model, optimizer, protocol, history, best, consumed)
        checkpoint(out / "resume.pt", model, optimizer, protocol, history, best, consumed)
        save_json(out / "history.json", history)
        print(f"SEGMENT {args.change} {segment + 1}/30 score={score:.6f} seconds={training_seconds:.1f} best={best['segment']}", flush=True)
        if args.change == "control":
            differences = {"train": abs(train_evaluation["loss"] - control[100]["train"]["loss"]),
                           "validation": abs(score - control[100]["validation"]["loss"])}
            passed = max(differences.values()) <= .003 and max(r["absolute_difference"] for r in calibration_samples) <= .005
            save_json(out / "result.json", {"status": "calibration_passed" if passed else "calibration_failed",
                      "protocol": protocol, "differences": differences, "samples": calibration_samples,
                      "checkpoint_sha256": sha(out / "best-model.pt")})
            if not passed:
                raise RuntimeError("Fresh control diverged from historical control; do not reuse its curve")
            print("CALIBRATION PASSED", flush=True)
            return
        decision = staged_decision(history, control) if args.quality_followup else None
        if decision:
            save_json(out / f"decision-{(segment + 1) * 100}.json", decision)
            print(f"DECISION {json.dumps(decision)}", flush=True)
            if decision["stop"]:
                save_json(out / "result.json", {"status": "stopped_by_quality_budget", "protocol": protocol,
                          "decision": decision, "history": history, "best": best,
                          "checkpoint_sha256": sha(out / "best-model.pt")})
                return
    if len(history) < 30:
        print(f"PAUSED {out} after {len(history)} saved segments; full quality result pending", flush=True)
        return
    if not samplers and consumed != total_bytes:
        raise RuntimeError("Unequal final byte budget")
    del optimizer, model
    gc.collect()
    torch.mps.empty_cache()
    saved = torch.load(out / "best-model.pt", map_location="cpu", weights_only=False)
    model = restore_model(saved).to(device)
    del saved
    final_validation = evaluate(model, data, device)
    if samplers:
        independent = evaluate_random_batches(model, RandomWindowSampler(validation_tokens, 8, 256, 1837), device, 100)
    else:
        independent = final_validation
    audit = generation_audit(model, data, device, out / "generation-audit.json", use_cache=not args.quality_followup)
    result = {"status": "completed_not_promoted", "protocol": protocol, "best": best, "history": history,
              "validation": independent, "exact_byte_validation": final_validation, "audit": audit,
              "parameters": model.parameter_count(), "checkpoint_sha256": sha(out / "best-model.pt"),
              "initial_sha256": sha(out / "initial-model.pt")}
    save_json(out / "result.json", result)
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
