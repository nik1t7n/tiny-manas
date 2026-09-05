"""O15/O16: resumable, budget-matched real-corpus training on native MPS."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import time

import torch

from build_manas_expansion import OUT as BUNDLE
from experiment_tokenizers import optimizer_for, update
from followup_common import (ROOT, FixedEvaluation, evaluate, generation_audit, save_json,
                             sha, baseline_model, synchronize)
from manas_gpt.config import ModelConfig, load_config
from manas_gpt.data import load_tokenizer, TOKENIZER_SHA256
from manas_gpt.experiment import learning_rate, require_mps, seed_everything, environment_info
from manas_gpt.model import ManasGPT


class Corpus:
    def __init__(self):
        self.manifest = json.loads((BUNDLE / "manifest.json").read_text())
        if sha(BUNDLE / "tokens.pt") != self.manifest["tokens_sha256"]:
            raise RuntimeError("Frozen token bundle identity changed")
        self.arrays = torch.load(BUNDLE / "tokens.pt", weights_only=True)
        self.docs = {d["id"]: d for d in self.manifest["documents"]}

    def keys(self, names):
        return [s["key"] for n in names for s in self.docs[n]["segments"]]


class BookEvaluation(FixedEvaluation):
    def __init__(self, corpus, name="orozbakov-4", full=False):
        keys = corpus.keys([name])
        if len(keys) != 1:
            raise RuntimeError("Evaluation must retain one whole, unpruned book")
        self.ids = corpus.arrays[keys[0]].long()
        self.tokenizer = load_tokenizer()
        self.lengths = torch.tensor([len(self.tokenizer.id_to_token(i)) for i in range(32768)])
        spans = [(i, min(i+128, len(self.ids))) for i in range(512, len(self.ids), 128)]
        if not full and len(spans) > 256:
            indices = torch.linspace(0, len(spans)-1, 256).round().long().tolist()
            if len(set(indices)) != 256:
                raise RuntimeError("Selection targets overlap")
            spans = [spans[i] for i in indices]
        self.spans = spans
        self.info = {"dataset": name, "split": corpus.docs[name]["split"],
            "source_sha256": corpus.docs[name]["text_sha256"], "tokenizer_sha256": TOKENIZER_SHA256,
            "prefix_tokens": 512, "stride": 128, "windows": len(spans),
            "targets": sum(b-a for a,b in spans),
            "scored_bytes": sum(int(self.lengths[self.ids[a:b]].sum()) for a,b in spans),
            "target_spans": spans, "target_coverage": "once; identical across contexts",
            "full_book_scoring": full, "independent_document_holdout": True,
            "independent_narrator_holdout": name == "mamay"}


class Sampler:
    def __init__(self, corpus, mixture, context):
        self.corpus, self.context = corpus, context
        self.batch = 2048 // context
        self.rng = torch.Generator().manual_seed(1338)
        old = corpus.keys(["karalaev-train"])
        new = corpus.keys(["orozbakov-2", "orozbakov-3", "orozbakov-5"])
        self.groups = [old, new] if mixture == "expanded" else [old]
        self.weights = [torch.tensor([len(corpus.arrays[k])-context for k in g], dtype=torch.float64) for g in self.groups]
        if any((w <= 0).any() for w in self.weights):
            raise RuntimeError("A segment cannot supply the registered context")

    def next(self):
        x, y = [], []
        for row in range(self.batch):
            group = row % len(self.groups)  # exact 50/50 when expanded
            index = torch.multinomial(self.weights[group], 1, generator=self.rng).item()
            ids = self.corpus.arrays[self.groups[group][index]]
            start = torch.randint(len(ids)-self.context, (1,), generator=self.rng).item()
            x.append(ids[start:start+self.context].long())
            y.append(ids[start+1:start+self.context+1].long())
        return {"x": torch.stack(x), "y": torch.stack(y), "targets": 2048}


def decision(history, control):
    step = history[-1]["step"]
    if step not in (900, 1200, 1500):
        return None
    deltas = [h["primary"]["loss"] - control[h["step"]]["primary"]["loss"] for h in history]
    recent, previous = statistics.mean(deltas[-3:]), statistics.mean(deltas[-6:-3])
    improvement = previous - recent
    stop = (recent > -.02 and improvement < .01) or (recent > .05 and min(deltas[-3:]) > .03 and improvement < .01)
    if step == 1500 and not (recent <= -.02 and max(deltas[-3:]) < 0):
        stop = True
    return {"step": step, "stop": stop, "recent_mean_delta": recent,
            "previous_mean_delta": previous, "gap_improvement": improvement,
            "last_three_deltas": deltas[-3:], "interpretation": "budget heuristic; not proof of eventual inferiority"}


def save_checkpoint(path, model, optimizer, sampler, protocol, history, best):
    payload = {"model": {k: v.detach().cpu() for k,v in model.state_dict().items()},
        "model_config": asdict(model.config), "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.rng.get_state(), "cpu_rng": torch.get_rng_state(),
        "mps_rng": torch.mps.get_rng_state(), "protocol": protocol, "history": history,
        "step": history[-1]["step"], "best": best}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixture", choices=("old", "expanded"), required=True)
    p.add_argument("--context", type=int, choices=(256,512), default=256)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control", type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus()
    base_config = load_config(ROOT / "configs/manas01-27m.toml")
    config = replace(base_config.model, vocab_size=32768, position_encoding="rope", block_size=args.context)
    initial_path = BUNDLE / "initial-model.pt"
    if not initial_path.exists():
        torch.manual_seed(1337)
        initial = ManasGPT(replace(config, block_size=256))
        torch.save({"model": initial.state_dict(), "model_config": asdict(initial.config),
                    "artifact_kind": "fresh_initialization", "seed": 1337}, initial_path)
        del initial
    protocol = {"mixture": args.mixture, "context": args.context, "seed": 1337,
        "max_steps": 3000, "targets_per_step": 4096, "precision_train": "bf16", "precision_eval": "fp32",
        "bundle_sha256": sha(BUNDLE / "manifest.json"), "initial_sha256": sha(initial_path),
        "runner_sha256": sha(__file__), "helpers_sha256": sha(Path(__file__).with_name("followup_common.py")),
        "training_helpers_sha256": sha(Path(__file__).with_name("experiment_tokenizers.py")),
        "config": base_config.as_dict(), "model_config": asdict(config),
        "control_sha256": sha(args.control / "result.json") if args.control else None}
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        if result["protocol"] != protocol:
            raise RuntimeError("Completed result provenance differs")
        print("REUSE", out, flush=True)
        return
    control = {}
    if args.control:
        control_result = json.loads((args.control / "result.json").read_text())
        if control_result["status"] != "completed":
            raise RuntimeError("A full-budget control is required")
        control = {r["step"]: r for r in control_result["history"]}
    elif args.mixture != "old" or args.context != 256:
        p.error("Only the old-data 256 control runs without --control")
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    model = ManasGPT(config)
    model.load_state_dict(torch.load(initial_path, weights_only=False)["model"])
    model.to(device)
    optimizer = optimizer_for(model, base_config)
    sampler = Sampler(corpus, args.mixture, args.context)
    familiar, primary = FixedEvaluation(), BookEvaluation(corpus)
    seed_everything(1337)
    history, best = [], {"loss": float("inf"), "step": 0}
    if (out / "resume.pt").exists():
        resume = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if resume["protocol"] != protocol:
            raise RuntimeError("Resume provenance differs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        sampler.rng.set_state(resume["sampler_rng"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.mps.set_rng_state(resume["mps_rng"])
        history, best = resume["history"], resume["best"]
        del resume
    save_json(out / "protocol.json", {"protocol": protocol, "environment": environment_info(device),
                                      "primary_evaluation": primary.info, "familiar_evaluation": familiar.info})
    start_step = history[-1]["step"] if history else 0
    segment_start, losses = time.perf_counter(), []
    stage = None
    for step in range(start_step+1, 3001):
        lr = learning_rate(base_config, step-1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, norm = update(model, optimizer, [sampler.next(), sampler.next()], device)
        losses.append(loss)
        if step % 20 == 0:
            print(f"TRAIN {args.mixture} T={args.context} step={step} loss={loss:.5f}", flush=True)
        if step % 100:
            continue
        duration = time.perf_counter()-segment_start
        row = {"step": step, "training_loss": statistics.mean(losses), "training_seconds": duration,
            "targets_per_second": 409600/duration, "learning_rate": lr,
            "primary": evaluate(model, primary, device, args.context),
            "familiar": evaluate(model, familiar, device, 256),
            "memory_bytes": {"allocated": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}}
        if args.context == 512:
            row["common_256"] = evaluate(model, primary, device, 256)
        history.append(row)
        if row["primary"]["loss"] < best["loss"]:
            best = {"loss": row["primary"]["loss"], "step": step}
            save_checkpoint(out / "best-model.pt", model, optimizer, sampler, protocol, history, best)
        save_checkpoint(out / "resume.pt", model, optimizer, sampler, protocol, history, best)
        save_json(out / "history.json", history)
        print(f"EVAL step={step} primary={row['primary']['loss']:.6f} familiar={row['familiar']['loss']:.6f} tok/s={row['targets_per_second']:.0f}", flush=True)
        if control:
            stage = decision(history, control)
            if stage:
                save_json(out / f"stage-{step}.json", stage)
                if stage["stop"]:
                    break
        segment_start, losses = time.perf_counter(), []
    result = {"status": "completed" if history[-1]["step"] == 3000 else "budget_stopped",
        "protocol": protocol, "best": best, "history": history, "stage_decision": stage,
        "targets_consumed": history[-1]["step"]*4096, "best_checkpoint_sha256": sha(out / "best-model.pt")}
    save_json(out / "result.json", result)
    print("RESULT", result["status"], "best", best, flush=True)


if __name__ == "__main__":
    main()
