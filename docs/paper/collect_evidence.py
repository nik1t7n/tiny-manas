"""Extract observed paper data. Does not import torch, train, or modify any run.

Run from any directory with Python 3.12+. The resulting small, public snapshot
excludes corpus payloads, credentials, model tensors and full generated texts.
"""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
SOURCES = {}


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def curve(run):
    path = f"runs/{run}/metrics.jsonl"
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(s) for s in raw.splitlines()]
    return [{"step": r["step"], "train": r["train"]["loss"],
             "validation": r["validation"]["loss"]}
            for r in rows if r["kind"] == "evaluation"]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


pilot = read("runs/pilot-10k-20260831T152349Z/summary.json")
base = read("runs/manas01-base-20260831T152803Z/summary.json")
original = read("reports/results.json")
paired = read("runs/optimization-02-full-20260904T053040Z/comparison.json")
rope = read("runs/quality-followup-rope-20260904/result.json")
rms = read("runs/quality-followup-rmsnorm-20260904/history.json")
rms_stop = read("runs/quality-followup-rmsnorm-20260904/decision-900.json")
swiglu = read("runs/optimization-08-swiglu-full-20260904/result.json")
test = read("runs/quality-followup-release-20260904/test-evaluation.json")
accepted = read("docs/experiments/accepted-state.json")
promotion = read("runs/quality-followup-rope-20260904/promotion.json")

write_csv("pilot.csv", curve("pilot-10k-20260831T152349Z"))
for name, run in [("base13", "manas01-base-20260831T152803Z"),
                  ("context512", "manas01-context512-20260831T154720Z"),
                  ("classic27", "manas01-27m-20260831T160739Z"),
                  ("bf16", "optimization-02-bf16-20260904T055741Z")]:
    write_csv(name + ".csv", curve(run))
for name, history in [("rope", rope["history"]), ("rmsnorm", rms),
                      ("swiglu", swiglu["history"])]:
    write_csv(name + ".csv", [{"step": 100 * h["segment"],
                               "train": h["train_evaluation"]["loss"],
                               "validation": h["validation"]["loss"]}
                              for h in history])
control_curve = {r["step"]: r["validation"] for r in curve("optimization-02-bf16-20260904T055741Z")}
for name, hist in [("rope", rope["history"]), ("rmsnorm", rms)]:
    write_csv(name + "-delta.csv", [{"step": h["segment"] * 100,
        "delta": h["validation"]["loss"] - control_curve[h["segment"] * 100]}
        for h in hist])

vocabulary = []
for size in [32768, 16384, 8192]:
    r = read(f"runs/optimization-05-tokenizers-20260904/{size}/result.json")
    vocabulary.append({"vocabulary": size, "parameters": r["parameters"],
        "best_epoch": r["best"]["epoch"], "updates": sum(h["updates"] for h in r["history"]),
        "bpb": r["best"]["bits_per_byte"], "token_loss": r["best"]["loss"],
        "scored_bytes": r["best"]["scored_bytes"], "targets": r["best"]["targets"],
        "training_seconds": r["training_seconds"], "bytes_per_second": r["training_bytes_per_second"],
        "allocation_bytes": r["sampled_memory"]["allocated"], "audit": r["audit"]})
    write_csv(f"vocab{size}.csv", [{"epoch": h["epoch"], "bpb": h["validation"]["bits_per_byte"]} for h in r["history"]])

systems = {}
for name, path in {
    "last_position": "runs/optimization-01-last-position/result.json",
    "compile": "runs/optimization-03-compile-20260904T062518Z/result.json",
    "checkpointing": "runs/optimization-04-checkpointing-20260904T062736Z/result.json",
    "cache": "runs/optimization-09-kv-cache-20260904/result.json",
    "gqa": "runs/optimization-10-gqa-20260904/result.json",
}.items():
    r = read(path)
    # Hashes give access to complete local evidence; retain only measured scalars
    # or compact timing summaries, never raw model outputs or absolute paths.
    systems[name] = {k: v for k, v in r.items() if k in [
        "parameters", "precision", "loss_delta_vs_adapted_mha", "loss_delta_vs_original_mha",
        "persistent_cache_reduction", "full_cache_storage", "single_token_128_past",
        "new_training_updates", "numeric_gates_passed"]}

audit = read("runs/quality-followup-rope-20260904/generation-audit.json")
audit_rows = [{"sample": i + 1, "prompt": s["prompt"], "seed": s["seed"],
              "repeated_trigram_ratio": s["repetition"]["repeated_trigram_ratio"]}
             for i, s in enumerate(audit["samples"])]
write_csv("rope-audit.csv", audit_rows)

snapshot = {
    "as_of": "2026-09-04", "evidence_scope": "Existing artifacts only; no training rerun for this paper.",
    "dataset": base["dataset"],
    "pilot": {k: pilot[k] for k in ["model", "initial", "final", "best_step", "completed_step", "elapsed_seconds"]},
    "original_experiments": original["experiments"],
    "precision_pair": {name: {"validation": {k: v for k, v in r["validation"].items() if k != "checkpoint"},
        "elapsed_seconds": r["training"]["elapsed_seconds"], "best_step": r["training"]["best_step"]}
        for name, r in paired["runs"].items()},
    "vocabularies": vocabulary,
    "rope": {k: rope[k] for k in ["parameters", "best", "validation", "exact_byte_validation", "audit", "checkpoint_sha256"]},
    "rmsnorm_stop": rms_stop,
    "swiglu": {k: swiglu[k] for k in ["parameters", "best", "validation", "exact_byte_validation", "audit"]},
    "test": {k: v for k, v in test.items() if k != "checkpoint"},
    "systems": systems,
    "accepted": {k: accepted[k] for k in ["parameters", "position_encoding", "normalization", "training_precision", "inference_precision", "execution", "activation_checkpointing", "release_commit", "inference_artifact_sha256", "inference_artifact_bytes"]},
    "source_sha256": SOURCES,
}
(OUT / "evidence.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
print(f"Extracted {len(SOURCES)} real artifacts into {OUT}; no model computations performed.")
