"""Publish compact O14-O17 evidence from completed real runs; no model calls."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "data"
SOURCES = {}


def read(relative):
    raw = (ROOT / relative).read_bytes()
    SOURCES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def compact_score(score):
    return {k: v for k, v in score.items() if k not in ("evaluation",)}


def main():
    bundle = read("runs/data-evaluation-20260905/bundle/manifest.json")
    snapshot = {"as_of": "2026-09-05", "bundle": bundle, "runs": {}, "decisions": {}}
    for name, path in {"old256": "data-control-20260905", "expanded256": "data-expanded-20260905",
                       "context512": "context512-20260905"}.items():
        run = read(f"runs/{path}/result.json")
        curves = [{"step": r["step"], "train": r["training_loss"], "primary": r["primary"]["loss"],
                   "familiar": r["familiar"]["loss"], "common256": r.get("common_256", r["primary"])["loss"],
                   "training_seconds": r["training_seconds"], "targets_per_second": r["targets_per_second"]}
                  for r in run["history"]]
        with (OUT / f"followup-{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(curves)
        selected = next(r for r in run["history"] if r["step"] == run["best"]["step"])
        snapshot["runs"][name] = {"status": run["status"], "protocol": run["protocol"],
            "best": run["best"], "stage_decision": run["stage_decision"],
            "targets_consumed": run["targets_consumed"], "checkpoint_sha256": run["best_checkpoint_sha256"],
            "selected_scores": {k: compact_score(selected[k]) for k in ("primary", "familiar", "common_256") if k in selected},
            "training_seconds": sum(r["training_seconds"] for r in run["history"]),
            "post_evaluation_memory_bytes": {k: max(r["memory_bytes"][k] for r in run["history"]) for k in ("allocated", "driver")}}
        audit_path = f"runs/{path}/generation-audit.json"
        if (ROOT / audit_path).exists():
            audit = read(audit_path)
            snapshot["runs"][name]["audit"] = audit["summary"]
    for name, path in [("data", "data-selection-20260905"), ("context", "context-selection-20260905")]:
        report = read(f"runs/{path}/selection.json")
        snapshot["decisions"][name] = {k: v for k,v in report.items() if k not in ("candidate", "control", "incumbent")}
    final = read("runs/data-context-final-20260905/final-evaluation.json")
    snapshot["final"] = {k: compact_score(v) if isinstance(v, dict) else v for k,v in final.items()}
    precision = read("runs/inference-bf16-20260905/result.json")
    snapshot["precision_incumbent"] = {"status": precision["status"], "protocol": precision["protocol"],
        "familiar_fp32_loss": precision["validation"]["fp32"]["loss"],
        "loss_delta": precision["validation_loss_delta"], "probabilities": precision["probabilities"],
        "cache_parity": precision["cache_parity"], "timing": precision["timing"], "gates": precision["gates"],
        "sampled_live_memory_reduction": precision["sampled_live_memory_reduction"]}
    latest = ROOT / "runs/inference-bf16-selected-20260905/result.json"
    if latest.exists():
        report = read(str(latest.relative_to(ROOT)))
        snapshot["precision_selected"] = {k: v for k,v in report.items() if k not in ("validation", "audits")}
    snapshot["source_sha256"] = SOURCES
    snapshot["format_diagnostic"] = read("runs/data-evaluation-20260905/format-diagnostic.json")
    (OUT / "followup-evidence.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    print(f"Extracted {len(SOURCES)} follow-up artifacts; no model computation.")


if __name__ == "__main__":
    main()
