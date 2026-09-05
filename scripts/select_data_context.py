"""Validation-only O15/O16 decisions and post-selection held-out reporting."""
import argparse
import gc
import json
from pathlib import Path

import torch

from experiment_data_context import Corpus, BookEvaluation, BUNDLE
from followup_common import (BASELINE, BASELINE_SHA, ROOT, FixedEvaluation, evaluate,
    generation_audit, save_json, sha)
from manas_gpt.audit import audit_summary, longest_copied_word_span
from manas_gpt.data import load_tokenizer
from manas_gpt.experiment import load_checkpoint, require_mps


def selected_row(result):
    return next(h for h in result["history"] if h["step"] == result["best"]["step"])


def audit_checkpoint(path, output, corpus, names, device):
    if output.exists():
        report = json.loads(output.read_text())
        if report.get("checkpoint_sha256") == sha(path) and report.get("training_documents") == names and len(report["samples"]) == 20:
            return report
        raise RuntimeError("Incomplete or mismatched generation audit; inspect before resuming")
    model, payload = load_checkpoint(path, device)
    del payload
    generation_audit(model, device, output)
    report = json.loads(output.read_text())
    tokenizer = load_tokenizer()
    training = [tokenizer.decode(corpus.arrays[k].tolist()) for k in corpus.keys(names)]
    for sample in report["samples"]:
        matches = [longest_copied_word_span(sample["continuation"], text) for text in training]
        sample["longest_copied_word_span"] = max(matches, key=lambda match: match["words"])
    report.update(checkpoint_sha256=sha(path), training_documents=names,
                  copying_boundary="maximum within one retained training segment, never concatenated across books",
                  summary=audit_summary(report["samples"]))
    save_json(output, report)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--final-checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    corpus = Corpus()
    provenance = {"bundle_sha256": sha(BUNDLE / "manifest.json"), "script_sha256": sha(__file__)}
    if args.final_checkpoint:
        dest = args.output / "final-evaluation.json"
        if dest.exists():
            raise FileExistsError("Final test was already evaluated; reuse its report")
        model, payload = load_checkpoint(args.final_checkpoint, device)
        del payload
        result = {"checkpoint_sha256": sha(args.final_checkpoint), "provenance": provenance,
                  "selection_closed_before_test": True, "context": model.config.block_size}
        for name, data in [("orozbakov4_full_validation", BookEvaluation(corpus, full=True)),
                           ("mamay_full_test", BookEvaluation(corpus, "mamay", full=True)),
                           ("original_test", FixedEvaluation("test"))]:
            result[name] = evaluate(model, data, device, model.config.block_size)
            print(name, result[name]["loss"], flush=True)
            save_json(args.output / "final-evaluation-in-progress.json", result)
        save_json(dest, result)
        return
    if not args.candidate or not args.control:
        p.error("Provide candidate/control or a final checkpoint")
    candidate = json.loads((args.candidate / "result.json").read_text())
    control = json.loads((args.control / "result.json").read_text())
    if control["status"] != "completed":
        raise RuntimeError("Unfinished control")
    report = {"provenance": provenance, "candidate_sha256": sha(args.candidate / "best-model.pt"),
              "control_sha256": sha(args.control / "best-model.pt"), "candidate_status": candidate["status"]}
    if candidate["status"] != "completed":
        report.update(decision="not_promoted_budget_stop", reason="Preregistered full-budget requirement not met")
        save_json(args.output / "selection.json", report)
        print(report["decision"], flush=True)
        return
    cache = BUNDLE.parent / "incumbent-validation.json"
    if cache.exists():
        incumbent = json.loads(cache.read_text())
        if incumbent["checkpoint_sha256"] != BASELINE_SHA or incumbent["bundle_sha256"] != provenance["bundle_sha256"]:
            raise RuntimeError("Incumbent evaluation identity mismatch")
    else:
        if sha(BASELINE) != BASELINE_SHA:
            raise RuntimeError("Incumbent checkpoint identity mismatch")
        model, payload = load_checkpoint(BASELINE, device)
        del payload
        incumbent = {"checkpoint_sha256": BASELINE_SHA, "bundle_sha256": provenance["bundle_sha256"],
                     "primary": evaluate(model, BookEvaluation(corpus), device, 256),
                     "familiar": evaluate(model, FixedEvaluation(), device, 256)}
        save_json(cache, incumbent)
        del model
        gc.collect()
        torch.mps.empty_cache()
    c, r = selected_row(candidate), selected_row(control)
    gates = {"primary_gain": c["primary"]["loss"] <= r["primary"]["loss"]-.02,
             "familiar_vs_control": c["familiar"]["loss"] <= r["familiar"]["loss"]+.02,
             "familiar_vs_incumbent": c["familiar"]["loss"] <= incumbent["familiar"]["loss"]+.02,
             "primary_vs_incumbent": c["primary"]["loss"] <= incumbent["primary"]["loss"]+.02}
    if "common_256" in c:
        gates["common_context"] = c["common_256"]["loss"] <= r["primary"]["loss"]+.02
    report.update(candidate=c, control=r, incumbent=incumbent, gates=gates,
                  primary_delta=c["primary"]["loss"]-r["primary"]["loss"])
    report["decision"] = "not_promoted_prediction_gate"
    if all(gates.values()):
        audits = {}
        for name, directory, result in [("control", args.control, control), ("candidate", args.candidate, candidate)]:
            names = ["karalaev-train"]
            if result["protocol"]["mixture"] == "expanded":
                names += ["orozbakov-2", "orozbakov-3", "orozbakov-5"]
            audit = audit_checkpoint(directory / "best-model.pt", directory / "generation-audit.json", corpus, names, device)
            audits[name] = audit["summary"]
        report["audits"] = audits
        gates["generation_repetition"] = audits["candidate"]["mean_repeated_trigram_ratio"] <= audits["control"]["mean_repeated_trigram_ratio"]+.01
        report["decision"] = "passed_pending_full_text_review" if gates["generation_repetition"] else "not_promoted_generation_gate"
    save_json(args.output / "selection.json", report)
    print(report["decision"], gates, flush=True)


if __name__ == "__main__":
    main()
