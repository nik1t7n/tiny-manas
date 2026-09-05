"""Post-hoc validation attribution after the observed new-book loss divergence.

No training, data transformation, test access, or selection-rule change.
"""
import gc
import json

import torch
from torch.nn import functional as F

from experiment_data_context import Corpus, BookEvaluation
from followup_common import ROOT, save_json, sha
from manas_gpt.experiment import load_checkpoint, require_mps


@torch.inference_mode()
def main():
    output = ROOT / "runs/data-evaluation-20260905/format-diagnostic.json"
    if output.exists():
        raise FileExistsError("Reuse the completed diagnostic instead of repeating it")
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(.65)
    data = BookEvaluation(Corpus())
    newline = torch.tensor(["\n" in data.tokenizer.decode([i]) for i in range(32768)], device=device)
    result = {"post_hoc": True, "used_for_selection": False, "split": "validation",
              "reason": "Old-data book loss rose while familiar loss fell; verse line formatting differs",
              "evaluation": data.info, "script_sha256": sha(__file__), "runs": {}}
    for name in ("data-control-20260905", "data-expanded-20260905"):
        directory = ROOT / "runs" / name
        training = json.loads((directory / "result.json").read_text())
        if training["status"] != "completed":
            raise RuntimeError("The diagnostic compares two completed equal-budget runs")
        model, payload = load_checkpoint(directory / "resume.pt", device)
        if payload["step"] != 3000 or payload["protocol"] != training["protocol"]:
            raise RuntimeError("Final checkpoint identity does not match the recorded full run")
        del payload
        model.eval()
        sums, counts = [0., 0.], [0, 0]
        for x, y, _ in data.batches(256):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            active = y != -100
            losses = F.cross_entropy(logits[active].float(), y[active], reduction="none")
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite attribution loss")
            kind = newline[y[active]]
            for i, mask in enumerate((~kind, kind)):
                sums[i] += losses[mask].sum().item()
                counts[i] += mask.sum().item()
        total = sum(sums)/sum(counts)
        expected = training["history"][-1]["primary"]["loss"]
        if sum(counts) != data.info["targets"] or abs(total-expected) > 1e-5:
            raise RuntimeError("Diagnostic did not reproduce the frozen final validation score")
        row = {"checkpoint_sha256": sha(directory / "resume.pt"), "step": 3000, "total_loss": total,
               "groups": {key: {"targets": counts[i], "summed_nll": sums[i], "mean_loss": sums[i]/counts[i],
                                 "contribution_to_total_loss": sums[i]/sum(counts)}
                          for i, key in enumerate(("other_tokens", "newline_bearing_tokens"))}}
        result["runs"][name] = row
        print(name, json.dumps(row["groups"]), flush=True)
        del model
        gc.collect()
        torch.mps.empty_cache()
    save_json(output, result)


if __name__ == "__main__":
    main()
