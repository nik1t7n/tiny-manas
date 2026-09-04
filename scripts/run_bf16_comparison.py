"""Run the full preregistered pair sequentially; never overwrite accepted weights."""
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import traceback

import torch

from manas_gpt.audit import run_generation_audit
from manas_gpt.experiment import environment_info, evaluate_checkpoint, require_mps, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    device = require_mps()
    torch.mps.set_per_process_memory_fraction(0.65)
    destination = ROOT / "runs" / ("optimization-02-full-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    destination.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / "docs/experiments/00-baseline-manifest.json").read_text())
    # Test split is deliberately not loaded or evaluated during model selection.
    for relative in ("data/processed/manas01-full/train.bin", "data/processed/manas01-full/validation.bin", "data/tokenizer/kyrgyz-byte-bpe-v1.json"):
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != manifest["files"][relative]["sha256"]:
            raise RuntimeError(f"Frozen input mismatch: {relative}")
    state = {"environment": environment_info(device), "status": "running", "runs": {},
             "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / "src/manas_gpt").glob("*.py")}}
    state_path = destination / "comparison.json"

    def save():
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    save()
    print(f"COMPARISON {destination}", flush=True)
    try:
        original = (ROOT / "configs/manas01-27m.toml").read_text()
        for precision in ("fp32", "bf16"):
            config_path = destination / f"{precision}.toml"
            text = original.replace('name = "manas01-27m"', f'name = "optimization-02-{precision}"')
            text = text.replace("[training]", f'[training]\nprecision = "{precision}"')
            text = text.replace("early_stop_patience = 6", "early_stop_patience = 0")
            config_path.write_text(text)
            state["active_precision"] = precision
            save()
            print(f"START {precision} config={config_path}", flush=True)
            summary = train(config_path)
            run = Path(summary["run_dir"])
            state["runs"][precision] = {"training": summary}
            save()
            gc.collect()
            torch.mps.empty_cache()
            checkpoint = run / "best-model.pt"
            state["runs"][precision]["validation"] = evaluate_checkpoint(checkpoint, "validation", 100)
            state["runs"][precision]["audit"] = run_generation_audit(checkpoint, run / "generation-audit.json", 256, 0.8, 40)
            save()
            print(f"DONE {precision}: {json.dumps(state['runs'][precision]['validation'])}", flush=True)
            gc.collect()
            torch.mps.empty_cache()
        state["status"] = "completed_measurements_pending_review"
        save()
        print(f"COMPLETE {state_path}", flush=True)
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
