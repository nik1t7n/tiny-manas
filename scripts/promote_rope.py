"""Export the measured RoPE winner after real native/cache parity checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from architecture_candidates import restore_model
from experiment_tokenizers import INCUMBENT, INCUMBENT_SHA, ROOT, save_json, sha
from manas_gpt.config import ModelConfig, load_config
from manas_gpt.data import load_split
from manas_gpt.experiment import export_inference_checkpoint, load_checkpoint, seed_everything
from manas_gpt.kv_cache import CacheSession
from manas_gpt.model import ManasGPT


def main():
    run = ROOT / "runs/quality-followup-rope-20260904"
    checkpoint = run / "best-model.pt"
    result = json.loads((run / "result.json").read_text())
    expected = "c43e76c6307ce345da3983e04132b95291270c07bb7c38763c48b3ef394b6473"
    if (sha(checkpoint) != expected or result["checkpoint_sha256"] != expected
        or result["status"] != "completed_not_promoted"
        or result["validation"]["loss"] > 4.345779147148132 - .02
        or result["audit"]["samples"] != 20 or sha(INCUMBENT) != INCUMBENT_SHA):
        raise RuntimeError("Pinned research winner or its quality evidence differs")
    output = ROOT / "artifacts/tiny-manas-27m-rope-20260904.pt"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["protocol"]["architecture_changes"] != ["rope"]:
        raise ValueError("This promotion is the isolated RoPE arm, not a combined architecture")
    config = replace(ModelConfig(**saved["model_config"]), position_encoding="rope")
    torch.set_num_threads(2)
    model = ManasGPT(config).eval()
    model.load_state_dict(saved["model"], strict=True)
    reference = restore_model(saved).eval()
    recipe = load_config(ROOT / "configs/manas01-27m-rope.toml")
    tokens = load_split(recipe.data.dataset, "validation")
    x = tokens[:256][None]
    evidence = {"research_checkpoint_sha256": expected, "device": "cpu", "position_encoding": "rope"}
    with torch.no_grad():
        native_logits, _ = model(x)
        reference_logits, _ = reference(x)
        evidence["native_logit_max_error"] = (native_logits - reference_logits).abs().max().item()
        if not torch.equal(native_logits, reference_logits):
            raise AssertionError("Native RoPE differs from the measured candidate")
        # One real continuation traverses prefill, one-token decode and window overflow.
        prompt = tokens[:251][None]
        cached = model.generate(prompt, 12, 1.0, 1, use_cache=True)
        uncached = reference.generate(prompt, 12, 1.0, 1, use_cache=False)
        if not torch.equal(cached, uncached):
            raise AssertionError("Cached greedy continuation changed across window overflow")
        evidence["cached_greedy_equal_across_overflow"] = True
        session = CacheSession(model)
        logits = session.prefill(prompt)
        expected_logits, _ = reference(prompt, last_position_only=True)
        errors = [(logits - expected_logits).abs().max().item()]
        for length in range(252, 257):
            logits = session.decode(tokens[length - 1:length][None])
            expected_logits, _ = reference(tokens[:length][None], last_position_only=True)
            errors.append((logits - expected_logits).abs().max().item())
            if not torch.allclose(logits, expected_logits, atol=1e-4, rtol=1e-4):
                raise AssertionError("Cached RoPE decode changed logits")
        evidence["cache_max_logit_error"] = max(errors)
        evidence["cache_storage"] = session.storage()
        # Preserve the original fresh shared initialization, not just checkpoint loading.
        seed_everything(1337)
        fresh = ManasGPT(config)
        initial = torch.load(run / "initial-model.pt", map_location="cpu", weights_only=False)
        if any(not torch.equal(value, fresh.state_dict()[key]) for key, value in initial["model"].items()):
            raise AssertionError("Native fresh initialization differs from the controlled candidate")
        evidence["shared_fresh_initialization_equal"] = True
    incumbent = torch.load(INCUMBENT, map_location="cpu", weights_only=False)
    canonical = {
        "schema_version": 1, "model": saved["model"], "model_config": asdict(config),
        "experiment_config": recipe.as_dict(), "dataset_metadata": incumbent["dataset_metadata"],
        "step": result["best"]["segment"] * 100, "best_validation_loss": result["best"]["score"],
        "research_provenance": {"source_checkpoint_sha256": expected, "protocol": saved["protocol"],
                                "quality_result_sha256": sha(run / "result.json"),
                                "promotion_config_note": "Explicit native RoPE config; original measured protocol retained"},
    }
    # Original research checkpoint remains byte-for-byte unchanged.
    torch.save(canonical, run / "native-checkpoint.pt")
    evidence["export"] = export_inference_checkpoint(run / "native-checkpoint.pt", output)
    loaded, payload = load_checkpoint(output, torch.device("cpu"))
    loaded.eval()
    with torch.no_grad():
        if not torch.equal(loaded(x)[0], native_logits):
            raise AssertionError("The ordinary exported-checkpoint loader changed predictions")
    if payload["research_provenance"]["source_checkpoint_sha256"] != expected:
        raise AssertionError("Export lost research provenance")
    evidence["native_export_reload_equal"] = True
    save_json(run / "promotion.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
