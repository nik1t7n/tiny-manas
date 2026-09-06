from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model, save_model
from tokenizers import Tokenizer

from manas_gpt.config import ModelConfig
from manas_gpt.model import ManasGPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "huggingface"
RUNTIME_FILES = (
    "__init__.py",
    "checkpointing.py",
    "config.py",
    "kv_cache.py",
    "model.py",
    "rotary.py",
)
TEMPLATE_FILES = ("generate.py", "requirements.txt", "LICENSE", "CITATION.cff")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_digest(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reproducible Tiny Manas Hub release")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--accepted-state", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    accepted_state_path = args.accepted_state.resolve()
    paper_path = args.paper.resolve()
    output = args.output.resolve()
    for path in (checkpoint, tokenizer_path, accepted_state_path, paper_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_digest(checkpoint, args.checkpoint_sha256)
    require_digest(tokenizer_path, args.tokenizer_sha256)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**payload["model_config"])
        state = dict(payload["model"])
        if model_config.tie_embeddings:
            state["lm_head.weight"] = state["token_embedding.weight"]
        model = ManasGPT(model_config)
        model.load_state_dict(state, strict=True)
        model.eval()

        source_commit = git_commit()
        weights_path = temporary / "model.safetensors"
        save_model(
            model,
            str(weights_path),
            metadata={
                "format": "pt",
                "model": "Tiny Manas",
                "source_checkpoint_sha256": args.checkpoint_sha256,
                "source_commit": source_commit,
            },
        )

        reloaded = ManasGPT(model_config)
        missing, unexpected = load_model(reloaded, weights_path, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Safetensors mismatch: missing={missing}, unexpected={unexpected}")
        reloaded.eval()

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        prompt_ids = tokenizer.encode("Манас", add_special_tokens=False).ids
        if not prompt_ids:
            raise RuntimeError("Fixed verification prompt encoded to zero tokens")
        inputs = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.inference_mode():
            expected_logits, _ = model(inputs, last_position_only=True)
            actual_logits, _ = reloaded(inputs, last_position_only=True)
        if not torch.equal(expected_logits, actual_logits):
            max_error = float((expected_logits - actual_logits).abs().max().item())
            raise RuntimeError(f"Safetensors logits are not exact; max error {max_error}")

        shutil.copy2(tokenizer_path, temporary / "tokenizer.json")
        shutil.copy2(paper_path, temporary / "tiny-manas-paper.pdf")
        for filename in TEMPLATE_FILES:
            shutil.copy2(TEMPLATE_DIR / filename, temporary / filename)

        runtime_dir = temporary / "src" / "manas_gpt"
        runtime_dir.mkdir(parents=True)
        for filename in RUNTIME_FILES:
            shutil.copy2(PROJECT_ROOT / "src" / "manas_gpt" / filename, runtime_dir / filename)

        accepted = json.loads(accepted_state_path.read_text(encoding="utf-8"))
        write_json(
            temporary / "config.json",
            {
                "schema_version": 1,
                "model_type": "tiny_manas",
                "architectures": ["ManasGPT"],
                "model_config": payload["model_config"],
                "parameters": model.parameter_count(),
                "torch_dtype": "float32",
                "weights": "model.safetensors",
                "tokenizer": "tokenizer.json",
            },
        )
        write_json(
            temporary / "generation_config.json",
            {
                "max_new_tokens": 64,
                "temperature": 0.8,
                "top_k": 40,
                "use_cache": True,
            },
        )
        write_json(
            temporary / "provenance.json",
            {
                "schema_version": 1,
                "source_repository": "https://github.com/nik1t7n/tiny-manas",
                "source_commit": source_commit,
                "source_checkpoint": checkpoint.name,
                "source_checkpoint_sha256": args.checkpoint_sha256,
                "source_checkpoint_step": payload.get("step"),
                "source_checkpoint_best_validation_loss": payload.get("best_validation_loss"),
                "tokenizer_sha256": args.tokenizer_sha256,
                "accepted_model": {
                    "parameters": accepted["parameters"],
                    "position_encoding": accepted["position_encoding"],
                    "context": payload["model_config"]["block_size"],
                    "training_precision": accepted["training_precision"],
                    "inference_precision": accepted["inference_precision"],
                    "validation_loss": accepted["validation_loss"],
                    "final_test": accepted["final_test"],
                },
                "training_dataset": {
                    "document": "Manas01",
                    "performer_author": "Sayakbai Karalaev",
                    "corpus_license": "CC BY-NC-SA 4.0",
                    "source_text_included": False,
                },
                "conversion_verification": {
                    "prompt": "Манас",
                    "prompt_tokens": len(prompt_ids),
                    "fp32_logits_exact": True,
                },
            },
        )

        weights_sha = digest(weights_path)
        weights_bytes = weights_path.stat().st_size
        readme = (TEMPLATE_DIR / "README.template.md").read_text(encoding="utf-8")
        replacements = {
            "{{SOURCE_CHECKPOINT_SHA256}}": args.checkpoint_sha256,
            "{{MODEL_SAFETENSORS_SHA256}}": weights_sha,
            "{{TOKENIZER_SHA256}}": args.tokenizer_sha256,
            "{{SOURCE_COMMIT}}": source_commit,
        }
        for marker, value in replacements.items():
            readme = readme.replace(marker, value)
        if "{{" in readme or "}}" in readme:
            raise RuntimeError("Unresolved marker in rendered model card")
        (temporary / "README.md").write_text(readme, encoding="utf-8")

        files = {}
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            files[str(path.relative_to(temporary))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        write_json(
            temporary / "release-manifest.json",
            {
                "schema_version": 1,
                "repository": "Nik1t7n/tiny-manas",
                "source_commit": source_commit,
                "conversion_verification": "exact_fp32_logits",
                "files": files,
            },
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "files": len(files) + 1,
                    "model_safetensors_bytes": weights_bytes,
                    "model_safetensors_sha256": weights_sha,
                    "exact_fp32_logits": True,
                },
                indent=2,
            )
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
