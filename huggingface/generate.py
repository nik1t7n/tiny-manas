from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from manas_gpt.config import ModelConfig  # noqa: E402
from manas_gpt.model import ManasGPT  # noqa: E402


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Tiny Manas continuation")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    config_payload = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    model = ManasGPT(ModelConfig(**config_payload["model_config"]))
    missing, unexpected = load_model(model, ROOT / "model.safetensors", strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Weight mismatch: missing={missing}, unexpected={unexpected}")

    device = _device(args.device)
    model.eval().to(device)
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("The prompt encoded to zero tokens")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        generated = model.generate(
            token_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            generator=generator,
            use_cache=True,
        )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
