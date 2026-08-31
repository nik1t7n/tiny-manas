from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import prepare_dataset
from .experiment import evaluate_checkpoint, generate_from_checkpoint, train


def _json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manas-gpt", description="Tiny Manas research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="download and prepare pinned Manas data")
    prepare_parser.add_argument("--config", type=Path, required=True)

    train_parser = subparsers.add_parser("train", help="train a configured Tiny Manas model")
    train_parser.add_argument("--config", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="prepare data and train in one command")
    run_parser.add_argument("--config", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a local checkpoint")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    evaluate_parser.add_argument("--batches", type=int, default=50)

    generate_parser = subparsers.add_parser("generate", help="generate from a local checkpoint")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt")
    generate_parser.add_argument("--max-new-tokens", type=int)
    generate_parser.add_argument("--temperature", type=float)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        config = load_config(args.config)
        _json(prepare_dataset(config.data))
    elif args.command == "train":
        _json(train(args.config))
    elif args.command == "run":
        config = load_config(args.config)
        prepare_dataset(config.data)
        _json(train(args.config))
    elif args.command == "evaluate":
        _json(evaluate_checkpoint(args.checkpoint, args.split, args.batches))
    elif args.command == "generate":
        _json(
            generate_from_checkpoint(
                args.checkpoint,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.seed,
            )
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
