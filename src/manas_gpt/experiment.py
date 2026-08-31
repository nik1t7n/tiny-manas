from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .config import ExperimentConfig, ModelConfig, load_config
from .data import RandomWindowSampler, load_metadata, load_split, load_tokenizer
from .model import ManasGPT
from .paths import PROJECT_ROOT, RUNS_DIR


def require_mps() -> torch.device:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower()
    if fallback not in {"", "0", "false", "no"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must not be enabled for Tiny Manas")
    if not torch.backends.mps.is_built():
        raise RuntimeError("Installed PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available on this machine")
    return torch.device("mps")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def environment_info(device: torch.device) -> dict[str, Any]:
    recommended = torch.mps.recommended_max_memory()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_recommended_max_memory_bytes": recommended,
        "git_commit": git_commit(),
    }


def learning_rate(config: ExperimentConfig, step: int) -> float:
    training = config.training
    if training.warmup_steps and step < training.warmup_steps:
        return training.learning_rate * (step + 1) / training.warmup_steps
    if step >= training.max_steps:
        return training.min_learning_rate
    decay_span = max(1, training.max_steps - training.warmup_steps)
    ratio = (step - training.warmup_steps) / decay_span
    ratio = min(max(ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return training.min_learning_rate + coefficient * (
        training.learning_rate - training.min_learning_rate
    )

@torch.no_grad()
def evaluate_random_batches(
    model: ManasGPT,
    sampler: RandomWindowSampler,
    device: torch.device,
    batches: int,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for _ in range(batches):
        x, y = sampler.next(device)
        logits, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Evaluation loss was unexpectedly None")
        losses.append(float(loss.item()))
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = y.reshape(-1)
        correct_top1 += int((flat_logits.argmax(dim=-1) == flat_targets).sum().item())
        top5 = flat_logits.topk(k=min(5, flat_logits.size(-1)), dim=-1).indices
        correct_top5 += int((top5 == flat_targets[:, None]).any(dim=-1).sum().item())
        total += flat_targets.numel()
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "top1_accuracy": correct_top1 / total,
        "top5_accuracy": correct_top5 / total,
        "tokens": float(total),
    }


@torch.no_grad()
def generate_text(
    model: ManasGPT,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    tokenizer = load_tokenizer()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens")
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    generated = model.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    result_ids = generated[0].tolist()
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "all_token_ids": result_ids,
        "new_tokens": len(result_ids) - len(prompt_ids),
        "temperature": temperature,
        "top_k": top_k,
        "elapsed_seconds": elapsed,
        "tokens_per_second": max_new_tokens / elapsed,
        "text": tokenizer.decode(result_ids),
    }


def _cpu_state_dict(model: ManasGPT) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: ManasGPT,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    dataset_metadata: dict[str, Any],
    step: int,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model": _cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "experiment_config": config.as_dict(),
            "dataset_metadata": dataset_metadata,
            "step": step,
            "best_validation_loss": best_validation_loss,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ManasGPT, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ManasGPT(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    device = require_mps()
    seed_everything(config.run.seed)

    metadata = load_metadata(config.data.dataset)
    vocab_size = int(metadata["tokenizer"]["vocab_size"])
    model_config = config.model.with_vocab_size(vocab_size)
    model = ManasGPT(model_config).to(device)
    optimizer = model.configure_optimizer(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
    )

    train_tokens = load_split(config.data.dataset, "train")
    validation_tokens = load_split(config.data.dataset, "validation")
    train_sampler = RandomWindowSampler(
        train_tokens,
        config.training.batch_size,
        model_config.block_size,
        seed=config.run.seed + 1,
        fixed=config.training.fixed_batch,
    )
    train_eval_sampler = (
        train_sampler
        if config.training.fixed_batch
        else RandomWindowSampler(
            train_tokens,
            config.training.batch_size,
            model_config.block_size,
            seed=config.run.seed + 2,
            fixed=False,
        )
    )
    validation_sampler = RandomWindowSampler(
        validation_tokens,
        config.training.batch_size,
        model_config.block_size,
        seed=config.run.seed + 3,
        fixed=False,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"{config.run.name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.path, run_dir / "config.toml")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(device), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "architecture.json").write_text(
        json.dumps(model.architecture_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    generation_prompt = config.generation.prompt
    if config.training.fixed_batch:
        fixed_x, _ = train_sampler.next(device)
        generation_prompt = load_tokenizer().decode(fixed_x[0, :8].cpu().tolist())

    initial_train = evaluate_random_batches(
        model, train_eval_sampler, device, config.training.eval_batches
    )
    initial_validation = evaluate_random_batches(
        model, validation_sampler, device, config.training.eval_batches
    )
    initial_generation = generate_text(
        model,
        generation_prompt,
        min(config.generation.max_new_tokens, 80),
        config.generation.temperature,
        config.generation.top_k,
        config.run.seed,
        device,
    )
    (run_dir / "generation-initial.json").write_text(
        json.dumps(initial_generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    best_validation_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0
    peak_allocated = 0
    peak_driver = 0
    stopped_reason = "max_steps"
    started = time.perf_counter()

    for step in range(1, config.training.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            x, y = train_sampler.next(device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Training loss was unexpectedly None")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}: {loss.item()}")
            accumulated_loss += float(loss.item())
            (loss / config.training.gradient_accumulation_steps).backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {gradient_norm.item()}")
        current_lr = learning_rate(config, step - 1)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()

        mean_step_loss = accumulated_loss / config.training.gradient_accumulation_steps
        allocated = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        peak_allocated = max(peak_allocated, allocated)
        peak_driver = max(peak_driver, driver)

        if step % config.training.log_interval == 0 or step == 1:
            torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            trained_tokens = (
                step
                * config.training.gradient_accumulation_steps
                * config.training.batch_size
                * model_config.block_size
            )
            _append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "kind": "train",
                    "step": step,
                    "loss": mean_step_loss,
                    "learning_rate": current_lr,
                    "gradient_norm": float(gradient_norm.item()),
                    "elapsed_seconds": elapsed,
                    "trained_tokens": trained_tokens,
                    "tokens_per_second": trained_tokens / elapsed,
                    "mps_allocated_bytes": allocated,
                    "mps_driver_bytes": driver,
                },
            )

        should_evaluate = step % config.training.eval_interval == 0 or step == config.training.max_steps
        if should_evaluate:
            train_metrics = evaluate_random_batches(
                model, train_eval_sampler, device, config.training.eval_batches
            )
            validation_metrics = evaluate_random_batches(
                model, validation_sampler, device, config.training.eval_batches
            )
            _append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "kind": "evaluation",
                    "step": step,
                    "train": train_metrics,
                    "validation": validation_metrics,
                },
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_step = step
                evaluations_without_improvement = 0
                save_checkpoint(
                    run_dir / "best-model.pt",
                    model,
                    optimizer,
                    config,
                    metadata,
                    step,
                    best_validation_loss,
                )
            else:
                evaluations_without_improvement += 1

            if config.training.fixed_batch and config.training.target_train_loss > 0:
                if train_metrics["loss"] <= config.training.target_train_loss:
                    stopped_reason = "target_train_loss"
                    break
            if (
                not config.training.fixed_batch
                and config.training.early_stop_patience > 0
                and evaluations_without_improvement >= config.training.early_stop_patience
            ):
                stopped_reason = "early_stopping"
                break

    completed_step = step
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    final_train = evaluate_random_batches(
        model, train_eval_sampler, device, config.training.eval_batches
    )
    final_validation = evaluate_random_batches(
        model, validation_sampler, device, config.training.eval_batches
    )
    save_checkpoint(
        run_dir / "final-model.pt",
        model,
        optimizer,
        config,
        metadata,
        completed_step,
        best_validation_loss,
    )
    final_generation = generate_text(
        model,
        generation_prompt,
        config.generation.max_new_tokens,
        config.generation.temperature,
        config.generation.top_k,
        config.run.seed + 100,
        device,
    )
    (run_dir / "generation-final.json").write_text(
        json.dumps(final_generation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    trained_tokens = (
        completed_step
        * config.training.gradient_accumulation_steps
        * config.training.batch_size
        * model_config.block_size
    )
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "config": config.as_dict(),
        "model": model.architecture_dict(),
        "dataset": metadata,
        "initial": {"train": initial_train, "validation": initial_validation},
        "final": {"train": final_train, "validation": final_validation},
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "completed_step": completed_step,
        "stopped_reason": stopped_reason,
        "elapsed_seconds": elapsed,
        "trained_tokens": trained_tokens,
        "tokens_per_second": trained_tokens / elapsed,
        "peak_mps_allocated_bytes": peak_allocated,
        "peak_mps_driver_bytes": peak_driver,
        "initial_generation": initial_generation["text"],
        "final_generation": final_generation["text"],
        "generation_prompt": generation_prompt,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def evaluate_checkpoint(checkpoint: str | Path, split: str, batches: int) -> dict[str, Any]:
    device = require_mps()
    model, payload = load_checkpoint(checkpoint, device)
    experiment = payload["experiment_config"]
    dataset_name = experiment["data"]["dataset"]
    tokens = load_split(dataset_name, split)
    sampler = RandomWindowSampler(
        tokens,
        experiment["training"]["batch_size"],
        model.config.block_size,
        seed=experiment["run"]["seed"] + 500,
    )
    metrics = evaluate_random_batches(model, sampler, device, batches)
    return {"checkpoint": str(checkpoint), "split": split, "batches": batches, **metrics}


def generate_from_checkpoint(
    checkpoint: str | Path,
    prompt: str | None,
    max_new_tokens: int | None,
    temperature: float | None,
    top_k: int | None,
    seed: int | None,
) -> dict[str, Any]:
    device = require_mps()
    model, payload = load_checkpoint(checkpoint, device)
    generation = payload["experiment_config"]["generation"]
    run = payload["experiment_config"]["run"]
    return generate_text(
        model,
        prompt if prompt is not None else generation["prompt"],
        max_new_tokens if max_new_tokens is not None else generation["max_new_tokens"],
        temperature if temperature is not None else generation["temperature"],
        top_k if top_k is not None else generation["top_k"],
        seed if seed is not None else run["seed"],
        device,
    )
