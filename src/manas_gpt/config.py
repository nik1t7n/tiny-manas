from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    device: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    max_tokens: int
    train_fraction: float
    validation_fraction: float

    @property
    def test_fraction(self) -> float:
        return max(0.0, 1.0 - self.train_fraction - self.validation_fraction)


@dataclass(frozen=True)
class ModelConfig:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    ffn_multiplier: int
    dropout: float
    bias: bool
    tie_embeddings: bool
    vocab_size: int = 0

    def with_vocab_size(self, vocab_size: int) -> ModelConfig:
        return ModelConfig(**{**self.__dict__, "vocab_size": vocab_size})


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    log_interval: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip: float
    fixed_batch: bool
    early_stop_patience: int
    target_train_loss: float


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    path: Path
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.__dict__,
            "data": {**self.data.__dict__, "test_fraction": self.data.test_fraction},
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "config_path": str(self.path),
            "config_sha256": self.sha256,
        }


def _require_table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing TOML table [{name}]")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    payload = config_path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))

    config = ExperimentConfig(
        run=RunConfig(**_require_table(raw, "run")),
        data=DataConfig(**_require_table(raw, "data")),
        model=ModelConfig(**_require_table(raw, "model")),
        training=TrainingConfig(**_require_table(raw, "training")),
        generation=GenerationConfig(**_require_table(raw, "generation")),
        path=config_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.run.device != "mps":
        raise ValueError("Tiny Manas requires run.device = 'mps'")
    if config.run.seed < 0:
        raise ValueError("run.seed must be non-negative")
    if config.data.max_tokens != 0 and config.data.max_tokens < 1000:
        raise ValueError("data.max_tokens must be 0 for full data or at least 1000")
    if not 0.5 <= config.data.train_fraction < 1.0:
        raise ValueError("data.train_fraction must be in [0.5, 1.0)")
    if not 0.0 < config.data.validation_fraction < 0.5:
        raise ValueError("data.validation_fraction must be in (0.0, 0.5)")
    if config.data.test_fraction < -1e-9:
        raise ValueError("data split fractions exceed 1.0")
    if config.model.n_embd % config.model.n_head != 0:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if config.model.block_size < 2:
        raise ValueError("model.block_size must be at least 2")
    if config.model.n_layer < 1 or config.model.n_head < 1 or config.model.n_embd < 1:
        raise ValueError("model dimensions must be positive")
    if config.model.ffn_multiplier < 1:
        raise ValueError("model.ffn_multiplier must be positive")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    training = config.training
    positive_ints = {
        "batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "max_steps": training.max_steps,
        "eval_interval": training.eval_interval,
        "eval_batches": training.eval_batches,
        "log_interval": training.log_interval,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f"training.{name} must be positive")
    if training.warmup_steps < 0 or training.warmup_steps > training.max_steps:
        raise ValueError("training.warmup_steps must be between 0 and max_steps")
    if not 0 < training.min_learning_rate <= training.learning_rate:
        raise ValueError("learning rates must satisfy 0 < min <= peak")
    if training.gradient_clip <= 0:
        raise ValueError("training.gradient_clip must be positive")
    if training.fixed_batch and config.model.dropout != 0.0:
        raise ValueError("fixed-batch overfit requires model.dropout = 0.0")
    if config.generation.max_new_tokens < 1:
        raise ValueError("generation.max_new_tokens must be positive")
    if config.generation.temperature <= 0:
        raise ValueError("generation.temperature must be positive")
    if config.generation.top_k < 1:
        raise ValueError("generation.top_k must be positive")
