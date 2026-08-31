from __future__ import annotations

import asyncio
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

from .data import TOKENIZER_SHA256, file_digest, load_tokenizer
from .experiment import load_checkpoint
from .model import ManasGPT


ACCEPTED_CHECKPOINT_SHA256 = "cc415e95a70d5b93a02042afdf96441b38ba529da2152febe16edc46a3c5f1a1"


@dataclass(frozen=True)
class ServiceSettings:
    checkpoint: Path
    tokenizer: Path
    api_token: str
    checkpoint_sha256: str
    device: torch.device

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        checkpoint = os.environ.get("MANAS_CHECKPOINT_PATH", "").strip()
        tokenizer = os.environ.get("MANAS_TOKENIZER_PATH", "").strip()
        api_token = os.environ.get("MANAS_API_TOKEN", "")
        expected = os.environ.get("MANAS_CHECKPOINT_SHA256", ACCEPTED_CHECKPOINT_SHA256).strip()
        device_name = os.environ.get("MANAS_DEVICE", "cpu").strip().lower()
        if not checkpoint or not tokenizer or not api_token:
            raise RuntimeError(
                "MANAS_CHECKPOINT_PATH, MANAS_TOKENIZER_PATH, and MANAS_API_TOKEN are required"
            )
        if device_name not in {"cpu", "mps"}:
            raise RuntimeError("MANAS_DEVICE must be cpu or mps")
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MANAS_DEVICE=mps but Apple MPS is unavailable")
        return cls(
            checkpoint=Path(checkpoint).resolve(),
            tokenizer=Path(tokenizer).resolve(),
            api_token=api_token,
            checkpoint_sha256=expected,
            device=torch.device(device_name),
        )


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=300)
    max_new_tokens: int = Field(default=80, ge=16, le=128)
    temperature: float = Field(default=0.8, ge=0.5, le=1.2)
    top_k: int = Field(default=40, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class GenerateResponse(BaseModel):
    continuation: str
    seed: int
    new_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


runtime: dict[str, Any] = {}
generation_lock = asyncio.Lock()


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = file_digest(path, "sha256")
    if actual != expected_sha256:
        raise RuntimeError(f"{label} hash mismatch: expected {expected_sha256}, got {actual}")


@torch.inference_mode()
def _generate(payload: GenerateRequest) -> GenerateResponse:
    model: ManasGPT = runtime["model"]
    tokenizer: Tokenizer = runtime["tokenizer"]
    device: torch.device = runtime["settings"].device
    prompt_ids = tokenizer.encode(payload.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise HTTPException(status_code=422, detail="Prompt produced no tokens")
    if len(prompt_ids) >= model.config.block_size:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt is too long for the {model.config.block_size}-token context",
        )
    seed = payload.seed if payload.seed is not None else secrets.randbelow(2_147_483_648)
    generator = torch.Generator(device=device).manual_seed(seed)
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    generated = model.generate(
        token_ids,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        generator=generator,
    )
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    continuation_ids = generated[0, len(prompt_ids) :].tolist()
    return GenerateResponse(
        continuation=tokenizer.decode(continuation_ids),
        seed=seed,
        new_tokens=len(continuation_ids),
        elapsed_seconds=elapsed,
        tokens_per_second=len(continuation_ids) / elapsed,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = ServiceSettings.from_environment()
    _verify_file(settings.checkpoint, settings.checkpoint_sha256, "checkpoint")
    _verify_file(settings.tokenizer, TOKENIZER_SHA256, "tokenizer")
    if settings.device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model, checkpoint = load_checkpoint(settings.checkpoint, settings.device)
    model.eval()
    tokenizer = load_tokenizer(settings.tokenizer)
    runtime.update(
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        checkpoint_step=checkpoint["step"],
    )
    prompt_ids = tokenizer.encode("Манас", add_special_tokens=False).ids
    with torch.inference_mode():
        model(torch.tensor([prompt_ids], dtype=torch.long, device=settings.device))
    yield
    runtime.clear()


app = FastAPI(
    title="Tiny Manas inference",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    model: ManasGPT | None = runtime.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {
        "status": "ok",
        "model": "tiny-manas-27m",
        "parameters": model.parameter_count(),
        "context": model.config.block_size,
        "checkpoint_step": runtime["checkpoint_step"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    payload: GenerateRequest,
    x_manas_token: Annotated[str | None, Header()] = None,
) -> GenerateResponse:
    settings: ServiceSettings | None = runtime.get("settings")
    if settings is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if x_manas_token is None or not secrets.compare_digest(x_manas_token, settings.api_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if generation_lock.locked():
        raise HTTPException(status_code=429, detail="The model is busy; try again shortly")
    async with generation_lock:
        return await asyncio.to_thread(_generate, payload)
