"""Isolated O07/O08 candidates, pending real MPS and full-quality comparisons."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class FP32RMSNorm(nn.RMSNorm):
    def __init__(self, width):
        super().__init__(width, eps=1e-5, dtype=torch.float32)

    def forward(self, x):
        # Explicit FP32 input keeps reduction precision independent of autocast.
        return super().forward(x.float()).to(x.dtype)


def with_rmsnorm(reference):
    if any(parameter.device.type != "cpu" for parameter in reference.parameters()):
        raise ValueError("Build the isolated candidate from a CPU reference before timing")
    model = copy.deepcopy(reference)
    for block in model.blocks:
        for name in ("ln_attention", "ln_ffn"):
            old = getattr(block, name)
            norm = FP32RMSNorm(model.config.n_embd)
            with torch.no_grad():
                norm.weight.copy_(old.weight)
            setattr(block, name, norm)
    norm = FP32RMSNorm(model.config.n_embd)
    with torch.no_grad():
        norm.weight.copy_(model.final_norm.weight)
    model.final_norm = norm
    return model


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config, generator):
        super().__init__()
        numerator = 2 * config.ffn_multiplier * config.n_embd
        if numerator % 3:
            raise ValueError("Exact matrix-budget matching requires an integral hidden width")
        self.hidden = numerator // 3
        self.gate = nn.Linear(config.n_embd, self.hidden, bias=config.bias)
        self.up = nn.Linear(config.n_embd, self.hidden, bias=config.bias)
        self.down = nn.Linear(self.hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        for module in (self.gate, self.up, self.down):
            std = .02 / math.sqrt(2 * config.n_layer) if module is self.down else .02
            nn.init.normal_(module.weight, mean=0.0, std=std, generator=generator)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


def with_swiglu(reference, ffn_seed=1337):
    if any(parameter.device.type != "cpu" for parameter in reference.parameters()):
        raise ValueError("Build the isolated candidate from a CPU reference before timing")
    model = copy.deepcopy(reference)
    generator = torch.Generator(device="cpu").manual_seed(ffn_seed)
    for block in model.blocks:
        block.ffn = SwiGLUFeedForward(model.config, generator)
    return model
