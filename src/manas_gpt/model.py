from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .checkpointing import checkpoint_block
from .kv_cache import CacheSession


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        q = q.view(batch, time, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_size).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.output_dropout(self.output(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_multiplier * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln_ffn = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_attention(x))
        return x + self.ffn(self.ln_ffn(x))


class ManasGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size < 1:
            raise ValueError("ModelConfig.vocab_size must be set before model construction")
        self.config = config
        self.activation_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.output.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if last_position_only and targets is not None:
            raise ValueError("last_position_only cannot be used with training targets")
        batch, time = token_ids.shape
        if time > self.config.block_size:
            raise ValueError(
                f"Sequence length {time} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint_block(block, x)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, -1:, :] if last_position_only else x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        generator: torch.Generator | None = None,
        *,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if temperature <= 0 or top_k <= 0 or max_new_tokens < 0:
            raise ValueError("Require positive temperature/top_k and nonnegative length")
        cache = CacheSession(self) if use_cache else None
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            if cache is None:
                logits, _ = self(context, last_position_only=True)
            elif cache.layers is None or cache.length == self.config.block_size:
                # Rebuild after cropping: retained tokens change position IDs,
                # and their deeper representations must forget the evicted prefix.
                logits = cache.prefill(context)
            else:
                logits = cache.decode(token_ids[:, -1:])
            next_logits = logits[:, -1, :] / temperature
            k = min(top_k, next_logits.size(-1))
            threshold = torch.topk(next_logits, k).values[:, [-1]]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def parameter_count(self, non_embedding: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.position_embedding.weight.numel()
        return count

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
    ) -> torch.optim.AdamW:
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        decay = [parameter for parameter in parameters.values() if parameter.dim() >= 2]
        no_decay = [parameter for parameter in parameters.values() if parameter.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    def architecture_dict(self) -> dict[str, Any]:
        return {**asdict(self.config), "parameters": self.parameter_count()}
