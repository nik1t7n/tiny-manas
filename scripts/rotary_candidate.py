"""Research-only O06 candidate; not a change to the accepted model or loader."""
import torch
from torch import nn
from torch.nn import functional as F

from manas_gpt.model import ManasGPT


def candidate_from(reference, device):
    candidate = RotaryGPT(reference.config)
    state = {name: value.detach().cpu() for name, value in reference.state_dict().items()
             if not name.startswith("position_embedding.")}
    candidate.load_state_dict(state)
    if any(not torch.equal(value, candidate.state_dict()[name]) for name, value in state.items()):
        raise AssertionError("RoPE changed shared initial parameters")
    return candidate.to(device)


def rotary_tables(positions, head_size):
    if head_size % 2:
        raise ValueError("Adjacent-pair RoPE requires an even head size")
    frequencies = 10000.0 ** (-torch.arange(0, head_size, 2, device=positions.device, dtype=torch.float32) / head_size)
    angles = positions.float()[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def rotate_pairs(x, cos, sin):
    # Rotation arithmetic is FP32; SDPA still receives the original Q/K dtype.
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)


class RotaryAttention(nn.Module):
    def __init__(self, reference, block_size):
        super().__init__()
        self.n_head, self.n_embd, self.dropout = reference.n_head, reference.n_embd, reference.dropout
        # Preserve the exact shared weights rather than initialize extra projections.
        self.qkv, self.output = reference.qkv, reference.output
        self.output_dropout = reference.output_dropout
        cos, sin = rotary_tables(torch.arange(block_size), self.n_embd // self.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        q, k, v = (value.view(batch, time, self.n_head, head_size).transpose(1, 2) for value in (q, k, v))
        q = rotate_pairs(q, self.cos[:time], self.sin[:time])
        k = rotate_pairs(k, self.cos[:time], self.sin[:time])
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                                 is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.output_dropout(self.output(attended))


class RotaryGPT(ManasGPT):
    def __init__(self, config):
        super().__init__(config)
        del self.position_embedding
        for block in self.blocks:
            block.attention = RotaryAttention(block.attention, config.block_size)

    def forward(self, token_ids, targets=None, *, last_position_only=False):
        if self.activation_checkpointing:
            raise RuntimeError("O06 is preregistered without activation checkpointing")
        if last_position_only and targets is not None:
            raise ValueError("last_position_only cannot be used with training targets")
        if token_ids.shape[1] > self.config.block_size:
            raise ValueError("Sequence exceeds the declared context")
        x = self.embedding_dropout(self.token_embedding(token_ids))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, -1:, :] if last_position_only else x)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def parameter_count(self, non_embedding=False):
        # The parent's historical non_embedding option only subtracts learned positions.
        return sum(parameter.numel() for parameter in self.parameters())
