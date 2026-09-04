"""Adjacent-pair RoPE, shared by the measured candidate and native inference."""
import torch
from torch import nn
from torch.nn import functional as F


def rotary_tables(positions, head_size):
    if head_size % 2:
        raise ValueError("Adjacent-pair RoPE requires an even head size")
    frequencies = 10000.0 ** (-torch.arange(0, head_size, 2, device=positions.device, dtype=torch.float32) / head_size)
    angles = positions.float()[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def rotate_pairs(x, cos, sin):
    # Rotation arithmetic is FP32; SDPA receives the incoming Q/K dtype.
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)


class RotaryAttention(nn.Module):
    def __init__(self, reference, block_size):
        super().__init__()
        self.n_head, self.n_embd, self.dropout = reference.n_head, reference.n_embd, reference.dropout
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
