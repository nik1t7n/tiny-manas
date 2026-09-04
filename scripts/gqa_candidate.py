"""O10 research candidate: mean-pooled K/V with explicit MPS head expansion."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from rotary_candidate import rotate_pairs


class GroupedAttention(nn.Module):
    def __init__(self, reference, kv_heads):
        super().__init__()
        self.n_head, self.n_embd = reference.n_head, reference.n_embd
        self.n_kv_head = kv_heads
        self.dropout = reference.dropout
        if kv_heads < 1 or self.n_head % kv_heads or reference.qkv.out_features != 3 * self.n_embd:
            raise ValueError("Convert ordinary MHA with a KV-head count dividing query heads")
        self.head_size = self.n_embd // self.n_head
        self.kv_width = kv_heads * self.head_size
        self.qkv = nn.Linear(self.n_embd, self.n_embd + 2 * self.kv_width,
                             bias=reference.qkv.bias is not None)
        groups = self.n_head // kv_heads
        weights = reference.qkv.weight.detach().view(3, self.n_head, self.head_size, self.n_embd)
        pooled = [weights[0].reshape(self.n_embd, self.n_embd)]
        for index in (1, 2):
            pooled.append(weights[index].reshape(kv_heads, groups, self.head_size, self.n_embd)
                          .mean(dim=1).reshape(self.kv_width, self.n_embd))
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat(pooled))
            if self.qkv.bias is not None:
                biases = reference.qkv.bias.detach().view(3, self.n_head, self.head_size)
                values = [biases[0].reshape(self.n_embd)]
                for index in (1, 2):
                    values.append(biases[index].reshape(kv_heads, groups, self.head_size).mean(1).flatten())
                self.qkv.bias.copy_(torch.cat(values))
        self.output, self.output_dropout = reference.output, reference.output_dropout
        if hasattr(reference, "cos"):
            self.register_buffer("cos", reference.cos.clone(), persistent=False)
            self.register_buffer("sin", reference.sin.clone(), persistent=False)

    def forward(self, x):
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).split((self.n_embd, self.kv_width, self.kv_width), dim=-1)
        q = q.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        k, v = (value.view(batch, time, self.n_kv_head, self.head_size).transpose(1, 2) for value in (k, v))
        if hasattr(self, "cos"):
            q = rotate_pairs(q, self.cos[:time], self.sin[:time])
            k = rotate_pairs(k, self.cos[:time], self.sin[:time])
        if self.n_kv_head != self.n_head:
            # Declared implementation, not a fallback: temporary eight-head inputs.
            repeats = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.output_dropout(self.output(attended))


def from_mha(reference, kv_heads=2):
    if any(parameter.device.type != "cpu" for parameter in reference.parameters()):
        raise ValueError("Convert the explicit CPU checkpoint before timing")
    model = copy.deepcopy(reference)
    for block in model.blocks:
        block.attention = GroupedAttention(block.attention, kv_heads)
    return model
