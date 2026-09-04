"""Research-only O06 candidate; not a change to the accepted model or loader."""
import torch
from torch.nn import functional as F

from manas_gpt.model import ManasGPT
from manas_gpt.rotary import RotaryAttention, rotate_pairs, rotary_tables


def candidate_from(reference, device):
    candidate = RotaryGPT(reference.config)
    state = {name: value.detach().cpu() for name, value in reference.state_dict().items()
             if not name.startswith("position_embedding.")}
    candidate.load_state_dict(state)
    if any(not torch.equal(value, candidate.state_dict()[name]) for name, value in state.items()):
        raise AssertionError("RoPE changed shared initial parameters")
    return candidate.to(device)


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
