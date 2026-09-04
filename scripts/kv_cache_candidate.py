"""Research-only, request-local MHA cache for the O09 equivalence experiment."""
import torch
from torch.nn import functional as F

from rotary_candidate import rotate_pairs


class CacheSession:
    """One immutable eval model, one request; full-window overflow is explicit."""

    def __init__(self, model):
        if model.training:
            raise ValueError("KV caching requires model.eval(); training stays uncached")
        for block in model.blocks:
            if block.attention.qkv.out_features != 3 * model.config.n_embd:
                raise ValueError("O09 currently supports the declared MHA projections only")
        self.model = model
        self.layers = None
        self.length = 0
        self.batch_size = None

    @torch.no_grad()
    def _forward(self, token_ids, past, offset):
        model = self.model
        if model.training:
            raise ValueError("The cached model must remain in evaluation mode")
        batch, time = token_ids.shape
        x = model.token_embedding(token_ids)
        if hasattr(model, "position_embedding"):
            positions = torch.arange(offset, offset + time, device=token_ids.device)
            x = x + model.position_embedding(positions)[None]
        x = model.embedding_dropout(x)
        layers = []
        for index, block in enumerate(model.blocks):
            attention = block.attention
            features = block.ln_attention(x)
            q, k, v = attention.qkv(features).split(model.config.n_embd, dim=-1)
            dim = model.config.n_embd // attention.n_head
            q, k, v = (value.view(batch, time, attention.n_head, dim).transpose(1, 2)
                       for value in (q, k, v))
            if hasattr(attention, "cos"):
                cos, sin = attention.cos[offset:offset + time], attention.sin[offset:offset + time]
                q, k = rotate_pairs(q, cos, sin), rotate_pairs(k, cos, sin)
            if past is not None:
                k = torch.cat((past[index][0], k), dim=-2)
                v = torch.cat((past[index][1], v), dim=-2)
            else:
                # Own K/V storage, not views retaining the entire combined QKV output.
                k = k.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
            layers.append((k, v))
            # Decode has one query and no future keys. A top-left rectangular
            # causal mask would incorrectly hide almost all cached positions.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                       is_causal=past is None)
            attended = attended.transpose(1, 2).contiguous().view(batch, time, model.config.n_embd)
            x = x + attention.output_dropout(attention.output(attended))
            x = x + block.ffn(block.ln_ffn(x))
        logits = model.lm_head(model.final_norm(x)[:, -1:, :])
        return logits, layers

    def prefill(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.model.config.block_size:
            raise ValueError("Prefill requires a nonempty (B,T) prompt within the context window")
        # Starting/rebuilding a prompt invalidates all previous request state.
        self.layers, self.length, self.batch_size = None, 0, None
        logits, layers = self._forward(token_ids, None, 0)
        self.layers, self.length, self.batch_size = layers, token_ids.shape[1], token_ids.shape[0]
        return logits

    def decode(self, token_ids):
        if self.layers is None:
            raise ValueError("Call prefill before decode")
        if token_ids.shape != (self.batch_size, 1):
            raise ValueError("Decode accepts exactly one new token per existing batch row")
        if self.length >= self.model.config.block_size:
            raise ValueError("Full cache: explicitly prefill the cropped window instead")
        logits, layers = self._forward(token_ids, self.layers, self.length)
        self.layers = layers
        self.length += 1
        return logits

    def storage(self):
        tensors = [] if self.layers is None else [value for pair in self.layers for value in pair]
        unique = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in tensors}
        return {"positions": self.length, "logical_bytes": sum(value.numel() * value.element_size() for value in tensors),
                "storage_bytes": sum(unique.values()), "dtype": str(tensors[0].dtype) if tensors else None,
                "allocation": "dynamic concatenation; no preallocated unused capacity"}


@torch.no_grad()
def cached_generate(model, token_ids, max_new_tokens, temperature=.8, top_k=40, generator=None, *, greedy=False):
    if max_new_tokens < 0 or temperature <= 0 or top_k <= 0:
        raise ValueError("Require nonnegative length and positive temperature/top_k")
    session = CacheSession(model)
    for _ in range(max_new_tokens):
        if session.layers is None or session.length == model.config.block_size:
            # Rebuilding at overflow is the declared crop semantics, not eviction
            # with stale positions or hidden historical context in deeper states.
            logits = session.prefill(token_ids[:, -model.config.block_size:])
        else:
            logits = session.decode(token_ids[:, -1:])
        scores = logits[:, -1] / temperature
        if greedy:
            next_token = scores.argmax(dim=-1, keepdim=True)
        else:
            threshold = torch.topk(scores, min(top_k, scores.size(-1))).values[:, [-1]]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(scores, dim=-1), 1, generator=generator)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids
