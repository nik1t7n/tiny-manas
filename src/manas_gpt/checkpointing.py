"""Non-reentrant checkpointing with explicit MPS dropout-RNG preservation."""
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _MPSRNGContext:
    def __init__(self, state: dict[str, torch.Tensor], replay: bool) -> None:
        self.state = state
        self.replay = replay

    def __enter__(self) -> None:
        if self.replay:
            self.outer = torch.mps.get_rng_state().clone()
            torch.mps.set_rng_state(self.state["forward"])
        else:
            self.state["forward"] = torch.mps.get_rng_state().clone()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        # This also runs when early-stop recomputation exits through an exception.
        if self.replay:
            torch.mps.set_rng_state(self.outer)
        return False


def _mps_rng_contexts() -> tuple[_MPSRNGContext, _MPSRNGContext]:
    state: dict[str, torch.Tensor] = {}
    return _MPSRNGContext(state, False), _MPSRNGContext(state, True)


def checkpoint_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "mps":
        raise RuntimeError("This checkpointing path is validated for MPS training only")
    return checkpoint(
        block,
        x,
        use_reentrant=False,
        preserve_rng_state=True,
        context_fn=_mps_rng_contexts,
    )
