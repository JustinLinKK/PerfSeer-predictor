"""Legacy symbolic-FX comparison path; never used as production v3 capture."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


def capture_fx_diagnostic(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> GraphModule:
    if kwargs:
        raise ValueError("legacy symbolic FX diagnostic does not support keyword inputs")
    traced = symbolic_trace(model)
    with torch.no_grad():
        ShapeProp(traced).propagate(*args)
    return traced


__all__ = ["capture_fx_diagnostic"]

