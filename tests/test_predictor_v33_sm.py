"""Behavioral requirements for the CPU-sized V3.3 SM specialization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "predictor_v3.3" / "source"
V32_SOURCE = ROOT / "predictor_v3.2" / "source"
sys.path[:0] = [str(SOURCE), str(V32_SOURCE)]

from perfseer_v33 import model, training  # noqa: E402
from perfseer_v32.training import regression_loss as v32_regression_loss  # noqa: E402


def test_v33_sm_head_uses_three_finite_gated_experts() -> None:
    """V3.3 must give each SM prediction a lightweight mixture of experts."""
    head_type = getattr(model, "GatedSMExpertHead", None)
    assert head_type is not None, "V3.3 needs a dedicated GatedSMExpertHead"
    head = head_type(hidden=8, dropout=0.0, experts=3)
    prediction = head(torch.randn(5, 8))
    assert prediction.shape == (5, 1)
    assert len(head.experts) == 3
    assert torch.isfinite(prediction).all()


def test_v33_loss_gives_sm_error_more_weight_than_v32() -> None:
    """An isolated SM miss must be penalized more than under V3.2's six-way average."""
    loss_function = getattr(training, "sm_emphasized_regression_loss", None)
    assert loss_function is not None, "V3.3 needs an SM-emphasized regression loss"
    target = torch.full((2, 12), 10.0)
    target[:, 3] = 50.0
    prediction = target.clone()
    prediction[:, 3] = 40.0
    assert loss_function(prediction, target) > v32_regression_loss(prediction, target)


def test_v33_source_includes_the_operator_registry() -> None:
    """The V3.3 source package must contain the registry required by its encoder."""
    assert (SOURCE / "perfseer_v3" / "op_registry_v3.yaml").is_file()
