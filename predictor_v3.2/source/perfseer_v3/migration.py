"""Shadow comparison, canary policy, and explicit rollback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .runtime import SchedulerPredictionV3


@dataclass(frozen=True)
class ShadowComparison:
    v2_prediction: tuple[float, ...] | None
    v3_result: SchedulerPredictionV3
    fallback_prediction: tuple[float, ...] | None
    v2_v3_absolute_difference: tuple[float, ...] | None
    v3_fallback_absolute_difference: tuple[float, ...] | None


def compare_shadow(
    *,
    v2_prediction: Sequence[float] | None,
    v3_result: SchedulerPredictionV3,
    fallback_prediction: Sequence[float] | None,
) -> ShadowComparison:
    def difference(
        left: Sequence[float] | None,
        right: Sequence[float] | None,
    ) -> tuple[float, ...] | None:
        if left is None or right is None:
            return None
        if len(left) != len(right):
            raise ValueError("shadow predictions must use the same target order")
        return tuple(float(value) for value in np.abs(np.asarray(left) - np.asarray(right)))

    return ShadowComparison(
        v2_prediction=None if v2_prediction is None else tuple(v2_prediction),
        v3_result=v3_result,
        fallback_prediction=(
            None if fallback_prediction is None else tuple(fallback_prediction)
        ),
        v2_v3_absolute_difference=difference(v2_prediction, v3_result.prediction),
        v3_fallback_absolute_difference=difference(v3_result.prediction, fallback_prediction),
    )


@dataclass(frozen=True)
class CanaryPolicy:
    minimum_confidence: float = 0.8
    maximum_unknown_gpu_cost_proxy_fraction: float = 0.0
    accept_ok_with_unknowns: bool = False


@dataclass(frozen=True)
class MigrationDecision:
    route: str
    reason: str
    rollback_target: str


def canary_decision(
    result: SchedulerPredictionV3,
    policy: CanaryPolicy | None = None,
) -> MigrationDecision:
    policy = policy or CanaryPolicy()
    accepted_status = result.status == "ok" or (
        result.status == "ok_with_unknowns" and policy.accept_ok_with_unknowns
    )
    if not accepted_status:
        return MigrationDecision("fallback", f"status={result.status}", "perfseer_v2")
    if result.confidence is None or result.confidence < policy.minimum_confidence:
        return MigrationDecision("fallback", "confidence below canary policy", "perfseer_v2")
    if (
        result.unknown_gpu_cost_proxy_fraction
        > policy.maximum_unknown_gpu_cost_proxy_fraction
    ):
        return MigrationDecision("fallback", "unknown-cost fraction above canary policy", "perfseer_v2")
    return MigrationDecision("perfseer_v3", "high-confidence supported canary", "perfseer_v2")


__all__ = [
    "CanaryPolicy",
    "MigrationDecision",
    "ShadowComparison",
    "canary_decision",
    "compare_shadow",
]

