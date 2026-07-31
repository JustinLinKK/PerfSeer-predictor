"""Reproducible production bundle containing only the 18K end-to-end manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .fingerprints import canonical_sha256, canonical_value
from .quota import load_quota_plan
from .sampler import TargetManifest, build_target_manifest


PLANNING_BUNDLE_VERSION = "perfseer_v3_a10g_dataset_planning_bundle_v2"


class PlanningBundleError(ValueError):
    """Raised when the target-only production bundle is inconsistent."""


@dataclass(frozen=True)
class DatasetPlanningBundle:
    version: str
    target_manifest: TargetManifest
    observation_protocol_sha256: str
    production_corpus: str
    local_qa_corpora_excluded: bool
    configuration_binding_state: str
    training_approved: bool

    def identity_payload(self) -> Mapping[str, Any]:
        return canonical_value(
            {
                "version": self.version,
                "target_manifest_sha256": canonical_sha256(asdict(self.target_manifest)),
                "observation_protocol_sha256": self.observation_protocol_sha256,
                "production_corpus": self.production_corpus,
                "local_qa_corpora_excluded": self.local_qa_corpora_excluded,
                "configuration_binding_state": self.configuration_binding_state,
                "training_approved": self.training_approved,
            }
        )

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(self.identity_payload())

    def validate(self) -> None:
        if self.version != PLANNING_BUNDLE_VERSION:
            raise PlanningBundleError("planning bundle version mismatch")
        self.target_manifest.validate()
        if self.observation_protocol_sha256 != load_quota_plan().protocol.sha256:
            raise PlanningBundleError("observation protocol binding drifted")
        if self.production_corpus != "end_to_end_18k":
            raise PlanningBundleError("production bundle may contain only end-to-end rows")
        if self.local_qa_corpora_excluded is not True:
            raise PlanningBundleError("operation/composite local QA must be excluded")
        if self.configuration_binding_state != "task_materialization_required":
            raise PlanningBundleError("local planning bundle must await task materialization")
        if self.training_approved is not False:
            raise PlanningBundleError("local planning bundle cannot approve training")

    def to_summary(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(
            {
                "version": self.version,
                "bundle_sha256": canonical_sha256(self.identity_payload()),
                "target": self.target_manifest._summary_unchecked(),
                "observation_protocol_sha256": self.observation_protocol_sha256,
                "production_corpus": self.production_corpus,
                "local_qa_corpora_excluded": self.local_qa_corpora_excluded,
                "configuration_binding_state": self.configuration_binding_state,
                "training_approved": self.training_approved,
            }
        )


def build_dataset_planning_bundle() -> DatasetPlanningBundle:
    target = build_target_manifest()
    result = DatasetPlanningBundle(
        version=PLANNING_BUNDLE_VERSION,
        target_manifest=target,
        observation_protocol_sha256=load_quota_plan().protocol.sha256,
        production_corpus="end_to_end_18k",
        local_qa_corpora_excluded=True,
        configuration_binding_state="task_materialization_required",
        training_approved=False,
    )
    result.validate()
    return result


__all__ = [
    "PLANNING_BUNDLE_VERSION",
    "DatasetPlanningBundle",
    "PlanningBundleError",
    "build_dataset_planning_bundle",
]
