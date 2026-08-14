"""Process-wide immutable identity profile for the 18K labeler.

The V100 implementation remains the default.  The A10 entry point sets the
profile before importing dataset-pack modules, which keeps identities stable
inside fresh worker processes and prevents mixed-corpus objects in one process.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


PROFILE_ENVIRONMENT_VARIABLE = "PERFSEER_LABELER_PROFILE"


@dataclass(frozen=True)
class LabelerProfile:
    name: str
    target_hardware_id: str
    hardware_family_id: str
    identity_token: str
    display_name: str
    quota_version: str
    task_registry_version: str
    model_registry_version: str
    workload_config_version: str
    label_run_record_version: str
    epoch_measurement_version: str
    target_manifest_version: str
    candidate_version: str
    planner_version: str
    batch_plan_version: str
    compatibility_version: str
    generated_lineage_version: str
    generated_lineage_registry_version: str
    precision_pattern: tuple[str, ...]

    @property
    def is_a10(self) -> bool:
        return self.name in {
            "legacy_a10g",
            "native_a10",
            "native_a10_speech_v2",
            "native_a10_nonvision_4gpu_v1",
        }

    @property
    def is_native_a10(self) -> bool:
        return self.name in {
            "native_a10",
            "native_a10_speech_v2",
            "native_a10_nonvision_4gpu_v1",
        }

    @property
    def uses_speech_v2(self) -> bool:
        return self.name in {
            "native_a10_speech_v2",
            "native_a10_nonvision_4gpu_v1",
        }

    @property
    def is_nonvision_4gpu(self) -> bool:
        return self.name == "native_a10_nonvision_4gpu_v1"


_A10_PRECISION_PATTERN = (
    *("fp32_tf32",) * 5,
    *("bf16",) * 6,
    *("fp16_grad_scaler",) * 5,
    *("mixed_structured",) * 4,
)

PROFILES = {
    "v100": LabelerProfile(
        name="v100",
        target_hardware_id="nvidia_tesla_v100_sxm2_32gb_nrp",
        hardware_family_id="nvidia_volta_v100_32gb_family_v1",
        identity_token="v100",
        display_name="NRP Tesla V100 SXM2 32GB",
        quota_version="perfseer_v3_v100_18k_quota_v2",
        task_registry_version="perfseer_v3_v100_task_registry_v1",
        model_registry_version="perfseer_v3_v100_model_registry_v1",
        workload_config_version="perfseer_v3_v100_workload_config_v2",
        label_run_record_version="perfseer_v3_v100_label_run_v2",
        epoch_measurement_version="perfseer_v3_v100_epoch_measurement_v1",
        target_manifest_version="perfseer_v3_v100_18k_target_manifest_v8",
        candidate_version="perfseer_v3_v100_candidate_v8",
        planner_version="perfseer_v3_v100_constrained_sampler_v8",
        batch_plan_version="perfseer_v3_v100_batch_plan_v2",
        compatibility_version="perfseer_v3_v100_compatibility_v3",
        generated_lineage_version="perfseer_v3_v100_generated_lineage_v2",
        generated_lineage_registry_version="perfseer_v3_v100_generated_lineage_registry_v2",
        precision_pattern=(
            *("fp32_ieee",) * 5,
            *("fp16_grad_scaler",) * 11,
            *("mixed_structured",) * 4,
        ),
    ),
    "legacy_a10g": LabelerProfile(
        name="legacy_a10g",
        target_hardware_id="nvidia_a10g_24gb_aws_g5",
        hardware_family_id="nvidia_ampere_a10g_24gb_aws_g5_v1",
        identity_token="a10g",
        display_name="AWS NVIDIA A10G 24GB reference",
        quota_version="perfseer_v3_a10g_18k_quota_v2",
        task_registry_version="perfseer_v3_a10g_task_registry_v1",
        model_registry_version="perfseer_v3_a10g_model_registry_v1",
        workload_config_version="perfseer_v3_a10g_workload_config_v2",
        label_run_record_version="perfseer_v3_a10g_label_run_v2",
        epoch_measurement_version="perfseer_v3_a10g_epoch_measurement_v1",
        target_manifest_version="perfseer_v3_a10g_18k_target_manifest_v8",
        candidate_version="perfseer_v3_a10g_candidate_v8",
        planner_version="perfseer_v3_a10g_constrained_sampler_v8",
        batch_plan_version="perfseer_v3_a10g_batch_plan_v2",
        compatibility_version="perfseer_v3_a10g_compatibility_v3",
        generated_lineage_version="perfseer_v3_a10g_generated_lineage_v2",
        generated_lineage_registry_version="perfseer_v3_a10g_generated_lineage_registry_v2",
        precision_pattern=_A10_PRECISION_PATTERN,
    ),
    "native_a10": LabelerProfile(
        name="native_a10",
        target_hardware_id="nvidia_a10_24gb_nrp",
        hardware_family_id="nvidia_ampere_a10_24gb_nrp_v1",
        identity_token="nrp_a10",
        display_name="NRP NVIDIA A10 24GB",
        quota_version="perfseer_v3_nrp_a10_18k_quota_v1",
        task_registry_version="perfseer_v3_nrp_a10_task_registry_v1",
        model_registry_version="perfseer_v3_nrp_a10_model_registry_v1",
        workload_config_version="perfseer_v3_nrp_a10_workload_config_v1",
        label_run_record_version="perfseer_v3_nrp_a10_label_run_v1",
        epoch_measurement_version="perfseer_v3_nrp_a10_epoch_measurement_v1",
        target_manifest_version="perfseer_v3_nrp_a10_18k_target_manifest_v1",
        candidate_version="perfseer_v3_nrp_a10_candidate_v1",
        planner_version="perfseer_v3_nrp_a10_constrained_sampler_v1",
        batch_plan_version="perfseer_v3_nrp_a10_batch_plan_v1",
        compatibility_version="perfseer_v3_nrp_a10_compatibility_v1",
        generated_lineage_version="perfseer_v3_nrp_a10_generated_lineage_v1",
        generated_lineage_registry_version="perfseer_v3_nrp_a10_generated_lineage_registry_v1",
        precision_pattern=_A10_PRECISION_PATTERN,
    ),
    "native_a10_speech_v2": LabelerProfile(
        name="native_a10_speech_v2",
        target_hardware_id="nvidia_a10_24gb_nrp",
        hardware_family_id="nvidia_ampere_a10_24gb_nrp_v1",
        identity_token="nrp_a10_speech_v2",
        display_name="NRP NVIDIA A10 24GB (Speech substitution V2)",
        # Candidate-independent planning versions intentionally remain V1.  A
        # registry rebound changes only the 1,300 audio rows; changing these
        # versions would needlessly re-identify all 18,000 candidates.
        quota_version="perfseer_v3_nrp_a10_18k_quota_v1",
        task_registry_version="perfseer_v3_nrp_a10_speech_task_registry_v2",
        model_registry_version="perfseer_v3_nrp_a10_speech_model_registry_v2",
        workload_config_version="perfseer_v3_nrp_a10_workload_config_v1",
        label_run_record_version="perfseer_v3_nrp_a10_speech_label_run_v2",
        epoch_measurement_version="perfseer_v3_nrp_a10_epoch_measurement_v1",
        target_manifest_version="perfseer_v3_nrp_a10_speech_18k_target_manifest_v2",
        candidate_version="perfseer_v3_nrp_a10_candidate_v1",
        planner_version="perfseer_v3_nrp_a10_constrained_sampler_v1",
        batch_plan_version="perfseer_v3_nrp_a10_batch_plan_v1",
        compatibility_version="perfseer_v3_nrp_a10_compatibility_v1",
        generated_lineage_version="perfseer_v3_nrp_a10_generated_lineage_v1",
        generated_lineage_registry_version="perfseer_v3_nrp_a10_generated_lineage_registry_v1",
        precision_pattern=_A10_PRECISION_PATTERN,
    ),
    "native_a10_nonvision_4gpu_v1": LabelerProfile(
        name="native_a10_nonvision_4gpu_v1",
        target_hardware_id="nvidia_a10_24gb_nrp",
        hardware_family_id="nvidia_ampere_a10_24gb_nrp_v1",
        identity_token="nrp_a10_nonvision_4gpu_v1",
        display_name="NRP NVIDIA A10 24GB non-vision four-worker campaign",
        quota_version="perfseer_v3_nrp_a10_nonvision_11200_quota_v1",
        task_registry_version="perfseer_v3_nrp_a10_nonvision_task_registry_v1",
        model_registry_version="perfseer_v3_nrp_a10_nonvision_model_registry_v1",
        workload_config_version="perfseer_v3_nrp_a10_nonvision_workload_config_v1",
        label_run_record_version="perfseer_v3_nrp_a10_nonvision_label_run_v1",
        epoch_measurement_version="perfseer_v3_nrp_a10_nonvision_epoch_measurement_v1",
        target_manifest_version="perfseer_v3_nrp_a10_nonvision_11200_target_manifest_v1",
        candidate_version="perfseer_v3_nrp_a10_nonvision_candidate_v1",
        planner_version="perfseer_v3_nrp_a10_nonvision_projected_sampler_v1",
        batch_plan_version="perfseer_v3_nrp_a10_effective_batch_plan_v1",
        compatibility_version="perfseer_v3_nrp_a10_nonvision_compatibility_v1",
        generated_lineage_version="perfseer_v3_nrp_a10_generated_lineage_v1",
        generated_lineage_registry_version="perfseer_v3_nrp_a10_generated_lineage_registry_v1",
        precision_pattern=_A10_PRECISION_PATTERN,
    ),
}


def active_profile() -> LabelerProfile:
    name = os.environ.get(PROFILE_ENVIRONMENT_VARIABLE, "v100")
    try:
        return PROFILES[name]
    except KeyError as error:
        raise RuntimeError(
            f"unsupported {PROFILE_ENVIRONMENT_VARIABLE} value {name!r}"
        ) from error


PROFILE = active_profile()


__all__ = [
    "PROFILE",
    "PROFILE_ENVIRONMENT_VARIABLE",
    "PROFILES",
    "LabelerProfile",
    "active_profile",
]
