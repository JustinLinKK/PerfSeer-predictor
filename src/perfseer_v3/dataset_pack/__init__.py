"""Source-only AWS A10G dataset collection pack for PerfSeer V3.

Public objects are resolved lazily so the production entrypoint can verify its
locked environment before importing CUDA-heavy training modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


DATASET_PACK_VERSION = "perfseer_v3_a10g_dataset_pack_v2"
TARGET_HARDWARE_ID = "nvidia_a10g_24gb_aws_g5"

_LAZY_EXPORTS = {
    "canonical_sha256": (".fingerprints", "canonical_sha256"),
    "file_sha256": (".fingerprints", "file_sha256"),
    "DEFAULT_QUOTA_CONFIG_PATH": (".quota", "DEFAULT_QUOTA_CONFIG_PATH"),
    "QuotaPlan": (".quota", "QuotaPlan"),
    "load_quota_plan": (".quota", "load_quota_plan"),
    "CorpusLayer": (".contracts", "CorpusLayer"),
    "EpochMeasurement": (".contracts", "EpochMeasurement"),
    "LabelRunRecord": (".contracts", "LabelRunRecord"),
    "TelemetrySample": (".contracts", "TelemetrySample"),
    "WorkloadConfiguration": (".contracts", "WorkloadConfiguration"),
    "ModelRegistry": (".model_registry", "ModelRegistry"),
    "load_model_registry": (".model_registry", "load_model_registry"),
    "ModelFactoryAudit": (".model_golden", "ModelFactoryAudit"),
    "run_model_factory_audit": (".model_golden", "run_model_factory_audit"),
    "OperationSupportContract": (".operation_support", "OperationSupportContract"),
    "build_operation_support_contract": (
        ".operation_support",
        "build_operation_support_contract",
    ),
    "TaskRegistry": (".task_registry", "TaskRegistry"),
    "load_task_registry": (".task_registry", "load_task_registry"),
    "CampaignFinalization": (".finalization", "CampaignFinalization"),
    "FinalizationError": (".finalization", "FinalizationError"),
    "build_campaign_finalization": (
        ".finalization",
        "build_campaign_finalization",
    ),
    "finalize_workspace": (".finalization", "finalize_workspace"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))


__all__ = [
    "DATASET_PACK_VERSION",
    "DEFAULT_QUOTA_CONFIG_PATH",
    "CorpusLayer",
    "CampaignFinalization",
    "ModelRegistry",
    "ModelFactoryAudit",
    "QuotaPlan",
    "OperationSupportContract",
    "EpochMeasurement",
    "FinalizationError",
    "LabelRunRecord",
    "TaskRegistry",
    "TelemetrySample",
    "TARGET_HARDWARE_ID",
    "canonical_sha256",
    "build_operation_support_contract",
    "build_campaign_finalization",
    "file_sha256",
    "finalize_workspace",
    "load_quota_plan",
    "load_model_registry",
    "load_task_registry",
    "run_model_factory_audit",
    "WorkloadConfiguration",
]
