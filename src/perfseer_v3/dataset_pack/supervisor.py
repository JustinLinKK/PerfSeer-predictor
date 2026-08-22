"""Fresh-process dispatch, cleanup proof, and label-record publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence

import torch

from .v100_runner import FiveEpochRunResult, TelemetryReading
from .contracts import (
    AttemptStatus,
    FailureStage,
    FingerprintBundle,
    GpuCleanupEvidence,
    LABEL_RUN_RECORD_VERSION,
    LOCAL_VALIDATION_RETAINED_EPOCH_SPREAD_LIMIT,
    LabelRunRecord,
    PRODUCTION_RETAINED_EPOCH_SPREAD_LIMIT,
)
from .fingerprints import canonical_sha256
from .label_worker import (
    ASSIGNED_GPU_UUID_ENV,
    WORKER_ENVELOPE_VERSION,
    _diagnostic_message,
)
from .labeler_profile import PROFILE
from .a10_crosswalk import (
    REFERENCE_MANIFEST_SHA256,
    semantic_distribution_signature,
)
from .a10_speech_crosswalk import (
    NATIVE_V1_CROSSWALK_SHA256,
    NATIVE_V1_MANIFEST_SHA256,
)
from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract
from .disaster_substitution import (
    DISASTER_TASK_ID,
    load_disaster_substitution_contract,
)
from .operation_support import build_operation_support_contract
from .prepared_view import PreparedViewManifest
from .sampler import TargetCandidate
from .storage import atomic_write_json
from .task_registry import TaskRegistryEntry


SUPERVISOR_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_attempt_supervisor_v2"
    if PROFILE.uses_disaster_v2
    else "perfseer_v3_v100_attempt_supervisor_v2"
)
CAMPAIGN_ENVIRONMENT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_campaign_environment_v2"
    if PROFILE.uses_disaster_v2
    else "perfseer_v3_v100_campaign_environment_v1"
)
CAMPAIGN_ENVIRONMENT_RECOVERY_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_disaster_campaign_environment_recovery_v1"
)
PRODUCTION_CLEANUP_RELEASE_TOLERANCE_MIB = 64.0
LOCAL_VALIDATION_CLEANUP_RELEASE_TOLERANCE_MIB = 4_096.0
_CAMPAIGN_PACKAGES = (
    "appdirs",
    "kaggle",
    "networkx",
    "numpy",
    "nvidia-ml-py",
    "ogb",
    "pandas",
    "pillow",
    "py7zr",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "soundfile",
    "tensorflow-cpu",
    "torch-geometric",
    "torchaudio",
    "torchvision",
    "transformers",
    "triton",
)


class SupervisorError(RuntimeError):
    pass


class TransientAttemptError(SupervisorError):
    """A candidate-local timeout/crash that may be retried after cleanup."""


def await_worker_futures(futures: Sequence[Any]) -> tuple[Any, ...]:
    """Observe every worker result before propagating the first batch failure."""

    results: list[Any] = []
    failures: list[Exception] = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception as error:
            failures.append(error)
    if failures:
        raise failures[0]
    return tuple(results)


def validate_v100_gpu_identity(
    name: str,
    compute_capability: Sequence[int],
    total_memory_bytes: int,
) -> None:
    normalized_name = "".join(
        character for character in str(name).upper() if character.isalnum()
    )
    if (
        "TESLAV100SXM2" not in normalized_name
        or tuple(compute_capability) != (7, 0)
        or type(total_memory_bytes) is not int
        or not 30 * 1024**3 <= total_memory_bytes <= 34 * 1024**3
    ):
        raise SupervisorError(
            "GPU must be a Tesla V100 SXM2 32GB with compute capability 7.0"
        )


def validate_a10_gpu_identity(
    name: str,
    compute_capability: Sequence[int],
    total_memory_bytes: int,
) -> None:
    normalized_name = "".join(
        character for character in str(name).upper() if character.isalnum()
    )
    if (
        normalized_name != "NVIDIAA10"
        or tuple(compute_capability) != (8, 6)
        or type(total_memory_bytes) is not int
        or not 22 * 1024**3 <= total_memory_bytes <= 26 * 1024**3
    ):
        raise SupervisorError(
            "GPU must be exactly an NVIDIA A10 with compute capability 8.6 and 22--26 GiB"
        )


def validate_rtx5090_gpu_identity(
    name: str,
    compute_capability: Sequence[int],
    total_memory_bytes: int,
) -> None:
    normalized_name = "".join(
        character for character in str(name).upper() if character.isalnum()
    )
    if (
        normalized_name != "NVIDIAGEFORCERTX5090"
        or tuple(compute_capability) != (12, 0)
        or type(total_memory_bytes) is not int
        or not 30 * 1024**3 <= total_memory_bytes <= 34 * 1024**3
    ):
        raise SupervisorError(
            "local validation requires exactly one GeForce RTX 5090 with "
            "compute capability 12.0 and 30--34 GiB"
        )


class GpuProbe(Protocol):
    physical_index: int
    gpu_uuid: str
    hardware_fingerprint: str
    hardware_provenance: Mapping[str, Any]

    def read(self) -> TelemetryReading: ...


@dataclass(frozen=True)
class PhysicalGpuSlot:
    physical_index: int
    gpu_uuid: str
    hardware_fingerprint: str
    hardware_provenance: Mapping[str, Any]


class ParentNvmlProbe:
    def __init__(self, physical_index: int, *, hardware_mode: str = "production-a10") -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(name, bytes):
                name = name.decode()
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if hardware_mode == "local-rtx5090":
                validate_rtx5090_gpu_identity(str(name), capability, int(memory.total))
            elif PROFILE.is_native_a10:
                validate_a10_gpu_identity(str(name), capability, int(memory.total))
            else:
                validate_v100_gpu_identity(str(name), capability, int(memory.total))
            self._pynvml = pynvml
            self._handle = handle
            self.physical_index = physical_index
            self.gpu_uuid = str(uuid)
            self.hardware_provenance = {
                "target_hardware_id": PROFILE.target_hardware_id,
                "name": str(name),
                "uuid": self.gpu_uuid,
                "total_memory_bytes": int(memory.total),
                "compute_capability": list(capability),
                "hardware_mode": hardware_mode,
            }
            self.hardware_fingerprint = canonical_sha256(self.hardware_provenance)
        except SupervisorError:
            raise
        except Exception as error:
            raise SupervisorError("cannot qualify physical GPU with NVML") from error

    def read(self) -> TelemetryReading:
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        return TelemetryReading(
            float(utilization.gpu),
            float(utilization.memory),
            float(memory.used / 1024**2),
            0,
            tuple(sorted(int(row.pid) for row in processes)),
        )


def discover_v100_probes() -> tuple[ParentNvmlProbe, ...]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception as error:
        raise SupervisorError("NVML cannot enumerate V100 workers") from error
    probes = tuple(ParentNvmlProbe(index) for index in range(count))
    if len(probes) != 4:
        raise SupervisorError(
            f"production labeling requires exactly four V100 workers; discovered {len(probes)}"
        )
    if len({row.gpu_uuid for row in probes}) != 4:
        raise SupervisorError("V100 worker UUIDs are duplicated")
    hardware = {
        (
            "".join(
                character
                for character in str(row.hardware_provenance["name"]).upper()
                if character.isalnum()
            ),
            tuple(row.hardware_provenance["compute_capability"]),
            row.hardware_provenance["total_memory_bytes"],
        )
        for row in probes
    }
    if len(hardware) != 1:
        raise SupervisorError("production labeling refuses mixed GPU hardware")
    return probes


def discover_a10_probe() -> ParentNvmlProbe:
    if not PROFILE.is_native_a10:
        raise SupervisorError("A10 discovery requires a native A10 process profile")
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception as error:
        raise SupervisorError("NVML cannot enumerate the A10 worker") from error
    if count != 1:
        raise SupervisorError(
            f"production labeling requires exactly one visible A10; discovered {count}"
        )
    return ParentNvmlProbe(0)


def discover_a10_probes() -> tuple[ParentNvmlProbe, ...]:
    """Qualify exactly four homogeneous A10s for the non-vision controller."""

    if not PROFILE.is_nonvision_4gpu:
        raise SupervisorError("four-A10 discovery requires the non-vision profile")
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception as error:
        raise SupervisorError("NVML cannot enumerate four A10 workers") from error
    probes = tuple(
        ParentNvmlProbe(index, hardware_mode="production-a10")
        for index in range(count)
    )
    if len(probes) != 4:
        raise SupervisorError(
            f"production labeling requires exactly four visible A10s; discovered {len(probes)}"
        )
    if len({row.gpu_uuid for row in probes}) != 4:
        raise SupervisorError("A10 worker UUIDs are duplicated")
    identities = {
        (
            row.hardware_provenance["name"],
            tuple(row.hardware_provenance["compute_capability"]),
            row.hardware_provenance["total_memory_bytes"],
        )
        for row in probes
    }
    if len(identities) != 1:
        raise SupervisorError("production labeling refuses mixed A10 hardware")
    return probes


def discover_local_rtx5090_probe() -> ParentNvmlProbe:
    """Qualify the one-GPU, non-production validation environment."""

    if not PROFILE.is_nonvision_4gpu:
        raise SupervisorError("RTX 5090 validation requires the non-vision profile")
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception as error:
        raise SupervisorError("NVML cannot enumerate the RTX 5090 worker") from error
    if count != 1:
        raise SupervisorError(
            f"local validation requires exactly one visible RTX 5090; discovered {count}"
        )
    return ParentNvmlProbe(0, hardware_mode="local-rtx5090")


def environment_provenance() -> Mapping[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
    except Exception:
        driver = "unavailable"
    packages = {}
    for name in _CAMPAIGN_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "container_digest": os.environ.get("PERFSEER_CONTAINER_DIGEST", "not_supplied"),
        "source_revision": os.environ.get(
            "PERFSEER_BUILD_SOURCE_REVISION", "not_supplied"
        ),
        "source_tree_sha256": os.environ.get(
            "PERFSEER_BUILD_SOURCE_TREE_SHA256", "not_supplied"
        ),
        "dependency_lock_sha256": os.environ.get(
            "PERFSEER_BUILD_DEPENDENCY_LOCK_SHA256", "not_supplied"
        ),
        "image_identity": os.environ.get(
            "PERFSEER_BUILD_IMAGE_IDENTITY", "not_supplied"
        ),
        "packages": packages,
    }


def lock_campaign_environment(
    workspace: str | Path,
    *,
    recovery_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Freeze the origin environment or an authorized recovery environment."""

    environment = environment_provenance()
    value = {
        "version": CAMPAIGN_ENVIRONMENT_VERSION,
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
    }
    path = Path(workspace).resolve() / "state" / "campaign_environment.json"
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SupervisorError("campaign environment lock is unreadable") from error
        if stored != value:
            if (
                not PROFILE.uses_disaster_v2
                or recovery_identity is None
                or not isinstance(stored, Mapping)
                or set(stored) != {"version", "environment", "environment_sha256"}
                or stored.get("version") != CAMPAIGN_ENVIRONMENT_VERSION
                or not isinstance(stored.get("environment"), Mapping)
                or stored.get("environment_sha256")
                != canonical_sha256(stored.get("environment"))
            ):
                raise SupervisorError(
                    "campaign environment changed since collection started"
                )
            identity_sha256 = recovery_identity.get("identity_sha256")
            if (
                not isinstance(identity_sha256, str)
                or len(identity_sha256) != 64
                or environment.get("source_revision")
                != recovery_identity.get("repository_revision")
                or environment.get("container_digest")
                != recovery_identity.get("image_digest")
            ):
                raise SupervisorError(
                    "recovery environment differs from its active run identity"
                )
            recovery: dict[str, Any] = {
                "version": CAMPAIGN_ENVIRONMENT_RECOVERY_VERSION,
                "run_identity_sha256": identity_sha256,
                "origin_environment_sha256": stored["environment_sha256"],
                "environment": environment,
                "environment_sha256": value["environment_sha256"],
            }
            recovery["recovery_sha256"] = canonical_sha256(recovery)
            directory = (
                path.parent
                / "campaign_environment_recoveries"
                / identity_sha256
            )
            if directory.exists() and (
                directory.is_symlink() or not directory.is_dir()
            ):
                raise SupervisorError("campaign environment recovery path is unsafe")
            for existing_path in directory.iterdir() if directory.is_dir() else ():
                if (
                    existing_path.is_symlink()
                    or not existing_path.is_file()
                    or existing_path.suffix != ".json"
                ):
                    raise SupervisorError(
                        "campaign environment recovery history is unsafe"
                    )
                try:
                    existing = json.loads(existing_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise SupervisorError(
                        "campaign environment recovery history is unreadable"
                    ) from error
                if not isinstance(existing, Mapping):
                    raise SupervisorError(
                        "campaign environment recovery history differs"
                    )
                existing_payload = dict(existing)
                claimed = existing_payload.pop("recovery_sha256", None)
                if (
                    set(existing)
                    != {
                        "version",
                        "run_identity_sha256",
                        "origin_environment_sha256",
                        "environment",
                        "environment_sha256",
                        "recovery_sha256",
                    }
                    or not isinstance(existing.get("environment"), Mapping)
                    or existing_path.stem != existing.get("environment_sha256")
                    or existing.get("version")
                    != CAMPAIGN_ENVIRONMENT_RECOVERY_VERSION
                    or existing.get("run_identity_sha256") != identity_sha256
                    or existing.get("origin_environment_sha256")
                    != stored["environment_sha256"]
                    or existing.get("environment_sha256")
                    != canonical_sha256(existing.get("environment"))
                    or claimed != canonical_sha256(existing_payload)
                ):
                    raise SupervisorError(
                        "campaign environment recovery history differs"
                    )
            recovery_path = directory / f"{value['environment_sha256']}.json"
            if recovery_path.is_file():
                try:
                    existing_recovery = json.loads(
                        recovery_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise SupervisorError(
                        "campaign environment recovery is unreadable"
                    ) from error
                if existing_recovery != recovery:
                    raise SupervisorError("campaign environment recovery differs")
            elif recovery_path.exists() or recovery_path.is_symlink():
                raise SupervisorError(
                    "campaign environment recovery path is not a regular file"
                )
            else:
                atomic_write_json(recovery_path, recovery)
    elif path.exists() or path.is_symlink():
        raise SupervisorError("campaign environment lock path is not a regular file")
    else:
        atomic_write_json(path, value)
    return value


def _environment_fingerprint() -> str:
    return canonical_sha256(environment_provenance())


def _fingerprints(
    candidate: TargetCandidate,
    view: PreparedViewManifest,
    hardware_sha256: str,
) -> FingerprintBundle:
    support = build_operation_support_contract()
    graph_sha = canonical_sha256(
        {
            "source_sha256": candidate.source_sha256,
            "factory_id": candidate.factory_id,
            "architecture_parameters": candidate.architecture_parameters,
            "input_signature": candidate.input_signature,
            "training_step_id": candidate.training_step_id,
        }
    )
    return FingerprintBundle(
        source_sha256=candidate.source_sha256,
        graph_sha256=graph_sha,
        environment_sha256=_environment_fingerprint(),
        hardware_sha256=hardware_sha256,
        support_contract_sha256=support.sha256,
        dataset_sha256=view.dataset_fingerprint,
    )


def build_accepted_label_record(
    candidate: TargetCandidate,
    task_entry: TaskRegistryEntry,
    view: PreparedViewManifest,
    run: FiveEpochRunResult,
    cleanup: GpuCleanupEvidence,
    *,
    attempt_index: int,
    production_eligible: bool = True,
) -> LabelRunRecord:
    run.validate()
    cleanup.validate()
    if not cleanup.passed:
        raise SupervisorError("successful child cannot be accepted without cleanup proof")
    fingerprints = _fingerprints(candidate, view, run.hardware_sha256)
    draft = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id=canonical_sha256(
            {"candidate_id": candidate.candidate_id, "attempt_index": attempt_index}
        ),
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.ACCEPTED,
        failure_stage=FailureStage.NONE,
        fingerprints=fingerprints,
        gpu_uuid=run.gpu_uuid,
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=str(candidate.execution["mode"]),
        compile_completed_before_epoch_1=run.compile_completed_before_epoch_1,
        expected_train_examples=task_entry.expected_train_examples,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=run.completed_epochs,
        finite_loss_epochs=run.finite_loss_epochs,
        finite_gradient_epochs=run.finite_gradient_epochs,
        epoch_measurements=run.epoch_measurements,
        cleanup=cleanup,
        targets=None,
        production_eligible=production_eligible,
    )
    record = replace(draft, targets=draft.aggregate_targets())
    record.validate_against_configuration(candidate, task_entry)
    return record


def build_failed_label_record(
    candidate: TargetCandidate,
    task_entry: TaskRegistryEntry,
    view: PreparedViewManifest,
    cleanup: GpuCleanupEvidence,
    *,
    status: AttemptStatus,
    failure_stage: FailureStage,
    gpu_uuid: str,
    hardware_sha256: str,
    attempt_index: int,
    run: FiveEpochRunResult | None = None,
) -> LabelRunRecord:
    if status == AttemptStatus.ACCEPTED or failure_stage == FailureStage.NONE:
        raise SupervisorError("failed label record requires rejected status/stage")
    record = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id=canonical_sha256(
            {
                "candidate_id": candidate.candidate_id,
                "attempt_index": attempt_index,
                "status": status.value,
                "failure_stage": failure_stage.value,
            }
        ),
        configuration_id=candidate.candidate_id,
        status=status,
        failure_stage=failure_stage,
        fingerprints=_fingerprints(candidate, view, hardware_sha256),
        gpu_uuid=gpu_uuid,
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=str(candidate.execution["mode"]),
        compile_completed_before_epoch_1=(
            False if run is None else run.compile_completed_before_epoch_1
        ),
        expected_train_examples=task_entry.expected_train_examples,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=() if run is None else run.completed_epochs,
        finite_loss_epochs=() if run is None else run.finite_loss_epochs,
        finite_gradient_epochs=() if run is None else run.finite_gradient_epochs,
        epoch_measurements=() if run is None else run.epoch_measurements,
        cleanup=cleanup,
        targets=None,
    )
    record.validate_against_configuration(candidate, task_entry)
    return record


def _bind_native_a10_provenance(
    record: LabelRunRecord,
    candidate: TargetCandidate,
    task_entry: TaskRegistryEntry,
    workspace: Path,
    *,
    production_eligible: bool = True,
) -> LabelRunRecord:
    if not PROFILE.is_native_a10:
        return record
    nonvision = PROFILE.is_nonvision_4gpu
    speech_v2 = PROFILE.uses_speech_v2
    disaster_v2 = PROFILE.uses_disaster_v2
    mapping_path = workspace / "state" / (
        "a10_disaster_crosswalk.jsonl"
        if disaster_v2
        else "a10_nonvision_crosswalk.jsonl"
        if nonvision
        else "a10_speech_v2_crosswalk.jsonl"
        if speech_v2
        else "a10_crosswalk.jsonl"
    )
    original_id = None
    v1_id = None
    lineage_row: Mapping[str, Any] | None = None
    root_candidate_id = candidate.candidate_id
    repair_history = candidate.mutation_specification.get("oom_repair_history", ())
    replacement = candidate.mutation_specification.get("quota_replacement")
    if isinstance(replacement, Mapping):
        root_candidate_id = str(replacement["target_candidate_id"])
    elif repair_history:
        root_candidate_id = str(repair_history[-1]["root_candidate_id"])
    if mapping_path.is_file():
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            native_key = (
                "v2_candidate_id"
                if disaster_v2
                else "nonvision_candidate_id"
                if nonvision
                else "native_v2_candidate_id"
                if speech_v2
                else "native_nrp_a10_candidate_id"
            )
            if row.get(native_key) == root_candidate_id:
                original_id = (
                    row.get("v1_candidate_id")
                    if disaster_v2
                    else row.get("original_a10g_candidate_id")
                )
                v1_id = (
                    row.get("v1_candidate_id")
                    if disaster_v2
                    else row.get("native_v1_candidate_id")
                )
                lineage_row = row
                break
    if original_id is None:
        raise SupervisorError("native A10 record has no frozen reference crosswalk row")
    if disaster_v2:
        from .a10_disaster_crosswalk import (
            PREDECESSOR_CROSSWALK_SHA256,
            PREDECESSOR_MANIFEST_SHA256,
        )

        if not isinstance(lineage_row, Mapping) or not isinstance(v1_id, str):
            raise SupervisorError("Disaster V2 record has no V1-to-V2 lineage row")
        try:
            materialization_state = json.loads(
                (workspace / "task_cache" / "materialization_state.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SupervisorError(
                "Disaster V2 record cannot bind source archive provenance"
            ) from error
        if (
            not isinstance(materialization_state, Mapping)
            or materialization_state.get("task_id") != task_entry.task_id
            or materialization_state.get("archive_sha256") is None
            or materialization_state.get("stage") != "view_ready"
            or materialization_state.get("dataset_fingerprint")
            != record.fingerprints.dataset_sha256
        ):
            raise SupervisorError("Disaster V2 materialization provenance is incomplete")
        state_payload = dict(materialization_state)
        state_sha256 = state_payload.pop("state_sha256", None)
        if state_sha256 != canonical_sha256(state_payload):
            raise SupervisorError("Disaster V2 materialization state hash differs")
        archive_sha256 = materialization_state.get("archive_sha256")
        remote_inventory_sha256 = materialization_state.get(
            "remote_inventory_sha256"
        )
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (archive_sha256, remote_inventory_sha256)
        ):
            raise SupervisorError("Disaster V2 source provenance digests are invalid")
        speech_source_lock_sha256 = None
        disaster_source_lock_sha256 = None
        if task_entry.task_id in {SPEECH_TASK_ID, DISASTER_TASK_ID}:
            lock_path = (
                workspace
                / "state"
                / "source_locks"
                / f"{task_entry.task_id}.json"
            )
            try:
                source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SupervisorError(
                    f"{task_entry.task_id} has no immutable source lock"
                ) from error
            if not isinstance(source_lock, Mapping) or (
                source_lock.get("archive_sha256") != archive_sha256
                or source_lock.get("remote_inventory_sha256")
                != remote_inventory_sha256
            ):
                raise SupervisorError("source lock differs from materialization")
            lock_sha256 = source_lock.get("lock_sha256")
            if not isinstance(lock_sha256, str) or len(lock_sha256) != 64:
                raise SupervisorError("source-lock digest is invalid")
            if task_entry.task_id == SPEECH_TASK_ID:
                speech_source_lock_sha256 = lock_sha256
            else:
                disaster_source_lock_sha256 = lock_sha256
        disaster_contract = load_disaster_substitution_contract()
        speech_contract = load_speech_substitution_contract()
        reference_provenance = {
            "predecessor_candidate_id": v1_id,
            "v1_root_candidate_id": v1_id,
            "v2_root_candidate_id": root_candidate_id,
            "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
            "predecessor_crosswalk_sha256": PREDECESSOR_CROSSWALK_SHA256,
            "substitution_contract_sha256": disaster_contract.sha256,
            "disaster_substitution_contract_sha256": disaster_contract.sha256,
            "speech_substitution_contract_sha256": speech_contract.sha256,
            "row_classification": lineage_row["row_classification"],
            "old_task_id": lineage_row["old_task_id"],
            "new_task_id": lineage_row["new_task_id"],
            "old_semantic_signature": lineage_row["old_semantic_signature"],
            "new_semantic_signature": lineage_row["new_semantic_signature"],
            "task_independent_compute_signature": lineage_row[
                "new_compute_signature"
            ],
            "resolved_semantic_distribution_signature": (
                semantic_distribution_signature(candidate)
            ),
            "dataset_substitution": task_entry.task_id == DISASTER_TASK_ID,
            "source_archive_sha256": archive_sha256,
            "remote_inventory_sha256": remote_inventory_sha256,
            "speech_source_lock_sha256": speech_source_lock_sha256,
            "disaster_source_lock_sha256": disaster_source_lock_sha256,
        }
    elif speech_v2:
        if not isinstance(lineage_row, Mapping) or not isinstance(v1_id, str):
            raise SupervisorError("speech V2 record has no three-way lineage row")
        try:
            materialization_state = json.loads(
                (workspace / "task_cache" / "materialization_state.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            raise SupervisorError(
                "speech V2 record cannot bind source archive provenance"
            ) from error
        if (
            not isinstance(materialization_state, Mapping)
            or materialization_state.get("task_id") != task_entry.task_id
            or materialization_state.get("archive_sha256") is None
            or materialization_state.get("stage") != "view_ready"
            or materialization_state.get("dataset_fingerprint")
            != record.fingerprints.dataset_sha256
        ):
            raise SupervisorError("speech V2 materialization provenance is incomplete")
        state_payload = dict(materialization_state)
        state_sha256 = state_payload.pop("state_sha256", None)
        if state_sha256 != canonical_sha256(state_payload):
            raise SupervisorError("speech V2 materialization state hash differs")
        archive_sha256 = materialization_state.get("archive_sha256")
        remote_inventory_sha256 = materialization_state.get(
            "remote_inventory_sha256"
        )
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (archive_sha256, remote_inventory_sha256)
        ):
            raise SupervisorError("speech V2 source provenance digests are invalid")
        source_lock_sha256 = None
        if task_entry.task_id == SPEECH_TASK_ID:
            try:
                source_lock = json.loads(
                    (
                        workspace
                        / "state"
                        / "source_locks"
                        / f"{SPEECH_TASK_ID}.json"
                    ).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise SupervisorError("speech dataset has no immutable source lock") from error
            if not isinstance(source_lock, Mapping) or (
                source_lock.get("archive_sha256") != archive_sha256
                or source_lock.get("remote_inventory_sha256")
                != remote_inventory_sha256
            ):
                raise SupervisorError("speech source lock differs from materialization")
            source_lock_sha256 = source_lock.get("lock_sha256")
            if not isinstance(source_lock_sha256, str) or len(source_lock_sha256) != 64:
                raise SupervisorError("speech source-lock digest is invalid")
        substitution = load_speech_substitution_contract()
        reference_provenance: Mapping[str, Any] = {
            "original_a10g_candidate_id": original_id,
            "native_v1_candidate_id": v1_id,
            (
                "nonvision_root_candidate_id"
                if nonvision
                else "native_v2_root_candidate_id"
            ): root_candidate_id,
            "original_a10g_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "native_v1_manifest_sha256": NATIVE_V1_MANIFEST_SHA256,
            "native_v1_crosswalk_sha256": NATIVE_V1_CROSSWALK_SHA256,
            "substitution_contract_sha256": substitution.sha256,
            "row_classification": lineage_row.get(
                "row_classification", lineage_row.get("disposition")
            ),
            "old_semantic_signature": lineage_row["old_semantic_signature"],
            "new_semantic_signature": lineage_row["new_semantic_signature"],
            "task_independent_compute_signature": lineage_row.get(
                "task_independent_compute_signature",
                lineage_row.get("new_nonbatch_semantic_signature"),
            ),
            "resolved_semantic_distribution_signature": semantic_distribution_signature(
                candidate
            ),
            "dataset_substitution": task_entry.task_id == SPEECH_TASK_ID,
            "source_archive_sha256": archive_sha256,
            "remote_inventory_sha256": remote_inventory_sha256,
            "speech_source_lock_sha256": source_lock_sha256,
        }
        if nonvision:
            from .a10_nonvision_crosswalk import PREDECESSOR_MANIFEST_SHA256

            reference_provenance = {
                **reference_provenance,
                "predecessor_candidate_id": lineage_row[
                    "predecessor_candidate_id"
                ],
                "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
                "historical_ordinal": lineage_row["historical_ordinal"],
                "nonvision_ordinal": lineage_row["nonvision_ordinal"],
                "lineage_disposition": lineage_row["disposition"],
                "old_nonbatch_semantic_signature": lineage_row[
                    "old_nonbatch_semantic_signature"
                ],
                "new_nonbatch_semantic_signature": lineage_row[
                    "new_nonbatch_semantic_signature"
                ],
            }
    else:
        reference_provenance = {
            "original_a10g_candidate_id": original_id,
            "native_root_candidate_id": root_candidate_id,
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "semantic_distribution_signature": semantic_distribution_signature(candidate),
        }
    environment = environment_provenance()
    bound = replace(
        record,
        production_eligible=production_eligible,
        reference_provenance=reference_provenance,
        build_identity={
            key: str(environment[key])
            for key in (
                "source_revision",
                "source_tree_sha256",
                "dependency_lock_sha256",
                "image_identity",
            )
        },
    )
    bound.validate_against_configuration(candidate, task_entry)
    return bound


def wait_for_gpu_cleanup(
    probe: GpuProbe,
    baseline: TelemetryReading,
    *,
    release_tolerance_mib: float = 64.0,
    stable_dwell_seconds: float = 2.0,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 0.1,
    child_process_tree_exited: bool = True,
) -> GpuCleanupEvidence:
    started = time.monotonic()
    stable_since: float | None = None
    latest = probe.read()
    while time.monotonic() - started < timeout_seconds:
        latest = probe.read()
        clean = (
            latest.device_used_vram_mib
            <= baseline.device_used_vram_mib + release_tolerance_mib
            and not latest.compute_process_ids
        )
        if clean:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_dwell_seconds:
                break
        else:
            stable_since = None
        time.sleep(poll_seconds)
    dwell = 0.0 if stable_since is None else time.monotonic() - stable_since
    evidence = GpuCleanupEvidence(
        pre_sample_device_used_vram_mib=baseline.device_used_vram_mib,
        post_cleanup_device_used_vram_mib=latest.device_used_vram_mib,
        release_tolerance_mib=release_tolerance_mib,
        cleanup_seconds=time.monotonic() - started,
        stable_dwell_seconds=dwell,
        child_process_tree_exited=child_process_tree_exited,
        owned_gpu_processes_remaining=len(latest.compute_process_ids),
        passed=(
            not latest.compute_process_ids
            and child_process_tree_exited
            and latest.device_used_vram_mib
            <= baseline.device_used_vram_mib + release_tolerance_mib
            and dwell >= stable_dwell_seconds
        ),
    )
    evidence.validate()
    return evidence


def cleanup_release_tolerance_mib(*, production_eligible: bool) -> float:
    """Keep headless A10 cleanup strict while allowing local display-memory drift."""

    return (
        PRODUCTION_CLEANUP_RELEASE_TOLERANCE_MIB
        if production_eligible
        else LOCAL_VALIDATION_CLEANUP_RELEASE_TOLERANCE_MIB
    )


def retained_epoch_spread_limit(*, production_eligible: bool) -> float:
    """Keep the corpus gate strict; tolerate desktop jitter only in non-production."""

    return (
        PRODUCTION_RETAINED_EPOCH_SPREAD_LIMIT
        if production_eligible
        else LOCAL_VALIDATION_RETAINED_EPOCH_SPREAD_LIMIT
    )


def build_worker_failure_diagnostic(
    *,
    candidate_id: str,
    run_id: str,
    attempt_index: int,
    return_code: int | None,
    child_log: str,
    worker_diagnostic: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind a worker's bounded failure reason to its durable failed run."""

    diagnostic = {
        "version": f"{SUPERVISOR_VERSION}_worker_failure_diagnostic_v1",
        "candidate_id": candidate_id,
        "run_id": run_id,
        "attempt_index": attempt_index,
        "return_code": return_code,
        "child_log": child_log,
        "failure_stage": worker_diagnostic.get("failure_stage"),
        "reason_code": worker_diagnostic.get("reason_code"),
        "reason_message": worker_diagnostic.get("reason_message"),
    }
    return {
        **diagnostic,
        "diagnostic_sha256": canonical_sha256(diagnostic),
    }


def worker_failure_summary(worker_diagnostic: Mapping[str, Any]) -> tuple[str, str]:
    """Return a bounded, redacted controller-visible worker failure summary."""

    reason_code = str(worker_diagnostic.get("reason_code") or "unknown_error")
    raw_message = worker_diagnostic.get("reason_message")
    reason_message = _diagnostic_message(
        RuntimeError(
            raw_message
            if isinstance(raw_message, str) and raw_message
            else "worker supplied no diagnostic message"
        )
    )
    return reason_code, reason_message


def child_gpu_environment(probe: GpuProbe) -> Mapping[str, str]:
    """Bind a child to one parent-qualified physical GPU."""

    if type(probe.physical_index) is not int or probe.physical_index < 0:
        raise SupervisorError("assigned physical GPU index is invalid")
    if (
        not isinstance(probe.gpu_uuid, str)
        or not probe.gpu_uuid.startswith("GPU-")
        or any(character.isspace() for character in probe.gpu_uuid)
    ):
        raise SupervisorError("assigned physical GPU UUID is invalid")
    return {
        "CUDA_VISIBLE_DEVICES": str(probe.physical_index),
        ASSIGNED_GPU_UUID_ENV: probe.gpu_uuid,
    }


def terminate_lingering_process_group(
    process_group_id: int,
    *,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.1,
) -> bool:
    """Kill residual descendants and prove the whole process group disappeared."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return True
    os.killpg(process_group_id, signal.SIGKILL)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        time.sleep(poll_seconds)
    return False


def _load_worker_envelope(path: Path, candidate_id: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SupervisorError("label child produced no valid result envelope") from error
    if not isinstance(value, Mapping) or set(value) != {
        "version", "candidate_id", "status", "payload", "envelope_sha256"
    }:
        raise SupervisorError("label child result envelope schema differs")
    unhashed = {key: value[key] for key in value if key != "envelope_sha256"}
    if (
        value["version"] != WORKER_ENVELOPE_VERSION
        or value["candidate_id"] != candidate_id
        or value["envelope_sha256"] != canonical_sha256(unhashed)
    ):
        raise SupervisorError("label child result envelope identity/hash differs")
    return value


@dataclass
class AttemptSupervisor:
    workspace: Path
    timeout_seconds: float | None = None
    no_progress_seconds: float | None = None
    transient_retries: int | None = None
    production_eligible: bool = True

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.timeout_seconds is None:
            self.timeout_seconds = 2 * 60 * 60 if PROFILE.is_nonvision_4gpu else 6 * 60 * 60
        if self.no_progress_seconds is None:
            self.no_progress_seconds = 10 * 60 if PROFILE.is_nonvision_4gpu else self.timeout_seconds
        if self.transient_retries is None:
            self.transient_retries = 1 if PROFILE.is_nonvision_4gpu else 0
        if self.timeout_seconds < 0.05 or self.no_progress_seconds < 0.05:
            raise SupervisorError("attempt timeout must be positive")
        if type(self.transient_retries) is not int or self.transient_retries < 0:
            raise SupervisorError("transient retry count must be nonnegative")

    def run(
        self,
        candidate: TargetCandidate,
        task_entry: TaskRegistryEntry,
        view: PreparedViewManifest,
        *,
        public_directory: Path,
        prepared_directory: Path,
        archive_sha256: str,
        probe: GpuProbe,
        attempt_index: int,
    ) -> LabelRunRecord:
        for retry_index in range(self.transient_retries + 1):
            try:
                return self._run_once(
                    candidate,
                    task_entry,
                    view,
                    public_directory=public_directory,
                    prepared_directory=prepared_directory,
                    archive_sha256=archive_sha256,
                    probe=probe,
                    attempt_index=attempt_index + retry_index,
                    retry_index=retry_index,
                )
            except TransientAttemptError:
                if retry_index >= self.transient_retries:
                    raise
        raise AssertionError("transient retry loop did not return")

    def _run_once(
        self,
        candidate: TargetCandidate,
        task_entry: TaskRegistryEntry,
        view: PreparedViewManifest,
        *,
        public_directory: Path,
        prepared_directory: Path,
        archive_sha256: str,
        probe: GpuProbe,
        attempt_index: int,
        retry_index: int,
    ) -> LabelRunRecord:
        baseline = probe.read()
        if baseline.compute_process_ids:
            raise SupervisorError("assigned GPU is not idle before child launch")
        dispatch = self.workspace / "state" / "dispatch" / f"{candidate.candidate_id}.json"
        output = self.workspace / "attempts" / "staging" / f"{candidate.candidate_id}.json"
        atomic_write_json(dispatch, candidate.to_dict())
        output.unlink(missing_ok=True)
        environment = os.environ.copy()
        for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            environment.pop(name, None)
        credential_sandbox_root = self.workspace / "state" / "credential_sandboxes"
        credential_sandbox_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        credential_prefix = f"{candidate.candidate_id[:12]}-"
        for stale in credential_sandbox_root.glob(f"{credential_prefix}*"):
            if stale.is_symlink() or not stale.is_dir():
                raise SupervisorError("stale credential sandbox path is unsafe")
            shutil.rmtree(stale)
        credential_directory = Path(
            tempfile.mkdtemp(
                prefix=credential_prefix,
                dir=credential_sandbox_root,
            )
        )
        os.chmod(credential_directory, 0o700)
        compiler_cache_directory = (
            self.workspace
            / "cache"
            / "compiler_attempts"
            / f"{candidate.candidate_id}-{attempt_index}"
        )
        if compiler_cache_directory.exists():
            shutil.rmtree(compiler_cache_directory)
        compiler_cache_directory.mkdir(parents=True)
        environment["KAGGLE_CONFIG_DIR"] = str(credential_directory)
        environment.update(child_gpu_environment(probe))
        environment["PERFSEER_HARDWARE_MODE"] = (
            "production-a10" if self.production_eligible else "local-rtx5090"
        )
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache_directory / "inductor")
        environment["TRITON_CACHE_DIR"] = str(compiler_cache_directory / "triton")
        log_directory = self.workspace / "attempts" / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_fd, raw_log_path = tempfile.mkstemp(
            prefix=f"{candidate.candidate_id}-{attempt_index}-",
            suffix=".log",
            dir=log_directory,
        )
        log_path = Path(raw_log_path)
        log_stream = os.fdopen(log_fd, "wb")
        os.chmod(log_path, 0o600)
        heartbeat = (
            self.workspace
            / "state"
            / "heartbeats"
            / f"{candidate.candidate_id}-{attempt_index}.heartbeat"
        )
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.unlink(missing_ok=True)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "perfseer_v3.dataset_pack.label_worker",
                    "--candidate",
                    str(dispatch),
                    "--public",
                    str(public_directory),
                    "--prepared",
                    str(prepared_directory),
                    "--archive-sha256",
                    archive_sha256,
                    "--output",
                    str(output),
                    "--heartbeat",
                    str(heartbeat),
                ],
                env=environment,
                start_new_session=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            log_stream.close()
            shutil.rmtree(credential_directory)
            shutil.rmtree(compiler_cache_directory)
            raise
        timed_out = False
        timeout_reason = ""
        started = time.monotonic()
        if not hasattr(process, "poll"):
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                timeout_reason = "absolute_attempt_limit"
        else:
            while process.poll() is None:
                now = time.monotonic()
                if now - started >= self.timeout_seconds:
                    timed_out = True
                    timeout_reason = "absolute_attempt_limit"
                    break
                try:
                    heartbeat_age = time.time() - heartbeat.stat().st_mtime
                except OSError:
                    heartbeat_age = now - started
                if heartbeat_age >= self.no_progress_seconds:
                    timed_out = True
                    timeout_reason = "heartbeat_no_progress"
                    break
                time.sleep(0.25)
        if timed_out:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log_stream.close()
        process_tree_exited = terminate_lingering_process_group(process.pid)
        try:
            cleanup = wait_for_gpu_cleanup(
                probe,
                baseline,
                release_tolerance_mib=cleanup_release_tolerance_mib(
                    production_eligible=self.production_eligible
                ),
                child_process_tree_exited=process_tree_exited,
            )
        finally:
            shutil.rmtree(credential_directory)
            shutil.rmtree(compiler_cache_directory)
            heartbeat.unlink(missing_ok=True)
        def publish_failure(
            record: LabelRunRecord,
            *,
            worker_diagnostic: Mapping[str, Any] | None = None,
        ) -> LabelRunRecord:
            record = _bind_native_a10_provenance(
                record,
                candidate,
                task_entry,
                self.workspace,
                production_eligible=self.production_eligible,
            )
            path = self.workspace / "attempts" / "failed" / f"{record.run_id}.json"
            if worker_diagnostic is not None:
                atomic_write_json(
                    self.workspace
                    / "attempts"
                    / "diagnostics"
                    / f"{record.run_id}.json",
                    build_worker_failure_diagnostic(
                        candidate_id=candidate.candidate_id,
                        run_id=record.run_id,
                        attempt_index=attempt_index,
                        return_code=process.returncode,
                        child_log=str(log_path.relative_to(self.workspace)),
                        worker_diagnostic=worker_diagnostic,
                    ),
                )
            output.unlink(missing_ok=True)
            dispatch.unlink(missing_ok=True)
            atomic_write_json(path, asdict(record))
            return record

        if not cleanup.passed:
            failure = {
                "version": SUPERVISOR_VERSION,
                "candidate_id": candidate.candidate_id,
                "failure_kind": "gpu_cleanup_integrity_failure",
                "child_log": str(log_path.relative_to(self.workspace)),
                "cleanup": asdict(cleanup),
            }
            atomic_write_json(
                self.workspace
                / "state"
                / "global_failures"
                / f"gpu-cleanup-{candidate.candidate_id}.json",
                {**failure, "failure_sha256": canonical_sha256(failure)},
            )
            raise SupervisorError(
                "GPU cleanup failed; aborting campaign; "
                f"child log: {log_path}; evidence: {failure['cleanup']}"
            )
        if timed_out:
            if not PROFILE.is_nonvision_4gpu:
                raise SupervisorError(
                    "label child timed out; aborting campaign instead of consuming "
                    f"a quota replacement; child log: {log_path}"
                )
            if retry_index < self.transient_retries:
                raise TransientAttemptError(
                    f"label child {timeout_reason}; retrying once after cleanup; "
                    f"child log: {log_path}"
                )
            return publish_failure(
                build_failed_label_record(
                    candidate,
                    task_entry,
                    view,
                    cleanup,
                    status=AttemptStatus.TIMED_OUT,
                    failure_stage=FailureStage.TIMEOUT,
                    gpu_uuid=probe.gpu_uuid,
                    hardware_sha256=probe.hardware_fingerprint,
                    attempt_index=attempt_index,
                ),
                worker_diagnostic={
                    "failure_stage": FailureStage.TIMEOUT.value,
                    "reason_code": f"supervisor.{timeout_reason}",
                    "reason_message": (
                        "label worker exceeded its no-progress or absolute time limit"
                    ),
                },
            )
        try:
            envelope = _load_worker_envelope(output, candidate.candidate_id)
        except SupervisorError as error:
            if not PROFILE.is_nonvision_4gpu:
                raise SupervisorError(
                    "label child exited without a valid diagnostic envelope; aborting "
                    "instead of consuming a quota replacement; "
                    f"return code: {process.returncode}; child log: {log_path}"
                ) from error
            if retry_index < self.transient_retries:
                raise TransientAttemptError(
                    "label child exited without a valid envelope; retrying once after "
                    f"cleanup; return code: {process.returncode}; child log: {log_path}"
                ) from error
            return publish_failure(
                build_failed_label_record(
                    candidate,
                    task_entry,
                    view,
                    cleanup,
                    status=AttemptStatus.INTERRUPTED,
                    failure_stage=FailureStage.FORWARD,
                    gpu_uuid=probe.gpu_uuid,
                    hardware_sha256=probe.hardware_fingerprint,
                    attempt_index=attempt_index,
                ),
                worker_diagnostic={
                    "failure_stage": FailureStage.FORWARD.value,
                    "reason_code": "supervisor.invalid_worker_envelope",
                    "reason_message": (
                        "label worker exited without a valid hashed result envelope"
                    ),
                },
            )
        if envelope["status"] != "success" or process.returncode != 0:
            reason_code, reason_message = worker_failure_summary(envelope["payload"])
            if envelope["payload"].get("global_integrity_failure") is True:
                raise SupervisorError(
                    f"label child reported global integrity failure "
                    f"{reason_code}: {reason_message}; child log: {log_path}"
                )
            if envelope["status"] != "oom":
                if not PROFILE.is_nonvision_4gpu:
                    raise SupervisorError(
                        f"label child failed with {reason_code or 'unknown_error'}; aborting "
                        "instead of consuming a quota replacement; "
                        f"return code: {process.returncode}; child log: {log_path}"
                    )
                raw_stage = str(envelope["payload"].get("failure_stage", "forward"))
                stage = (
                    FailureStage(raw_stage)
                    if raw_stage in {row.value for row in FailureStage}
                    else FailureStage.FORWARD
                )
                return publish_failure(
                    build_failed_label_record(
                        candidate,
                        task_entry,
                        view,
                        cleanup,
                        status=AttemptStatus.QUARANTINED,
                        failure_stage=stage,
                        gpu_uuid=probe.gpu_uuid,
                        hardware_sha256=probe.hardware_fingerprint,
                        attempt_index=attempt_index,
                    ),
                    worker_diagnostic=envelope["payload"],
                )
            raw_stage = str(envelope["payload"].get("failure_stage", "forward"))
            stage = (
                FailureStage(raw_stage)
                if raw_stage in {row.value for row in FailureStage}
                else FailureStage.FORWARD
            )
            return publish_failure(
                build_failed_label_record(
                    candidate,
                    task_entry,
                    view,
                    cleanup,
                    status=AttemptStatus.OOM,
                    failure_stage=stage,
                    gpu_uuid=probe.gpu_uuid,
                    hardware_sha256=probe.hardware_fingerprint,
                    attempt_index=attempt_index,
                ),
                worker_diagnostic=envelope["payload"],
            )
        run = FiveEpochRunResult.from_dict(envelope["payload"])
        if run.gpu_uuid != probe.gpu_uuid or run.hardware_sha256 != probe.hardware_fingerprint:
            raise SupervisorError("child hardware differs from assigned physical GPU")
        epoch_times = tuple(float(row.epoch_ms) for row in run.epoch_measurements)
        mean = sum(epoch_times) / len(epoch_times)
        relative_spread = (
            (max(epoch_times) - min(epoch_times)) / mean
            if mean > 0
            else float("inf")
        )
        spread_limit = retained_epoch_spread_limit(
            production_eligible=self.production_eligible
        )
        if (
            mean <= 0
            or relative_spread > spread_limit
        ):
            return publish_failure(
                build_failed_label_record(
                    candidate,
                    task_entry,
                    view,
                    cleanup,
                    status=AttemptStatus.QUARANTINED,
                    failure_stage=FailureStage.STABILITY,
                    gpu_uuid=run.gpu_uuid,
                    hardware_sha256=run.hardware_sha256,
                    attempt_index=attempt_index,
                    run=run,
                ),
                worker_diagnostic={
                    "failure_stage": FailureStage.STABILITY.value,
                    "reason_code": "supervisor.retained_epoch_spread",
                    "reason_message": (
                        f"retained epoch spread {relative_spread:.6f} exceeded "
                        f"limit {spread_limit:.6f}"
                    ),
                },
            )
        record = build_accepted_label_record(
            candidate,
            task_entry,
            view,
            run,
            cleanup,
            attempt_index=attempt_index,
            production_eligible=self.production_eligible,
        )
        record = _bind_native_a10_provenance(
            record,
            candidate,
            task_entry,
            self.workspace,
            production_eligible=self.production_eligible,
        )
        current_environment_provenance = environment_provenance()
        hardware_provenance = getattr(probe, "hardware_provenance", None)
        if (
            not isinstance(hardware_provenance, Mapping)
            or canonical_sha256(current_environment_provenance)
            != record.fingerprints.environment_sha256
            or canonical_sha256(hardware_provenance)
            != record.fingerprints.hardware_sha256
        ):
            raise SupervisorError("accepted run provenance payloads differ from their hashes")
        atomic_write_json(
            self.workspace
            / "provenance"
            / "environment"
            / f"{record.fingerprints.environment_sha256}.json",
            current_environment_provenance,
        )
        atomic_write_json(
            self.workspace
            / "provenance"
            / "hardware"
            / f"{record.fingerprints.hardware_sha256}.json",
            hardware_provenance,
        )
        record_path = self.workspace / "attempts" / "accepted" / f"{record.configuration_id}.json"
        output.unlink(missing_ok=True)
        dispatch.unlink(missing_ok=True)
        atomic_write_json(record_path, asdict(record))
        log_path.unlink(missing_ok=True)
        return record


__all__ = [
    "AttemptSupervisor",
    "CAMPAIGN_ENVIRONMENT_VERSION",
    "GpuProbe",
    "ParentNvmlProbe",
    "PhysicalGpuSlot",
    "SUPERVISOR_VERSION",
    "SupervisorError",
    "TransientAttemptError",
    "await_worker_futures",
    "build_accepted_label_record",
    "build_failed_label_record",
    "child_gpu_environment",
    "discover_v100_probes",
    "discover_a10_probe",
    "discover_a10_probes",
    "discover_local_rtx5090_probe",
    "environment_provenance",
    "lock_campaign_environment",
    "wait_for_gpu_cleanup",
    "validate_a10_gpu_identity",
    "validate_rtx5090_gpu_identity",
    "validate_v100_gpu_identity",
    "worker_failure_summary",
]
