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
    LabelRunRecord,
)
from .fingerprints import canonical_sha256
from .label_worker import WORKER_ENVELOPE_VERSION
from .labeler_profile import PROFILE
from .a10_crosswalk import (
    REFERENCE_MANIFEST_SHA256,
    semantic_distribution_signature,
)
from .operation_support import build_operation_support_contract
from .prepared_view import PreparedViewManifest
from .sampler import TargetCandidate
from .storage import atomic_write_json
from .task_registry import TaskRegistryEntry


SUPERVISOR_VERSION = "perfseer_v3_v100_attempt_supervisor_v2"
CAMPAIGN_ENVIRONMENT_VERSION = "perfseer_v3_v100_campaign_environment_v1"
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
    def __init__(self, physical_index: int) -> None:
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
            if PROFILE.name == "native_a10":
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
    if PROFILE.name != "native_a10":
        raise SupervisorError("A10 discovery requires the native_a10 process profile")
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


def lock_campaign_environment(workspace: str | Path) -> Mapping[str, Any]:
    """Freeze one software/driver identity for every accepted campaign row."""

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
            raise SupervisorError("campaign environment changed since collection started")
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
) -> LabelRunRecord:
    if PROFILE.name != "native_a10":
        return record
    mapping_path = workspace / "state" / "a10_crosswalk.jsonl"
    original_id = None
    root_candidate_id = candidate.candidate_id
    repair_history = candidate.mutation_specification.get("oom_repair_history", ())
    replacement = candidate.mutation_specification.get("quota_replacement")
    if repair_history:
        root_candidate_id = str(repair_history[-1]["root_candidate_id"])
    elif isinstance(replacement, Mapping):
        root_candidate_id = str(replacement["target_candidate_id"])
    if mapping_path.is_file():
        for line in mapping_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("native_nrp_a10_candidate_id") == root_candidate_id:
                original_id = row.get("original_a10g_candidate_id")
                break
    if original_id is None:
        raise SupervisorError("native A10 record has no frozen reference crosswalk row")
    environment = environment_provenance()
    bound = replace(
        record,
        production_eligible=True,
        reference_provenance={
            "original_a10g_candidate_id": original_id,
            "native_root_candidate_id": root_candidate_id,
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "semantic_distribution_signature": semantic_distribution_signature(candidate),
        },
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
    timeout_seconds: int = 6 * 60 * 60

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.timeout_seconds < 1:
            raise SupervisorError("attempt timeout must be positive")

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
        environment["CUDA_VISIBLE_DEVICES"] = str(probe.physical_index)
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
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        finally:
            log_stream.close()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process_tree_exited = True
        else:
            process_tree_exited = False
            os.killpg(process.pid, signal.SIGKILL)
        try:
            cleanup = wait_for_gpu_cleanup(
                probe, baseline, child_process_tree_exited=process_tree_exited
            )
        finally:
            shutil.rmtree(credential_directory)
            shutil.rmtree(compiler_cache_directory)
        def publish_failure(record: LabelRunRecord) -> LabelRunRecord:
            record = _bind_native_a10_provenance(
                record, candidate, task_entry, self.workspace
            )
            path = self.workspace / "attempts" / "failed" / f"{record.run_id}.json"
            output.unlink(missing_ok=True)
            dispatch.unlink(missing_ok=True)
            atomic_write_json(path, asdict(record))
            return record

        if not cleanup.passed:
            raise SupervisorError(
                f"GPU cleanup failed; aborting campaign; child log: {log_path}"
            )
        if timed_out:
            raise SupervisorError(
                "label child timed out; aborting campaign instead of consuming "
                f"a quota replacement; child log: {log_path}"
            )
        try:
            envelope = _load_worker_envelope(output, candidate.candidate_id)
        except SupervisorError as error:
            raise SupervisorError(
                "label child exited without a valid diagnostic envelope; aborting "
                "instead of consuming a quota replacement; "
                f"return code: {process.returncode}; child log: {log_path}"
            ) from error
        if envelope["status"] != "success" or process.returncode != 0:
            reason_code = str(envelope["payload"].get("reason_code", ""))
            if envelope["status"] != "oom":
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
                    status=AttemptStatus.OOM,
                    failure_stage=stage,
                    gpu_uuid=probe.gpu_uuid,
                    hardware_sha256=probe.hardware_fingerprint,
                    attempt_index=attempt_index,
                )
            )
        run = FiveEpochRunResult.from_dict(envelope["payload"])
        if run.gpu_uuid != probe.gpu_uuid or run.hardware_sha256 != probe.hardware_fingerprint:
            raise SupervisorError("child hardware differs from assigned physical GPU")
        epoch_times = tuple(float(row.epoch_ms) for row in run.epoch_measurements)
        mean = sum(epoch_times) / len(epoch_times)
        if mean <= 0 or (max(epoch_times) - min(epoch_times)) / mean > 0.10:
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
                )
            )
        record = build_accepted_label_record(
            candidate,
            task_entry,
            view,
            run,
            cleanup,
            attempt_index=attempt_index,
        )
        record = _bind_native_a10_provenance(
            record, candidate, task_entry, self.workspace
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
    "await_worker_futures",
    "build_accepted_label_record",
    "build_failed_label_record",
    "discover_v100_probes",
    "discover_a10_probe",
    "environment_provenance",
    "lock_campaign_environment",
    "wait_for_gpu_cleanup",
    "validate_a10_gpu_identity",
    "validate_v100_gpu_identity",
]
