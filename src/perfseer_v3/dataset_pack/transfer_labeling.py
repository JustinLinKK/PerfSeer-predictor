"""Paired target-GPU labeling built on the existing five-epoch model runner."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..baseline import canonical_json
from ..graph_ir_v3 import GraphIRV3
from ..hardware import (
    HardwareProfileV3,
    graph_hardware_id,
    require_specific_hardware_id,
)
from ..hardware_transfer import (
    BaseTransferLineageV3,
    TRANSFER_LABEL_BUDGETS,
    TRANSFER_TARGET_NAMES,
    base_transfer_lineage_json_schema,
)
from ..version import TRANSFER_MANIFEST_VERSION
from .a10g_runner import FiveEpochRunResult, TelemetryBackend, run_five_epoch_training
from .sampler import TargetCandidate
from .task_registry import TaskRegistryEntry


TRANSFER_ATTEMPT_VERSION = "perfseer_v3_target_attempt_v1"
TARGET_TRAINING_MANIFEST_VERSION = "perfseer_v3_target_training_manifest_v1"
TARGET_MEMORY_PROBE_PLAN_VERSION = "perfseer_v3_target_memory_probe_plan_v1"


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, context: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")


def _validate_targets(values: Sequence[float], *, context: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(TRANSFER_TARGET_NAMES):
        raise ValueError(f"{context} must follow the six-target order")
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"{context} must be finite and nonnegative")
    return result


def _allocator_failure_payload(error: BaseException) -> dict[str, Any]:
    allocator_state: dict[str, int] = {}
    if torch.cuda.is_available():
        for name, getter in (
            ("memory_allocated_bytes", torch.cuda.memory_allocated),
            ("memory_reserved_bytes", torch.cuda.memory_reserved),
            ("max_memory_allocated_bytes", torch.cuda.max_memory_allocated),
            ("max_memory_reserved_bytes", torch.cuda.max_memory_reserved),
        ):
            try:
                allocator_state[name] = int(getter())
            except (RuntimeError, AssertionError):
                pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            allocator_state.update(
                {
                    "device_free_bytes": int(free_bytes),
                    "device_total_bytes": int(total_bytes),
                }
            )
        except (RuntimeError, AssertionError):
            pass
    return {
        "exception_type": type(error).__name__,
        "message": str(error)[:4096],
        "allocator_state": allocator_state,
    }


def deterministic_memory_probe_batches(
    original_microbatch_size: int,
    eligible_microbatch_sizes: Sequence[int],
    *,
    maximum_microbatch_size: int | None = None,
) -> tuple[int, ...]:
    """Return larger power-of-two probe steps after the exact paired batch."""

    if original_microbatch_size < 1 or original_microbatch_size & (
        original_microbatch_size - 1
    ):
        raise ValueError("target memory-probe origin must be a positive power of two")
    eligible = tuple(sorted({int(value) for value in eligible_microbatch_sizes}))
    if any(value < 1 for value in eligible) or original_microbatch_size not in eligible:
        raise ValueError("target memory-probe origin is outside its eligible batch ladder")
    ceiling = (
        max(eligible)
        if maximum_microbatch_size is None
        else int(maximum_microbatch_size)
    )
    if ceiling < original_microbatch_size:
        raise ValueError("target memory-probe ceiling is below the paired batch")
    result: list[int] = []
    next_batch = original_microbatch_size * 2
    while next_batch <= ceiling and next_batch in eligible:
        result.append(next_batch)
        next_batch *= 2
    return tuple(result)


def build_memory_probe_candidates(
    candidate: TargetCandidate,
    *,
    maximum_microbatch_size: int | None = None,
) -> tuple[TargetCandidate, ...]:
    """Derive hash-valid workload candidates for one target probe ladder."""

    candidate.validate()
    batches = deterministic_memory_probe_batches(
        candidate.microbatch_size,
        candidate.batch_plan.eligible_ladder,
        maximum_microbatch_size=maximum_microbatch_size,
    )
    probes: list[TargetCandidate] = []
    for ladder_index, batch_size in enumerate(batches, 1):
        mutation = dict(candidate.mutation_specification)
        mutation["target_memory_probe"] = {
            "version": TARGET_MEMORY_PROBE_PLAN_VERSION,
            "base_configuration_id": candidate.candidate_id,
            "ladder_index": ladder_index,
            "original_microbatch_size": candidate.microbatch_size,
            "measured_microbatch_size": batch_size,
        }
        draft = replace(
            candidate,
            candidate_id="0" * 64,
            batch_plan=replace(
                candidate.batch_plan,
                selected_microbatch=batch_size,
            ),
            microbatch_size=batch_size,
            mutation_specification=mutation,
        )
        probe = replace(draft, candidate_id=_sha256(draft.unhashed_payload()))
        probe.validate()
        probes.append(probe)
    return tuple(probes)


def aggregate_target_run(run: FiveEpochRunResult) -> tuple[float, ...]:
    run.validate()
    epoch_times = tuple(float(row.epoch_ms) for row in run.epoch_measurements)
    samples = tuple(
        sample for row in run.epoch_measurements for sample in row.telemetry_samples
    )
    if not samples:
        raise ValueError("target run contains no telemetry samples")
    total_duration = sum(sample.duration_s for sample in samples)
    if total_duration <= 0:
        raise ValueError("target telemetry duration must be positive")
    ordered_sm = sorted(sample.sm_util_percent for sample in samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered_sm)) - 1)
    return _validate_targets(
        (
            sum(epoch_times) / len(epoch_times),
            sum(sample.sm_util_percent * sample.duration_s for sample in samples)
            / total_duration,
            ordered_sm[p95_index],
            max(sample.device_used_vram_mib for sample in samples),
            max(float(row.peak_torch_reserved_mib) for row in run.epoch_measurements),
            max(sample.memory_controller_util_percent for sample in samples),
        ),
        context="target run",
    )


def materialize_target_conditioned_graph(
    base_graph_path: str | Path,
    output_path: str | Path,
    *,
    target_profile: HardwareProfileV3,
    paired_graph_signature: str,
    base_hardware_id: str = "nvidia_a10g_24gb_aws_g5",
    measured_configuration_id: str | None = None,
) -> Path:
    """Reuse workload IR while replacing only the explicit hardware profile.

    The paired A10 graph signature is retained separately for grouped leakage
    checks. The target graph obtains its own content hash because hardware
    metadata is part of GraphIR integrity.
    """

    target_profile.validate(require_complete_signature=True)
    target_hardware_id = require_specific_hardware_id(
        target_profile.hardware_id, context="target-conditioned graph hardware_id"
    )
    expected_base = require_specific_hardware_id(
        base_hardware_id, context="paired graph base_hardware_id"
    )
    base_graph = GraphIRV3.load(base_graph_path)
    if graph_hardware_id(base_graph.metadata) != expected_base:
        raise ValueError("paired graph does not target the declared base GPU")
    if not paired_graph_signature:
        raise ValueError("paired graph signature cannot be empty")
    if base_graph.graph_sha256 != paired_graph_signature and not measured_configuration_id:
        raise ValueError("paired graph signature differs from the frozen base graph")
    if measured_configuration_id and base_graph.graph_sha256 == paired_graph_signature:
        raise ValueError(
            "changed-configuration target probes cannot reuse the paired base graph"
        )
    raw = base_graph.to_dict()
    metadata = dict(raw.get("metadata") or {})
    for legacy_name in (
        "hardware_features",
        "hardware_environment",
        "hardware_microbenchmarks",
        "hardware_id",
    ):
        metadata.pop(legacy_name, None)
    metadata.update(
        {
            "target_hardware_id": target_hardware_id,
            "hardware_profile": target_profile.canonical_payload,
            "hardware_profile_sha256": target_profile.sha256,
            "paired_base_hardware_id": expected_base,
            "paired_base_graph_signature": paired_graph_signature,
            "measured_workload_graph_signature": base_graph.graph_sha256,
            **(
                {"measured_configuration_id": measured_configuration_id}
                if measured_configuration_id
                else {}
            ),
        }
    )
    raw["metadata"] = metadata
    target_graph = GraphIRV3.from_dict(raw)
    if graph_hardware_id(target_graph.metadata) != target_hardware_id:
        raise ValueError("target-conditioned graph hardware identity was not preserved")
    return target_graph.save(output_path)


@dataclass(frozen=True)
class TransferLabelAttemptV3:
    base_configuration_id: str
    paired_configuration_id: str
    split: str
    base_hardware_id: str
    target_hardware_id: str
    hardware_profile_sha256: str
    base_targets: tuple[float, ...]
    status: str
    failure_stage: str
    target_targets: tuple[float, ...] | None
    original_microbatch_size: int
    measured_microbatch_size: int
    repaired_from_configuration_id: str | None
    run_payload: dict[str, Any] | None
    graph_path: str | None = None
    measured_configuration_id: str | None = None
    failure_payload: dict[str, Any] | None = None
    attempt_version: str = TRANSFER_ATTEMPT_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransferLabelAttemptV3":
        raw = dict(value)
        declared_sha256 = raw.pop("attempt_sha256", None)
        raw["base_targets"] = tuple(float(item) for item in raw["base_targets"])
        if raw.get("target_targets") is not None:
            raw["target_targets"] = tuple(
                float(item) for item in raw["target_targets"]
            )
        if raw.get("run_payload") is not None:
            raw["run_payload"] = dict(raw["run_payload"])
        if raw.get("failure_payload") is not None:
            raw["failure_payload"] = dict(raw["failure_payload"])
        attempt = cls(**raw)
        attempt.validate()
        if declared_sha256 is not None and declared_sha256 != attempt.sha256:
            raise ValueError("target attempt content hash mismatch")
        return attempt

    def validate(self) -> None:
        if self.attempt_version != TRANSFER_ATTEMPT_VERSION:
            raise ValueError("target attempt version mismatch")
        if not self.base_configuration_id or not self.paired_configuration_id:
            raise ValueError("target attempt configuration IDs cannot be empty")
        base = require_specific_hardware_id(
            self.base_hardware_id, context="target attempt base_hardware_id"
        )
        target = require_specific_hardware_id(
            self.target_hardware_id, context="target attempt target_hardware_id"
        )
        if base == target:
            raise ValueError("paired target attempt must use a new GPU")
        if self.split not in {"train", "validation", "test", "memory_probe"}:
            raise ValueError("target attempt split is invalid")
        _require_sha256(
            self.hardware_profile_sha256,
            context="target attempt hardware profile hash",
        )
        _validate_targets(self.base_targets, context="paired A10 targets")
        if self.status not in {"accepted", "oom", "quarantined"}:
            raise ValueError("target attempt status is invalid")
        if self.status == "accepted":
            if self.failure_stage != "none" or self.target_targets is None:
                raise ValueError("accepted target attempts require six labels and no failure")
            _validate_targets(self.target_targets, context="target labels")
            if self.run_payload is None:
                raise ValueError("accepted target attempts require the five-epoch run payload")
            if self.failure_payload is not None:
                raise ValueError("accepted target attempts cannot retain a failure payload")
        elif self.target_targets is not None or self.failure_stage == "none":
            raise ValueError("failed target attempts retain a stage and no fabricated targets")
        if self.status == "oom":
            if not isinstance(self.failure_payload, Mapping):
                raise ValueError("OOM target attempts require allocator failure evidence")
            allocator_state = self.failure_payload.get("allocator_state")
            if not isinstance(allocator_state, Mapping):
                raise ValueError("OOM target attempts require allocator state")
            if not self.failure_payload.get("exception_type"):
                raise ValueError("OOM target attempts require an exception type")
        if self.original_microbatch_size < 1 or self.measured_microbatch_size < 1:
            raise ValueError("target attempt batch sizes must be positive")
        if (
            self.measured_microbatch_size > self.original_microbatch_size
            and self.split != "memory_probe"
        ):
            raise ValueError("only a target memory probe may increase microbatch size")
        if (
            self.measured_microbatch_size != self.original_microbatch_size
            and not self.repaired_from_configuration_id
        ):
            raise ValueError("changed-batch target attempts must retain probe/repair lineage")
        measured_id = self.measured_configuration_id or self.base_configuration_id
        if self.measured_microbatch_size != self.original_microbatch_size:
            smaller = min(self.measured_microbatch_size, self.original_microbatch_size)
            larger = max(self.measured_microbatch_size, self.original_microbatch_size)
            ratio = larger // smaller
            if larger % smaller or ratio & (ratio - 1):
                raise ValueError(
                    "target memory probes and repairs require a power-of-two batch ladder"
                )
            if measured_id == self.base_configuration_id:
                raise ValueError(
                    "changed-batch target attempts require a distinct measured configuration"
                )
        if self.split != "memory_probe" and measured_id != self.base_configuration_id:
            raise ValueError("selected paired labels must measure the exact A10 configuration")
        if (
            self.split != "memory_probe"
            and self.measured_microbatch_size != self.original_microbatch_size
        ):
            raise ValueError("target batch repairs are retained only as memory probes")

    @property
    def sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def run_target_paired_attempt(
    candidate: TargetCandidate,
    task_entry: TaskRegistryEntry,
    *,
    split: str,
    base_targets: Sequence[float],
    target_profile: HardwareProfileV3,
    telemetry_backend: TelemetryBackend,
    public_directory: str | Path,
    prepared_directory: str | Path,
    archive_sha256: str,
    original_configuration_id: str | None = None,
    original_microbatch_size: int | None = None,
    graph_path: str | None = None,
) -> TransferLabelAttemptV3:
    """Reuse the exact five-epoch execution path for one paired target attempt."""

    candidate.validate()
    target_profile.validate(require_complete_signature=True)
    if candidate.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
        raise ValueError("transfer labeling must preserve the frozen A10 configuration identity")
    if getattr(telemetry_backend, "hardware_profile_sha256", None) != target_profile.sha256:
        raise ValueError("telemetry backend does not match the frozen target profile")
    base_values = _validate_targets(base_targets, context="paired A10 targets")
    original_id = original_configuration_id or candidate.candidate_id
    original_batch = original_microbatch_size or candidate.microbatch_size
    repaired_from = None if candidate.candidate_id == original_id else original_id
    paired_id = _sha256(
        {
            "base_configuration_id": original_id,
            "measured_candidate_id": candidate.candidate_id,
            "target_hardware_id": target_profile.hardware_id,
            "hardware_profile_sha256": target_profile.sha256,
        }
    )
    try:
        run = run_five_epoch_training(
            candidate,
            task_entry,
            public_directory=public_directory,
            prepared_directory=prepared_directory,
            archive_sha256=archive_sha256,
            telemetry_backend=telemetry_backend,
        )
    except torch.OutOfMemoryError as error:
        attempt = TransferLabelAttemptV3(
            base_configuration_id=original_id,
            paired_configuration_id=paired_id,
            split=split,
            base_hardware_id="nvidia_a10g_24gb_aws_g5",
            target_hardware_id=target_profile.hardware_id,
            hardware_profile_sha256=target_profile.sha256,
            base_targets=base_values,
            status="oom",
            failure_stage="allocator",
            target_targets=None,
            original_microbatch_size=original_batch,
            measured_microbatch_size=candidate.microbatch_size,
            repaired_from_configuration_id=repaired_from,
            run_payload=None,
            graph_path=graph_path,
            measured_configuration_id=candidate.candidate_id,
            failure_payload=_allocator_failure_payload(error),
        )
    else:
        attempt = TransferLabelAttemptV3(
            base_configuration_id=original_id,
            paired_configuration_id=paired_id,
            split=split,
            base_hardware_id="nvidia_a10g_24gb_aws_g5",
            target_hardware_id=target_profile.hardware_id,
            hardware_profile_sha256=target_profile.sha256,
            base_targets=base_values,
            status="accepted",
            failure_stage="none",
            target_targets=aggregate_target_run(run),
            original_microbatch_size=original_batch,
            measured_microbatch_size=candidate.microbatch_size,
            repaired_from_configuration_id=repaired_from,
            run_payload=dict(run.to_dict()),
            graph_path=graph_path,
            measured_configuration_id=candidate.candidate_id,
        )
    attempt.validate()
    return attempt


def build_target_training_manifest(
    subset_manifest: Mapping[str, Any],
    attempts: Sequence[TransferLabelAttemptV3],
) -> dict[str, Any]:
    if subset_manifest.get("manifest_version") != TRANSFER_MANIFEST_VERSION:
        raise ValueError("transfer subset manifest version mismatch")
    declared_subset_sha256 = str(subset_manifest.get("subset_sha256", ""))
    unhashed_subset = dict(subset_manifest)
    unhashed_subset.pop("subset_sha256", None)
    if declared_subset_sha256 != _sha256(unhashed_subset):
        raise ValueError("transfer subset manifest content hash mismatch")
    base_lineage = BaseTransferLineageV3.from_dict(
        subset_manifest.get("base_lineage", {})
    )
    budget = int(subset_manifest.get("label_budget", 0))
    if budget not in TRANSFER_LABEL_BUDGETS:
        raise ValueError("transfer subset label budget is invalid")
    target_hardware_id = require_specific_hardware_id(
        subset_manifest.get("target_hardware_id"),
        context="target training manifest hardware ID",
    )
    base_hardware_id = require_specific_hardware_id(
        subset_manifest.get("base_hardware_id"),
        context="target training manifest base hardware ID",
    )
    if base_hardware_id != "nvidia_a10g_24gb_aws_g5":
        raise ValueError("target training manifest must derive from the frozen A10G corpus")
    selected = {
        str(row["configuration_id"]): str(row["split"])
        for row in subset_manifest.get("selection", ())
    }
    if len(selected) != budget:
        raise ValueError("transfer subset selection count differs from label budget")
    memory_probe_contract = {
        str(row.get("configuration_id", "")): str(row.get("grouped_split", ""))
        for row in subset_manifest.get("memory_boundary_probes", ())
    }
    memory_probe_ids = set(memory_probe_contract)
    if not memory_probe_ids or not memory_probe_ids.issubset(selected):
        raise ValueError(
            "target memory probes must be a nonempty subset of the frozen selection"
        )
    if any(memory_probe_contract[item] != selected[item] for item in memory_probe_ids):
        raise ValueError("target memory-probe grouped split differs from the selection")
    for attempt in attempts:
        attempt.validate()
        if attempt.base_hardware_id != base_hardware_id:
            raise ValueError("target attempts contain another base GPU")
        if attempt.target_hardware_id != target_hardware_id:
            raise ValueError("target attempts contain another GPU")
        if attempt.base_configuration_id not in selected:
            raise ValueError("target attempt is outside the frozen subset/probe contract")
        expected_split = selected[attempt.base_configuration_id]
        if attempt.split == "memory_probe":
            if attempt.base_configuration_id not in memory_probe_ids:
                raise ValueError(
                    "target memory attempt is outside the frozen probe contract"
                )
        elif attempt.split != expected_split:
            raise ValueError("target attempt split differs from the frozen subset")
        if not attempt.graph_path:
            raise ValueError(
                "every retained target attempt requires a target-conditioned graph path"
            )
    accepted_by_base: dict[str, TransferLabelAttemptV3] = {}
    for attempt in attempts:
        if (
            attempt.status == "accepted"
            and attempt.base_configuration_id in selected
            and attempt.split == selected[attempt.base_configuration_id]
        ):
            if attempt.base_configuration_id in accepted_by_base:
                raise ValueError("target subset has duplicate accepted measurements")
            accepted_by_base[attempt.base_configuration_id] = attempt
    missing = sorted(set(selected) - set(accepted_by_base))
    if missing:
        raise ValueError(
            f"target subset is missing {len(missing)} successful paired measurements"
        )
    profile_hashes = {attempt.hardware_profile_sha256 for attempt in attempts}
    if len(profile_hashes) != 1:
        raise ValueError("target attempts use multiple hardware profiles")
    rows = [
        {
            "sample_id": attempt.paired_configuration_id,
            "base_configuration_id": base_id,
            "split": selected[base_id],
            "graph_path": (
                attempt.graph_path
                or next(
                    (
                        str(item.get("graph_path"))
                        for item in subset_manifest.get("selection", ())
                        if str(item.get("configuration_id")) == base_id
                        and item.get("graph_path") not in (None, "")
                    ),
                    "",
                )
            ),
            "source_group": next(
                str(item["source_group"])
                for item in subset_manifest.get("selection", ())
                if str(item.get("configuration_id")) == base_id
            ),
            "graph_signature": next(
                str(item["graph_signature"])
                for item in subset_manifest.get("selection", ())
                if str(item.get("configuration_id")) == base_id
            ),
            "base_hardware_id": attempt.base_hardware_id,
            "target_hardware_id": attempt.target_hardware_id,
            "base_target": list(attempt.base_targets),
            "target": list(attempt.target_targets or ()),
            "attempt_sha256": attempt.sha256,
        }
        for base_id, attempt in sorted(accepted_by_base.items())
    ]
    if any(not row["graph_path"] for row in rows):
        raise ValueError("every accepted target label requires a target-conditioned graph path")
    oom_attempts = [
        {
            **json.loads(canonical_json(asdict(attempt))),
            "attempt_sha256": attempt.sha256,
        }
        for attempt in attempts
        if attempt.status == "oom"
    ]
    memory_probe_attempts = [
        {
            **json.loads(canonical_json(asdict(attempt))),
            "attempt_sha256": attempt.sha256,
        }
        for attempt in attempts
        if attempt.split == "memory_probe"
    ]
    payload: dict[str, Any] = {
        "manifest_version": TARGET_TRAINING_MANIFEST_VERSION,
        "source_subset_sha256": subset_manifest["subset_sha256"],
        "base_lineage": base_lineage.to_dict(),
        "label_budget": budget,
        "base_hardware_id": base_hardware_id,
        "target_hardware_id": target_hardware_id,
        "hardware_profile_sha256": next(iter(profile_hashes)),
        "deployment": {
            "target_hardware_id": target_hardware_id,
            "hardware_allowlist": [target_hardware_id],
            "precision_allowlist": ["float32", "float16", "bfloat16", "mixed"],
            "capture_quality_allowlist": ["strict"],
            "optimizer_allowlist": ["adamw", "sgd"],
            "scheduler_allowlist": ["none", "cosine", "linear"],
            "training_mode_allowlist": ["training"],
        },
        "samples": rows,
        "oom_attempts": oom_attempts,
        "memory_probe_attempts": memory_probe_attempts,
        "failed_attempts_retained": True,
    }
    payload["manifest_sha256"] = _sha256(payload)
    return payload


def target_training_manifest_json_schema() -> dict[str, Any]:
    target_vector = {
        "type": "array",
        "minItems": 6,
        "maxItems": 6,
        "items": {"type": "number", "minimum": 0},
    }
    attempt_schema = {
        "type": "object",
        "required": [
            "base_configuration_id",
            "paired_configuration_id",
            "split",
            "base_hardware_id",
            "target_hardware_id",
            "hardware_profile_sha256",
            "base_targets",
            "status",
            "failure_stage",
            "target_targets",
            "original_microbatch_size",
            "measured_microbatch_size",
            "repaired_from_configuration_id",
            "run_payload",
            "graph_path",
            "measured_configuration_id",
            "failure_payload",
            "attempt_version",
            "attempt_sha256",
        ],
        "properties": {
            "base_configuration_id": {"type": "string", "minLength": 1},
            "paired_configuration_id": {"type": "string", "minLength": 1},
            "split": {"enum": ["train", "validation", "test", "memory_probe"]},
            "base_hardware_id": {"type": "string", "minLength": 1},
            "target_hardware_id": {"type": "string", "minLength": 1},
            "hardware_profile_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "base_targets": target_vector,
            "status": {"enum": ["accepted", "oom", "quarantined"]},
            "failure_stage": {"type": "string", "minLength": 1},
            "target_targets": {"oneOf": [target_vector, {"type": "null"}]},
            "original_microbatch_size": {"type": "integer", "minimum": 1},
            "measured_microbatch_size": {"type": "integer", "minimum": 1},
            "repaired_from_configuration_id": {
                "type": ["string", "null"]
            },
            "run_payload": {"type": ["object", "null"]},
            "graph_path": {"type": "string", "minLength": 1},
            "measured_configuration_id": {"type": ["string", "null"]},
            "failure_payload": {"type": ["object", "null"]},
            "attempt_version": {"const": TRANSFER_ATTEMPT_VERSION},
            "attempt_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "oom"}}},
                "then": {
                    "properties": {
                        "failure_payload": {
                            "type": "object",
                            "required": ["exception_type", "allocator_state"],
                        }
                    }
                },
            }
        ],
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TARGET_TRAINING_MANIFEST_VERSION,
        "type": "object",
        "required": [
            "manifest_version",
            "source_subset_sha256",
            "base_lineage",
            "label_budget",
            "base_hardware_id",
            "target_hardware_id",
            "hardware_profile_sha256",
            "deployment",
            "samples",
            "oom_attempts",
            "memory_probe_attempts",
            "failed_attempts_retained",
            "manifest_sha256",
        ],
        "properties": {
            "manifest_version": {"const": TARGET_TRAINING_MANIFEST_VERSION},
            "source_subset_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "base_lineage": base_transfer_lineage_json_schema(),
            "label_budget": {"enum": list(TRANSFER_LABEL_BUDGETS)},
            "base_hardware_id": {"type": "string", "minLength": 1},
            "target_hardware_id": {"type": "string", "minLength": 1},
            "hardware_profile_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "deployment": {
                "type": "object",
                "required": ["target_hardware_id", "hardware_allowlist"],
            },
            "samples": {
                "type": "array",
                "minItems": 128,
                "maxItems": 1024,
                "items": {
                    "type": "object",
                    "required": [
                        "sample_id",
                        "base_configuration_id",
                        "split",
                        "graph_path",
                        "source_group",
                        "graph_signature",
                        "base_hardware_id",
                        "target_hardware_id",
                        "base_target",
                        "target",
                        "attempt_sha256",
                    ],
                    "properties": {
                        "sample_id": {"type": "string", "minLength": 1},
                        "base_configuration_id": {"type": "string", "minLength": 1},
                        "split": {"enum": ["train", "validation", "test"]},
                        "graph_path": {"type": "string", "minLength": 1},
                        "source_group": {"type": "string", "minLength": 1},
                        "graph_signature": {"type": "string", "minLength": 1},
                        "base_hardware_id": {"type": "string", "minLength": 1},
                        "target_hardware_id": {"type": "string", "minLength": 1},
                        "base_target": target_vector,
                        "target": target_vector,
                        "attempt_sha256": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "oom_attempts": {
                "type": "array",
                "items": {
                    "allOf": [
                        {"$ref": "#/$defs/transfer_attempt"},
                        {"properties": {"status": {"const": "oom"}}},
                    ]
                },
            },
            "memory_probe_attempts": {
                "type": "array",
                "items": {
                    "allOf": [
                        {"$ref": "#/$defs/transfer_attempt"},
                        {"properties": {"split": {"const": "memory_probe"}}},
                    ]
                },
            },
            "failed_attempts_retained": {"const": True},
            "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "$defs": {"transfer_attempt": attempt_schema},
        "additionalProperties": True,
    }


__all__ = [
    "TARGET_TRAINING_MANIFEST_VERSION",
    "TARGET_MEMORY_PROBE_PLAN_VERSION",
    "TRANSFER_ATTEMPT_VERSION",
    "TransferLabelAttemptV3",
    "aggregate_target_run",
    "build_memory_probe_candidates",
    "build_target_training_manifest",
    "deterministic_memory_probe_batches",
    "materialize_target_conditioned_graph",
    "run_target_paired_attempt",
    "target_training_manifest_json_schema",
]
