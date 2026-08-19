"""Fresh-process entry point for exactly one V100 configuration attempt."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
import re
import threading
import traceback
from typing import Any, Mapping, Sequence

import torch

from .v100_runner import NvmlTelemetryBackend, run_five_epoch_training
from .fingerprints import canonical_sha256
from .sampler import target_candidate_from_dict
from .storage import atomic_write_json
from .task_registry import load_task_registry
from ..hardware import HardwareProfileV3
from ..version import TRANSFER_MANIFEST_VERSION
from .transfer_labeling import (
    materialize_target_conditioned_graph,
    run_target_paired_attempt,
)


WORKER_ENVELOPE_VERSION = "perfseer_v3_v100_label_worker_envelope_v1"
ASSIGNED_GPU_UUID_ENV = "PERFSEER_ASSIGNED_GPU_UUID"


class LabelWorkerError(RuntimeError):
    pass


def _diagnostic_message(error: BaseException) -> str:
    """Return a bounded, single-line failure reason without credential values."""

    value = " ".join(str(error).split()) or error.__class__.__name__
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "<redacted>")
    value = re.sub(
        r"(?i)(token|key|password|secret)(\s*[=:]\s*)[^\s,;]+",
        r"\1\2<redacted>",
        value,
    )
    return value[:2_000]


def _failure_stage(error: BaseException) -> str:
    value = f"{error.__class__.__name__} {error}".lower()
    for stage, tokens in (
        ("compile", ("compile", "inductor", "triton")),
        ("loss", ("loss",)),
        ("backward", ("backward", "gradient", "grad")),
        ("optimizer", ("optimizer", "scheduler", "parameter update")),
        ("telemetry", ("telemetry", "nvml")),
        ("stability", ("non-finite", "nonfinite", "nan", "inf")),
    ):
        if any(token in value for token in tokens):
            return stage
    return "forward"


def _global_integrity_failure(error: BaseException) -> bool:
    value = f"{error.__class__.__name__} {error}".lower()
    return any(
        token in value
        for token in (
            "foreign gpu",
            "foreign process",
            "nvml",
            "hardware differs",
            "hardware identity",
            "not bound to",
            "archive drift",
            "dataset fingerprint",
        )
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabelWorkerError("worker input JSON is unreadable") from error
    if not isinstance(value, Mapping):
        raise LabelWorkerError("worker input JSON must be an object")
    return value


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise LabelWorkerError(f"base labels line {line_number} is not an object")
            rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("samples", value.get("rows"))
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise LabelWorkerError("base labels must be JSONL or a JSON row array")
    return list(value)


def resolve_transfer_inputs(
    candidate_id: str,
    subset_manifest: Mapping[str, Any],
    base_label_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[float, ...]]:
    """Bind one worker attempt to an integrity-checked frozen paired row."""

    if subset_manifest.get("manifest_version") != TRANSFER_MANIFEST_VERSION:
        raise LabelWorkerError("worker transfer subset version mismatch")
    unhashed = dict(subset_manifest)
    declared_hash = str(unhashed.pop("subset_sha256", ""))
    if declared_hash != canonical_sha256(unhashed):
        raise LabelWorkerError("worker transfer subset content hash mismatch")
    matches = [
        row
        for row in subset_manifest.get("selection", ())
        if str(row.get("configuration_id")) == candidate_id
    ]
    if len(matches) != 1:
        raise LabelWorkerError("candidate is not exactly once in the frozen transfer subset")
    base_matches = [
        row
        for row in base_label_rows
        if candidate_id
        in {
            str(row.get("configuration_id", "")),
            str(row.get("root_configuration_id", "")),
            str(row.get("sample_id", "")),
        }
    ]
    if len(base_matches) != 1:
        raise LabelWorkerError("candidate does not have exactly one frozen V100 label row")
    raw_targets = base_matches[0].get("target_values", base_matches[0].get("target"))
    if not isinstance(raw_targets, (list, tuple)):
        raise LabelWorkerError("frozen V100 row has no six-target values")
    targets = tuple(float(value) for value in raw_targets)
    if len(targets) != 6 or any(not math.isfinite(value) or value < 0 for value in targets):
        raise LabelWorkerError("frozen V100 row violates the six-target contract")
    return matches[0], targets


def _physical_assignment() -> tuple[int, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or not visible.isascii() or not visible.isdecimal():
        raise LabelWorkerError(
            "label worker requires exactly one numeric CUDA_VISIBLE_DEVICES token"
        )
    assigned_uuid = os.environ.get(ASSIGNED_GPU_UUID_ENV, "")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", assigned_uuid) is None:
        raise LabelWorkerError("label worker requires one assigned physical GPU UUID")
    # CUDA addresses this one visible physical device as logical device zero,
    # while NVML continues to use the physical index supplied by the parent.
    return int(visible), assigned_uuid


def _start_heartbeat(path: Path | None) -> tuple[threading.Event, threading.Thread | None]:
    stop = threading.Event()
    if path is None:
        return stop, None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    return stop, None


def _envelope(candidate_id: str, *, status: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = {
        "version": WORKER_ENVELOPE_VERSION,
        "candidate_id": candidate_id,
        "status": status,
        "payload": payload,
    }
    return {**value, "envelope_sha256": canonical_sha256(value)}


def run_worker(arguments: argparse.Namespace) -> int:
    candidate = target_candidate_from_dict(_load_mapping(arguments.candidate))
    entry = next(
        (row for row in load_task_registry().entries if row.task_id == candidate.task_id),
        None,
    )
    if entry is None:
        raise LabelWorkerError("candidate task is absent from the frozen registry")
    heartbeat_stop, heartbeat_thread = _start_heartbeat(arguments.heartbeat)
    try:
        expected_profile = None
        if arguments.target_hardware_profile is not None:
            profile_payload = _load_mapping(arguments.target_hardware_profile)
            expected_profile = HardwareProfileV3.from_dict(profile_payload)
            expected_hash = profile_payload.get("hardware_profile_sha256")
            if expected_hash is not None and expected_hash != expected_profile.sha256:
                raise LabelWorkerError("target hardware profile content hash mismatch")
        physical_gpu_index, assigned_gpu_uuid = _physical_assignment()
        backend = NvmlTelemetryBackend(
            physical_gpu_index,
            expected_gpu_uuid=assigned_gpu_uuid,
            expected_hardware_profile=expected_profile,
        )
        if arguments.transfer_subset is not None:
            if expected_profile is None or arguments.base_labels is None:
                raise LabelWorkerError(
                    "transfer mode requires --target-hardware-profile and --base-labels"
                )
            subset = _load_mapping(arguments.transfer_subset)
            base_configuration_id = arguments.base_configuration_id or candidate.candidate_id
            selection, base_targets = resolve_transfer_inputs(
                base_configuration_id,
                subset,
                _load_rows(arguments.base_labels),
            )
            if expected_profile.hardware_id != subset.get("target_hardware_id"):
                raise LabelWorkerError("target profile and transfer subset hardware IDs differ")
            if not arguments.memory_probe and candidate.candidate_id != base_configuration_id:
                raise LabelWorkerError(
                    "selected paired labels must execute the exact frozen V100 configuration"
                )
            if arguments.memory_probe and arguments.base_configuration_id is None:
                raise LabelWorkerError("memory probes require --base-configuration-id")
            base_graph_raw = selection.get("graph_path")
            if not base_graph_raw or arguments.target_graph_output is None:
                raise LabelWorkerError(
                    "transfer mode requires a selected base graph and --target-graph-output"
                )
            base_graph_path = (arguments.transfer_subset.parent / str(base_graph_raw)).resolve()
            measured_configuration_id = None
            if arguments.memory_probe and candidate.candidate_id != base_configuration_id:
                if arguments.memory_probe_graph is None:
                    raise LabelWorkerError(
                        "changed-batch memory probes require --memory-probe-graph"
                    )
                base_graph_path = arguments.memory_probe_graph.resolve()
                measured_configuration_id = candidate.candidate_id
            target_graph_path = materialize_target_conditioned_graph(
                base_graph_path,
                arguments.target_graph_output,
                target_profile=expected_profile,
                paired_graph_signature=str(selection["graph_signature"]),
                base_hardware_id=str(subset["base_hardware_id"]),
                measured_configuration_id=measured_configuration_id,
            )
            attempt = run_target_paired_attempt(
                candidate,
                entry,
                split=("memory_probe" if arguments.memory_probe else str(selection["split"])),
                base_targets=base_targets,
                target_profile=expected_profile,
                telemetry_backend=backend,
                public_directory=arguments.public,
                prepared_directory=arguments.prepared,
                archive_sha256=arguments.archive_sha256,
                original_configuration_id=base_configuration_id,
                original_microbatch_size=arguments.original_microbatch_size,
                graph_path=str(target_graph_path),
            )
            payload = {**asdict(attempt), "attempt_sha256": attempt.sha256}
            worker_status = "success" if attempt.status == "accepted" else attempt.status
            atomic_write_json(
                arguments.output,
                _envelope(candidate.candidate_id, status=worker_status, payload=payload),
            )
            return 0 if attempt.status == "accepted" else 20
        result = run_five_epoch_training(
            candidate,
            entry,
            public_directory=arguments.public,
            prepared_directory=arguments.prepared,
            archive_sha256=arguments.archive_sha256,
            telemetry_backend=backend,
            progress_callback=(
                (lambda: arguments.heartbeat.touch(exist_ok=True))
                if arguments.heartbeat is not None
                else None
            ),
        )
        atomic_write_json(
            arguments.output,
            _envelope(candidate.candidate_id, status="success", payload=result.to_dict()),
        )
        return 0
    except torch.OutOfMemoryError:
        atomic_write_json(
            arguments.output,
            _envelope(
                candidate.candidate_id,
                status="oom",
                payload={"failure_stage": "allocator", "reason_code": "cuda_out_of_memory"},
            ),
        )
        return 20
    except Exception as error:
        # The parent redirects this process to a private 0600 attempt log.
        # Keep the full stack there while the durable/exportable diagnostic
        # carries only the bounded, redacted single-line reason below.
        traceback.print_exc()
        atomic_write_json(
            arguments.output,
            _envelope(
                candidate.candidate_id,
                status="failed",
                payload={
                    "failure_stage": _failure_stage(error),
                    "reason_code": f"{error.__class__.__module__}.{error.__class__.__name__}",
                    "reason_message": _diagnostic_message(error),
                    "global_integrity_failure": _global_integrity_failure(error),
                },
            ),
        )
        return 21
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--target-hardware-profile", type=Path)
    parser.add_argument("--transfer-subset", type=Path)
    parser.add_argument("--base-labels", type=Path)
    parser.add_argument("--target-graph-output", type=Path)
    parser.add_argument("--base-configuration-id")
    parser.add_argument("--original-microbatch-size", type=int)
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--memory-probe-graph", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LabelWorkerError",
    "ASSIGNED_GPU_UUID_ENV",
    "WORKER_ENVELOPE_VERSION",
    "_diagnostic_message",
    "_physical_assignment",
    "main",
    "resolve_transfer_inputs",
    "run_worker",
]
