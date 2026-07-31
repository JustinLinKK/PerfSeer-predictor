"""Fresh-process entry point for exactly one A10G configuration attempt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .a10g_runner import NvmlTelemetryBackend, run_five_epoch_training
from .fingerprints import canonical_sha256
from .sampler import target_candidate_from_dict
from .storage import atomic_write_json
from .task_registry import load_task_registry


WORKER_ENVELOPE_VERSION = "perfseer_v3_a10g_label_worker_envelope_v1"


class LabelWorkerError(RuntimeError):
    pass


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LabelWorkerError("worker input JSON is unreadable") from error
    if not isinstance(value, Mapping):
        raise LabelWorkerError("worker input JSON must be an object")
    return value


def _physical_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise LabelWorkerError(
            "label worker requires CUDA_VISIBLE_DEVICES to contain one physical GPU index"
        )
    return int(visible)


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
    try:
        backend = NvmlTelemetryBackend(_physical_index())
        result = run_five_epoch_training(
            candidate,
            entry,
            public_directory=arguments.public,
            prepared_directory=arguments.prepared,
            archive_sha256=arguments.archive_sha256,
            telemetry_backend=backend,
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
        atomic_write_json(
            arguments.output,
            _envelope(
                candidate.candidate_id,
                status="failed",
                payload={
                    "failure_stage": "forward",
                    "reason_code": f"{error.__class__.__module__}.{error.__class__.__name__}",
                },
            ),
        )
        return 21
    finally:
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LabelWorkerError", "WORKER_ENVELOPE_VERSION", "main", "run_worker"]
