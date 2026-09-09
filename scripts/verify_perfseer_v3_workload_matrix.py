"""Execute every declared v3 workload without producing training labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json, sha256_bytes
from perfseer_v3.splits import grouped_split, split_manifest_payload
from perfseer_v3.workloads import (
    WorkloadDescriptor,
    build_workload,
    default_composites,
    default_microbenchmarks,
    default_source_workloads,
    manifest_payload,
)


def _to_device(value: Any, device: torch.device, *, leaf_grad: bool) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        if leaf_grad and moved.is_floating_point() and value.requires_grad:
            moved = moved.detach().requires_grad_(True)
        return moved
    if isinstance(value, tuple):
        return tuple(_to_device(item, device, leaf_grad=leaf_grad) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device, leaf_grad=leaf_grad) for item in value]
    if isinstance(value, dict):
        return {
            key: _to_device(item, device, leaf_grad=leaf_grad)
            for key, item in value.items()
        }
    return value


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensors(item)]
    if isinstance(value, dict):
        return [
            tensor
            for key in sorted(value, key=str)
            for tensor in _tensors(value[key])
        ]
    return []


def _assert_finite(value: Any, *, workload_id: str) -> None:
    for tensor in _tensors(value):
        if tensor.is_floating_point() or tensor.is_complex():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{workload_id}: produced nonfinite output")


def _execute(
    descriptor: WorkloadDescriptor,
    *,
    device: torch.device,
) -> None:
    instance = build_workload(descriptor)
    model = instance.model.to(device)
    model.train(descriptor.phase == "training")
    args = _to_device(
        instance.args,
        device,
        leaf_grad=descriptor.phase == "training",
    )
    kwargs = _to_device(
        instance.kwargs,
        device,
        leaf_grad=descriptor.phase == "training",
    )
    target = _to_device(instance.target, device, leaf_grad=False)
    if descriptor.phase == "forward":
        with torch.no_grad():
            output = model(*args, **kwargs)
        _assert_finite(output, workload_id=descriptor.workload_id)
        return

    if instance.loss_fn is None:
        raise RuntimeError(f"{descriptor.workload_id}: training loss is missing")
    optimizer = (
        instance.optimizer_factory(model.parameters())
        if instance.optimizer_factory is not None
        else None
    )
    actual_optimizer = (
        optimizer.__class__.__name__.lower()
        if optimizer is not None
        else None
    )
    if actual_optimizer != descriptor.optimizer:
        raise RuntimeError(
            f"{descriptor.workload_id}: optimizer mismatch "
            f"{actual_optimizer!r} != {descriptor.optimizer!r}"
        )
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = model(*args, **kwargs)
    loss = instance.loss_fn(output, target)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError(f"{descriptor.workload_id}: loss must be finite scalar")
    loss.backward()
    gradient_tensors = [
        tensor
        for tensor in (*model.parameters(), *_tensors(args), *_tensors(kwargs))
        if tensor.is_floating_point() and tensor.requires_grad
    ]
    if not any(tensor.grad is not None for tensor in gradient_tensors):
        raise RuntimeError(f"{descriptor.workload_id}: backward produced no gradients")
    for tensor in gradient_tensors:
        if tensor.grad is not None and not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{descriptor.workload_id}: backward produced nonfinite gradients")
    if optimizer is not None:
        optimizer.step()
    _assert_finite(output, workload_id=descriptor.workload_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "perfseer_v3_workload_execution.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="execute only the first N workloads; zero executes the full matrix",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        raise ValueError("limit cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    descriptors = tuple(
        sorted(
            (
                *default_microbenchmarks(),
                *default_composites(),
                *default_source_workloads(),
            ),
            key=lambda row: row.workload_id,
        )
    )
    selected = descriptors[: args.limit] if args.limit else descriptors
    failures: list[dict[str, str]] = []
    completed: list[WorkloadDescriptor] = []
    for index, descriptor in enumerate(selected):
        torch.manual_seed(index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(index)
        try:
            _execute(descriptor, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            completed.append(descriptor)
        except Exception as exc:
            failures.append(
                {
                    "workload_id": descriptor.workload_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            if device.type == "cuda" and index % 100 == 99:
                torch.cuda.empty_cache()

    full_manifest = manifest_payload(descriptors)
    split_manifest = split_manifest_payload(grouped_split(descriptors))
    report: dict[str, Any] = {
        "report_version": "perfseer_v3_workload_execution_v1",
        "device": {
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "pytorch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "full_matrix": args.limit == 0,
        "declared_workloads": len(descriptors),
        "selected_workloads": len(selected),
        "completed_workloads": len(completed),
        "failures": failures,
        "data_layer_counts": dict(
            sorted(Counter(row.data_layer for row in completed).items())
        ),
        "phase_counts": dict(
            sorted(Counter(row.phase for row in completed).items())
        ),
        "precision_counts": dict(
            sorted(Counter(row.dtype for row in completed).items())
        ),
        "optimizer_counts": dict(
            sorted(Counter(row.optimizer or "none" for row in completed).items())
        ),
        "workload_manifest_sha256": full_manifest["sha256"],
        "split_manifest_sha256": split_manifest["sha256"],
    }
    report["report_sha256"] = sha256_bytes(
        canonical_json(report).encode("utf-8")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed={len(completed)}/{len(selected)} "
        f"failures={len(failures)} device={device.type} "
        f"report_sha256={report['report_sha256']}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
