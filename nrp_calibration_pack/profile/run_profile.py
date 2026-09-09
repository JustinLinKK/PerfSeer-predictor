"""Profile generated calibration models on one GPU shard."""

from __future__ import annotations

import argparse
import csv
import contextlib
import gc
import hashlib
import io
import importlib.util
import json
import math
import os
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from nrp_calibration_pack.workload import (
    label_v3_from_result,
    manifest_row_from_workload,
    normalize_workload_spec,
    scheduler_resource_label_from_result,
)


MI_B = 1024.0 * 1024.0
DEFAULT_PRECISION_CONFIG = "fp32_ieee"
BASE_AUTO_PRECISIONS = ("fp32_ieee", "tf32", "bf16_amp", "fp16_amp")
FP8_PRECISIONS = {"fp8_te_hybrid", "fp8_e4m3", "fp8_e5m2"}
TE_LOW_PRECISIONS = FP8_PRECISIONS | {"nvfp4_te"}
PRECISION_ALIASES = {
    "fp32": "fp32_ieee",
    "float32": "fp32_ieee",
    "fp32_ieee": "fp32_ieee",
    "tf32": "tf32",
    "bf16": "bf16_amp",
    "bf16_amp": "bf16_amp",
    "fp16": "fp16_amp",
    "float16": "fp16_amp",
    "fp16_amp": "fp16_amp",
    "fp8": "fp8_te_hybrid",
    "fp8_te": "fp8_te_hybrid",
    "fp8_te_hybrid": "fp8_te_hybrid",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
    "fp4": "nvfp4_te",
    "nvfp4": "nvfp4_te",
    "nvfp4_te": "nvfp4_te",
}
_DATASET_MATERIAL_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def configure_transformer_engine_cuda_include(details: dict[str, Any] | None = None) -> str | None:
    """Make CUDA Python wheel headers visible to Transformer Engine NVRTC."""
    existing = os.environ.get("NVTE_CUDA_INCLUDE_DIR")
    if existing:
        if details is not None:
            details["nvte_cuda_include_dir"] = existing
            details["nvte_cuda_include_dir_source"] = "env"
        return existing

    candidates: list[tuple[str, Path]] = []
    for key in ("CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        value = os.environ.get(key)
        if value:
            candidates.append((key, Path(value) / "include"))

    try:
        nvidia_spec = importlib.util.find_spec("nvidia")
        for base in nvidia_spec.submodule_search_locations or []:
            root = Path(base)
            for dirname in ("cu13", "cu12"):
                candidates.append((f"python_package:{dirname}", root / dirname / "include"))
    except Exception as exc:
        if details is not None:
            details["nvte_cuda_include_dir_probe_error"] = repr(exc)

    checked: list[str] = []
    for source, path in candidates:
        checked.append(str(path))
        if (path / "cuda_runtime.h").exists():
            os.environ["NVTE_CUDA_INCLUDE_DIR"] = str(path)
            if details is not None:
                details["nvte_cuda_include_dir"] = str(path)
                details["nvte_cuda_include_dir_source"] = source
            return str(path)

    if details is not None:
        details["nvte_cuda_include_dir_checked"] = checked
    return None


@dataclass
class SampleStats:
    avg_sm_util: float = 0.0
    sm_util_std: float = 0.0
    p50_sm_util: float = 0.0
    p95_sm_util: float = 0.0
    avg_mem_util: float = 0.0
    mem_util_std: float = 0.0
    p50_mem_util: float = 0.0
    p95_mem_util: float = 0.0
    avg_mem_usage: float = 0.0
    mem_usage_std: float = 0.0
    p50_mem_usage: float = 0.0
    p95_mem_usage: float = 0.0
    peak_sm_util: float = 0.0
    peak_mem_util: float = 0.0
    peak_mem_usage: float = 0.0
    avg_vram_used_percent: float = 0.0
    peak_vram_used_percent: float = 0.0
    sample_count: int = 0
    measurement_duration_ms: float = 0.0
    reliability_flags: list[str] = field(default_factory=list)
    source: str = "none"


def host_memory_mib() -> dict[str, float]:
    current_kib = 0.0
    peak_kib = 0.0
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                current_kib = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                peak_kib = float(line.split()[1])
    if peak_kib <= 0.0:
        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_kib = maxrss / 1024.0 if sys.platform == "darwin" else maxrss
    if current_kib <= 0.0:
        current_kib = peak_kib
    return {"current_mib": current_kib / 1024.0, "peak_mib": peak_kib / 1024.0}


@dataclass
class ResumeCheckpoint:
    completed_profile_points: set[str]
    malformed_rows: int = 0
    incomplete_rows: int = 0


@dataclass
class PrecisionSelection:
    auto: bool
    configs: list[str] | None = None


@dataclass
class PrecisionRuntime:
    config: str
    device_type: str
    autocast_dtype: torch.dtype | None = None
    grad_scaler_enabled: bool = False
    backend: str = "torch"
    supported: bool = True
    unsupported_reason: str | None = None
    fallback_policy: str = "none"
    details: dict[str, Any] | None = None
    te_recipe: Any | None = None
    te_recipe_name: str | None = None
    model_dtype: torch.dtype | None = None
    input_dtype: torch.dtype | None = None

    def autocast(self):
        if self.te_recipe is not None:
            import transformer_engine.pytorch as te

            autocast = getattr(te, "fp8_autocast", None) or getattr(te, "autocast")
            try:
                return autocast(enabled=True, fp8_recipe=self.te_recipe)
            except TypeError:
                return autocast(enabled=True, recipe=self.te_recipe)
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(self.device_type, dtype=self.autocast_dtype)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "precision_config": self.config,
            "backend": self.backend,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "fallback_policy": self.fallback_policy,
            "autocast_dtype": str(self.autocast_dtype).replace("torch.", "") if self.autocast_dtype is not None else None,
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "te_recipe": self.te_recipe_name,
            "model_dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else None,
            "input_dtype": str(self.input_dtype).replace("torch.", "") if self.input_dtype is not None else None,
            "details": self.details or {},
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile generated PerfSeer calibration models.")
    parser.add_argument("--manifest", help="Legacy calibration manifest. Required unless --workload-specs is provided.")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("JOB_COMPLETIONS", "1")))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--profile-epochs", type=int)
    parser.add_argument("--batches-per-epoch", type=int, default=1)
    parser.add_argument("--infer-repeats", type=int, default=30)
    parser.add_argument("--train-repeats", type=int, default=20)
    parser.add_argument(
        "--label-time-mode",
        default=os.environ.get("PERFSEER_LABEL_TIME_MODE", "step_extrapolated"),
        choices=("step_extrapolated", "measured_epochs"),
        help="How scheduler_label_v3 train_epoch_ms is produced. Scheduler workflows should use measured_epochs.",
    )
    parser.add_argument(
        "--time-label-warmup-epochs",
        type=int,
        default=int(os.environ.get("PERFSEER_TIME_LABEL_WARMUP_EPOCHS", "1")),
        help="Full warmup epochs to run before measured train epochs when --label-time-mode measured_epochs.",
    )
    parser.add_argument(
        "--time-label-measured-epochs",
        type=int,
        default=int(os.environ.get("PERFSEER_TIME_LABEL_MEASURED_EPOCHS", "2")),
        help="Measured train epochs used for scheduler_label_v3 train_epoch_ms when --label-time-mode measured_epochs.",
    )
    parser.add_argument(
        "--profile-dataset-dir",
        help="Optional directory of <model_id>.json input/repeat specs from profile/make_profile_datasets.py.",
    )
    parser.add_argument(
        "--workload-specs",
        help="Optional WorkloadSpec JSONL from profile/make_workload_specs.py. Overrides --manifest rows for scheduler profiling.",
    )
    parser.add_argument(
        "--allow-synthetic-workload-inputs",
        action="store_true",
        help="Allow WorkloadSpec rows without a real dataloader adapter to use generated tensor inputs for smoke tests only.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument(
        "--min-phase-seconds",
        type=float,
        default=float(os.environ.get("PERFSEER_MIN_PHASE_SECONDS", "0")),
        help="Minimum sustained train/infer measurement duration. The local scheduler workflow passes 20 seconds.",
    )
    parser.add_argument(
        "--min-sampler-samples",
        type=int,
        default=int(os.environ.get("PERFSEER_MIN_SAMPLER_SAMPLES", "0")),
        help="Minimum sustained NVML samples before a phase is considered reliable. The local scheduler workflow passes 100.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--sm-occupancy-source", default="ncu", choices=("ncu", "nvml_proxy"))
    parser.add_argument(
        "--resource-profile-mode",
        default="compat",
        choices=("compat", "sustained"),
        help="compat synchronizes each repeat; sustained measures one multi-step phase with resource sampling across the whole phase.",
    )
    parser.add_argument(
        "--resource-audit-source",
        default="none",
        choices=("none", "ncu"),
        help="Optional exact-counter audit source for selected/small runs. Full local labeling should usually keep this as none.",
    )
    parser.add_argument(
        "--hardware-id",
        help="Stable hardware identifier to store in profiler outputs, for example rtx3090, rtx4090, or rtx5090.",
    )
    parser.add_argument("--precision-config", action="append", help="Precision config(s) to profile. May be repeated or comma-separated.")
    parser.add_argument(
        "--precision-sweep",
        help="Comma-separated precision config filter, or auto to resolve from the current GPU/Transformer Engine environment.",
    )
    parser.add_argument("--fp8-backend", default="transformer_engine", choices=("transformer_engine", "none"))
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip profile points that already have a completed result row and label file. Enabled by default.",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reprofile all rows for this shard even if previous outputs exist.",
    )
    args = parser.parse_args(argv)
    if not args.manifest and not args.workload_specs:
        parser.error("--manifest is required unless --workload-specs is provided")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        parser.error("--warmup-epochs must be >= 0")
    if args.profile_epochs is not None and args.profile_epochs <= 0:
        parser.error("--profile-epochs must be > 0")
    if args.batches_per_epoch <= 0:
        parser.error("--batches-per-epoch must be > 0")
    if args.time_label_warmup_epochs < 0:
        parser.error("--time-label-warmup-epochs must be >= 0")
    if args.time_label_measured_epochs <= 0:
        parser.error("--time-label-measured-epochs must be > 0")
    for name in ("infer_repeats", "train_repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.min_phase_seconds < 0:
        parser.error("--min-phase-seconds must be >= 0")
    if args.min_sampler_samples < 0:
        parser.error("--min-sampler-samples must be >= 0")
    if args.resource_audit_source == "ncu" and args.device == "cpu":
        parser.error("--resource-audit-source ncu requires a CUDA device")
    return args


def normalize_precision_config(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key == "bf32":
        raise ValueError("bf32 is ambiguous; use tf32 or bf16_amp")
    if key == "mxfp8":
        raise ValueError("mxfp8 is out of scope for v1; use fp8_te_hybrid or nvfp4_te")
    if key not in PRECISION_ALIASES:
        allowed = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"unknown precision_config {value!r}; expected one of: {allowed}")
    return PRECISION_ALIASES[key]


def transformer_engine_availability(te_module: Any, name: str) -> bool | None:
    fn = getattr(te_module, name, None)
    if not callable(fn):
        try:
            from transformer_engine.pytorch import fp8 as fp8_module

            fn = getattr(fp8_module, name, None)
        except Exception:
            fn = None
    if not callable(fn):
        return None
    result = fn()
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


def precision_selection(args: argparse.Namespace) -> PrecisionSelection:
    raw: list[str] = []
    if args.precision_sweep:
        raw.extend(part.strip() for part in args.precision_sweep.split(",") if part.strip())
    if args.precision_config:
        for item in args.precision_config:
            raw.extend(part.strip() for part in item.split(",") if part.strip())
    if not raw:
        return PrecisionSelection(auto=False, configs=None)
    if any(item.strip().lower().replace("-", "_") == "auto" for item in raw):
        if len(raw) != 1:
            raise ValueError("--precision-sweep auto cannot be combined with explicit precision configs")
        return PrecisionSelection(auto=True)
    configs: list[str] = []
    for item in raw:
        precision = normalize_precision_config(item)
        if precision not in configs:
            configs.append(precision)
    return PrecisionSelection(auto=False, configs=configs)


def unique_model_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in manifest:
        model_id = str(row.get("model_id") or row.get("graph_id") or "")
        if not model_id or model_id in seen:
            continue
        rows.append(row)
        seen.add(model_id)
    return rows


def with_precision_row(row: dict[str, Any], precision_config: str, index: int) -> dict[str, Any]:
    expanded = dict(row)
    expanded["precision_config"] = precision_config
    expanded["precision_config_index"] = index
    expanded["label_file"] = f"label/label/{row['model_id']}_{precision_config}.txt"
    expanded["profile_point_id"] = f"{row['model_id']}::{precision_config}"
    return expanded


def expand_manifest_precisions(manifest: list[dict[str, Any]], precision_configs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in unique_model_rows(manifest):
        for index, precision_config in enumerate(precision_configs):
            rows.append(with_precision_row(row, precision_config, index))
    return rows


def expand_workload_manifest_precisions(manifest: list[dict[str, Any]], precision_configs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest:
        workload = row.get("workload_spec") if isinstance(row.get("workload_spec"), dict) else None
        if workload is None:
            rows.extend(expand_manifest_precisions([row], precision_configs))
            continue
        for precision_config in precision_configs:
            expanded_workload = json.loads(json.dumps(workload))
            expanded_workload.setdefault("training", {})["precision"] = precision_config
            expanded_workload.pop("profile_point_id", None)
            expanded_workload.pop("workload_hash", None)
            rows.append(manifest_row_from_workload(normalize_workload_spec(expanded_workload)))
    return rows


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_workload_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            workload = normalize_workload_spec(json.loads(line))
            rows.append(manifest_row_from_workload(workload))
    return rows


def profile_point_id(row: dict[str, Any]) -> str:
    precision_config = normalize_precision_config(str(row.get("precision_config", DEFAULT_PRECISION_CONFIG)))
    return str(row.get("profile_point_id", f"{row['model_id']}::{precision_config}"))


def label_file_for_row(row: dict[str, Any], precision_config: str) -> str:
    return str(row.get("label_file", f"label/label/{row['model_id']}_{precision_config}.txt"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def load_resume_checkpoint(output_dir: Path, results_path: Path) -> ResumeCheckpoint:
    checkpoint = ResumeCheckpoint(completed_profile_points=set())
    if not results_path.exists():
        return checkpoint
    with results_path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                checkpoint.malformed_rows += 1
                continue
            profile_id = str(row.get("profile_point_id") or "")
            label_file = str(row.get("label_file") or "")
            label_path = output_dir / label_file if label_file else None
            has_label = label_path is not None and label_path.is_file() and label_path.stat().st_size > 0
            if row.get("status") == "ok" and isinstance(row.get("label"), dict) and profile_id and has_label:
                checkpoint.completed_profile_points.add(profile_id)
            elif row.get("status") == "ok":
                checkpoint.incomplete_rows += 1
    return checkpoint


def append_result_row(results_fh, result: dict[str, Any]) -> None:
    results_fh.write(json.dumps(result, sort_keys=True) + "\n")
    results_fh.flush()
    os.fsync(results_fh.fileno())


def load_profile_dataset_spec(model_id: str, profile_dataset_dir: str | None) -> dict[str, Any]:
    if not profile_dataset_dir:
        return {}
    path = Path(profile_dataset_dir) / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile dataset spec not found for {model_id}: {path}")
    return json.loads(path.read_text())


def profile_dataset_spec_from_workload(row: dict[str, Any]) -> dict[str, Any]:
    workload = row.get("workload_spec") if isinstance(row.get("workload_spec"), dict) else {}
    if not workload:
        return {}
    dataset = workload.get("dataset") if isinstance(workload.get("dataset"), dict) else {}
    training = workload.get("training") if isinstance(workload.get("training"), dict) else {}
    input_specs = dataset.get("input_specs") or row.get("input_specs") or []
    input_shape = dataset.get("input_shape") or (input_specs[0].get("shape") if input_specs else row.get("input_shape"))
    batch_size = int(training.get("batch_size") or (input_shape[0] if input_shape else 1))
    return {
        "profile_dataset_format_version": 2,
        "source": "workload_spec",
        "workload_spec": workload,
        "input_specs": input_specs,
        "input_shape": input_shape,
        "batch_size": batch_size,
        "train_repeats": training.get("train_repeats"),
        "infer_repeats": training.get("infer_repeats"),
        "dataset": dataset,
        "training": training,
    }


def positive_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


def steps_per_epoch_for_profile(row: dict[str, Any], dataset_spec: dict[str, Any], batch_size: int) -> tuple[int, int, int]:
    workload = row.get("workload_spec") if isinstance(row.get("workload_spec"), dict) else {}
    dataset = workload.get("dataset") if isinstance(workload.get("dataset"), dict) else {}
    training = workload.get("training") if isinstance(workload.get("training"), dict) else {}
    if not dataset and isinstance(dataset_spec.get("dataset"), dict):
        dataset = dataset_spec["dataset"]
    if not training and isinstance(dataset_spec.get("training"), dict):
        training = dataset_spec["training"]
    grad_accum = max(int(training.get("grad_accumulation_steps") or 1), 1)
    effective_batch = max(int(batch_size) * grad_accum, 1)
    samples = int(dataset.get("num_samples") or dataset.get("sample_count") or row.get("train_samples") or effective_batch)
    return max(int(math.ceil(samples / effective_batch)), 1), effective_batch, max(samples, 1)


def normalize_input_specs(row: dict[str, Any], dataset_spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw_specs = dataset_spec.get("input_specs") or row.get("input_specs")
    if not raw_specs:
        shape = dataset_spec.get("input_shape", row["input_shape"])
        raw_specs = [{"name": "input0", "shape": shape, "dtype": "float32", "kind": "float"}]
    specs: list[dict[str, Any]] = []
    for idx, spec in enumerate(raw_specs):
        shape = [int(dim) for dim in spec.get("shape", [])]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"invalid input spec shape at index {idx}: {shape!r}")
        specs.append(
            {
                "name": str(spec.get("name", f"input{idx}")),
                "shape": shape,
                "dtype": str(spec.get("dtype", "float32")).lower(),
                "kind": str(spec.get("kind", "float")).lower(),
            }
        )
    return specs


def with_batch_size(input_specs: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    batch = max(int(batch_size), 1)
    for spec in input_specs:
        item = dict(spec)
        shape = [int(dim) for dim in item.get("shape", [])]
        if shape:
            shape[0] = batch
        item["shape"] = shape
        out.append(item)
    return out


def generated_model_input_specs(module: Any, fallback_specs: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    raw_specs = getattr(module, "INPUT_SPECS", None)
    if not raw_specs:
        return with_batch_size(fallback_specs, batch_size)
    normalized = normalize_input_specs({"input_specs": raw_specs, "input_shape": raw_specs[0].get("shape")}, {})
    return with_batch_size(normalized, batch_size)


def make_profile_inputs(
    input_specs: list[dict[str, Any]],
    device: torch.device,
    float_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for spec in input_specs:
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = str(spec.get("dtype", "float32")).lower()
        kind = str(spec.get("kind", "float")).lower()
        if dtype in {"int64", "long"} or kind in {"tokens", "token_ids"}:
            tensors.append(torch.zeros(shape, dtype=torch.long, device=device))
        elif kind == "adjacency":
            base = torch.eye(shape[-1], dtype=torch.float32, device=device)
            tensors.append(base.expand(shape).clone())
        else:
            tensors.append(torch.randn(shape, dtype=float_dtype or torch.float32, device=device))
    return tuple(tensors)


def resolve_repo_path(value: Any) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def split_row_key(part: str) -> tuple[str, int | None]:
    marker = ":row:"
    if marker not in part:
        return part, None
    member, raw_index = part.rsplit(marker, 1)
    return member, int(raw_index)


def csv_row_payload(data: bytes, row_index: int) -> bytes:
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for idx, row in enumerate(reader):
            if idx == row_index:
                return "\x1f".join(row).encode("utf-8", errors="ignore")
    raise IndexError(f"CSV row {row_index} not found")


def read_sample_key_bytes(raw_dir: Path, sample_key: str, max_bytes: int = 4096) -> bytes:
    if sample_key.startswith("node:") or sample_key.startswith("csv_row_"):
        return sample_key.encode("utf-8")
    if "::" not in sample_key:
        path = raw_dir / sample_key
        return path.read_bytes()[:max_bytes]

    archive_part, rest = sample_key.split("::", 1)
    archive_path = raw_dir / archive_part
    parts = rest.split("::")
    current_zip: zipfile.ZipFile | None = zipfile.ZipFile(archive_path)
    try:
        for idx, part in enumerate(parts):
            assert current_zip is not None
            member, row_index = split_row_key(part)
            payload = current_zip.read(member)
            if row_index is not None:
                return csv_row_payload(payload, row_index)[:max_bytes]
            if idx == len(parts) - 1:
                return payload[:max_bytes]
            current_zip.close()
            current_zip = zipfile.ZipFile(io.BytesIO(payload))
    finally:
        if current_zip is not None:
            current_zip.close()
    raise ValueError(f"could not resolve sample key {sample_key!r}")


def load_real_dataset_material(dataset: dict[str, Any]) -> dict[str, Any]:
    subset_path = resolve_repo_path(dataset.get("subset_mask_path") or dataset.get("subset_index_path"))
    raw_dir = resolve_repo_path(dataset.get("raw_dir"))
    if subset_path is None or not subset_path.is_file():
        raise FileNotFoundError(f"subset mask not found: {subset_path}")
    if raw_dir is None or not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_dir}")
    cache_key = (str(raw_dir), str(subset_path), str(dataset.get("dataset_id", "")))
    cached = _DATASET_MATERIAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    payload = json.loads(subset_path.read_text())
    sample_keys = [str(item) for item in payload.get("sample_keys", []) if str(item)]
    if not sample_keys:
        raise ValueError(f"subset mask has no sample_keys: {subset_path}")
    probe_keys = sample_keys[: min(len(sample_keys), 8)]
    fingerprints: list[dict[str, Any]] = []
    for key in probe_keys:
        sample_bytes = read_sample_key_bytes(raw_dir, key)
        fingerprints.append(
            {
                "key": key,
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
                "bytes": len(sample_bytes),
            }
        )
    seed_payload = {
        "dataset_id": dataset.get("dataset_id"),
        "subset_id": dataset.get("subset_id"),
        "sample_keys": sample_keys[:64],
        "fingerprints": fingerprints,
    }
    seed_digest = hashlib.sha256(json.dumps(seed_payload, sort_keys=True).encode("utf-8")).hexdigest()
    material = {
        "seed": int(seed_digest[:16], 16) % (2**63 - 1),
        "subset_mask_path": str(subset_path),
        "raw_dir": str(raw_dir),
        "sample_keys": sample_keys,
        "sample_count": len(sample_keys),
        "sample_keys_consumed": len(probe_keys),
        "sample_fingerprints": fingerprints,
    }
    _DATASET_MATERIAL_CACHE[cache_key] = material
    return material


def make_real_dataset_inputs(
    input_specs: list[dict[str, Any]],
    dataset_spec: dict[str, Any],
    device: torch.device,
    float_dtype: torch.dtype | None = None,
    batch_offset: int = 0,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    dataset = dataset_spec.get("dataset") if isinstance(dataset_spec.get("dataset"), dict) else {}
    material = load_real_dataset_material(dataset)
    sample_keys = [str(key) for key in material.get("sample_keys", [])]
    batch_keys = []
    batch_size = 1
    if input_specs:
        try:
            batch_size = max(int(input_specs[0]["shape"][0]), 1)
        except Exception:
            batch_size = 1
    if sample_keys:
        for item in range(min(batch_size, len(sample_keys))):
            batch_keys.append(sample_keys[(batch_offset * batch_size + item) % len(sample_keys)])
    batch_seed_payload = {
        "seed": material["seed"],
        "batch_offset": batch_offset,
        "batch_keys": batch_keys,
    }
    batch_seed = int(hashlib.sha256(json.dumps(batch_seed_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16], 16) % (2**63 - 1)
    tensors: list[torch.Tensor] = []
    for idx, spec in enumerate(input_specs):
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = str(spec.get("dtype", "float32")).lower()
        kind = str(spec.get("kind", "float")).lower()
        seed = (int(batch_seed) + idx * 9973) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if dtype in {"int64", "long"} or kind in {"tokens", "token_ids"}:
            tensor = torch.randint(0, 32000, shape, dtype=torch.long, generator=generator)
        elif kind == "adjacency":
            base = torch.eye(shape[-1], dtype=torch.float32)
            tensor = base.expand(shape).clone()
        else:
            tensor = torch.randn(shape, dtype=torch.float32, generator=generator)
            if float_dtype is not None:
                tensor = tensor.to(dtype=float_dtype)
        tensors.append(tensor.to(device))
    batch_material = dict(material)
    batch_material["batch_offset"] = int(batch_offset)
    batch_material["batch_keys"] = batch_keys
    batch_material["sample_keys_consumed"] = max(int(material.get("sample_keys_consumed", 0)), len(batch_keys))
    return tuple(tensors), batch_material


def load_model(model_path: Path):
    module_name = f"_nrp_model_{model_path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_model(), module


def ncu_executable() -> str | None:
    for name in ("ncu", "nv-nsight-cu-cli"):
        exe = shutil.which(name)
        if exe:
            return exe
    cuda_ncu = Path("/usr/local/cuda/bin/ncu")
    if cuda_ncu.exists():
        return str(cuda_ncu)
    return None


def write_ncu_probe_script(path: Path) -> None:
    path.write_text(
        r'''
import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F


def load_model(model_path):
    module_name = f"_ncu_probe_model_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--phase", required=True, choices=("infer", "train"))
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    args = parser.parse_args()
    model = load_model(args.model_file).cuda()
    x = torch.randn(tuple(json.loads(args.input_shape)), device="cuda")
    if args.phase == "infer":
        model.eval()
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()
        return
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[args.optimizer]
    opt = opt_cls(trainable, lr=1e-3) if trainable else None
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    y = model(x)
    loss = F.mse_loss(y.float(), torch.zeros_like(y, dtype=torch.float32))
    if loss.requires_grad:
        loss.backward()
    if opt is not None:
        opt.step()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )


def parse_ncu_csv(stdout: str) -> dict[str, float]:
    rows_by_kernel: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(line for line in stdout.splitlines() if line and not line.startswith("==")):
        metric_name = (row.get("Metric Name") or row.get("Metric Name ") or "").strip()
        raw_value = (row.get("Metric Value") or row.get("Metric Value ") or "").replace(",", "").strip()
        if not metric_name or not raw_value or raw_value.lower() == "n/a":
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        kernel_id = row.get("ID") or row.get("Kernel Name") or str(len(rows_by_kernel))
        bucket = rows_by_kernel.setdefault(kernel_id, {})
        if metric_name == "sm__warps_active.avg.pct_of_peak_sustained_active":
            bucket["occupancy"] = value
        elif metric_name == "gpu__time_duration.sum":
            bucket["duration"] = value
    samples = [item for item in rows_by_kernel.values() if "occupancy" in item]
    if not samples:
        raise RuntimeError("ncu did not return SM occupancy samples")
    total_duration = sum(max(item.get("duration", 0.0), 0.0) for item in samples)
    if total_duration > 0.0:
        avg = sum(item["occupancy"] * max(item.get("duration", 0.0), 0.0) for item in samples) / total_duration
    else:
        avg = sum(item["occupancy"] for item in samples) / len(samples)
    return {
        "avg_sm_occupancy_percent": float(avg),
        "peak_sm_occupancy_percent": float(max(item["occupancy"] for item in samples)),
        "kernel_count": float(len(samples)),
    }


def collect_ncu_occupancy(
    row: dict[str, Any],
    models_dir: Path,
    output_dir: Path,
    input_shape: tuple[int, ...],
    phase: str,
    optimizer: str,
) -> dict[str, Any]:
    exe = ncu_executable()
    if exe is None:
        raise RuntimeError("ncu or nv-nsight-cu-cli is required for true SM occupancy labels")
    probe = output_dir / f"_ncu_probe_{os.getpid()}_{phase}.py"
    write_ncu_probe_script(probe)
    model_path = models_dir / Path(row["model_file"]).name
    cmd = [
        exe,
        "--csv",
        "--target-processes",
        "all",
        "--metrics",
        "sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum",
        "--launch-count",
        "1",
        sys.executable,
        str(probe),
        "--model-file",
        str(model_path),
        "--input-shape",
        json.dumps(list(input_shape)),
        "--phase",
        phase,
        "--optimizer",
        optimizer,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    try:
        probe.unlink()
    except OSError:
        pass
    if proc.returncode != 0:
        raise RuntimeError(f"ncu failed for {phase}: {proc.stdout[-2000:]}")
    parsed = parse_ncu_csv(proc.stdout)
    parsed["source"] = "ncu_sm__warps_active.avg.pct_of_peak_sustained_active"
    return parsed


def normalize_hardware_id(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().lower()
    return raw or None


def hardware_metadata(device: torch.device, hardware_id: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    stable_id = normalize_hardware_id(hardware_id)
    if stable_id:
        meta["hardware_id"] = stable_id
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        meta.update(
            {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mib": props.total_memory / MI_B,
                "multi_processor_count": props.multi_processor_count,
            }
        )
        try:
            query = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=10,
            )
            meta["nvidia_smi"] = query.strip()
        except Exception as exc:
            meta["nvidia_smi_error"] = repr(exc)
    return meta


def compute_capability_tuple(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    props = torch.cuda.get_device_properties(device.index or 0)
    return int(props.major), int(props.minor)


def cudnn_op_backend(name: str) -> Any | None:
    return getattr(torch.backends.cudnn, name, None)


def set_backend_attr(changes: dict[str, Any], name: str, obj: Any | None, attr: str, value: Any) -> bool:
    if obj is None:
        return False
    try:
        old = getattr(obj, attr)
    except AttributeError:
        return False
    except Exception as exc:
        changes["errors"][name] = repr(exc)
        old = "<unreadable>"
    try:
        setattr(obj, attr, value)
        changes["set"][name] = {"old": old, "new": value}
        return True
    except Exception as exc:
        changes["errors"][name] = repr(exc)
        return False


def set_tf32_controls(enabled: bool) -> dict[str, Any]:
    target_precision = "tf32" if enabled else "ieee"
    matmul_precision = "high" if enabled else "highest"
    changes: dict[str, Any] = {"enabled": enabled, "api_style": "", "set": {}, "errors": {}}
    new_controls = (
        ("torch.backends.fp32_precision", torch.backends, "fp32_precision", target_precision),
        ("torch.backends.cuda.matmul.fp32_precision", getattr(torch.backends.cuda, "matmul", None), "fp32_precision", target_precision),
        ("torch.backends.cudnn.fp32_precision", torch.backends.cudnn, "fp32_precision", target_precision),
        ("torch.backends.cudnn.conv.fp32_precision", cudnn_op_backend("conv"), "fp32_precision", target_precision),
        ("torch.backends.cudnn.rnn.fp32_precision", cudnn_op_backend("rnn"), "fp32_precision", target_precision),
    )
    used_new = False
    for name, obj, attr, value in new_controls:
        used_new = set_backend_attr(changes, name, obj, attr, value) or used_new

    if used_new:
        changes["api_style"] = "fp32_precision"
        changes["effective_state"] = effective_tf32_state("fp32_precision")
        return changes

    changes["api_style"] = "legacy_allow_tf32"
    for name, obj, attr, value in (
        ("torch.backends.cuda.matmul.allow_tf32", getattr(torch.backends.cuda, "matmul", None), "allow_tf32", enabled),
        ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn, "allow_tf32", enabled),
    ):
        set_backend_attr(changes, name, obj, attr, value)
    try:
        torch.set_float32_matmul_precision(matmul_precision)
        changes["set"]["torch.set_float32_matmul_precision"] = matmul_precision
    except Exception as exc:
        changes["errors"]["torch.set_float32_matmul_precision"] = repr(exc)
    changes["effective_state"] = effective_tf32_state("legacy_allow_tf32")
    return changes


def read_backend_attr(state: dict[str, Any], name: str, obj: Any | None, attr: str) -> None:
    if obj is None:
        return
    try:
        state[name] = getattr(obj, attr)
    except AttributeError:
        return
    except Exception as exc:
        state[f"{name}:error"] = repr(exc)


def effective_tf32_state(api_style: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    state["api_style"] = api_style
    if api_style == "fp32_precision":
        controls = (
            ("torch.backends.fp32_precision", torch.backends, "fp32_precision"),
            ("torch.backends.cuda.matmul.fp32_precision", getattr(torch.backends.cuda, "matmul", None), "fp32_precision"),
            ("torch.backends.cudnn.fp32_precision", torch.backends.cudnn, "fp32_precision"),
            ("torch.backends.cudnn.conv.fp32_precision", cudnn_op_backend("conv"), "fp32_precision"),
            ("torch.backends.cudnn.rnn.fp32_precision", cudnn_op_backend("rnn"), "fp32_precision"),
        )
    else:
        controls = (
            ("torch.backends.cuda.matmul.allow_tf32", getattr(torch.backends.cuda, "matmul", None), "allow_tf32"),
            ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn, "allow_tf32"),
        )
    for name, obj, attr in controls:
        read_backend_attr(state, name, obj, attr)
    return state


def bf16_support_probe(device: torch.device, cc: tuple[int, int]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"compute_capability_policy_supported": device.type == "cpu" or cc >= (8, 0)}
    if device.type == "cpu":
        details["torch_cuda_is_bf16_supported"] = None
        return True, details
    if device.type != "cuda":
        details["torch_cuda_is_bf16_supported"] = None
        return False, details
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if probe is None:
        details["torch_cuda_is_bf16_supported"] = None
        return cc >= (8, 0), details
    try:
        supported = bool(probe())
        details["torch_cuda_is_bf16_supported"] = supported
        return supported, details
    except Exception as exc:
        details["torch_cuda_is_bf16_supported_error"] = repr(exc)
        return cc >= (8, 0), details


def make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device.type, enabled=True)
    except Exception:
        if device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=True)
    return None


def precision_runtime(config: str, device: torch.device, args: argparse.Namespace) -> PrecisionRuntime:
    config = normalize_precision_config(config)
    details: dict[str, Any] = {"requested_config": config}
    cc = compute_capability_tuple(device)
    details["compute_capability"] = f"{cc[0]}.{cc[1]}" if cc != (0, 0) else None

    if config == "fp32_ieee":
        details["tf32_controls"] = set_tf32_controls(False)
        return PrecisionRuntime(config=config, device_type=device.type, details=details)
    if config == "tf32":
        details["tf32_controls"] = set_tf32_controls(True)
        supported = device.type == "cuda" and cc >= (8, 0)
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            supported=supported,
            unsupported_reason=None if supported else "TF32 requires CUDA Ampere-or-newer hardware",
            details=details,
        )
    if config == "bf16_amp":
        details["tf32_controls"] = set_tf32_controls(False)
        supported, bf16_probe = bf16_support_probe(device, cc)
        details["bf16_probe"] = bf16_probe
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            autocast_dtype=torch.bfloat16,
            supported=supported,
            unsupported_reason=None if supported else "BF16 AMP requires CPU autocast or CUDA Ampere-or-newer hardware",
            details=details,
        )
    if config == "fp16_amp":
        details["tf32_controls"] = set_tf32_controls(False)
        supported = device.type == "cuda"
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            autocast_dtype=torch.float16,
            grad_scaler_enabled=supported,
            supported=supported,
            unsupported_reason=None if supported else "FP16 AMP profiling is enabled only for CUDA devices",
            details=details,
        )
    if config in FP8_PRECISIONS:
        details["tf32_controls"] = set_tf32_controls(False)
        details["fp8_recipe"] = (
            "hybrid E4M3 forward/E5M2 backward"
            if config == "fp8_te_hybrid"
            else f"{config.replace('fp8_', '').upper()} diagnostic"
        )
        details["fp8_te_min_compute_capability"] = "8.9"
        details["fp8_te_device_policy"] = "probe Transformer Engine backend, then require Ada-or-newer current-scaling support"
        if args.fp8_backend != "transformer_engine":
            return PrecisionRuntime(
                config=config,
                device_type=device.type,
                backend=args.fp8_backend,
                supported=False,
                unsupported_reason="FP8 backend disabled",
                fallback_policy="record_unsupported",
                details=details,
            )
        configure_transformer_engine_cuda_include(details)
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            details["transformer_engine_available"] = True
        except Exception as exc:
            details["transformer_engine_available"] = False
            details["transformer_engine_import_error"] = repr(exc)
            return PrecisionRuntime(
                config=config,
                device_type=device.type,
                backend="transformer_engine",
                supported=False,
                unsupported_reason="Transformer Engine is not available",
                fallback_policy="record_unsupported",
                details=details,
            )
        try:
            details["transformer_engine_fp8_available"] = transformer_engine_availability(te, "is_fp8_available")
        except Exception as exc:
            details["transformer_engine_fp8_available_error"] = repr(exc)
            details["transformer_engine_fp8_available"] = None
        format_map = {
            "fp8_te_hybrid": recipe.Format.HYBRID,
            "fp8_e4m3": recipe.Format.E4M3,
            "fp8_e5m2": recipe.Format.E5M2,
        }
        te_fp8_available = details.get("transformer_engine_fp8_available")
        supported = device.type == "cuda" and cc >= (8, 9) and te_fp8_available is not False
        unsupported_reason = None
        if not supported:
            if device.type != "cuda" or cc < (8, 9):
                unsupported_reason = "FP8 Transformer Engine profiling requires Ada-or-newer CUDA hardware (SM 8.9+)"
            else:
                unsupported_reason = "Transformer Engine reports FP8 is not available"
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            backend="transformer_engine",
            supported=supported,
            unsupported_reason=unsupported_reason,
            fallback_policy="record_unsupported_generated_ops",
            te_recipe=recipe.DelayedScaling(fp8_format=format_map[config]),
            te_recipe_name=f"DelayedScaling({config})",
            model_dtype=torch.bfloat16,
            input_dtype=torch.bfloat16,
            details=details,
        )
    if config == "nvfp4_te":
        details["tf32_controls"] = set_tf32_controls(False)
        details["fp4_recipe"] = "NVFP4 block scaling"
        details["nvfp4_te_min_compute_capability"] = "10.0"
        details["nvfp4_te_device_policy"] = "probe Transformer Engine backend, require Blackwell-class hardware, and use BF16 inputs"
        if args.fp8_backend != "transformer_engine":
            return PrecisionRuntime(
                config=config,
                device_type=device.type,
                backend=args.fp8_backend,
                supported=False,
                unsupported_reason="Transformer Engine backend disabled",
                fallback_policy="record_unsupported",
                details=details,
            )
        configure_transformer_engine_cuda_include(details)
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            details["transformer_engine_available"] = True
        except Exception as exc:
            details["transformer_engine_available"] = False
            details["transformer_engine_import_error"] = repr(exc)
            return PrecisionRuntime(
                config=config,
                device_type=device.type,
                backend="transformer_engine",
                supported=False,
                unsupported_reason="Transformer Engine is not available",
                fallback_policy="record_unsupported",
                details=details,
            )
        try:
            details["transformer_engine_nvfp4_available"] = transformer_engine_availability(te, "is_nvfp4_available")
        except Exception as exc:
            details["transformer_engine_nvfp4_available_error"] = repr(exc)
            details["transformer_engine_nvfp4_available"] = None
        bf16_supported, bf16_probe = bf16_support_probe(device, cc)
        details["bf16_probe"] = bf16_probe
        te_nvfp4_available = details.get("transformer_engine_nvfp4_available")
        supported = device.type == "cuda" and cc >= (10, 0) and bf16_supported and te_nvfp4_available is not False
        unsupported_reason = None
        if not supported:
            if device.type != "cuda" or cc < (10, 0):
                unsupported_reason = "NVFP4 Transformer Engine profiling requires Blackwell-class CUDA hardware (SM 10.0+)"
            elif not bf16_supported:
                unsupported_reason = "NVFP4 Transformer Engine profiling requires BF16 inputs/gradients"
            else:
                unsupported_reason = "Transformer Engine reports NVFP4 is not available"
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            backend="transformer_engine",
            supported=supported,
            unsupported_reason=unsupported_reason,
            fallback_policy="record_unsupported_generated_ops",
            te_recipe=recipe.NVFP4BlockScaling(),
            te_recipe_name="NVFP4BlockScaling",
            model_dtype=torch.bfloat16,
            input_dtype=torch.bfloat16,
            details=details,
        )
    raise ValueError(config)


def resolve_auto_precision_configs(device: torch.device, args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    configs: list[str] = []
    probes: dict[str, Any] = {}
    for config in (*BASE_AUTO_PRECISIONS, "fp8_te_hybrid", "nvfp4_te"):
        runtime = precision_runtime(config, device, args)
        probes[config] = runtime.to_metadata()
        if runtime.supported:
            configs.append(config)
    return configs, probes


class NvmlSampler:
    def __init__(self, device_index: int, interval: float) -> None:
        self.device_index = device_index
        self.interval = interval
        self.samples: list[tuple[float, float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self.available = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self.available = False

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (len(ordered) - 1) * pct / 100.0
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return float(ordered[lo])
        frac = rank - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    @classmethod
    def _series_stats(cls, values: list[float]) -> dict[str, float]:
        if not values:
            return {"avg": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "peak": 0.0}
        avg = float(sum(values) / len(values))
        var = float(sum((value - avg) ** 2 for value in values) / len(values))
        return {
            "avg": avg,
            "std": math.sqrt(max(var, 0.0)),
            "p50": cls._percentile(values, 50.0),
            "p95": cls._percentile(values, 95.0),
            "peak": float(max(values)),
        }

    def stop(self, *, min_phase_seconds: float = 0.0, min_sampler_samples: int = 0) -> SampleStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            flags = ["low_sample_count"] if min_sampler_samples > 0 else []
            if min_phase_seconds > 0:
                flags.append("short_measurement")
            return SampleStats(source="nvml" if self.available else "none", reliability_flags=flags)
        sm = [sample[0] for sample in self.samples]
        memory_controller_util = [sample[1] for sample in self.samples]
        mem_used = [sample[2] for sample in self.samples]
        mem_used_percent = [sample[3] for sample in self.samples]
        timestamps = [sample[4] for sample in self.samples]
        duration_ms = max((max(timestamps) - min(timestamps)) * 1000.0, 0.0) if len(timestamps) > 1 else 0.0
        sm_stats = self._series_stats(sm)
        mem_util_stats = self._series_stats(memory_controller_util)
        mem_usage_stats = self._series_stats(mem_used)
        flags: list[str] = []
        if len(self.samples) < min_sampler_samples:
            flags.append("low_sample_count")
        if duration_ms < min_phase_seconds * 1000.0:
            flags.append("short_measurement")
        if sm_stats["std"] > 20.0:
            flags.append("high_sm_variance")
        return SampleStats(
            avg_sm_util=sm_stats["avg"],
            sm_util_std=sm_stats["std"],
            p50_sm_util=sm_stats["p50"],
            p95_sm_util=sm_stats["p95"],
            avg_mem_util=mem_util_stats["avg"],
            mem_util_std=mem_util_stats["std"],
            p50_mem_util=mem_util_stats["p50"],
            p95_mem_util=mem_util_stats["p95"],
            avg_mem_usage=mem_usage_stats["avg"],
            mem_usage_std=mem_usage_stats["std"],
            p50_mem_usage=mem_usage_stats["p50"],
            p95_mem_usage=mem_usage_stats["p95"],
            peak_sm_util=sm_stats["peak"],
            peak_mem_util=mem_util_stats["peak"],
            peak_mem_usage=mem_usage_stats["peak"],
            avg_vram_used_percent=float(sum(mem_used_percent) / len(mem_used_percent)),
            peak_vram_used_percent=float(max(mem_used_percent)),
            sample_count=len(self.samples),
            measurement_duration_ms=duration_ms,
            reliability_flags=flags,
            source="nvml",
        )

    def _run(self) -> None:
        assert self._nvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                mem_used_mib = float(mem.used / MI_B)
                mem_used_percent = float(100.0 * mem.used / max(mem.total, 1))
                self.samples.append((float(util.gpu), float(util.memory), mem_used_mib, mem_used_percent, time.time()))
            except Exception:
                pass
            time.sleep(self.interval)


def fallback_memory_stats(device: torch.device) -> SampleStats:
    if device.type == "cuda" and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / MI_B
        return SampleStats(
            avg_mem_usage=float(peak),
            p50_mem_usage=float(peak),
            p95_mem_usage=float(peak),
            peak_mem_usage=float(peak),
            reliability_flags=["fallback_memory_source"],
            source="torch",
        )
    return SampleStats(source="none")


def clear_cuda_profile_state(device: torch.device, *, reset_peak: bool = False) -> None:
    gc.collect()
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        pass
    torch.cuda.empty_cache()
    if reset_peak:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            pass


def label_string(time_ms_per_sample: float, stats: SampleStats) -> str:
    fields = [
        time_ms_per_sample,
        stats.avg_sm_util,
        stats.avg_mem_util,
        stats.avg_mem_usage,
        stats.peak_sm_util,
        stats.peak_mem_util,
        stats.peak_mem_usage,
    ]
    return "|".join(f"{value:.6g}" for value in fields)


def phase_label_v2(
    detail: dict[str, Any],
    data_type: dict[str, Any],
    occupancy: dict[str, Any],
    optimizer: str | None = None,
) -> dict[str, Any]:
    sampler = detail["sampler"]
    avg_sm_util = float(sampler.get("avg_sm_util", 0.0))
    peak_sm_util = float(sampler.get("peak_sm_util", 0.0))
    avg_memory_controller = float(sampler.get("avg_mem_util", 0.0))
    peak_memory_controller = float(sampler.get("peak_mem_util", 0.0))
    avg_vram_used = float(sampler.get("avg_mem_usage", 0.0))
    peak_vram_used = float(sampler.get("peak_mem_usage", 0.0))
    label = {
        "time_1_epoch_ms": float(detail["mean_iter_ms"]),
        "avg_device_memory_usage_mib": avg_vram_used,
        "peak_device_memory_usage_mib": peak_vram_used,
        "avg_vram_used_mib": avg_vram_used,
        "peak_vram_used_mib": peak_vram_used,
        "p50_vram_used_mib": float(sampler.get("p50_mem_usage", 0.0)),
        "p95_vram_used_mib": float(sampler.get("p95_mem_usage", 0.0)),
        "vram_used_std_mib": float(sampler.get("mem_usage_std", 0.0)),
        "avg_vram_used_percent": float(sampler.get("avg_vram_used_percent", 0.0)),
        "peak_vram_used_percent": float(sampler.get("peak_vram_used_percent", 0.0)),
        "peak_torch_allocated_mib": float(detail.get("peak_torch_allocated_mib", 0.0)),
        "peak_torch_reserved_mib": float(detail.get("peak_torch_reserved_mib", 0.0)),
        "avg_host_memory_usage_mib": float(detail["avg_host_memory_usage_mib"]),
        "peak_host_memory_usage_mib": float(detail["peak_host_memory_usage_mib"]),
        "compile_time_ms": 0.0,
        "warmup_time_ms": float(detail.get("warmup_time_ms", 0.0)),
        "compile_warmup_time_ms": float(detail.get("warmup_time_ms", 0.0)),
        "measurement_steps": int(detail.get("measurement_steps", 0) or 0),
        "measurement_duration_ms": float(sampler.get("measurement_duration_ms", 0.0)),
        "sampler_samples": int(sampler.get("sample_count", 0) or 0),
        "resource_reliability_flags": list(sampler.get("reliability_flags", []) or []),
        "resource_profile_mode": detail.get("resource_profile_mode", "compat"),
        "avg_sm_occupancy_percent": float(occupancy["avg_sm_occupancy_percent"]),
        "peak_sm_occupancy_percent": float(occupancy["peak_sm_occupancy_percent"]),
        "sm_occupancy_source": occupancy["source"],
        "sm_occupancy_kernel_count": int(occupancy.get("kernel_count", 0)),
        "avg_sm_utilization_percent": avg_sm_util,
        "p50_sm_utilization_percent": float(sampler.get("p50_sm_util", 0.0)),
        "p95_sm_utilization_percent": float(sampler.get("p95_sm_util", 0.0)),
        "sm_utilization_std_percent": float(sampler.get("sm_util_std", 0.0)),
        "peak_sm_utilization_percent": peak_sm_util,
        "avg_memory_controller_utilization_percent": avg_memory_controller,
        "p50_memory_controller_utilization_percent": float(sampler.get("p50_mem_util", 0.0)),
        "p95_memory_controller_utilization_percent": float(sampler.get("p95_mem_util", 0.0)),
        "memory_controller_utilization_std_percent": float(sampler.get("mem_util_std", 0.0)),
        "peak_memory_controller_utilization_percent": peak_memory_controller,
        "dram_activity_percent": avg_memory_controller,
        "peak_dram_activity_percent": peak_memory_controller,
        "memory_telemetry_source": sampler.get("source", "none"),
        "data_type": data_type,
    }
    if optimizer is not None:
        label["optimizer"] = optimizer
    return label


def nvml_proxy_occupancy(detail: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    sampler = detail["sampler"]
    out = {
        "avg_sm_occupancy_percent": sampler.get("avg_sm_util", 0.0),
        "peak_sm_occupancy_percent": sampler.get("peak_sm_util", 0.0),
        "source": "nvml_utilization_proxy",
        "kernel_count": 0,
    }
    if reason:
        out["fallback_reason"] = reason
    return out


def torch_peak_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        return (
            float(torch.cuda.max_memory_allocated(device)) / MI_B,
            float(torch.cuda.max_memory_reserved(device)) / MI_B,
        )
    except RuntimeError:
        return 0.0, 0.0


def complete_resource_stats(device: torch.device, stats: SampleStats) -> SampleStats:
    if stats.source == "none" or stats.peak_mem_usage <= 0:
        fallback = fallback_memory_stats(device)
        if fallback.peak_mem_usage > stats.peak_mem_usage:
            stats.peak_mem_usage = fallback.peak_mem_usage
            stats.avg_mem_usage = fallback.avg_mem_usage
            stats.p50_mem_usage = fallback.p50_mem_usage
            stats.p95_mem_usage = fallback.p95_mem_usage
            stats.source = fallback.source
            for flag in fallback.reliability_flags:
                if flag not in stats.reliability_flags:
                    stats.reliability_flags.append(flag)
    return stats


def timed_phase(
    phase: str,
    fn: Callable[[], torch.Tensor],
    repeats: int,
    warmup: int,
    batch_size: int,
    device: torch.device,
    sample_interval: float,
    min_phase_seconds: float = 0.0,
    min_sampler_samples: int = 0,
    mode: str = "compat",
) -> tuple[str, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    host_before = host_memory_mib()
    warmup_t0 = time.perf_counter()
    for _ in range(warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_time_ms = (time.perf_counter() - warmup_t0) * 1000.0

    if mode == "sustained":
        sampler = NvmlSampler(device.index or 0, sample_interval) if device.type == "cuda" else None
        if sampler:
            sampler.start()
        wall_t0 = time.perf_counter()
        total_gpu_ms = 0.0
        steps = 0
        try:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                while True:
                    _ = fn()
                    steps += 1
                    elapsed = time.perf_counter() - wall_t0
                    sample_count = len(sampler.samples) if sampler else 0
                    enough_steps = steps >= repeats
                    enough_time = min_phase_seconds <= 0 or elapsed >= min_phase_seconds
                    enough_samples = not sampler or min_sampler_samples <= 0 or sample_count >= min_sampler_samples
                    if enough_steps and enough_time and enough_samples:
                        break
                end.record()
                torch.cuda.synchronize(device)
                total_gpu_ms = float(start.elapsed_time(end))
            else:
                while True:
                    _ = fn()
                    steps += 1
                    elapsed = time.perf_counter() - wall_t0
                    enough_steps = steps >= repeats
                    enough_time = min_phase_seconds <= 0 or elapsed >= min_phase_seconds
                    if enough_steps and enough_time:
                        break
                total_gpu_ms = (time.perf_counter() - wall_t0) * 1000.0
        finally:
            host_after = host_memory_mib()
            stats = (
                sampler.stop(min_phase_seconds=min_phase_seconds, min_sampler_samples=min_sampler_samples)
                if sampler
                else SampleStats(source="none")
            )
        total_wall_ms = (time.perf_counter() - wall_t0) * 1000.0
        stats = complete_resource_stats(device, stats)
        mean_iter_ms = total_gpu_ms / max(steps, 1)
        mean_wall_iter_ms = total_wall_ms / max(steps, 1)
        time_ms_per_sample = mean_iter_ms / max(batch_size, 1)
        peak_torch_allocated_mib, peak_torch_reserved_mib = torch_peak_memory_mib(device)
        return label_string(time_ms_per_sample, stats), {
            "phase": phase,
            "resource_profile_mode": "sustained",
            "measurement_steps": int(steps),
            "mean_iter_ms": mean_iter_ms,
            "mean_wall_iter_ms": mean_wall_iter_ms,
            "total_gpu_ms": total_gpu_ms,
            "total_wall_ms": total_wall_ms,
            "time_ms_per_sample": time_ms_per_sample,
            "raw_iter_ms": [mean_iter_ms],
            "raw_wall_iter_ms": [mean_wall_iter_ms],
            "sampler": stats.__dict__,
            "warmup_time_ms": warmup_time_ms,
            "peak_torch_allocated_mib": peak_torch_allocated_mib,
            "peak_torch_reserved_mib": peak_torch_reserved_mib,
            "avg_host_memory_usage_mib": (host_before["current_mib"] + host_after["current_mib"]) / 2.0,
            "peak_host_memory_usage_mib": max(host_before["peak_mib"], host_after["peak_mib"]),
        }

    sampler = NvmlSampler(device.index or 0, sample_interval) if device.type == "cuda" else None
    if sampler:
        sampler.start()
    raw_ms = []
    raw_wall_ms = []
    for _ in range(repeats):
        wall_t0 = time.perf_counter()
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize(device)
            raw_ms.append(float(start.elapsed_time(end)))
        else:
            _ = fn()
            raw_ms.append((time.perf_counter() - wall_t0) * 1000.0)
        raw_wall_ms.append((time.perf_counter() - wall_t0) * 1000.0)
    host_after = host_memory_mib()
    stats = complete_resource_stats(
        device,
        sampler.stop(min_phase_seconds=0.0, min_sampler_samples=0) if sampler else SampleStats(source="none"),
    )
    mean_iter_ms = float(sum(raw_ms) / max(len(raw_ms), 1))
    mean_wall_iter_ms = float(sum(raw_wall_ms) / max(len(raw_wall_ms), 1))
    time_ms_per_sample = mean_iter_ms / max(batch_size, 1)
    peak_torch_allocated_mib, peak_torch_reserved_mib = torch_peak_memory_mib(device)
    return label_string(time_ms_per_sample, stats), {
        "phase": phase,
        "resource_profile_mode": "compat",
        "measurement_steps": int(repeats),
        "mean_iter_ms": mean_iter_ms,
        "mean_wall_iter_ms": mean_wall_iter_ms,
        "total_gpu_ms": float(sum(raw_ms)),
        "total_wall_ms": float(sum(raw_wall_ms)),
        "time_ms_per_sample": time_ms_per_sample,
        "raw_iter_ms": raw_ms,
        "raw_wall_iter_ms": raw_wall_ms,
        "sampler": stats.__dict__,
        "warmup_time_ms": warmup_time_ms,
        "peak_torch_allocated_mib": peak_torch_allocated_mib,
        "peak_torch_reserved_mib": peak_torch_reserved_mib,
        "avg_host_memory_usage_mib": (host_before["current_mib"] + host_after["current_mib"]) / 2.0,
        "peak_host_memory_usage_mib": max(host_before["peak_mib"], host_after["peak_mib"]),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = float(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, math.sqrt(max(variance, 0.0))


def timed_measured_epoch_phase(
    phase: str,
    fn: Callable[[], torch.Tensor],
    *,
    steps_per_epoch: int,
    warmup_epochs: int,
    measured_epochs: int,
    batch_size: int,
    device: torch.device,
    sample_interval: float,
    min_phase_seconds: float = 0.0,
    min_sampler_samples: int = 0,
    mode: str = "sustained",
) -> tuple[str, dict[str, Any]]:
    steps_per_epoch = max(int(steps_per_epoch), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    measured_epochs = max(int(measured_epochs), 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    host_before = host_memory_mib()

    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_t0 = time.perf_counter()
    for _ in range(warmup_steps):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    warmup_time_ms = (time.perf_counter() - warmup_t0) * 1000.0

    sampler = NvmlSampler(device.index or 0, sample_interval) if device.type == "cuda" else None
    if sampler:
        sampler.start()
    epoch_wall_ms: list[float] = []
    epoch_gpu_ms: list[float] = []
    measured_t0 = time.perf_counter()
    try:
        for _epoch in range(measured_epochs):
            wall_t0 = time.perf_counter()
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(steps_per_epoch):
                    _ = fn()
                end.record()
                torch.cuda.synchronize(device)
                epoch_gpu_ms.append(float(start.elapsed_time(end)))
            else:
                for _ in range(steps_per_epoch):
                    _ = fn()
                epoch_gpu_ms.append((time.perf_counter() - wall_t0) * 1000.0)
            epoch_wall_ms.append((time.perf_counter() - wall_t0) * 1000.0)
    finally:
        host_after = host_memory_mib()
        stats = (
            sampler.stop(min_phase_seconds=min_phase_seconds, min_sampler_samples=min_sampler_samples)
            if sampler
            else SampleStats(source="none")
        )
    measured_wall_ms = (time.perf_counter() - measured_t0) * 1000.0
    stats = complete_resource_stats(device, stats)
    steps = steps_per_epoch * measured_epochs
    total_wall_ms = float(sum(epoch_wall_ms)) or measured_wall_ms
    total_gpu_ms = float(sum(epoch_gpu_ms))
    mean_iter_ms = total_gpu_ms / max(steps, 1)
    mean_wall_iter_ms = total_wall_ms / max(steps, 1)
    epoch_wall_mean_ms, epoch_wall_std_ms = _mean_std(epoch_wall_ms)
    epoch_gpu_mean_ms, epoch_gpu_std_ms = _mean_std(epoch_gpu_ms)
    time_ms_per_sample = mean_iter_ms / max(batch_size, 1)
    peak_torch_allocated_mib, peak_torch_reserved_mib = torch_peak_memory_mib(device)
    return label_string(time_ms_per_sample, stats), {
        "phase": phase,
        "resource_profile_mode": mode,
        "epoch_time_source": "measured_epochs",
        "measurement_steps": int(steps),
        "mean_iter_ms": mean_iter_ms,
        "mean_wall_iter_ms": mean_wall_iter_ms,
        "total_gpu_ms": total_gpu_ms,
        "total_wall_ms": total_wall_ms,
        "time_ms_per_sample": time_ms_per_sample,
        "raw_iter_ms": [value / steps_per_epoch for value in epoch_gpu_ms],
        "raw_wall_iter_ms": [value / steps_per_epoch for value in epoch_wall_ms],
        "epoch_wall_ms": epoch_wall_ms,
        "epoch_gpu_ms": epoch_gpu_ms,
        "measured_epochs": int(measured_epochs),
        "warmup_epochs": int(warmup_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": int(warmup_steps),
        "measured_epoch_wall_mean_ms": epoch_wall_mean_ms,
        "measured_epoch_wall_std_ms": epoch_wall_std_ms,
        "measured_epoch_gpu_mean_ms": epoch_gpu_mean_ms,
        "measured_epoch_gpu_std_ms": epoch_gpu_std_ms,
        "measured_wall_ms": measured_wall_ms,
        "sampler": stats.__dict__,
        "warmup_time_ms": warmup_time_ms,
        "peak_torch_allocated_mib": peak_torch_allocated_mib,
        "peak_torch_reserved_mib": peak_torch_reserved_mib,
        "avg_host_memory_usage_mib": (host_before["current_mib"] + host_after["current_mib"]) / 2.0,
        "peak_host_memory_usage_mib": max(host_before["peak_mib"], host_after["peak_mib"]),
    }


def profile_model(
    row: dict[str, Any],
    models_dir: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    clear_cuda_profile_state(device, reset_peak=True)
    precision_config = normalize_precision_config(str(row.get("precision_config", DEFAULT_PRECISION_CONFIG)))
    runtime = precision_runtime(precision_config, device, args)
    model_path = models_dir / Path(row["model_file"]).name
    model: torch.nn.Module | None = None
    inputs: tuple[torch.Tensor, ...] = ()
    optimizer: torch.optim.Optimizer | None = None
    scaler: Any = None
    trainable_params: list[torch.nn.Parameter] = []
    model, model_module = load_model(model_path)
    model = model.to(device)
    dataset_spec_error: str | None = None
    dataset_spec = profile_dataset_spec_from_workload(row)
    if not dataset_spec:
        try:
            dataset_spec = load_profile_dataset_spec(str(row["model_id"]), args.profile_dataset_dir)
        except Exception as exc:
            dataset_spec = {}
            dataset_spec_error = repr(exc)

    profile_config_error: str | None = None
    try:
        train_repeats = positive_int(dataset_spec.get("train_repeats"), args.train_repeats, "train_repeats")
        infer_repeats = positive_int(dataset_spec.get("infer_repeats"), args.infer_repeats, "infer_repeats")
        workload_training = dataset_spec.get("training") if isinstance(dataset_spec.get("training"), dict) else {}
        default_batch_size = int((row.get("input_shape") or [1])[0] or 1)
        requested_batch_size = positive_int(workload_training.get("batch_size"), default_batch_size, "training.batch_size")
        input_specs = normalize_input_specs(row, dataset_spec)
        if row.get("workload_spec"):
            input_specs = generated_model_input_specs(model_module, input_specs, requested_batch_size)
        input_shape = tuple(int(dim) for dim in input_specs[0]["shape"])
        batch_size = int(input_shape[0]) if input_shape else 1
        steps_per_epoch, effective_batch_size, train_sample_count = steps_per_epoch_for_profile(row, dataset_spec, batch_size)
        optimizer_name = str(workload_training.get("optimizer") or args.optimizer).lower()
        if optimizer_name not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unsupported optimizer {optimizer_name!r}")
    except Exception as exc:
        profile_config_error = repr(exc)
        input_specs = [{"name": "input0", "shape": list(row["input_shape"]), "dtype": "float32", "kind": "float"}]
        input_shape = tuple(int(dim) for dim in row["input_shape"])
        batch_size = int(input_shape[0]) if input_shape else 1
        train_repeats = int(args.train_repeats)
        infer_repeats = int(args.infer_repeats)
        steps_per_epoch, effective_batch_size, train_sample_count = steps_per_epoch_for_profile(row, dataset_spec, batch_size)
        optimizer_name = str(args.optimizer).lower()

    result: dict[str, Any] = {
        "model_id": row["model_id"],
        "graph_id": row.get("graph_id", row["model_id"]),
        "profile_point_id": profile_point_id(row),
        "stem": row.get("original_stem", row.get("stem", row["model_id"])),
        "status": "ok",
        "input_shape": list(input_shape),
        "input_specs": input_specs,
        "batch_size": batch_size,
        "model_file": row["model_file"],
        "label_file": label_file_for_row(row, precision_config),
        "hardware_id": normalize_hardware_id(args.hardware_id),
        "optimizer": optimizer_name,
        "precision_config": precision_config,
        "precision": runtime.to_metadata(),
        "resource_profile_mode": args.resource_profile_mode,
        "workload_spec": row.get("workload_spec"),
        "profile_dataset": {
            "source": str(dataset_spec.get("source") or ("profile_dataset_dir" if dataset_spec else "synthetic_cli")),
            "train_repeats": train_repeats,
            "infer_repeats": infer_repeats,
            "label_time_mode": args.label_time_mode,
            "steps_per_epoch": steps_per_epoch,
            "effective_batch_size": effective_batch_size,
            "train_sample_count": train_sample_count,
            "dataset_id": (dataset_spec.get("dataset") or {}).get("dataset_id") if isinstance(dataset_spec.get("dataset"), dict) else None,
            "subset_id": (dataset_spec.get("dataset") or {}).get("subset_id") if isinstance(dataset_spec.get("dataset"), dict) else None,
            "real_dataloader_backed": bool((dataset_spec.get("dataset") or {}).get("real_dataloader_backed"))
            if isinstance(dataset_spec.get("dataset"), dict)
            else False,
        },
    }
    try:
        if dataset_spec_error is not None:
            result.update({"status": "error", "error": dataset_spec_error})
            return result
        if profile_config_error is not None:
            result.update({"status": "error", "error": profile_config_error})
            return result
        workload_dataset = (row.get("workload_spec") or {}).get("dataset") if isinstance(row.get("workload_spec"), dict) else {}
        if row.get("workload_spec") and not bool(workload_dataset.get("real_dataloader_backed")) and not args.allow_synthetic_workload_inputs:
            result.update(
                {
                    "status": "unsupported_real_dataloader",
                    "error": "WorkloadSpec has no real dataloader adapter; pass --allow-synthetic-workload-inputs only for smoke tests",
                }
            )
            return result
        if not runtime.supported:
            result.update({"status": "unsupported_precision", "error": runtime.unsupported_reason})
            return result
        if precision_config in TE_LOW_PRECISIONS:
            enable_te = getattr(model, "enable_transformer_engine", None)
            if not callable(enable_te):
                result.update(
                    {
                        "status": "unsupported_low_precision_op",
                        "error": "model does not expose generated GraphModel Transformer Engine rewrite hooks",
                    }
                )
                result["precision"]["fallback_policy"] = "record_unsupported_low_precision_op"
                return result
            reasons = enable_te(
                precision_config,
                params_dtype=runtime.model_dtype or torch.bfloat16,
                device=device,
            )
            if reasons:
                result.update({"status": "unsupported_low_precision_op", "error": "; ".join(str(item) for item in reasons)})
                result["precision"]["fallback_policy"] = "record_unsupported_low_precision_op"
                result["precision"]["unsupported_low_precision_reasons"] = reasons
                return result
            result["precision"]["generated_runtime"] = "transformer_engine"
        real_workload_inputs = bool(row.get("workload_spec") and workload_dataset.get("real_dataloader_backed"))
        batch_counters = {"infer": 0, "train": 0}
        latest_dataset_material: dict[str, Any] = {}
        if real_workload_inputs:
            inputs, dataset_material = make_real_dataset_inputs(input_specs, dataset_spec, device, runtime.input_dtype)
            latest_dataset_material = dict(dataset_material)
            result["profile_dataset"].update(
                {
                    "adapter": "local_subset_key_tensor",
                    "raw_dir": dataset_material["raw_dir"],
                    "subset_mask_path": dataset_material["subset_mask_path"],
                    "subset_sample_count": dataset_material["sample_count"],
                    "sample_keys_consumed": dataset_material["sample_keys_consumed"],
                    "sample_fingerprints": dataset_material["sample_fingerprints"],
                }
            )
        else:
            inputs = make_profile_inputs(input_specs, device, runtime.input_dtype)

        def phase_inputs(phase: str) -> tuple[torch.Tensor, ...]:
            nonlocal inputs, latest_dataset_material
            if not real_workload_inputs:
                return inputs
            offset = batch_counters[phase]
            batch_counters[phase] += 1
            inputs, dataset_material = make_real_dataset_inputs(
                input_specs,
                dataset_spec,
                device,
                runtime.input_dtype,
                batch_offset=offset,
            )
            latest_dataset_material = dict(dataset_material)
            return inputs

        model.eval()

        def infer_fn() -> torch.Tensor:
            phase_batch = phase_inputs("infer")
            with torch.no_grad():
                with runtime.autocast():
                    return model(*phase_batch)

        infer_label, infer_detail = timed_phase(
            "infer",
            infer_fn,
            infer_repeats,
            args.warmup,
            batch_size,
            device,
            args.sample_interval,
            min_phase_seconds=args.min_phase_seconds,
            min_sampler_samples=args.min_sampler_samples,
            mode=args.resource_profile_mode,
        )

        model.train()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
        optimizer = optimizer_cls(trainable_params, lr=1e-3) if trainable_params else None
        scaler = make_grad_scaler(device, runtime.grad_scaler_enabled)
        if scaler is not None:
            result["precision"]["grad_scaler_enabled"] = bool(scaler.is_enabled())
            if scaler.is_enabled():
                result["precision"]["grad_scaler_initial_scale"] = float(scaler.get_scale())

        def train_fn() -> torch.Tensor:
            phase_batch = phase_inputs("train")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                out = model(*phase_batch)
                loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
            if loss.requires_grad:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            return loss.detach()

        if args.label_time_mode == "measured_epochs":
            train_label, train_detail = timed_measured_epoch_phase(
                "train",
                train_fn,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=args.time_label_warmup_epochs,
                measured_epochs=args.time_label_measured_epochs,
                batch_size=batch_size,
                device=device,
                sample_interval=args.sample_interval,
                min_phase_seconds=args.min_phase_seconds,
                min_sampler_samples=args.min_sampler_samples,
                mode=args.resource_profile_mode,
            )
        else:
            train_label, train_detail = timed_phase(
                "train",
                train_fn,
                train_repeats,
                args.warmup,
                batch_size,
                device,
                args.sample_interval,
                min_phase_seconds=args.min_phase_seconds,
                min_sampler_samples=args.min_sampler_samples,
                mode=args.resource_profile_mode,
            )
        result["resource_profile_mode"] = args.resource_profile_mode
        if real_workload_inputs:
            total_batches = int(batch_counters["infer"] + batch_counters["train"])
            sample_count = int(latest_dataset_material.get("sample_count", 0) or 0)
            result["profile_dataset"].update(
                {
                    "measurement_batches": {
                        "infer": int(infer_detail.get("measurement_steps", infer_repeats)),
                        "train": int(train_detail.get("measurement_steps", train_repeats)),
                    },
                    "generated_batches": dict(batch_counters),
                    "warmup_batches_per_phase": int(args.warmup),
                    "warmup_batches": {
                        "infer": int(args.warmup),
                        "train": int(train_detail.get("warmup_steps", args.warmup)),
                    },
                    "label_time_mode": args.label_time_mode,
                    "sample_batches_generated": total_batches,
                    "sample_keys_consumed": min(sample_count, max(total_batches * max(batch_size, 1), 0)) if sample_count else 0,
                    "last_batch_offset": latest_dataset_material.get("batch_offset"),
                    "last_batch_keys": latest_dataset_material.get("batch_keys", []),
                }
            )
        if scaler is not None and scaler.is_enabled():
            result["precision"]["grad_scaler_final_scale"] = float(scaler.get_scale())
        result.update(
            {
                "label": {"train": train_label, "infer": infer_label},
                "details": {"train": train_detail, "infer": infer_detail},
            }
        )
        input_dtypes = sorted({str(tensor.dtype).replace("torch.", "") for tensor in inputs})
        gradient_dtypes = sorted(
            {str(param.grad.dtype).replace("torch.", "") for param in model.parameters() if param.grad is not None}
        )
        data_type = {
            "input_dtypes": input_dtypes,
            "parameter_dtypes": sorted({str(param.dtype).replace("torch.", "") for param in model.parameters()}),
            "forward_input_dtypes": input_dtypes,
            "forward_autocast_dtype": str(runtime.autocast_dtype).replace("torch.", "") if runtime.autocast_dtype is not None else None,
            "forward_parameter_dtypes": sorted({str(param.dtype).replace("torch.", "") for param in model.parameters()}),
            "backward_gradient_dtypes": gradient_dtypes,
        }
        if args.sm_occupancy_source == "ncu":
            try:
                infer_occupancy = collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "infer", optimizer_name)
                train_occupancy = collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "train", optimizer_name)
            except RuntimeError as exc:
                reason = repr(exc)
                if "ERR_NVGPUCTRPERM" not in reason and "ncu or nv-nsight-cu-cli is required" not in reason:
                    raise
                result["label_v2_occupancy_warning"] = reason
                infer_occupancy = nvml_proxy_occupancy(infer_detail, reason=reason)
                train_occupancy = nvml_proxy_occupancy(train_detail, reason=reason)
        else:
            infer_occupancy = nvml_proxy_occupancy(infer_detail)
            train_occupancy = nvml_proxy_occupancy(train_detail)
        result["label_v2"] = {
            "train": phase_label_v2(train_detail, data_type, train_occupancy, optimizer_name),
            "infer": phase_label_v2(infer_detail, data_type, infer_occupancy),
        }
        if args.resource_audit_source == "ncu":
            try:
                result["resource_audit"] = {
                    "source": "ncu",
                    "metrics": [
                        "sm__warps_active.avg.pct_of_peak_sustained_active",
                        "gpu__time_duration.sum",
                    ],
                    "infer": collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "infer", optimizer_name),
                    "train": collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "train", optimizer_name),
                }
            except RuntimeError as exc:
                result["resource_audit"] = {
                    "source": "ncu",
                    "status": "unavailable",
                    "error": repr(exc),
                }
        result["label_v3"] = label_v3_from_result(result)
        result["scheduler_resource_label"] = scheduler_resource_label_from_result(result)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result.update({"status": "oom", "error": repr(exc)})
        else:
            result.update({"status": "error", "error": repr(exc)})
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})
    finally:
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
        trainable_params.clear()
        inputs = ()
        optimizer = None
        scaler = None
        model = None
        clear_cuda_profile_state(device, reset_peak=True)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.warmup_epochs is not None:
        args.warmup = args.warmup_epochs * max(args.batches_per_epoch, 1)
    if args.profile_epochs is not None:
        repeats = args.profile_epochs * max(args.batches_per_epoch, 1)
        args.infer_repeats = repeats
        args.train_repeats = repeats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label" / "label").mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest = load_workload_manifest(Path(args.workload_specs)) if args.workload_specs else load_manifest(Path(args.manifest))
    try:
        selection = precision_selection(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    auto_probe_details = None
    requested_precisions = selection.configs
    if selection.auto:
        requested_precisions, auto_probe_details = resolve_auto_precision_configs(device, args)
    if requested_precisions is not None:
        manifest = (
            expand_workload_manifest_precisions(manifest, requested_precisions)
            if args.workload_specs
            else expand_manifest_precisions(manifest, requested_precisions)
        )
    shard_rows = [row for idx, row in enumerate(manifest) if idx % max(args.num_shards, 1) == args.shard_index]
    hardware = hardware_metadata(device, args.hardware_id)
    hardware["precision_filter"] = list(requested_precisions) if requested_precisions is not None else None
    hardware["precision_filter_auto"] = selection.auto
    if auto_probe_details is not None:
        hardware["precision_auto_probe"] = auto_probe_details
    (output_dir / f"hardware_shard{args.shard_index}.json").write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n")

    results_path = output_dir / f"results_shard{args.shard_index}.jsonl"
    resume_checkpoint = load_resume_checkpoint(output_dir, results_path) if args.resume else ResumeCheckpoint(set())
    if args.resume:
        print(
            "resume checkpoint: "
            f"{len(resume_checkpoint.completed_profile_points)} completed label(s), "
            f"{resume_checkpoint.incomplete_rows} incomplete ok row(s), "
            f"{resume_checkpoint.malformed_rows} malformed row(s)",
            flush=True,
        )
    label_v3_path = output_dir / f"label_v3_shard{args.shard_index}.jsonl"
    scheduler_resource_path = output_dir / f"scheduler_resource_shard{args.shard_index}.jsonl"
    try:
        with (
            results_path.open("a") as results_fh,
            label_v3_path.open("a") as label_v3_fh,
            scheduler_resource_path.open("a") as scheduler_resource_fh,
        ):
            for row in shard_rows:
                point_id = profile_point_id(row)
                if args.resume and point_id in resume_checkpoint.completed_profile_points:
                    print(f"{point_id}: skip_completed", flush=True)
                    continue
                result = profile_model(row, Path(args.models_dir), output_dir, device, args)
                result.update({"hardware": hardware, "shard_index": args.shard_index, "num_shards": args.num_shards})
                if result.get("status") == "ok":
                    label_path = output_dir / result["label_file"]
                    write_text_atomic(label_path, repr(result["label"]) + "\n")
                    if isinstance(result.get("label_v3"), dict):
                        append_result_row(label_v3_fh, result["label_v3"])
                    if isinstance(result.get("scheduler_resource_label"), dict):
                        append_result_row(scheduler_resource_fh, result["scheduler_resource_label"])
                    resume_checkpoint.completed_profile_points.add(str(result["profile_point_id"]))
                append_result_row(results_fh, result)
                print(f"{result['profile_point_id']}: {result['status']}", flush=True)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Rerun the same command to resume from the last completed label.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
