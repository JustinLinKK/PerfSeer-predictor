"""Profile generated calibration models on one GPU shard."""

from __future__ import annotations

import argparse
import csv
import contextlib
import importlib.util
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


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


@dataclass
class SampleStats:
    avg_sm_util: float = 0.0
    avg_mem_util: float = 0.0
    avg_mem_usage: float = 0.0
    peak_sm_util: float = 0.0
    peak_mem_util: float = 0.0
    peak_mem_usage: float = 0.0
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
    parser.add_argument("--manifest", required=True)
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
        "--profile-dataset-dir",
        help="Optional directory of <model_id>.json input/repeat specs from profile/make_profile_datasets.py.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--sm-occupancy-source", default="ncu", choices=("ncu", "nvml_proxy"))
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
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.warmup_epochs is not None and args.warmup_epochs < 0:
        parser.error("--warmup-epochs must be >= 0")
    if args.profile_epochs is not None and args.profile_epochs <= 0:
        parser.error("--profile-epochs must be > 0")
    if args.batches_per_epoch <= 0:
        parser.error("--batches-per-epoch must be > 0")
    for name in ("infer_repeats", "train_repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
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


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
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


def positive_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


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
        self.samples: list[tuple[float, float, float, float]] = []
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

    def stop(self) -> SampleStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return SampleStats(source="nvml" if self.available else "none")
        sm = [sample[0] for sample in self.samples]
        mem_util = [sample[1] for sample in self.samples]
        mem_used = [sample[2] for sample in self.samples]
        return SampleStats(
            avg_sm_util=float(sum(sm) / len(sm)),
            avg_mem_util=float(sum(mem_util) / len(mem_util)),
            avg_mem_usage=float(sum(mem_used) / len(mem_used)),
            peak_sm_util=float(max(sm)),
            peak_mem_util=float(max(mem_util)),
            peak_mem_usage=float(max(mem_used)),
            source="nvml",
        )

    def _run(self) -> None:
        assert self._nvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                mem_used_mib = float(mem.used / MI_B)
                mem_util = float(100.0 * mem.used / max(mem.total, 1))
                self.samples.append((float(util.gpu), mem_util, mem_used_mib, time.time()))
            except Exception:
                pass
            time.sleep(self.interval)


def fallback_memory_stats(device: torch.device) -> SampleStats:
    if device.type == "cuda" and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / MI_B
        return SampleStats(avg_mem_usage=float(peak), peak_mem_usage=float(peak), source="torch")
    return SampleStats(source="none")


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
    label = {
        "time_1_epoch_ms": float(detail["mean_iter_ms"]),
        "avg_device_memory_usage_mib": float(sampler.get("avg_mem_usage", 0.0)),
        "peak_device_memory_usage_mib": float(sampler.get("peak_mem_usage", 0.0)),
        "avg_host_memory_usage_mib": float(detail["avg_host_memory_usage_mib"]),
        "peak_host_memory_usage_mib": float(detail["peak_host_memory_usage_mib"]),
        "compile_time_ms": 0.0,
        "warmup_time_ms": float(detail.get("warmup_time_ms", 0.0)),
        "compile_warmup_time_ms": float(detail.get("warmup_time_ms", 0.0)),
        "avg_sm_occupancy_percent": float(occupancy["avg_sm_occupancy_percent"]),
        "peak_sm_occupancy_percent": float(occupancy["peak_sm_occupancy_percent"]),
        "sm_occupancy_source": occupancy["source"],
        "sm_occupancy_kernel_count": int(occupancy.get("kernel_count", 0)),
        "avg_sm_utilization_percent": avg_sm_util,
        "peak_sm_utilization_percent": peak_sm_util,
        "dram_activity_percent": float(sampler.get("avg_mem_util", 0.0)),
        "peak_dram_activity_percent": float(sampler.get("peak_mem_util", 0.0)),
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


def timed_phase(
    phase: str,
    fn: Callable[[], torch.Tensor],
    repeats: int,
    warmup: int,
    batch_size: int,
    device: torch.device,
    sample_interval: float,
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

    sampler = NvmlSampler(device.index or 0, sample_interval) if device.type == "cuda" else None
    if sampler:
        sampler.start()
    raw_ms = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize(device)
            raw_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            _ = fn()
            raw_ms.append((time.perf_counter() - t0) * 1000.0)
    host_after = host_memory_mib()
    stats = sampler.stop() if sampler else SampleStats(source="none")
    if stats.source == "none" or stats.peak_mem_usage <= 0:
        fallback = fallback_memory_stats(device)
        if fallback.peak_mem_usage > stats.peak_mem_usage:
            stats.peak_mem_usage = fallback.peak_mem_usage
            stats.avg_mem_usage = fallback.avg_mem_usage
            stats.source = fallback.source
    mean_iter_ms = float(sum(raw_ms) / max(len(raw_ms), 1))
    time_ms_per_sample = mean_iter_ms / max(batch_size, 1)
    return label_string(time_ms_per_sample, stats), {
        "phase": phase,
        "mean_iter_ms": mean_iter_ms,
        "time_ms_per_sample": time_ms_per_sample,
        "raw_iter_ms": raw_ms,
        "sampler": stats.__dict__,
        "warmup_time_ms": warmup_time_ms,
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
    precision_config = normalize_precision_config(str(row.get("precision_config", DEFAULT_PRECISION_CONFIG)))
    runtime = precision_runtime(precision_config, device, args)
    model_path = models_dir / Path(row["model_file"]).name
    model, _module = load_model(model_path)
    model = model.to(device)
    dataset_spec_error: str | None = None
    try:
        dataset_spec = load_profile_dataset_spec(str(row["model_id"]), args.profile_dataset_dir)
    except Exception as exc:
        dataset_spec = {}
        dataset_spec_error = repr(exc)

    profile_config_error: str | None = None
    try:
        input_specs = normalize_input_specs(row, dataset_spec)
        input_shape = tuple(int(dim) for dim in input_specs[0]["shape"])
        batch_size = int(input_shape[0]) if input_shape else 1
        train_repeats = positive_int(dataset_spec.get("train_repeats"), args.train_repeats, "train_repeats")
        infer_repeats = positive_int(dataset_spec.get("infer_repeats"), args.infer_repeats, "infer_repeats")
    except Exception as exc:
        profile_config_error = repr(exc)
        input_specs = [{"name": "input0", "shape": list(row["input_shape"]), "dtype": "float32", "kind": "float"}]
        input_shape = tuple(int(dim) for dim in row["input_shape"])
        batch_size = int(input_shape[0]) if input_shape else 1
        train_repeats = int(args.train_repeats)
        infer_repeats = int(args.infer_repeats)

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
        "precision_config": precision_config,
        "precision": runtime.to_metadata(),
        "profile_dataset": {
            "source": "profile_dataset_dir" if dataset_spec else "synthetic_cli",
            "train_repeats": train_repeats,
            "infer_repeats": infer_repeats,
        },
    }
    try:
        if dataset_spec_error is not None:
            result.update({"status": "error", "error": dataset_spec_error})
            return result
        if profile_config_error is not None:
            result.update({"status": "error", "error": profile_config_error})
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
        inputs = make_profile_inputs(input_specs, device, runtime.input_dtype)
        model.eval()

        def infer_fn() -> torch.Tensor:
            with torch.no_grad():
                with runtime.autocast():
                    return model(*inputs)

        infer_label, infer_detail = timed_phase("infer", infer_fn, infer_repeats, args.warmup, batch_size, device, args.sample_interval)

        model.train()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[args.optimizer]
        optimizer = optimizer_cls(trainable_params, lr=1e-3) if trainable_params else None
        scaler = make_grad_scaler(device, runtime.grad_scaler_enabled)
        if scaler is not None:
            result["precision"]["grad_scaler_enabled"] = bool(scaler.is_enabled())
            if scaler.is_enabled():
                result["precision"]["grad_scaler_initial_scale"] = float(scaler.get_scale())

        def train_fn() -> torch.Tensor:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                out = model(*inputs)
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

        train_label, train_detail = timed_phase("train", train_fn, train_repeats, args.warmup, batch_size, device, args.sample_interval)
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
                infer_occupancy = collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "infer", args.optimizer)
                train_occupancy = collect_ncu_occupancy(row, models_dir, output_dir, input_shape, "train", args.optimizer)
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
            "train": phase_label_v2(train_detail, data_type, train_occupancy, args.optimizer),
            "infer": phase_label_v2(infer_detail, data_type, infer_occupancy),
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result.update({"status": "oom", "error": repr(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            result.update({"status": "error", "error": repr(exc)})
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
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
    manifest = load_manifest(Path(args.manifest))
    try:
        selection = precision_selection(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    auto_probe_details = None
    requested_precisions = selection.configs
    if selection.auto:
        requested_precisions, auto_probe_details = resolve_auto_precision_configs(device, args)
    if requested_precisions is not None:
        manifest = expand_manifest_precisions(manifest, requested_precisions)
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
    try:
        with results_path.open("a") as results_fh:
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
