"""Profile generated calibration models on one GPU shard."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
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

    def autocast(self):
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
    parser.add_argument("--infer-repeats", type=int, default=30)
    parser.add_argument("--train-repeats", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-config", action="append", help="Precision config(s) to profile. May be repeated or comma-separated.")
    parser.add_argument("--precision-sweep", help="Comma-separated precision config filter. Overrides manifest precision rows only by filtering them.")
    parser.add_argument("--fp8-backend", default="transformer_engine", choices=("transformer_engine", "none"))
    return parser.parse_args(argv)


def normalize_precision_config(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key == "bf32":
        raise ValueError("bf32 is ambiguous; use tf32 or bf16_amp")
    if key not in PRECISION_ALIASES:
        allowed = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"unknown precision_config {value!r}; expected one of: {allowed}")
    return PRECISION_ALIASES[key]


def precision_filter(args: argparse.Namespace) -> set[str] | None:
    raw: list[str] = []
    if args.precision_sweep:
        raw.extend(part.strip() for part in args.precision_sweep.split(",") if part.strip())
    if args.precision_config:
        for item in args.precision_config:
            raw.extend(part.strip() for part in item.split(",") if part.strip())
    if not raw:
        return None
    return {normalize_precision_config(item) for item in raw}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(model_path: Path):
    module_name = f"_nrp_model_{model_path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.make_model(), module


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
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
    if config in {"fp8_te_hybrid", "fp8_e4m3", "fp8_e5m2"}:
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
            import transformer_engine.pytorch as te  # noqa: F401

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
        supported = device.type == "cuda" and cc >= (8, 9)
        return PrecisionRuntime(
            config=config,
            device_type=device.type,
            backend="transformer_engine",
            supported=supported,
            unsupported_reason=None if supported else "FP8 Transformer Engine profiling requires Ada-or-newer CUDA hardware (SM 8.9+)",
            fallback_policy="record_unsupported_generated_ops",
            details=details,
        )
    raise ValueError(config)


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
    for _ in range(warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

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
    }


def profile_model(row: dict[str, Any], models_dir: Path, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    precision_config = normalize_precision_config(str(row.get("precision_config", DEFAULT_PRECISION_CONFIG)))
    runtime = precision_runtime(precision_config, device, args)
    model_path = models_dir / Path(row["model_file"]).name
    model, _module = load_model(model_path)
    model = model.to(device)
    input_shape = tuple(int(dim) for dim in row["input_shape"])
    x = torch.randn(input_shape, device=device)
    batch_size = int(input_shape[0]) if input_shape else 1

    result: dict[str, Any] = {
        "model_id": row["model_id"],
        "graph_id": row.get("graph_id", row["model_id"]),
        "profile_point_id": row.get("profile_point_id", f"{row['model_id']}::{precision_config}"),
        "stem": row.get("original_stem", row.get("stem", row["model_id"])),
        "status": "ok",
        "input_shape": list(input_shape),
        "batch_size": batch_size,
        "model_file": row["model_file"],
        "label_file": row.get("label_file", f"label/label/{row['model_id']}_{precision_config}.txt"),
        "precision_config": precision_config,
        "precision": runtime.to_metadata(),
    }
    try:
        if not runtime.supported:
            result.update({"status": "unsupported_precision", "error": runtime.unsupported_reason})
            return result
        if precision_config.startswith("fp8_"):
            result.update(
                {
                    "status": "unsupported_precision",
                    "error": "Generated GraphModel ops are not yet rewritten to Transformer Engine FP8 modules",
                }
            )
            result["precision"]["fallback_policy"] = "record_unsupported_generated_ops"
            return result
        model.eval()

        def infer_fn() -> torch.Tensor:
            with torch.no_grad():
                with runtime.autocast():
                    return model(x)

        infer_label, infer_detail = timed_phase("infer", infer_fn, args.infer_repeats, args.warmup, batch_size, device, args.sample_interval)

        model.train()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=1e-3) if trainable_params else None
        scaler = make_grad_scaler(device, runtime.grad_scaler_enabled)
        if scaler is not None:
            result["precision"]["grad_scaler_enabled"] = bool(scaler.is_enabled())
            if scaler.is_enabled():
                result["precision"]["grad_scaler_initial_scale"] = float(scaler.get_scale())

        def train_fn() -> torch.Tensor:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                out = model(x)
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

        train_label, train_detail = timed_phase("train", train_fn, args.train_repeats, args.warmup, batch_size, device, args.sample_interval)
        if scaler is not None and scaler.is_enabled():
            result["precision"]["grad_scaler_final_scale"] = float(scaler.get_scale())
        result.update(
            {
                "label": {"train": train_label, "infer": infer_label},
                "details": {"train": train_detail, "infer": infer_detail},
            }
        )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label" / "label").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest = load_manifest(Path(args.manifest))
    requested_precisions = precision_filter(args)
    if requested_precisions is not None:
        manifest = [
            row
            for row in manifest
            if normalize_precision_config(str(row.get("precision_config", DEFAULT_PRECISION_CONFIG))) in requested_precisions
        ]
    shard_rows = [row for idx, row in enumerate(manifest) if idx % max(args.num_shards, 1) == args.shard_index]
    hardware = hardware_metadata(device)
    hardware["precision_filter"] = sorted(requested_precisions) if requested_precisions is not None else None
    (output_dir / f"hardware_shard{args.shard_index}.json").write_text(json.dumps(hardware, indent=2, sort_keys=True) + "\n")

    results_path = output_dir / f"results_shard{args.shard_index}.jsonl"
    with results_path.open("a") as results_fh:
        for row in shard_rows:
            result = profile_model(row, Path(args.models_dir), device, args)
            result.update({"hardware": hardware, "shard_index": args.shard_index, "num_shards": args.num_shards})
            results_fh.write(json.dumps(result, sort_keys=True) + "\n")
            results_fh.flush()
            if result.get("status") == "ok":
                label_path = output_dir / result["label_file"]
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(repr(result["label"]) + "\n")
            print(f"{result['profile_point_id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
