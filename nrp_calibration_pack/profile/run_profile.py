"""Profile generated calibration models on one GPU shard."""

from __future__ import annotations

import argparse
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


@dataclass
class SampleStats:
    avg_sm_util: float = 0.0
    avg_mem_util: float = 0.0
    avg_mem_usage: float = 0.0
    peak_sm_util: float = 0.0
    peak_mem_util: float = 0.0
    peak_mem_usage: float = 0.0
    source: str = "none"


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
    return parser.parse_args(argv)


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
    model_path = models_dir / Path(row["model_file"]).name
    model, _module = load_model(model_path)
    model = model.to(device)
    input_shape = tuple(int(dim) for dim in row["input_shape"])
    x = torch.randn(input_shape, device=device)
    batch_size = int(input_shape[0]) if input_shape else 1

    result: dict[str, Any] = {
        "model_id": row["model_id"],
        "stem": row.get("original_stem", row.get("stem", row["model_id"])),
        "status": "ok",
        "input_shape": list(input_shape),
        "batch_size": batch_size,
        "model_file": row["model_file"],
        "label_file": row.get("label_file", f"label/label/{row['model_id']}.txt"),
    }
    try:
        model.eval()

        def infer_fn() -> torch.Tensor:
            with torch.no_grad():
                return model(x)

        infer_label, infer_detail = timed_phase("infer", infer_fn, args.infer_repeats, args.warmup, batch_size, device, args.sample_interval)

        model.train()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=1e-3) if trainable_params else None

        def train_fn() -> torch.Tensor:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
            if loss.requires_grad:
                loss.backward()
            if optimizer is not None:
                optimizer.step()
            return loss.detach()

        train_label, train_detail = timed_phase("train", train_fn, args.train_repeats, args.warmup, batch_size, device, args.sample_interval)
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
    shard_rows = [row for idx, row in enumerate(manifest) if idx % max(args.num_shards, 1) == args.shard_index]
    hardware = hardware_metadata(device)
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
            print(f"{result['model_id']}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
