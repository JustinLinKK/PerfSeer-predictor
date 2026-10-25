#!/usr/bin/env python3
"""Fresh local NVIDIA labeler with per-workload durable, hardware-bound receipts.

Never accepts a hardware-name override. Each workload runs in a fresh process.
Successful labels and honest OOM/error/timeout/contamination outcomes are atomic
checkpoint records; restarting skips completed records after checking run identity.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback

from generate import HERE, canonical, digest, sha

MIB = 1048576
SCHEMA = "perfseer-local-label-v1"
PROTOCOL = {"warmup_epochs": 1, "measured_epochs": 3, "repeats": 3, "steps_per_epoch": 1,
            "sample_interval_seconds": .01, "resource_min_seconds": 1.0, "resource_min_samples": 20,
            "seed": 20261024, "loss": "MSE(output.float(), zeros)", "input_kind": "synthetic",
            "time_mode": "synchronized measured epochs; sequential repeats on continuing model/optimizer state",
            "resource_mode": "separate sustained training after timing warmup",
            "nvml_busy_semantics": "device busy-time percent, not SM occupancy",
            "vram_semantics": "NVML device memory minus pre-worker baseline; raw device values retained",
            "historical_protocol_equivalent": False}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def nvml(index):
    import pynvml
    pynvml.nvmlInit()
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def detect(index=0):
    lib, handle = nvml(index)
    memory = lib.nvmlDeviceGetMemoryInfo(handle)
    name = decode(lib.nvmlDeviceGetName(handle))
    return {"gpu_name": name, "gpu_uuid": decode(lib.nvmlDeviceGetUUID(handle)),
            "hardware_id": re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + "_local",
            "nvml_index": index, "memory_total_bytes": memory.total,
            "driver_version": decode(lib.nvmlSystemGetDriverVersion())}


def idle_sample(index, count=5):
    lib, handle = nvml(index)
    samples = []
    for _ in range(count):
        samples.append({"memory_bytes": lib.nvmlDeviceGetMemoryInfo(handle).used,
                        "busy_percent": lib.nvmlDeviceGetUtilizationRates(handle).gpu})
        time.sleep(.1)
    return {"memory_bytes": statistics.median(x["memory_bytes"] for x in samples),
            "memory_span_bytes": max(x["memory_bytes"] for x in samples) - min(x["memory_bytes"] for x in samples),
            "busy_mean_percent": statistics.fmean(x["busy_percent"] for x in samples)}


def load_manifest(root):
    rows = [json.loads(line) for line in (root / "workloads.jsonl").open() if line.strip()]
    if len({r["profile_point_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate workload IDs")
    for row in rows:
        if sha(root / row["model_file"]) != row["source_sha256"]:
            raise ValueError("source hash mismatch: " + row["profile_point_id"])
        identity = {k: row[k] for k in ("architecture", "batch_size", "input_specs", "precision_config", "optimizer", "source_sha256", "runtime_sha256", "variant_source_sha256")}
        if digest(identity) != row["workload_sha256"]:
            raise ValueError("workload identity hash mismatch")
        if sha(root / row["variant_file"]) != row["variant_source_sha256"]:
            raise ValueError("variant source hash mismatch")
    if sha(root / "models/_model_runtime.py") != sha(HERE / "runtime/_model_runtime.py"):
        raise ValueError("packaged runtime differs from generated runtime")
    if sha(root / "models/_model_catalog.py") != sha(HERE / "runtime/_model_catalog.py"):
        raise ValueError("packaged catalog differs from generated catalog")
    return rows


class Sampler:
    def __init__(self, index, interval):
        self.lib, self.handle = nvml(index)
        self.interval, self.samples, self.errors = interval, [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                self.samples.append({"monotonic_ns": time.monotonic_ns(),
                                     "device_memory_bytes": self.lib.nvmlDeviceGetMemoryInfo(self.handle).used,
                                     "busy_percent": self.lib.nvmlDeviceGetUtilizationRates(self.handle).gpu})
            except Exception as exc:
                self.errors.append(repr(exc))
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive() or self.errors:
            raise RuntimeError("NVML sampling failed: " + repr(self.errors))


def worker(request, output):
    request = json.loads(Path(request).read_text())
    row, protocol, hardware = request["workload"], request["protocol"], request["hardware"]
    result = {"schema": SCHEMA, "profile_point_id": row["profile_point_id"], "workload_sha256": row["workload_sha256"],
              "hardware": hardware, "protocol": protocol, "run_sha256": request["run_sha256"],
              "source_sha256": row["variant_source_sha256"], "source_lineage": "standalone_variant", "started_at_unix": time.time()}
    try:
        import torch
        import torch.nn.functional as F
        sys.path.insert(0, str(HERE / "runtime"))
        from muon import make_training_optimizer
        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one visible CUDA GPU")
        if torch.cuda.get_device_name(0) != hardware["gpu_name"]:
            raise RuntimeError("CUDA/NVML hardware-name mismatch")
        cuda_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if cuda_uuid.lower().removeprefix("gpu-") != hardware["gpu_uuid"].lower().removeprefix("gpu-"):
            raise RuntimeError("CUDA/NVML GPU UUID mismatch")
        result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_uuid": cuda_uuid,
                             "python": sys.version, "cudnn": torch.backends.cudnn.version()}
        torch.manual_seed(protocol["seed"])
        torch.cuda.manual_seed_all(protocol["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        root = Path(request["generated_root"])
        path = root / row["variant_file"]
        if sha(path) != row["variant_source_sha256"]:
            raise RuntimeError("source changed before worker execution")
        spec = importlib.util.spec_from_file_location("workload_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.make_model().cuda().train()
        config = module.MODEL_CONFIG
        tensors = []
        for item in row["input_specs"]:
            shape = item["shape"]
            if item["dtype"] == "int64":
                tensor = torch.randint(0, int(config.get("vocab_size", 8192)), shape, device="cuda")
            else:
                tensor = torch.randn(shape, device="cuda")
                if item.get("kind") == "adjacency":
                    tensor = (tensor > 0).float()
                    tensor = tensor / tensor.sum(-1, keepdim=True).clamp_min(1)
            tensors.append(tensor)
        optimizer = make_training_optimizer(row["optimizer"], model, lr=1e-3)
        amp = row["precision_config"] == "fp16_amp"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                y = model(*tensors)
                loss = F.mse_loss(y.float(), torch.zeros_like(y, dtype=torch.float32))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite workload loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return float(loss.detach())

        steps = protocol["steps_per_epoch"]
        for _ in range(protocol["warmup_epochs"] * steps):
            step()
        torch.cuda.synchronize()
        repeat_records = []
        for repeat in range(protocol["repeats"]):
            epochs = []
            for epoch in range(protocol["measured_epochs"]):
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                began = time.perf_counter_ns()
                start.record()
                for _ in range(steps):
                    loss = step()
                end.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter_ns() - began) / 1e6
                epochs.append({"wall_ms": wall_ms, "gpu_ms": start.elapsed_time(end), "loss": loss})
            repeat_records.append({"repeat": repeat, "epochs": epochs})
        torch.cuda.reset_peak_memory_stats()
        with Sampler(hardware["nvml_index"], protocol["sample_interval_seconds"]) as sampler:
            began = time.monotonic()
            resource_steps = 0
            while time.monotonic() - began < protocol["resource_min_seconds"] or len(sampler.samples) < protocol["resource_min_samples"]:
                step()
                torch.cuda.synchronize()
                resource_steps += 1
                if time.monotonic() - began > max(30, protocol["resource_min_seconds"] * 10):
                    raise RuntimeError("insufficient NVML samples before resource deadline")
        samples = sampler.samples
        wall = statistics.fmean(e["wall_ms"] for r in repeat_records for e in r["epochs"])
        gpu = statistics.fmean(e["gpu_ms"] for r in repeat_records for e in r["epochs"])
        result.update(status="ok", repeats=repeat_records, resource_samples=samples, resource_steps=resource_steps,
                      training={"batch_size": row["batch_size"], "optimizer": row["optimizer"], "precision": row["precision_config"],
                                "steps_per_epoch": steps, "warmup_epochs": protocol["warmup_epochs"],
                                "measured_epochs": protocol["measured_epochs"], "repeats": protocol["repeats"]},
                      targets={"train_step_wall_ms": wall / steps, "train_step_gpu_ms": gpu / steps,
                               "train_epoch_wall_ms": wall, "train_epoch_gpu_ms": gpu,
                               "train_avg_nvml_busy_percent": statistics.fmean(x["busy_percent"] for x in samples),
                               "train_peak_nvml_busy_percent": max(x["busy_percent"] for x in samples),
                               "raw_device_avg_vram_bytes": statistics.fmean(x["device_memory_bytes"] for x in samples),
                               "raw_device_peak_vram_bytes": max(x["device_memory_bytes"] for x in samples),
                               "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    except BaseException as exc:
        result.update(status="oom" if "out of memory" in str(exc).lower() else "error", error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        result.pop("targets", None)
    result["finished_at_unix"] = time.time()
    atomic(output, result)


def binding(root, hardware, protocol):
    return {"schema": SCHEMA, "generated_manifest_sha256": sha(root / "workloads.jsonl"), "hardware": hardware,
            "protocol": protocol, "labeler_sha256": sha(__file__), "helper_sha256": sha(HERE / "generate.py"),
            "runtime_hashes": {p.name: sha(p) for p in (HERE / "runtime").glob("*.py")}}


def verify(output):
    run = json.loads((output / "run.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
    counts = {}
    for row in records:
        if row["run_sha256"] != digest(run) or row["hardware"] != run["hardware"]:
            raise ValueError("record run/hardware mismatch")
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] != "ok":
            if "targets" in row:
                raise ValueError("failed record has accepted targets")
            continue
        t, protocol = row["targets"], row["protocol"]
        if any(not math.isfinite(x) or x < 0 for x in t.values()):
            raise ValueError("invalid metric")
        if not 0 <= t["train_avg_nvml_busy_percent"] <= t["train_peak_nvml_busy_percent"] <= 100:
            raise ValueError("invalid busy percent")
        if t["train_avg_vram_bytes"] > t["train_peak_vram_bytes"]:
            raise ValueError("invalid memory ordering")
        for domain in ("wall", "gpu"):
            if not math.isclose(t[f"train_step_{domain}_ms"] * protocol["steps_per_epoch"], t[f"train_epoch_{domain}_ms"], rel_tol=1e-9):
                raise ValueError("epoch arithmetic mismatch")
        if len(row["repeats"]) != protocol["repeats"] or len(row["resource_samples"]) < protocol["resource_min_samples"]:
            raise ValueError("incomplete measurement protocol")
    receipt = {"passed": True, "records": len(records), "status_counts": counts, "run_sha256": digest(run),
               "hardware": run["hardware"], "zero_failed_records_promoted_to_labels": True}
    atomic(output / "verification.json", receipt)
    return receipt


def run(args):
    rows = load_manifest(args.generated)
    hardware = detect(args.nvml_index)
    output = args.output / hardware["hardware_id"]
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**PROTOCOL, "steps_per_epoch": args.steps_per_epoch,
                "max_idle_busy_percent": args.max_idle_busy, "max_baseline_drift_mib": args.max_drift_mib}
    identity = binding(args.generated, hardware, protocol)
    lock_path = args.gpu_lock or args.output / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("GPU lock is held; no workload started")
        run_path = output / "run.json"
        if run_path.exists() and json.loads(run_path.read_text()) != identity:
            raise ValueError("resume identity changed (GPU, protocol, source or labeler); use a new output directory")
        if not run_path.exists():
            atomic(run_path, identity)
        completed = 0
        for row in rows:
            if args.id and row["profile_point_id"] not in args.id:
                continue
            key = row["workload_sha256"]
            record_path = output / "records" / (key + ".json")
            if record_path.exists():
                continue
            if args.limit and completed >= args.limit:
                break
            pre = idle_sample(args.nvml_index)
            if pre["busy_mean_percent"] > args.max_idle_busy or pre["memory_span_bytes"] > args.max_drift_mib * MIB:
                raise SystemExit("GPU is busy or baseline unstable; no next workload started")
            request = {"workload": row, "protocol": protocol, "hardware": hardware,
                       "run_sha256": digest(identity), "generated_root": str(args.generated.resolve())}
            request_path = output / "work" / (key + ".request.json")
            worker_path = output / "work" / (key + ".worker.json")
            atomic(request_path, request)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": hardware["gpu_uuid"], "OMP_NUM_THREADS": "1"}
            started = time.time()
            with (output / "work" / (key + ".log")).open("w") as log:
                try:
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--request", str(request_path),
                                           "--worker-output", str(worker_path)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    exit_code, timeout = proc.returncode, False
                except subprocess.TimeoutExpired:
                    exit_code, timeout = None, True
            post = idle_sample(args.nvml_index)
            record = json.loads(worker_path.read_text()) if worker_path.exists() else {
                "schema": SCHEMA, "profile_point_id": row["profile_point_id"], "workload_sha256": key,
                "hardware": hardware, "run_sha256": digest(identity), "protocol": protocol,
                "status": "timeout" if timeout else "worker_crash", "started_at_unix": started}
            record.update(worker_exit_code=exit_code, worker_timeout=timeout, baseline_before=pre, baseline_after=post,
                          baseline_drift_bytes=post["memory_bytes"] - pre["memory_bytes"])
            if timeout or exit_code != 0:
                record.update(status="timeout" if timeout else "worker_crash")
                record.pop("targets", None)
            if record["status"] == "ok":
                raw = record["targets"]
                if abs(record["baseline_drift_bytes"]) > args.max_drift_mib * MIB or post["busy_mean_percent"] > args.max_idle_busy:
                    record["status"] = "contaminated"
                    record["discarded_targets"] = record.pop("targets")
                else:
                    for metric in ("avg", "peak"):
                        value = raw[f"raw_device_{metric}_vram_bytes"] - pre["memory_bytes"]
                        if value < 0:
                            raise ValueError("negative baseline-corrected memory; refusing to clamp")
                        raw[f"train_{metric}_vram_bytes"] = value
                        raw[f"train_{metric}_vram_mib"] = value / MIB
            atomic(record_path, record)
            completed += 1
            print(canonical({"profile_point_id": row["profile_point_id"], "status": record["status"], "completed_this_run": completed}), flush=True)
    print(canonical(verify(output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    detect_parser = sub.add_parser("detect")
    detect_parser.add_argument("--nvml-index", type=int, default=0)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--generated", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--gpu-lock", type=Path)
    run_parser.add_argument("--nvml-index", type=int, default=0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--id", action="append")
    run_parser.add_argument("--timeout", type=float, default=240)
    run_parser.add_argument("--steps-per-epoch", type=int, default=1)
    run_parser.add_argument("--max-idle-busy", type=float, default=10)
    run_parser.add_argument("--max-drift-mib", type=float, default=64)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--request", type=Path, required=True)
    worker_parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "detect":
        print(canonical(detect(args.nvml_index)))
    elif args.command == "run":
        if args.steps_per_epoch < 1:
            parser.error("steps-per-epoch must be positive")
        run(args)
    elif args.command == "worker":
        worker(args.request, args.worker_output)
    else:
        print(canonical(verify(args.output)))


if __name__ == "__main__":
    main()
