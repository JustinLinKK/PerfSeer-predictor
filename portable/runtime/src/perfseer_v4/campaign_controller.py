"""Finite, dependency-aware single-GPU stage admission and isolated workers."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from perfseer_v31.io import atomic_write, fingerprint, read_json
from .campaign import STAGES, final_evaluate, stage_paths, validate_config
from .campaign_data import verify_dataset


def admit(probes, *, free_bytes, total_bytes, maximum=2, fraction=.85):
    budget = min(free_bytes - 512 * 2 ** 20, total_bytes * fraction - (total_bytes - free_bytes))
    admitted, reserved = [], 0
    for probe in probes:
        # Include measured CUDA/context allocations outside PyTorch's allocator.
        # Serial probes may include other processes here; double counting them
        # is conservative and cannot over-admit a worker.
        need = int(probe["peak_reserved_bytes"] * 1.20 + 256 * 2 ** 20 + probe.get("non_allocator_bytes", 0))
        if reserved + need <= budget and len(admitted) < maximum:
            admitted.append(probe)
            reserved += need
    return admitted


def choose_three(two_score, three_score, gain=.10):
    return two_score > 0 and three_score >= two_score * (1 + gain)


@dataclass
class Worker:
    task: tuple
    process: object
    log: object
    result: Path
    started: float


def _start(config, config_path, task, *, probe=False, fraction=None, tag="run"):
    directory = Path(config["output"]) / "controller"
    directory.mkdir(parents=True, exist_ok=True)
    name = "-".join(task) + "-" + tag
    result = directory / (name + ".json")
    if result.exists():
        result.unlink()
    log = (directory / (name + ".log")).open("a", buffering=1)
    launcher = Path(__file__).resolve().parents[2] / "scripts/run_perfseer_v4.py"
    command = [sys.executable, "-u", str(launcher), "campaign-stage", "--config", str(config_path),
               "--source", task[0], "--proposal", task[1], "--stage", task[2], "--resume", "--result", str(result)]
    if probe:
        command.append("--probe")
    if fraction is not None:
        command.extend(("--memory-fraction", str(fraction)))
    env = {**os.environ, "OMP_NUM_THREADS": str(config["cpu_threads"]), "MKL_NUM_THREADS": str(config["cpu_threads"]),
           "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return Worker(task, process, log, result, time.monotonic())


def _finish(worker):
    worker.log.close()
    if worker.process.returncode != 0 or not worker.result.exists():
        log_path = Path(worker.log.name)
        tail = log_path.read_text(errors="replace")[-5000:]
        raise RuntimeError(f"worker {worker.task} exit={worker.process.returncode}; log={log_path}\n{tail}")
    return read_json(worker.result)


def _wait(workers, progress):
    while any(worker.process.poll() is None for worker in workers):
        for worker in workers:
            if worker.process.poll() is not None and worker.process.returncode != 0:
                _finish(worker)  # Fail promptly; do not await other long stages.
        progress(workers)
        time.sleep(2)
    return [_finish(worker) for worker in workers]


def _stop(workers):
    for worker in workers:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 290
    while any(w.process.poll() is None for w in workers) and time.monotonic() < deadline:
        time.sleep(1)
    for worker in workers:
        if worker.process.poll() is None:
            os.killpg(worker.process.pid, signal.SIGKILL)
        worker.process.wait()
        worker.log.close()


def run(config, config_path, *, resume=False):
    validate_config(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(config["cpu_threads"])
    import fcntl
    lock = (output / ".controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("campaign already has an active controller") from error
    identity = fingerprint(config)
    run_path = output / "campaign.json"
    if run_path.exists():
        if not resume or read_json(run_path)["fingerprint"] != identity:
            raise ValueError("existing campaign requires --resume with the same configuration")
    else:
        atomic_write(run_path, {"fingerprint": identity, "config": config})
    for source in (*config["sources"], "RTX5090"):
        manifest = verify_dataset(Path(config["data"]) / source)
        if bool(manifest["smoke_rows_per_split"]) != config["smoke"]:
            raise ValueError("smoke corpus cannot run as production")
    active = []
    stopped = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopped.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_report = [0.]
    def progress(workers):
        if stopped[0]:
            for worker in workers:
                if worker.process.poll() is None:
                    os.killpg(worker.process.pid, signal.SIGTERM)
            raise InterruptedError("controller interrupted; workers checkpointing")
        if time.monotonic() - last_report[0] >= 20:
            last_report[0] = time.monotonic()
            value = {"event": "controller", "time": time.time(), "active": [
                     {"source": w.task[0], "proposal": w.task[1], "stage": w.task[2], "pid": w.process.pid,
                      "seconds": time.monotonic() - w.started} for w in workers if w.process.poll() is None]}
            free, total = torch.cuda.mem_get_info()
            value.update(gpu_used_bytes=total - free, gpu_total_bytes=total)
            atomic_write(output / "progress.json", value)
            print(json.dumps(value), flush=True)
    verified = set()
    try:
        while len(verified) < 32:
            ready = []
            for source in config["sources"]:
                for proposal in config["proposals"]:
                    for stage in STAGES:
                        task = (source, proposal, stage)
                        if task not in verified:
                            ready.append(task)
                            break
            probes = []
            for task in ready:
                # Stage code verifies completed receipts against the current
                # teacher/source/data/code contract before skipping them.
                worker = _start(config, config_path, task, probe=True, tag="probe")
                active = [worker]
                value = _wait(active, progress)[0]
                active = []
                if value["status"] == "complete":
                    verified.add(task)
                else:
                    probes.append({**value, "task": task})
                if len(probes) >= config["concurrency"]["maximum"]:
                    break
            if not probes:
                continue
            free, total = torch.cuda.mem_get_info()
            candidates = admit(probes, free_bytes=free, total_bytes=total, maximum=config["concurrency"]["maximum"],
                               fraction=config["concurrency"]["memory_fraction"])
            if not candidates:
                raise RuntimeError("even one probed stage cannot be admitted with GPU headroom; outputs are resumable")
            count = min(len(candidates), config["concurrency"]["default"])
            evidence = {"time": time.time(), "candidates": [{"task": p["task"], "peak_reserved_bytes": p["peak_reserved_bytes"],
                        "non_allocator_bytes": p.get("non_allocator_bytes", 0)} for p in candidates],
                        "free_bytes": free, "total_bytes": total, "selected_concurrency": count}
            if len(candidates) >= 3 and config["concurrency"]["maximum"] >= 3:
                scores = {}
                for n in (2, 3):
                    active = [_start(config, config_path, p["task"], probe=True, tag=f"pilot-{n}",
                                     fraction=min(.85, (p["peak_reserved_bytes"] * 1.20 + 256 * 2 ** 20) / total))
                              for p in candidates[:n]]
                    try:
                        measurements = _wait(active, progress)
                    except RuntimeError as error:
                        _stop(active)
                        active = []
                        count = 1
                        evidence["pilot_failure"] = str(error)
                        break
                    active = []
                    scores[n] = sum(value["samples_per_second"] for value in measurements)
                if 3 in scores and choose_three(scores[2], scores[3], config["concurrency"]["throughput_gain"]):
                    count = 3
                evidence.update(aggregate_samples_per_second=scores, selected_concurrency=count)
            directory = output / "controller"
            atomic_write(directory / f"admission-{len(verified):02d}.json", evidence)
            active = [_start(config, config_path, p["task"], fraction=min(.85, (p["peak_reserved_bytes"] * 1.20 + 256 * 2 ** 20) / total))
                      for p in candidates[:count]]
            _wait(active, progress)
            verified.update(w.task for w in active)
            active = []
        return final_evaluate(config)
    except BaseException as error:
        atomic_write(output / "failure.json", {"status": "interrupted" if isinstance(error, InterruptedError) else "failed",
                     "error": str(error), "active": [list(w.task) for w in active], "time": time.time()})
        _stop(active)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
