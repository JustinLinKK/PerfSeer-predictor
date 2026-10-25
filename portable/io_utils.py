"""Small explicit-path utilities for portable inference and training entrypoints."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, default=float)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def log(**event):
    print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}, default=float), flush=True)


def summary(report):
    return {"score": report["score"],
            "timing_material": [report["timing"][c]["material"]["accuracy"] for c in ("wall", "gpu")],
            "timing_close": [report["timing"][c]["close_material"]["accuracy"] for c in ("wall", "gpu")],
            "busy_material": report["nvml_busy"]["material"]["accuracy"],
            "vram_within_100MB": report["vram"]["within_100MB"],
            "jct_ratio": float(np.mean([t["predicted_to_fcfs_mean_jct"] for t in report["online_replay"]["traces"].values()]))}


def rng_state():
    value = {"cpu": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_initialized():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value):
    torch.set_rng_state(value["cpu"])
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])
    if value.get("cuda"):
        torch.cuda.set_rng_state_all(value["cuda"])
