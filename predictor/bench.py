"""CPU inference benchmark helpers."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch


def configure_cpu_threads(num_threads: int = 0, interop_threads: int = 0) -> None:
    if num_threads and num_threads > 0:
        torch.set_num_threads(num_threads)
    if interop_threads and interop_threads > 0:
        torch.set_num_interop_threads(max(1, interop_threads))


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p95_ms": float("nan")}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def benchmark_forward(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    num_graphs: int = 1000,
    warmup: int = 20,
) -> dict[str, float]:
    """Measure model forward latency for batches supplied by ``loader``."""

    model.eval()
    latencies: list[float] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for _ in range(warmup if seen == 0 else 0):
                _ = model(batch)
            t0 = time.perf_counter()
            _ = model(batch)
            elapsed = (time.perf_counter() - t0) * 1000.0
            batch_graphs = int(getattr(batch, "num_graphs", 1))
            per_graph = elapsed / max(batch_graphs, 1)
            latencies.extend([per_graph] * batch_graphs)
            seen += batch_graphs
            if seen >= num_graphs:
                break
    return summarize_latencies(latencies[:num_graphs])
