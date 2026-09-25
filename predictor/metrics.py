"""Error metrics for optimized PerfSeer evaluation."""

from __future__ import annotations

import numpy as np


def _to_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def relative_errors(y, yhat, eps: float = 1e-8) -> np.ndarray:
    y = _to_numpy(y).reshape(-1)
    yhat = _to_numpy(yhat).reshape(-1)
    mask = np.abs(y) > eps
    if not np.any(mask):
        return np.zeros(0, dtype=np.float64)
    return np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])


def mape(y, yhat, eps: float = 1e-8) -> float:
    rel = relative_errors(y, yhat, eps)
    return float(np.mean(rel) * 100.0) if rel.size else float("nan")


def rmspe(y, yhat, eps: float = 1e-8) -> float:
    rel = relative_errors(y, yhat, eps)
    return float(np.sqrt(np.mean(rel**2)) * 100.0) if rel.size else float("nan")


def x_acc(y, yhat, x: float, eps: float = 1e-8) -> float:
    rel = relative_errors(y, yhat, eps)
    return float(np.mean(rel <= x / 100.0)) if rel.size else float("nan")


def all_metrics(y, yhat, eps: float = 1e-8) -> dict:
    return {
        "MAPE": mape(y, yhat, eps),
        "RMSPE": rmspe(y, yhat, eps),
        "5Acc": x_acc(y, yhat, 5.0, eps),
        "10Acc": x_acc(y, yhat, 10.0, eps),
    }


def metric_table(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[int, dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows: dict[int, dict] = {}
    for idx, name in enumerate(names):
        rows[idx] = {"name": name, **all_metrics(y_true[:, idx], y_pred[:, idx])}
    return rows
