"""Error metrics for PerfSeer, computed in the ORIGINAL metric space.

All helpers accept either numpy arrays or torch tensors and operate per-sample.
They guard against divide-by-zero by masking out targets whose absolute value
is below ``eps`` (such samples cannot yield a meaningful relative error).

Definitions (paper section 4.1):
    MAPE  = mean(|y - yhat| / |y|) * 100
    RMSPE = sqrt(mean(((y - yhat) / y) ** 2)) * 100
    xAcc  = fraction of samples with |y - yhat| / |y| <= x / 100
"""

from __future__ import annotations

import numpy as np


def _to_numpy(a) -> np.ndarray:
    """Convert torch tensors / lists to a flat float64 numpy array."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _relative_errors(y, yhat, eps: float = 1e-8) -> np.ndarray:
    """Return |y - yhat| / |y| for samples with |y| > eps."""
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    mask = np.abs(y) > eps
    if not np.any(mask):
        return np.zeros(0, dtype=np.float64)
    return np.abs(y[mask] - yhat[mask]) / np.abs(y[mask])


def mape(y, yhat, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (percent)."""
    rel = _relative_errors(y, yhat, eps)
    if rel.size == 0:
        return float("nan")
    return float(np.mean(rel) * 100.0)


def rmspe(y, yhat, eps: float = 1e-8) -> float:
    """Root Mean Squared Percentage Error (percent)."""
    rel = _relative_errors(y, yhat, eps)
    if rel.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(rel ** 2)) * 100.0)


def x_acc(y, yhat, x: float, eps: float = 1e-8) -> float:
    """Fraction of samples whose relative error <= x / 100 (range [0, 1])."""
    rel = _relative_errors(y, yhat, eps)
    if rel.size == 0:
        return float("nan")
    return float(np.mean(rel <= (x / 100.0)))


def all_metrics(y, yhat, eps: float = 1e-8) -> dict:
    """Convenience bundle: MAPE, RMSPE, 5%Acc, 10%Acc."""
    return {
        "MAPE": mape(y, yhat, eps),
        "RMSPE": rmspe(y, yhat, eps),
        "5Acc": x_acc(y, yhat, 5.0, eps),
        "10Acc": x_acc(y, yhat, 10.0, eps),
    }
