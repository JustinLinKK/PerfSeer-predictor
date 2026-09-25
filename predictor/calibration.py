"""Validation-only calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LinearCalibrator:
    """Per-metric affine calibration in standardized log space."""

    slope: np.ndarray
    intercept: np.ndarray

    def apply(self, pred_std: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred_std, dtype=np.float64)
        return pred * self.slope.reshape(1, -1) + self.intercept.reshape(1, -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "linear_std_log",
            "slope": self.slope.astype(float).tolist(),
            "intercept": self.intercept.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "LinearCalibrator | None":
        if not d or d.get("type") != "linear_std_log":
            return None
        return cls(np.asarray(d["slope"], dtype=np.float64), np.asarray(d["intercept"], dtype=np.float64))


def fit_linear_calibration(pred_std: np.ndarray, true_std: np.ndarray) -> LinearCalibrator:
    pred = np.asarray(pred_std, dtype=np.float64)
    true = np.asarray(true_std, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    slopes = np.ones(pred.shape[1], dtype=np.float64)
    intercepts = np.zeros(pred.shape[1], dtype=np.float64)
    for idx in range(pred.shape[1]):
        x = pred[:, idx]
        y = true[:, idx]
        if x.size >= 2 and np.std(x) > 1e-12:
            slopes[idx], intercepts[idx] = np.polyfit(x, y, deg=1)
    return LinearCalibrator(slopes, intercepts)


def fit_isotonic_calibration(pred_std: np.ndarray, true_std: np.ndarray):
    """Fit optional sklearn isotonic calibrators; callers serialize externally."""

    from sklearn.isotonic import IsotonicRegression

    pred = np.asarray(pred_std, dtype=np.float64)
    true = np.asarray(true_std, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    models = []
    for idx in range(pred.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pred[:, idx], true[:, idx])
        models.append(iso)
    return models
