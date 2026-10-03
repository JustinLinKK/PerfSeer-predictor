"""Independent reconstruction of exported seven-target prediction metrics."""

import math
import numpy as np

from perfseer_v31.io import read_json
from .version import TARGET_NAMES, METRIC_VERSION, HEAD_GROUPS, SM_INDICES, validate_targets


def verify_predictions(path, expected_rows=None):
    data = read_json(path)
    if data.get("metric_version") != METRIC_VERSION or tuple(data.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("prediction export contract differs")
    rows = data["rows"]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate prediction rows")
    if expected_rows is not None:
        expected = {row["sample_id"]: [row["targets"][name] for name in TARGET_NAMES] for row in expected_rows}
        if set(expected) != set(ids) or any(not np.array_equal(np.asarray(expected[row["sample_id"]], dtype=np.float32),
                                                             np.asarray(row["target"], dtype=np.float32)) for row in rows):
            raise ValueError("prediction coverage or labels differ")
    prediction = np.asarray([row["prediction"] for row in rows], dtype=np.float32).astype(np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float32).astype(np.float64)
    if prediction.shape != (len(rows), 7) or target.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid seven-target prediction dimensions or values")
    for values in target:
        validate_targets(values)
    error = np.abs(prediction - target)
    relative = error / np.maximum(np.abs(target), 1e-6)
    five, ten = (relative <= .05).sum(0).tolist(), (relative <= .1).sum(0).tolist()
    metrics = data["metrics"]
    if (metrics.get("metric_version") != METRIC_VERSION or tuple(metrics.get("target_names", ())) != TARGET_NAMES
            or metrics.get("rows") != len(rows) or metrics.get("within_5pct_count") != five
            or metrics.get("within_10pct_count") != ten):
        raise ValueError("exported metric counts differ")
    for key, values in (("mae", error.mean(0)), ("relative_mean_error", relative.mean(0))):
        if not np.allclose(metrics[key], values, rtol=1e-12, atol=1e-12):
            raise ValueError(f"exported {key} differs")
    for key, counts in (("within_5pct_accuracy", five), ("within_10pct_accuracy", ten)):
        if metrics[key] != dict(zip(TARGET_NAMES, [count / len(rows) for count in counts], strict=True)):
            raise ValueError("exported accuracy differs")
    if metrics.get("gate_passed") != all(20 * count >= 19 * len(rows) for count in five):
        raise ValueError("exported acceptance gate differs")
    denominator = target.astype(np.float32)
    denominator[:, SM_INDICES] = np.maximum(denominator[:, SM_INDICES], 1.)
    errors = np.abs(prediction.astype(np.float32) - target.astype(np.float32)) / denominator
    loss = np.mean([errors[:, group].mean() for group in HEAD_GROUPS])
    if not math.isclose(metrics["group_balanced_loss"], float(loss), rel_tol=2e-6, abs_tol=1e-7):
        raise ValueError("exported training loss differs")
    return {"status": "passed", "rows": len(rows), "target_names": list(TARGET_NAMES)}
