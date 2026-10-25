"""Immutable ABA-1 validation ranking; not a final teacher acceptance receipt."""
import numpy as np

TARGET_NAMES = ("train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
                "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib")


def timing_metrics(predictions, targets):
    p, y = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if (p.shape != y.shape or p.ndim != 2 or p.shape[1] != 6 or not len(p)
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or (p[:, [0, 1, 2, 4, 5]] <= 0).any() or (y[:, [0, 1, 2, 4, 5]] <= 0).any()
            or ((p[:, 3] < 0) | (p[:, 3] > 100)).any()
            or ((y[:, 3] < 0) | (y[:, 3] > 100)).any()):
        raise ValueError("expected aligned finite physically valid six-target arrays")
    rel = np.abs(p - y) / np.maximum(np.abs(y), 1e-6)
    counts = (rel <= .05).sum(0)
    hit = counts / len(y)
    memory_ok = bool((20 * counts[4:] >= 19 * len(y)).all())
    weighted = .70 * hit[:3].mean() + .25 * hit[4:].mean() + .05 * hit[3]
    zero = y[:, 3] == 0
    return {"rows": len(y), "hit5": dict(zip(TARGET_NAMES, hit.tolist())),
            "counts": counts.tolist(), "median_relative_error": np.median(rel, axis=0).tolist(),
            "p90_relative_error": np.quantile(rel, .9, axis=0).tolist(),
            "sm_mae_percentage_points": float(np.abs(p[:, 3] - y[:, 3]).mean()),
            "sm_zero_rows": int(zero.sum()),
            "sm_zero_false_positive_rows": int((p[zero, 3] > 5e-8).sum()),
            "vram_guard_passed": memory_ok, "weighted_score": float(weighted),
            "score": float(weighted if memory_ok else weighted - 1.),
            "all_six_validation_gate_passed": bool((20 * counts >= 19 * len(y)).all()),
            "test_status": "NOT_EVALUATED", "teacher_accepted": False}
