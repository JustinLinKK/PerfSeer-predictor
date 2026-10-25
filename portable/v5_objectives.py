"""Pointwise costs, pairwise training and decimal-100MB capacity semantics.

The six-column transport retains milliseconds, NVML percent and MiB internally.
Only VRAM targets change: conservative decimal-100MB buckets, minimum100MB.
Raw measurements are retained separately; no queue-dependent normalization.
"""
import math
import numpy as np

VERSION = "perfseer_v5_online_cost_100MB_v1"
BUCKET_BYTES = 100_000_000
MIB_BYTES = 1_048_576
BUCKET_MIB = BUCKET_BYTES / MIB_BYTES
TARGET_NAMES = ("train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
                "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib")


def vram_buckets_mib(value):
    """Return conservative bucket values and validity; never impute missing data."""
    a = np.asarray(value, dtype=np.float64)
    valid = np.isfinite(a) & (a >= 0)
    # Decimal bucket boundaries are exactly representable in binary MiB units.
    safe = np.where(valid, a, 0.)
    buckets = np.maximum(1., np.ceil(safe / BUCKET_MIB))
    return np.where(valid, buckets * BUCKET_MIB, np.nan), valid


def transform_targets(targets):
    raw = np.asarray(targets, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("expected six target columns")
    result = raw.copy()
    result[:, 4:6], memory_valid = vram_buckets_mib(raw[:, 4:6])
    valid = np.isfinite(raw)
    valid[:, :3] &= raw[:, :3] > 0
    valid[:, 3] &= (raw[:, 3] >= 0) & (raw[:, 3] <= 100)
    valid[:, 4:6] = memory_valid
    return result, valid


def quantize_predictions(predictions):
    p = np.asarray(predictions, dtype=np.float64).copy()
    p[:, 4:6], valid = vram_buckets_mib(p[:, 4:6])
    if not valid.all():
        raise ValueError("invalid predicted VRAM")
    return p


def pair_metrics(prediction, target, kind="time"):
    from scipy.stats import spearmanr
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    i, j = np.triu_indices(len(y), 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    valid = dy != 0
    credit = (dy * dp > 0).astype(float) + .5 * (dp == 0)
    if kind == "time":
        ratio = np.maximum(y[i], y[j]) / np.minimum(y[i], y[j])
        material, close = ratio >= 1.10, (ratio >= 1.10) & (ratio < 1.25)
    else:
        material, close = np.abs(dy) >= 5., (np.abs(dy) >= 5.) & (np.abs(dy) < 15.)
    def measure(mask):
        mask = mask & valid
        return {"pairs": int(mask.sum()), "correct_credit": float(credit[mask].sum()),
                "accuracy": float(credit[mask].mean()) if mask.any() else None}
    rho = float(spearmanr(y, p).statistic) if np.ptp(y) and np.ptp(p) else 0.
    return {"spearman": rho, "all": measure(valid), "material": measure(material),
            "close_material": measure(close)}


def replay_arrivals(prediction, observed, seed=5, load=.7, servers=2):
    """Nonpreemptive causal replay. Labels are simulator durations, not policy input.

    This is a fixed-one-step service proxy, not full-epoch or co-location validation.
    Arrival rate is trace construction; policies see only already-arrived jobs.
    """
    import heapq
    p, y = np.asarray(prediction, float), np.asarray(observed, float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y)); p, y = p[order], y[order]
    arrival = np.cumsum(rng.exponential(float(y.mean()) / (servers * load), len(y)))
    def simulate(policy):
        now = 0.; index = 0; waiting = []; running = []; results = []
        while index < len(y) or waiting or running:
            while running and running[0] <= now:
                heapq.heappop(running)
            while index < len(y) and arrival[index] <= now:
                key = arrival[index] if policy == "fcfs" else p[index] if policy == "predicted" else y[index]
                heapq.heappush(waiting, (float(key), index)); index += 1
            while waiting and len(running) < servers:
                _, job = heapq.heappop(waiting)
                finish = now + y[job]
                heapq.heappush(running, float(finish)); results.append(float(finish - arrival[job]))
            events = ([arrival[index]] if index < len(y) else []) + ([running[0]] if running else [])
            if events:
                now = min(events)
        return {"mean_jct": float(np.mean(results)), "p95_jct": float(np.quantile(results, .95))}
    values = {name: simulate(name) for name in ("fcfs", "predicted", "oracle")}
    values["predicted_to_fcfs_mean_jct"] = values["predicted"]["mean_jct"] / values["fcfs"]["mean_jct"]
    return values


def metrics(prediction, targets, raw_targets=None, families=None):
    p, y = np.asarray(prediction, float), np.asarray(targets, float)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 6 or not len(p):
        raise ValueError("prediction/target shape mismatch")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite scoring values")
    if (p[:, [0, 1, 2, 4, 5]] <= 0).any() or ((p[:, 3] < 0) | (p[:, 3] > 100)).any():
        raise ValueError("physically invalid prediction")
    q = quantize_predictions(p)
    bucket_error_mb = np.abs(q[:, 4:6] - y[:, 4:6]) * MIB_BYTES / 1e6
    raw = np.asarray(raw_targets, float) if raw_targets is not None else None
    if raw is not None and (raw.shape != y.shape or not np.isfinite(raw).all()):
        raise ValueError("invalid raw targets")
    reference = raw if raw is not None else y
    memory_error_mb = np.abs(q[:, 4:6] - reference[:, 4:6]) * MIB_BYTES / 1e6
    memory_counts = (memory_error_mb <= 100 + 1e-7).sum(0)
    memory_hit = memory_counts / len(y)
    time = [pair_metrics(p[:, c], y[:, c]) for c in (0, 1)]
    busy = pair_metrics(p[:, 3], y[:, 3], "busy")
    def accuracy(m, group):
        value = m[group]["accuracy"]
        return value if value is not None else 0.
    time_score = np.mean([.6 * accuracy(m, "material") + .4 * accuracy(m, "close_material") for m in time])
    busy_score = accuracy(busy, "material")
    traces = {str(load): replay_arrivals(p[:, 0], y[:, 0], seed=seed, load=load)
              for seed, load in [(5, .4), (17, .7), (29, .9)]}
    schedule_ratio = float(np.mean([r["predicted_to_fcfs_mean_jct"] for r in traces.values()]))
    schedule_score = float(np.clip(1. - schedule_ratio / 2., 0., 1.))
    score = float(.50 * time_score + .30 * np.mean(memory_hit) + .10 * busy_score + .10 * schedule_score)
    relative = np.abs(p-y) / np.maximum(np.abs(y), 1e-8)
    legacy_counts = (relative <= .05).sum(0)
    report = {"version": VERSION, "rows": len(y), "score": score,
        "timing": {"wall": time[0], "gpu": time[1]}, "nvml_busy": busy,
        "vram": {"unit": "decimal_MB", "bucket_MB": 100, "minimum_MB": 100,
                 "within_100MB_counts": memory_counts.tolist(), "within_100MB": memory_hit.tolist(),
                 "error_reference": "raw_v4_measurements" if raw is not None else "v5_bucket_targets_only",
                 "exact_bucket_accuracy": np.mean(bucket_error_mb < 1e-7, axis=0).tolist(),
                 "within_one_bucket_accuracy": np.mean(bucket_error_mb <= 100 + 1e-7, axis=0).tolist(),
                 "mae_MB": memory_error_mb.mean(0).tolist(), "p90_error_MB": np.quantile(memory_error_mb, .9, axis=0).tolist()},
        "online_replay": {"protocol": "causal_nonpreemptive_two_servers_measured_single_step_proxy", "traces": traces},
        "counts": {"memory_100MB": memory_counts.tolist(), "timing_material": [m["material"] for m in time], "busy_material": busy["material"]},
        "hit5": dict(zip(TARGET_NAMES, (legacy_counts/len(y)).tolist())),
        "legacy_hit5_diagnostic_only": True,
        "median_relative_error": dict(zip(TARGET_NAMES, np.median(relative, axis=0).tolist())),
        "p90_relative_error": dict(zip(TARGET_NAMES, np.quantile(relative, .9, axis=0).tolist())),
        "vram_guard_passed": bool((memory_hit >= .95).all()),
        "all_six_validation_gate_passed": False,
        "v5_validation_gate_passed": bool((memory_hit >= .95).all() and
             all(accuracy(m, "material") >= .95 and accuracy(m, "close_material") >= .85 for m in time)
             and busy_score >= .95 and schedule_ratio <= 1.0),
        "teacher_accepted": False, "test_status": "NOT_EVALUATED", "transfer_status": "NOT_EVALUATED",
        "distillation_status": "NOT_EVALUATED", "scheduling_production_status": "NOT_EVALUATED"}
    if raw is not None:
        if raw.shape != y.shape or not np.isfinite(raw).all():
            raise ValueError("invalid raw targets")
        shortfall = (raw[:, 4:6] - q[:, 4:6]) * MIB_BYTES / 1e6
        report["vram"]["raw_underprediction_over_100MB_fraction"] = (shortfall > 100).mean(0).tolist()
    if families is not None:
        f = np.asarray(families)
        if len(f) != len(y): raise ValueError("family alignment mismatch")
        report["within_family_wall"] = {str(g): pair_metrics(p[f==g, 0], y[f==g, 0]) for g in np.unique(f)}
    return report


def pairwise_loss(scores, targets, kind="time"):
    import torch
    from torch.nn import functional as F
    s, y = scores.float().reshape(-1), targets.float().reshape(-1)
    i, j = torch.triu_indices(len(s), len(s), 1, device=s.device)
    if not len(i): return s.sum() * 0.
    if kind == "time":
        delta = y.clamp_min(1e-8).log()[i] - y.clamp_min(1e-8).log()[j]
        keep = delta.abs() >= math.log(1.10)
        # Close-but-material pairs get equal priority to broadly separated pairs.
        weight = torch.where(delta.abs() < math.log(1.25), 2., 1.)
    else:
        delta = y[i] - y[j]; keep = delta.abs() >= 5.; weight = torch.ones_like(delta)
    if not keep.any(): return s.sum() * 0.
    return (F.softplus(-delta[keep].sign() * (s[i]-s[j])[keep]) * weight[keep]).mean()


def memory_interval_loss(prediction, target):
    """Train a continuous value inside the desired output bucket, not at its edge."""
    import torch
    from torch.nn import functional as F
    p = prediction.float() / BUCKET_MIB
    t = target.float() / BUCKET_MIB
    low, high = (t-1).clamp_min(0), t
    below, above = (low-p).clamp_min(0), (p-high).clamp_min(0)
    outside = 2 * F.smooth_l1_loss(below, torch.zeros_like(below)) + F.smooth_l1_loss(above, torch.zeros_like(above))
    center = (low+high)/2
    return outside + .05 * F.smooth_l1_loss(p, center)

