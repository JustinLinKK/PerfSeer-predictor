"""Masked seven-output campaign objectives; legacy dense contracts stay intact."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from .version import HEAD_GROUPS, TARGET_NAMES

VERSION = "perfseer_v4_masked_objective_v1"


def target_tensors(rows, device="cpu"):
    values, masks = [], []
    for row in rows:
        mask = row.get("target_mask", {name: True for name in TARGET_NAMES})
        if set(mask) != set(TARGET_NAMES) or any(type(v) is not bool for v in mask.values()):
            raise ValueError("seven explicit boolean target masks required")
        value = row["targets"]
        for index, name in enumerate(TARGET_NAMES):
            item = value[name]
            if not mask[name]:
                if item is not None:
                    raise ValueError("unobserved targets must be null")
            elif (isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                  or (not 0 <= item <= 100 if index == 3 else item <= 0)):
                raise ValueError("invalid observed target")
        values.append([value[name] if mask[name] else 1. for name in TARGET_NAMES])
        masks.append([mask[name] for name in TARGET_NAMES])
    return torch.tensor(values, dtype=torch.float32, device=device), torch.tensor(masks, dtype=torch.bool, device=device)


def denominators(target):
    result = target.float().clone()
    result[:, 3] = result[:, 3].clamp_min(1.)
    return result


def validate(prediction, target, mask):
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 7 or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("masked prediction contract requires [batch,7]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("nonfinite prediction or observed target")


def head_weights(mask):
    """Per-observation weights for the complete logical batch, before slicing."""
    counts = mask.sum(0)
    active = counts > 0
    heads = [indices for indices in HEAD_GROUPS if active[list(indices)].any()]
    if not heads:
        raise ValueError("batch contains no observed targets")
    weights = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    for indices in heads:
        valid = [i for i in indices if active[i]]
        for i in valid:
            weights[:, i] = mask[:, i] / counts[i] / len(valid) / len(heads)
    return weights


def loss(prediction, target, mask, *, weights=None):
    validate(prediction, target, mask)
    safe = torch.where(mask, target, torch.ones_like(target))
    errors = (prediction.float() - safe).abs() / denominators(safe)
    return (torch.where(mask, errors, 0.) * (head_weights(mask) if weights is None else weights)).sum()


def distill(student, teacher, target, mask, *, weights=None, relation_weight=.05):
    validate(teacher.prediction, target, mask)
    hard = loss(student.prediction, target, mask, weights=weights)
    safe = torch.where(mask, target, torch.ones_like(target))
    errors = (student.prediction.float() - teacher.prediction.detach().float()).abs() / denominators(safe)
    soft = (torch.where(mask, errors, 0.) * (head_weights(mask) if weights is None else weights)).sum()
    relation = hard * 0
    if relation_weight:
        if not torch.equal(student.phase_presence, teacher.phase_presence):
            raise ValueError("teacher/student phase presence differs")
        def relations(output):
            value = torch.cat((output.graph_embedding[:, :, None], output.phase_embedding), dim=2).float()
            value = F.normalize(value, dim=-1)
            return value @ value.transpose(-1, -2)
        presence = torch.cat((torch.ones_like(student.phase_presence[:, :, :1]), student.phase_presence), dim=2)
        pairs = presence[:, :, :, None] & presence[:, :, None, :]
        errors = F.smooth_l1_loss(relations(student), relations(teacher).detach(), reduction="none")
        relation = ((errors * pairs).sum((-1, -2)) / pairs.sum((-1, -2)).clamp_min(1)).mean()
    return .6 * hard + .4 * soft + relation_weight * relation


def metrics(prediction, target, mask):
    prediction, target, mask = prediction.detach().float().cpu(), target.detach().float().cpu(), mask.cpu()
    validate(prediction, target, mask)
    absolute = (prediction.double() - target.double()).abs()
    relative = absolute / target.double().abs().clamp_min(1e-6)
    counts = mask.sum(0).tolist()
    five = ((relative <= .05) & mask).sum(0).tolist()
    ten = ((relative <= .10) & mask).sum(0).tolist()
    def per_target(value, method):
        return [float(method(value[:, i][mask[:, i]])) if counts[i] else None for i in range(7)]
    return {"version": VERSION, "target_names": list(TARGET_NAMES), "rows": len(target),
            "observed_count": counts, "within_5pct_count": five, "within_10pct_count": ten,
            "mae": per_target(absolute, torch.mean), "relative_mean_error": per_target(relative, torch.mean),
            "relative_median_error": per_target(relative, torch.median),
            "relative_p95_error": per_target(relative, lambda x: torch.quantile(x, .95)),
            "selection_error": float(loss(prediction, target, mask)),
            "gate_passed": all(n > 0 and 20 * k >= 19 * n for k, n in zip(five, counts)),
            "unmeasured_targets": [name for name, n in zip(TARGET_NAMES, counts) if not n]}


def verify_prediction_report(report, source_rows=None):
    """Independent NumPy reconstruction; do not reuse the training loss/metrics."""
    rows = report["rows"]
    if "prediction_runtime" in report:
        from perfseer_v31.io import fingerprint
        runtime = report["prediction_runtime"]
        if (fingerprint(runtime) != report["prediction_runtime_sha256"] or sum(runtime["batch_sizes"]) != len(rows)
                or any(type(n) is not int or n < 1 for n in runtime["batch_sizes"])):
            raise ValueError("actual prediction runtime trace differs")
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate prediction rows")
    pred = np.asarray([r["prediction"] for r in rows], dtype=np.float32).astype(np.float64)
    target = np.asarray([r["target"] for r in rows], dtype=np.float32).astype(np.float64)
    mask = np.asarray([r["mask"] for r in rows], dtype=bool)
    if pred.shape != target.shape or mask.shape != target.shape or pred.shape[1] != 7 or not np.isfinite(pred).all():
        raise ValueError("invalid prediction export")
    if source_rows is not None:
        if [r["sample_id"] for r in rows] != [r["sample_id"] for r in source_rows]:
            raise ValueError("prediction rows differ from the frozen dataset")
        native_target = np.asarray([[r["targets"][name] if r["target_mask"][name] else 1. for name in TARGET_NAMES]
                                    for r in source_rows], dtype=np.float32).astype(np.float64)
        native_mask = np.asarray([[r["target_mask"][name] for name in TARGET_NAMES] for r in source_rows], dtype=bool)
        if not np.array_equal(target, native_target) or not np.array_equal(mask, native_mask):
            raise ValueError("prediction labels or masks differ from the frozen dataset")
    absolute = abs(pred - target)
    relative = absolute / np.maximum(abs(target), 1e-6)
    counts = mask.sum(0)
    expected = report["metrics"]
    for key, value in (("observed_count", counts), ("within_5pct_count", ((relative <= .05) & mask).sum(0)),
                       ("within_10pct_count", ((relative <= .1) & mask).sum(0))):
        if value.tolist() != expected[key]:
            raise ValueError(f"independent {key} differs")
    denom = target.copy()
    denom[:, 3] = np.maximum(denom[:, 3], 1.)
    heads = []
    for indices in HEAD_GROUPS:
        observed = [float(np.mean((absolute / denom)[:, i][mask[:, i]])) for i in indices if counts[i]]
        if observed:
            heads.append(float(np.mean(observed)))
    if not math.isclose(float(np.mean(heads)), expected["selection_error"], rel_tol=2e-6, abs_tol=1e-7):
        raise ValueError("independent selection error differs")
    for i, n in enumerate(counts):
        observed = relative[:, i][mask[:, i]]
        checks = {"mae": float(absolute[:, i][mask[:, i]].mean()) if n else None,
                  "relative_mean_error": float(observed.mean()) if n else None,
                  # torch.median uses the lower middle value for even counts.
                  "relative_median_error": float(np.sort(observed)[(n - 1) // 2]) if n else None,
                  "relative_p95_error": float(np.quantile(observed, .95)) if n else None}
        for key, value in checks.items():
            if (value is None and expected[key][i] is not None) or (value is not None and
                    not math.isclose(value, expected[key][i], rel_tol=1e-8, abs_tol=1e-8)):
                raise ValueError(f"independent {key} differs")
    if expected["unmeasured_targets"] != [name for name, n in zip(TARGET_NAMES, counts) if not n]:
        raise ValueError("independent missing-target list differs")
    passed = bool(all(n > 0 and 20 * k >= 19 * n for n, k in zip(counts, expected["within_5pct_count"])))
    if passed != expected["gate_passed"] or expected["rows"] != len(rows):
        raise ValueError("independent gate or row count differs")
    return {"status": "passed", "rows": len(rows), "observed_count": counts.tolist()}
