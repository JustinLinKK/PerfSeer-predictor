"""Binary, device-aware exact ridge/GP teachers for the full target corpus."""

import math

import numpy as np
import torch

from perfseer_v31.io import fingerprint
from perfseer_v32.calibration import LAMBDA_GRID, group_weights
from .residuals import LENGTH_SCALES, JITTER, _names, _matrix, residual_values
from .resource_model import RESOURCE_NAMES
from .version import TARGET_NAMES

VERSION = "perfseer_v4_campaign_correction_v1"


def kernel(a, b, length):
    # Avoid allocating [N,N,17] for the full 2,988-row fit.
    squared = (a.square().sum(1)[:, None] + b.square().sum(1)[None, :] - 2 * a @ b.T).clamp_min(0)
    return torch.exp(-.5 * squared / length ** 2)


def corrected(base, delta, index):
    if index < 3:
        result = base * delta.exp()
    elif index == 3:
        result = (base + 100 * delta).clamp(0, 100)
    else:
        result = (base + base.clamp_min(64.) * delta).clamp_min(0.)
    if not torch.isfinite(result).all() or (index < 3 and (result <= 0).any()):
        raise ValueError("nonfinite residual prediction")
    return result


def residual_rows(samples, predictions):
    return [{**sample["row"], "source_prediction": dict(zip(TARGET_NAMES, prediction)),
             "features": dict(zip(RESOURCE_NAMES, sample["features"].resources[0].tolist()))}
            for sample, prediction in zip(samples, predictions.tolist(), strict=True)]


def fit_correction(train, validation, base_train, base_validation, *, variant, binding, device,
                   regularizations=LAMBDA_GRID, lengths=LENGTH_SCALES, progress=None):
    if variant not in {"v4.1", "v4.3"}:
        raise ValueError("unsupported correction")
    if any(s["split"] != "train" for s in train) or any(s["split"] != "validation" for s in validation):
        raise ValueError("correction fitting cannot read test labels")
    groups = {s["group_id"] for s in train}
    if groups & {s["group_id"] for s in validation}:
        raise ValueError("correction group split leakage")
    rows, val_rows = residual_rows(train, base_train), residual_rows(validation, base_validation)
    tasks, shared = [], {}
    method = "gp" if variant == "v4.3" else "linear"
    for index, target in enumerate(TARGET_NAMES):
        fit_rows = [r for r in rows if r["target_mask"][target]]
        valid_rows = [r for r in val_rows if r["target_mask"][target]]
        if not fit_rows or not valid_rows:
            raise ValueError("all target outputs need fit and validation observations")
        names = _names(target, method, len({r["group_id"] for r in fit_rows}))
        weights = torch.as_tensor(group_weights(fit_rows), dtype=torch.float64, device=device)
        values = torch.as_tensor(_matrix(fit_rows, target, names), dtype=torch.float64, device=device)
        mean = (weights[:, None] * values).sum(0)
        scale = (weights[:, None] * (values - mean).square()).sum(0).sqrt()
        if method != "gp":
            mean[0], scale[0] = 0., 1.
        scale = torch.where(scale < 1e-12, 1., scale)
        phi = (values - mean) / scale
        val_phi = (torch.as_tensor(_matrix(valid_rows, target, names), dtype=torch.float64, device=device) - mean) / scale
        base = [r["source_prediction"][target] for r in fit_rows]
        truth = [r["targets"][target] for r in fit_rows]
        residual = torch.as_tensor(residual_values(base, truth, target), dtype=torch.float64, device=device)
        val_base = torch.tensor([r["source_prediction"][target] for r in valid_rows], dtype=torch.float64, device=device)
        val_truth = torch.tensor([r["targets"][target] for r in valid_rows], dtype=torch.float64, device=device)
        best, selected = None, None
        design_key = fingerprint({"names": names, "ids": [r["sample_id"] for r in fit_rows],
                                  "mean": mean.cpu().tolist(), "scale": scale.cpu().tolist()})
        for length in lengths if method == "gp" else (1.,):
            for regularization in regularizations:
                if regularization <= 0 or length <= 0:
                    raise ValueError("positive residual grid required")
                if method == "gp":
                    key = (design_key, float(length), float(regularization))
                    if key not in shared:
                        covariance = kernel(phi, phi, length)
                        covariance.diagonal().add_(regularization / weights + JITTER)
                        shared[key] = torch.linalg.cholesky(covariance)
                    cholesky = shared[key]
                    alpha = torch.cholesky_solve(residual[:, None], cholesky).flatten()
                    delta = kernel(val_phi, phi, length) @ alpha
                    parameters = {"fit_matrix": phi.cpu(), "alpha": alpha.cpu(), "cholesky": cholesky.cpu(),
                                  "fit_weights": weights.cpu(), "length_scale": float(length)}
                else:
                    a = torch.cat((weights.sqrt()[:, None] * phi, math.sqrt(regularization) * torch.eye(phi.shape[1], dtype=phi.dtype, device=device)))
                    b = torch.cat((weights.sqrt() * residual, torch.zeros(phi.shape[1], dtype=phi.dtype, device=device)))
                    coefficients = torch.linalg.lstsq(a, b).solution
                    delta = val_phi @ coefficients
                    parameters = {"coefficients": coefficients.cpu()}
                prediction = corrected(val_base, delta, index)
                error = float(((prediction - val_truth).abs() / (val_truth.clamp_min(1.) if index == 3 else val_truth)).mean())
                key = (error, float(regularization), float(length))
                if best is None or key < best:
                    best = key
                    selected = {"method": method, "feature_names": names, "mean": mean.cpu(), "scale": scale.cpu(),
                                "regularization": float(regularization), "validation_error": error, **parameters}
                if progress:
                    progress({"event": "correction_grid", "target": target, "regularization": regularization, "length": length, "error": error})
        tasks.append(selected)
    return {"version": VERSION, "proposal": variant, "binding": binding, "tasks": tasks,
            "fit_ids": [r["sample_id"] for r in rows], "validation_ids": [r["sample_id"] for r in val_rows],
            "grid": {"regularizations": list(regularizations), "length_scales": list(lengths) if method == "gp" else []}}


class CorrectedTeacher(torch.nn.Module):
    def __init__(self, base, correction):
        super().__init__()
        if correction.get("version") != VERSION:
            raise ValueError("correction version differs")
        self.base, self.correction = base.eval().requires_grad_(False), correction

    def predict_batch(self, batch):
        with torch.autocast(next(self.base.parameters()).device.type, enabled=False):
            output = self.base.predict_batch(batch)
        resources = batch.resources.double()
        values = []
        for index, task in enumerate(self.correction["tasks"]):
            base = output.prediction[:, index].double()
            physical = base * 2 ** 20 if index > 3 else base
            features = {name: resources[:, i] for i, name in enumerate(RESOURCE_NAMES)}
            features.update(intercept=torch.ones_like(base), log_source_prediction=torch.log1p(physical), source_prediction=base)
            phi = (torch.stack([features[name] for name in task["feature_names"]], 1) - task["mean"].to(base.device)) / task["scale"].to(base.device)
            if task["method"] == "gp":
                delta = kernel(phi, task["fit_matrix"].to(base.device), task["length_scale"]) @ task["alpha"].to(base.device)
            else:
                delta = phi @ task["coefficients"].to(base.device)
            values.append(corrected(base, delta, index))
        return output._replace(prediction=torch.stack(values, 1).float())
