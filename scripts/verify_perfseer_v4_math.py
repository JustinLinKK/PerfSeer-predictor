#!/usr/bin/env python
"""Deterministic v4 algebra checks; no training campaign or accuracy claim."""

from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

import run_perfseer_v32_transfer_labeling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perfseer_v32.calibration import group_weights, ridge
from perfseer_v4.adapters import TrainingHeadAdapter
from perfseer_v4.residuals import corrected_values, gp_fit, gp_predict, residual_values
from perfseer_v4.version import TARGET_NAMES


def run_checks():
    rng = np.random.default_rng(29)
    torch.manual_seed(29)
    checks = []

    def check(name, passed, detail):
        if not bool(passed):
            raise AssertionError((name, detail))
        checks.append({"name": name, "passed": True, "detail": detail})

    groups = ["a"] * 8 + ["b"] * 3 + ["c"]
    weights = group_weights([{"group_id": group} for group in groups])
    counts = Counter(groups)
    reference = np.asarray([1 / (len(counts) * counts[group]) for group in groups])
    check("group_balancing", np.array_equal(weights, reference) and all(
        math.isclose(sum(weight for group, weight in zip(groups, weights) if group == key), 1 / 3)
        for key in counts), {"groups": len(counts), "weight_sum": float(weights.sum())})

    phi = np.c_[np.ones(len(groups)), rng.normal(size=(len(groups), 4))]
    residual = rng.normal(size=len(groups))
    lam = .03
    coefficients = ridge(phi, residual, weights, lam)
    matrix = phi.T @ (weights[:, None] * phi) + lam * np.eye(phi.shape[1])
    rhs = phi.T @ (weights * residual)
    gradient = 2 * (matrix @ coefficients - rhs)
    check("ridge_stationarity", np.linalg.norm(gradient) < 1e-12,
          {"gradient_norm": float(np.linalg.norm(gradient))})
    check("ridge_normal_equation_parity", np.allclose(coefficients, np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12),
          "augmented least squares agrees with independently reconstructed normal equations")
    check("ridge_strict_convexity", np.linalg.eigvalsh(2 * matrix).min() >= 2 * lam - 1e-12,
          {"minimum_hessian_eigenvalue": float(np.linalg.eigvalsh(2 * matrix).min())})
    rank_deficient = np.c_[np.ones(8), np.ones(8), np.zeros(8)]
    degenerate = ridge(rank_deficient, np.ones(8), np.ones(8), lam)
    check("ridge_rank_deficient", np.isfinite(degenerate).all() and np.allclose(degenerate, [1 / (2 + lam), 1 / (2 + lam), 0]),
          {"coefficients": degenerate.tolist()})
    check("ridge_weight_normalization", np.allclose(coefficients, ridge(phi, residual, weights * 17, lam), rtol=1e-12, atol=1e-12),
          "multiplying all positive weights by a constant does not change the normalized objective")

    source = np.asarray([2., 1., 24.])
    truth = np.asarray([2.4, .8, 30.])
    timing_residual = np.log(truth / source)
    check("timing_residual_inverse", np.allclose(source * np.exp(timing_residual), truth, rtol=1e-14, atol=1e-14),
          "residual uses the actual frozen source prediction")
    mib = 1024 ** 2
    memory_source = np.asarray([16., 64., 256.]) * mib
    memory_truth = np.asarray([32., 48., 384.]) * mib
    memory_scale = np.maximum(memory_source, 64 * mib)
    memory_residual = (memory_truth - memory_source) / memory_scale
    check("memory_bytes_residual_inverse", np.array_equal(memory_source + memory_scale * memory_residual, memory_truth)
          and np.array_equal(memory_residual, [.25, -.25, .5]),
          {"memory_scale_bytes": memory_scale.tolist(), "residual": memory_residual.tolist()})
    sm_source, sm_truth = np.asarray([0., 20., 100.]), np.asarray([5., 10., 95.])
    sm_residual = (sm_truth - sm_source) / 100
    check("sm_residual_inverse", np.array_equal(np.clip(sm_source + 100 * sm_residual, 0, 100), sm_truth),
          "SM correction uses percentage points divided by 100")
    raw = np.asarray([-50., 0., 20., 150.])
    bounded_truth = np.asarray([0., 10., 25., 100.])
    check("physical_projection", np.all(np.abs(np.clip(raw, 0, 100) - bounded_truth) <= np.abs(raw - bounded_truth))
          and np.all(np.maximum(raw, 0) >= 0), "projection preserves physical bounds and cannot increase error against in-range truth")
    for target, base_values, truth_values, expected_residual in (
        (TARGET_NAMES[0], source, truth, timing_residual),
        (TARGET_NAMES[5], memory_source / mib, memory_truth / mib, memory_residual),
        (TARGET_NAMES[3], sm_source, sm_truth, sm_residual),
    ):
        actual_residual = residual_values(base_values, truth_values, target)
        actual_prediction, actual_raw = corrected_values(base_values, actual_residual, target)
        check("implemented_residual_inverse_" + target, np.allclose(actual_residual, expected_residual, rtol=1e-14, atol=1e-14)
              and np.allclose(actual_prediction, truth_values, rtol=1e-14, atol=1e-14)
              and np.allclose(actual_raw, truth_values, rtol=1e-14, atol=1e-14),
              "implemented residual and inverse agree with independent physical-unit equations")
    projected_memory, raw_memory = corrected_values([16., 256.], [-1., -2.], TARGET_NAMES[5])
    projected_sm, raw_sm = corrected_values([20., 90.], [-1., 1.], TARGET_NAMES[3])
    check("implemented_physical_projection", np.array_equal(projected_memory, [0., 0.])
          and np.array_equal(raw_memory, [-48., -256.]) and np.array_equal(projected_sm, [0., 100.])
          and np.array_equal(raw_sm, [-80., 190.]), "raw out-of-range corrections remain available for diagnostics")

    hidden, rank = 12, 8
    embedding = torch.randn(3, hidden, dtype=torch.float64)
    a = torch.randn(rank, hidden, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(hidden, rank, dtype=torch.float64, requires_grad=True)
    gamma = torch.zeros(3, hidden, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(3, hidden, dtype=torch.float64, requires_grad=True)
    adapted = (1 + gamma) * embedding + beta + embedding @ a.T @ b.T / rank
    check("adapter_identity_initialization", torch.equal(adapted, embedding), "zero FiLM outputs and B preserve the source embedding exactly")
    upstream = torch.randn_like(adapted)
    (adapted * upstream).sum().backward()
    expected_b = upstream.T @ embedding @ a.detach().T / rank
    check("adapter_initial_gradients", torch.equal(a.grad, torch.zeros_like(a))
          and torch.allclose(b.grad, expected_b, rtol=1e-12, atol=1e-12)
          and torch.equal(gamma.grad, upstream * embedding) and torch.equal(beta.grad, upstream)
          and b.grad.abs().sum() > 0,
          "B and FiLM can update at identity; A initially receives zero gradient")
    implemented_adapter = TrainingHeadAdapter(hidden, hardware_dim=6, rank=rank).double()
    hardware = torch.randn(3, 6, dtype=torch.float64)
    implemented_output = implemented_adapter(embedding, hardware)
    check("implemented_adapter_identity", torch.equal(implemented_output, embedding),
          "the actual TrainingHeadAdapter starts at exact identity")
    (implemented_output * upstream).sum().backward()
    expected_up = upstream.T @ embedding @ implemented_adapter.down.weight.detach().T / rank
    film_bias_gradient = torch.cat(((upstream * embedding).sum(0), upstream.sum(0)))
    check("implemented_adapter_initial_gradients", torch.equal(implemented_adapter.down.weight.grad,
          torch.zeros_like(implemented_adapter.down.weight)) and torch.allclose(
          implemented_adapter.up.weight.grad, expected_up, rtol=1e-12, atol=1e-12) and torch.allclose(
          implemented_adapter.film[-1].bias.grad, film_bias_gradient, rtol=1e-12, atol=1e-12),
          "actual adapter gradients agree with independent matrix calculus")

    x = np.asarray([[0., 0.], [0., 0.], [1., -1.], [2., 1.]])
    y = np.asarray([.2, .1, -.1, .3])
    gp_weights = np.asarray([.125, .125, .25, .5])
    queries = np.asarray([[0., 0.], [.5, 0.], [8., 8.]])
    length_scale, regularization, jitter = .7, .02, 1e-8

    def kernel(left, right):
        squared = ((left[:, None] - right[None, :]) ** 2).sum(-1)
        return np.exp(-squared / (2 * length_scale ** 2))

    k = kernel(x, x)
    cross = kernel(queries, x)
    covariance = k + regularization * np.diag(1 / gp_weights) + jitter * np.eye(len(x))
    chol = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    mean = cross @ alpha
    solved = np.linalg.solve(chol, cross.T)
    variance = 1 - (solved ** 2).sum(0)
    independent_mean = cross @ np.linalg.solve(covariance, y)
    independent_variance = np.diag(kernel(queries, queries) - cross @ np.linalg.solve(covariance, cross.T))
    check("weighted_gp_mean", np.allclose(mean, independent_mean, rtol=1e-12, atol=1e-12),
          {"posterior_mean": mean.tolist()})
    check("weighted_gp_latent_variance", np.allclose(variance, independent_variance, rtol=1e-12, atol=1e-12)
          and np.all((variance >= 0) & (variance <= 1)), {"latent_variance": variance.tolist()})
    check("weighted_gp_duplicate_inputs", np.isfinite(chol).all() and np.linalg.eigvalsh(covariance).min() > 0,
          "positive weighted observation noise makes duplicate-input covariance strictly positive definite")
    check("weighted_gp_far_field", abs(mean[-1]) < 1e-12 and abs(variance[-1] - 1) < 1e-12,
          "far-field residual mean tends to zero and latent variance tends to unit prior variance")
    fitted_gp = gp_fit(x, y, gp_weights, regularization, length_scale)
    implemented_mean, implemented_variance = gp_predict(fitted_gp, queries)
    check("implemented_weighted_gp_posterior", np.allclose(implemented_mean, independent_mean, rtol=1e-12, atol=1e-12)
          and np.allclose(implemented_variance, independent_variance, rtol=1e-12, atol=1e-12)
          and np.allclose(np.asarray(fitted_gp["cholesky"]) @ np.asarray(fitted_gp["cholesky"]).T,
                          covariance, rtol=1e-12, atol=1e-12),
          "actual GP covariance, posterior mean and latent variance agree with independent equations")
    restored_mean, restored_variance = gp_predict(json.loads(json.dumps(fitted_gp)), queries)
    check("gp_serialization_parity", np.array_equal(restored_mean, implemented_mean)
          and np.array_equal(restored_variance, implemented_variance), "JSON serialization preserves exact posterior predictions")
    log_mean, log_variance = .2, .3
    median = 2 * math.exp(log_mean)
    arithmetic_mean = 2 * math.exp(log_mean + log_variance / 2)
    check("lognormal_median_not_mean", arithmetic_mean > median and math.isclose(arithmetic_mean / median, math.exp(log_variance / 2)),
          {"source_times_exp_posterior_mean_is_median": median, "lognormal_arithmetic_mean": arithmetic_mean})

    return {"passed": len(checks), "failed": 0,
            "scope": "algebra and deterministic synthetic numerical checks only; no transfer accuracy evidence",
            "checks": checks}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
