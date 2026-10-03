"""Seven training-target losses, exact hit metrics, and training-only distillation."""

from dataclasses import fields, replace
import random
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from perfseer_v3.features import batch_graph_features
from perfseer_v31.io import atomic_write, fingerprint
from perfseer_v31.training import T1, S1, build_optimizers, build_schedulers

from .model import SeerNetV4, SeerNetV4Config
from .version import (CHECKPOINT_VERSION, OUTPUT_CONTRACT_VERSION, INPUT_SCHEMA_VERSION, FEATURE_VERSION,
                      LOSS_VERSION, METRIC_VERSION, SELECTION_VERSION, TARGET_NAMES, HEAD_GROUPS,
                      MODES, SM_INDICES, POSITIVE_INDICES)


class TrainingBatch(NamedTuple):
    training: object
    resources: object = None


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def validate_tensors(prediction, target):
    if prediction.ndim != 2 or prediction.shape != target.shape or target.size(1) != len(TARGET_NAMES):
        raise ValueError("predictions and targets must have shape [batch, 7]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite prediction or target")
    if (target[:, POSITIVE_INDICES] <= 0).any() or (target[:, SM_INDICES] < 0).any() or (target[:, SM_INDICES] > 100).any():
        raise ValueError("target outside physical range")


def loss_denominators(target):
    scale = target.float().clone()
    scale[:, SM_INDICES] = scale[:, SM_INDICES].clamp_min(1.0)
    return scale


def normalized_errors(prediction, target):
    prediction, target = prediction.float(), target.float()
    validate_tensors(prediction, target)
    return (prediction - target).abs() / loss_denominators(target)


def regression_loss(prediction, target):
    errors = normalized_errors(prediction, target)
    return torch.stack([errors[:, indices].mean() for indices in HEAD_GROUPS]).mean()


def quality_gate(metrics):
    if metrics.get("metric_version") != METRIC_VERSION or tuple(metrics.get("target_names", ())) != TARGET_NAMES:
        return False
    counts, rows = metrics.get("within_5pct_count", []), metrics.get("rows", 0)
    return (type(rows) is int and rows > 0 and len(counts) == len(TARGET_NAMES)
            and all(type(count) is int and 0 <= count <= rows and 20 * count >= 19 * rows for count in counts))


def selection_key(metrics, epoch):
    if metrics.get("metric_version") != METRIC_VERSION or tuple(metrics.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("invalid validation metric contract")
    rows = metrics.get("rows", 0)
    five, ten = metrics.get("within_5pct_count", []), metrics.get("within_10pct_count", [])
    if type(rows) is not int or rows <= 0 or len(five) != len(TARGET_NAMES) or len(ten) != len(TARGET_NAMES) or any(
        type(a) is not int or type(b) is not int or not 0 <= a <= b <= rows for a, b in zip(five, ten, strict=True)
    ):
        raise ValueError("invalid validation hit counts")
    return -min(five) / rows, -sum(five) / (len(TARGET_NAMES) * rows), -min(ten) / rows, -sum(ten) / (len(TARGET_NAMES) * rows), int(epoch)


def metrics_from_predictions(prediction, target):
    # Export these exact FP32 values; all evaluation arithmetic is float64.
    prediction, target = prediction.detach().float().cpu(), target.detach().float().cpu()
    validate_tensors(prediction, target)
    if len(target) == 0:
        raise ValueError("cannot evaluate an empty split")
    absolute = (prediction.double() - target.double()).abs()
    relative = absolute / target.double().abs().clamp_min(1e-6)
    rows = len(target)
    five, ten = (relative <= .05).sum(0).tolist(), (relative <= .1).sum(0).tolist()
    zeros = {}
    for index in SM_INDICES:
        mask = target[:, index] == 0
        zeros[TARGET_NAMES[index]] = {"rows": int(mask.sum()),
                                     "mae": float(absolute[mask, index].mean()) if mask.any() else None,
                                     "within_5pct_count": int((relative[mask, index] <= .05).sum())}
    result = {"metric_version": METRIC_VERSION, "target_names": list(TARGET_NAMES), "rows": rows,
              "within_5pct_count": five, "within_10pct_count": ten,
              "within_5pct_accuracy": dict(zip(TARGET_NAMES, [value / rows for value in five], strict=True)),
              "within_10pct_accuracy": dict(zip(TARGET_NAMES, [value / rows for value in ten], strict=True)),
              "mae": absolute.mean(0).tolist(), "relative_mean_error": relative.mean(0).tolist(),
              "loss_version": LOSS_VERSION, "group_balanced_loss": float(regression_loss(prediction, target)),
              "zero_sm": zeros}
    result["gate_passed"] = quality_gate(result)
    return result


def to_batch(samples, device):
    batches = {}
    for mode in MODES:
        batch = batch_graph_features([getattr(sample["features"], mode) for sample in samples])
        batches[mode] = replace(batch, **{field.name: getattr(batch, field.name).to(device)
                                         for field in fields(batch) if isinstance(getattr(batch, field.name), torch.Tensor)})
    resources = [getattr(sample["features"], "resources", None) for sample in samples]
    if any(value is not None for value in resources):
        if any(value is None for value in resources):
            raise ValueError("mixed resource feature availability")
        batches["resources"] = torch.cat(resources).to(device)
    return TrainingBatch(**batches)


def distillation_loss(student, teacher, targets, representation_weight=.05):
    hard = regression_loss(student.prediction, targets)
    validate_tensors(teacher.prediction, targets)
    soft_errors = (student.prediction.float() - teacher.prediction.detach().float()).abs() / loss_denominators(targets)
    soft = torch.stack([soft_errors[:, indices].mean() for indices in HEAD_GROUPS]).mean()
    if not torch.equal(student.phase_presence, teacher.phase_presence):
        raise ValueError("teacher/student phase presence differs")

    def relations(output):
        value = torch.cat((output.graph_embedding[:, :, None], output.phase_embedding), dim=2).float()
        value = F.normalize(value, dim=-1)
        return value @ value.transpose(-1, -2)

    presence = torch.cat((torch.ones_like(student.phase_presence[:, :, :1]), student.phase_presence), dim=2)
    mask = presence[:, :, :, None] & presence[:, :, None, :]
    relation_errors = F.smooth_l1_loss(relations(student), relations(teacher).detach(), reduction="none")
    relation = ((relation_errors * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)).mean()
    return .6 * hard + .4 * soft + representation_weight * relation


def train_batch(model, samples, optimizers, microbatch, *, teacher=None, amp=True):
    device = next(model.parameters()).device
    if teacher is not None:
        teacher.eval()
    # On allocation failure retry the complete logical batch, so no partial
    # gradients, dropped rows, or changes to the measured workload survive.
    initial_rng = rng_state()
    while True:
        restore_rng(initial_rng)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total = 0.0
        batch = targets = output = teacher_output = loss = weighted = None
        try:
            for start in range(0, len(samples), microbatch):
                subset = samples[start:start + microbatch]
                batch = to_batch(subset, device)
                targets = torch.stack([sample["target"] for sample in subset]).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                    output = model.predict_batch(batch)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_output = teacher.predict_batch(batch)
                loss = regression_loss(output.prediction, targets) if teacher is None else distillation_loss(output, teacher_output, targets)
                weighted = loss * (len(subset) / len(samples))
                weighted.backward()
                total += float(weighted.detach())
                batch = targets = output = teacher_output = loss = weighted = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            break
        except torch.cuda.OutOfMemoryError:
            batch = targets = output = teacher_output = loss = weighted = None
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch //= 2
            torch.cuda.empty_cache()
    for optimizer in optimizers:
        optimizer.step()
    return total, microbatch



def evaluation_config(model, microbatch, amp=True):
    device = next(model.parameters()).device
    return {"amp": bool(amp and device.type == "cuda"), "backbone_precision": "bfloat16" if amp and device.type == "cuda" else "float32",
            "head_precision": "float32", "metric_precision": "float64_from_float32",
            "device_type": device.type, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": str(torch.__version__), "requested_microbatch": microbatch,
            "cuda_version": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


@torch.no_grad()
def evaluate(model, samples, microbatch, *, amp=True, prediction_path=None, identity=None):
    if microbatch < 1 or not len(samples):
        raise ValueError("evaluation requires samples and a positive microbatch")
    model.eval()
    device = next(model.parameters()).device
    configuration = evaluation_config(model, microbatch, amp)
    start, predictions, targets, rows, batch_sizes = 0, [], [], [], []
    batch = target = prediction = None
    while start < len(samples):
        subset = samples[start:start + microbatch]
        try:
            batch = to_batch(subset, device)
            target = torch.stack([sample["target"] for sample in subset]).float()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=configuration["amp"]):
                prediction = model.predict_batch(batch).prediction.float().cpu()
            predictions.append(prediction)
            targets.append(target.cpu())
            if prediction_path is not None:
                rows.extend({"sample_id": sample["sample_id"], "prediction": prediction[i].tolist(),
                             "target": target[i].tolist(), "batch_index": len(batch_sizes)} for i, sample in enumerate(subset))
            batch_sizes.append(len(subset))
            start += len(subset)
            batch = target = prediction = None
        except torch.cuda.OutOfMemoryError:
            batch = target = prediction = None
            if microbatch == 1:
                raise
            microbatch //= 2
            torch.cuda.empty_cache()
    metrics = metrics_from_predictions(torch.cat(predictions), torch.cat(targets))
    metrics["evaluation"] = configuration
    if prediction_path is not None:
        atomic_write(prediction_path, {"metric_version": METRIC_VERSION, "target_names": list(TARGET_NAMES),
                     "identity": identity or {}, "evaluation": configuration, "batch_sizes": batch_sizes,
                     "metrics": metrics, "rows": rows}, compress=True)
    return metrics


def checkpoint_payload(model, *, role, epoch, normalization, dataset_fingerprint, optimizers=(), schedulers=(), **progress):
    return {"version": CHECKPOINT_VERSION, "output_contract": OUTPUT_CONTRACT_VERSION,
            "input_schema": INPUT_SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "metric_version": METRIC_VERSION, "loss_version": LOSS_VERSION, "selection_version": SELECTION_VERSION,
            "target_names": TARGET_NAMES, "model_variant": model.model_variant, "role": role, "epoch": epoch,
            **({"adapter_config": model.adapter_config} if model.model_variant == "v4.2" else {}),
            "model_config": model.config.to_dict(), "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "target_scales": model.target_scales.detach().cpu().clone(), "normalization": normalization,
            "normalization_sha256": fingerprint(normalization),
            "dataset_fingerprint": dataset_fingerprint, "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in schedulers], "rng": rng_state(), **progress}


def restore_model(payload, *, dataset_fingerprint=None, device="cpu"):
    expected = {"version": CHECKPOINT_VERSION, "output_contract": OUTPUT_CONTRACT_VERSION,
                "input_schema": INPUT_SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
                "metric_version": METRIC_VERSION, "loss_version": LOSS_VERSION, "selection_version": SELECTION_VERSION}
    if any(payload.get(key) != value for key, value in expected.items()) or tuple(payload.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("checkpoint is not a compatible v4 seven-output artifact; use explicit checkpoint conversion")
    if dataset_fingerprint is not None and payload["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("checkpoint dataset fingerprint differs")
    if payload.get("normalization_sha256") != fingerprint(payload["normalization"]):
        raise ValueError("checkpoint normalization hash differs")
    variant = payload.get("model_variant")
    if variant == "v4.3":
        from .resource_model import ResourceMLP, ResourceMLPConfig
        state = payload["model_state_dict"]
        model = ResourceMLP(ResourceMLPConfig(**payload["model_config"]), payload["target_scales"],
                            state["resource_mean"], state["resource_scale"])
    elif variant in {"v4.0", "v4.2"}:
        model = SeerNetV4(SeerNetV4Config(**payload["model_config"]), payload["target_scales"])
        if variant == "v4.2":
            from .adapters import HardwareAdaptedV4
            model = HardwareAdaptedV4(model, **payload["adapter_config"], require_trained_source=False)
    else:
        raise ValueError("unsupported model variant")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if not torch.equal(model.target_scales.cpu(), torch.as_tensor(payload["target_scales"]).cpu()):
        raise ValueError("checkpoint target scales differ from weights")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    return model.to(device)
