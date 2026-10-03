"""Scale-aligned objectives, paired optimizers, and versioned checkpoint contracts."""

import math
import random
from dataclasses import fields, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from perfseer_v3.features import batch_graph_features

from .model import SeerNetV31, SeerNetV31Config
from .version import CHECKPOINT_VERSION, OUTPUT_CONTRACT_VERSION, TARGET_NAMES


T1 = dict(hidden=1280, num_blocks=10, exact_embedding_dim=96, family_embedding_dim=48,
          hash_embedding_dim=24, overload_hash_embedding_dim=24, phase_embedding_dim=24,
          input_dtype_embedding_dim=16, dtype_embedding_dim=16, accumulation_dtype_embedding_dim=16,
          backend_embedding_dim=16, feature_quality_embedding_dim=8, layout_embedding_dim=12,
          rank_embedding_dim=12, optimizer_embedding_dim=8, optimizer_family_embedding_dim=8,
          optimizer_hash_embedding_dim=8, scheduler_embedding_dim=8, scheduler_family_embedding_dim=8,
          scheduler_hash_embedding_dim=8, node_identity_fusion="additive", pooling_mode="phase_aware", checkpoint_blocks=True)
S1 = dict(hidden=224, num_blocks=2, exact_embedding_dim=40, family_embedding_dim=20,
          hash_embedding_dim=10, overload_hash_embedding_dim=10, phase_embedding_dim=8,
          input_dtype_embedding_dim=8, dtype_embedding_dim=8, accumulation_dtype_embedding_dim=8,
          backend_embedding_dim=8, feature_quality_embedding_dim=4, layout_embedding_dim=6,
          rank_embedding_dim=6, node_identity_fusion="additive", pooling_mode="phase_aware")


def normalized_errors(prediction, target):
    prediction, target = prediction.float(), target.float()
    if prediction.ndim != 2 or prediction.shape != target.shape or target.size(1) != 3:
        raise ValueError("predictions and targets must have shape [batch, 3]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite prediction or target")
    if (target[:, (0, 2)] <= 0).any() or (target[:, 1] < 0).any() or (target[:, 1] > 100).any():
        raise ValueError("target outside physical range")
    scale = torch.stack((target[:, 0], torch.full_like(target[:, 1], 100), target[:, 2]), dim=1)
    return (prediction - target).abs() / scale


def regression_loss(prediction, target):
    return normalized_errors(prediction, target).mean()


def quality_gate(errors):
    values = torch.as_tensor(errors, dtype=torch.float64)
    return values.shape == (3,) and bool(torch.isfinite(values).all() and (values >= 0).all() and (values < 0.1).all())


def selection_key(errors, epoch):
    values = [float(value) for value in errors]
    if len(values) != 3 or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("invalid validation errors")
    return max(values), sum(values) / 3, int(epoch)


def build_optimizers(model, lr, weight_decay=1e-5):
    muon, adam_decay, adam_no_decay = [], [], []
    assigned = set()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = {"muon": [], "adamw_decay": [], "adamw_no_decay": []}
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in assigned:
                continue
            assigned.add(id(parameter))
            if isinstance(module, nn.Linear) and name == "weight" and not module_name.startswith("prediction_head"):
                muon.append(parameter)
                group = "muon"
            elif name == "bias" or isinstance(module, nn.LayerNorm) or parameter.ndim < 2:
                adam_no_decay.append(parameter)
                group = "adamw_no_decay"
            else:
                adam_decay.append(parameter)
                group = "adamw_decay"
            groups[group].append(names[id(parameter)])
    if assigned != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("optimizer partition missed parameters")
    if not muon:
        raise ValueError("Muon requires hidden matrix weights")
    optimizers = (
        torch.optim.Muon(muon, lr=lr, weight_decay=weight_decay, momentum=.95,
                         nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW([{"params": adam_decay, "weight_decay": weight_decay},
                           {"params": adam_no_decay, "weight_decay": 0.0}], lr=lr),
    )
    return optimizers, groups


def lr_multiplier(step, *, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def build_schedulers(optimizers, *, epochs, warmup_epochs, steps_per_epoch):
    return tuple(torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps=epochs * steps_per_epoch,
                                             warmup_steps=warmup_epochs * steps_per_epoch)
    ) for optimizer in optimizers)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def to_batch(samples, device):
    batch = batch_graph_features([sample["features"] for sample in samples])
    return replace(batch, **{field.name: getattr(batch, field.name).to(device)
                            for field in fields(batch) if isinstance(getattr(batch, field.name), torch.Tensor)})


def distillation_loss(student, teacher, targets, representation_weight=.05):
    hard = regression_loss(student.prediction, targets)
    soft = regression_loss(student.prediction, teacher.prediction.detach())
    def relations(output):
        # Relations among global/phase representations within each graph are
        # invariant to predictor microbatching and need no width projection.
        value = torch.cat((output.graph_embedding[:, None], output.phase_embedding), dim=1).float()
        value = F.normalize(value, dim=-1)
        return value @ value.transpose(1, 2)
    relation = F.smooth_l1_loss(relations(student), relations(teacher).detach())
    return .6 * hard + .4 * soft + representation_weight * relation


def train_batch(model, samples, optimizers, microbatch, *, teacher=None, amp=True):
    device = next(model.parameters()).device
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


@torch.no_grad()
def evaluate(model, samples, microbatch, *, amp=True):
    model.eval()
    device = next(model.parameters()).device
    errors, absolute = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    start = 0
    batch = target = prediction = None
    while start < len(samples):
        subset = samples[start:start + microbatch]
        try:
            batch = to_batch(subset, device)
            target = torch.stack([sample["target"] for sample in subset]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                prediction = model.predict_batch(batch).prediction
            errors += normalized_errors(prediction, target).double().sum(0).cpu()
            absolute += (prediction.float() - target.float()).abs().double().sum(0).cpu()
            start += len(subset)
            batch = target = prediction = None
        except torch.cuda.OutOfMemoryError:
            batch = target = prediction = None
            if microbatch == 1:
                raise
            microbatch //= 2
            torch.cuda.empty_cache()
    if not samples:
        raise ValueError("cannot evaluate an empty split")
    values = (errors / len(samples)).tolist()
    return {"errors": values, "scores": [1 - value for value in values],
            "mae": (absolute / len(samples)).tolist(), "gate_passed": quality_gate(values)}


def checkpoint_payload(model, *, role, epoch, normalization, dataset_fingerprint, optimizers=(), schedulers=(), **progress):
    return {"version": CHECKPOINT_VERSION, "output_contract": OUTPUT_CONTRACT_VERSION,
            "target_names": TARGET_NAMES, "role": role, "epoch": epoch,
            "model_config": model.config.to_dict(), "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "target_scales": model.target_scales.detach().cpu().clone(), "normalization": normalization,
            "dataset_fingerprint": dataset_fingerprint, "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in schedulers], "rng": rng_state(), **progress}


def restore_model(payload, *, dataset_fingerprint=None, device="cpu"):
    if payload.get("version") != CHECKPOINT_VERSION or payload.get("output_contract") != OUTPUT_CONTRACT_VERSION or tuple(payload.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("checkpoint is not a v3.1 three-output artifact")
    if dataset_fingerprint is not None and payload["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("checkpoint dataset fingerprint differs")
    model = SeerNetV31(SeerNetV31Config(**payload["model_config"]), payload["target_scales"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)
