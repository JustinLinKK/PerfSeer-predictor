#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import fields, replace
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'perfseer_v5_online_cost_100MB_v1'
OBJECTIVE = "rank"
MAX_NEW_STEPS = None
from v5_objectives import metrics as v5_metrics, quantize_predictions, pairwise_loss, memory_interval_loss
TARGET_NAMES = (
    "train_step_wall_ms",
    "train_step_gpu_ms",
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_avg_vram_mib",
    "train_peak_vram_mib",
)
ARCHITECTURE_SHA256 = "b729f2ced61bca0d7ce67ced6778270b99576c20b2660179a3830b421043a48e"
NORMALIZATION_SHA256 = "13197c7c5a3aab0017e0e0c72b0047cdb8c2647f81f0af776248441444279748"
FIT_SECONDS = 120 * 60
RUN = {}
MODE = "horus"
CONTINUE_FROM = None
from aba3_training_support import TrainingStream, capture_rng, restore_rng, campaign_window, attach_physics, physics_config
EVAL_STEPS = 1000
EFFECTIVE_BATCH = 8


def bootstrap(root):
    root = Path(root)
    additions = [str(root / "runtime" / "src"), str(root / "runtime" / "aliases")]
    for value in reversed(additions):
        if value not in sys.path:
            sys.path.insert(0, value)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(value, path):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def scatter_sum(source, index, size):
    result = source.new_zeros((size, source.size(-1)))
    if source.size(0):
        result.scatter_add_(0, index[:, None].expand(-1, source.size(-1)), source)
    return result


def scatter_mean(source, index, size):
    total = scatter_sum(source, index, size)
    count = source.new_zeros((size, 1))
    if source.size(0):
        count.scatter_add_(0, index[:, None], source.new_ones((source.size(0), 1)))
    return total / count.clamp_min(1)


def scatter_max(source, index, size):
    if not source.size(0):
        return source.new_zeros((size, source.size(-1)))
    result = source.new_full((size, source.size(-1)), -float("inf"))
    result = torch.scatter_reduce(
        result,
        0,
        index[:, None].expand(-1, source.size(-1)),
        source,
        reduce="amax",
        include_self=True,
    )
    return torch.where(torch.isinf(result), torch.zeros_like(result), result)


def mlp(input_dim, hidden, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, output_dim),
    )


def sanitize_batch(batch):
    zeros = {
        "op_family_id": torch.zeros_like(batch.op_family_id),
        "op_hash_id": torch.zeros_like(batch.op_hash_id),
        "op_overload_hash_id": torch.zeros_like(batch.op_overload_hash_id),
        "optimizer_family_id": torch.zeros_like(batch.optimizer_family_id),
        "optimizer_hash_id": torch.zeros_like(batch.optimizer_hash_id),
        "scheduler_id": torch.zeros_like(batch.scheduler_id),
        "scheduler_family_id": torch.zeros_like(batch.scheduler_family_id),
        "scheduler_hash_id": torch.zeros_like(batch.scheduler_hash_id),
    }
    return replace(batch, **zeros)


class DirectedEdgeResidual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.edge_update = mlp(4 * width, width, width, dropout)
        self.node_update = mlp(3 * width, width, width, dropout)
        self.edge_norm = nn.LayerNorm(width)
        self.node_norm = nn.LayerNorm(width)

    def forward(self, nodes, edges, global_context, edge_index, batch):
        if edge_index.size(1):
            source, destination = edge_index
            edge_graph = batch[source]
            delta = self.edge_update(
                torch.cat((edges, nodes[source], nodes[destination], global_context[edge_graph]), dim=-1)
            )
            edges = self.edge_norm(edges + delta)
            incoming = scatter_sum(edges, destination, nodes.size(0))
        else:
            incoming = nodes.new_zeros(nodes.shape)
        node_delta = self.node_update(torch.cat((nodes, incoming, global_context[batch]), dim=-1))
        return self.node_norm(nodes + node_delta), edges


class HardwareConditioner(nn.Module):
    def __init__(self, config, width, dropout):
        super().__init__()
        self.hardware_id = nn.Embedding(config.num_hardware_buckets, 16)
        self.network = nn.Sequential(
            nn.Linear(config.hardware_continuous_dim + config.hardware_missing_mask_dim + 16, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, width),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, batch):
        mask = batch.hardware_missing_mask.float()
        values = batch.hardware_cont.float() * (1 - mask)
        return self.network(torch.cat((values, mask, self.hardware_id(batch.hardware_id)), dim=-1))


class GraphCore(nn.Module):
    def __init__(self, base_config, width, dropout):
        super().__init__()
        from perfseer_v3.model import HierarchicalEdgeEncoder, HierarchicalNodeEncoder
        from perfseer_v4.seerblock_model import SeerBlockConfig

        values = base_config.to_dict() if hasattr(base_config, "to_dict") else dict(base_config)
        values.update(hidden=width, num_blocks=2, dropout=dropout)
        config = SeerBlockConfig(**values)
        self.width = width
        self.node_encoder = HierarchicalNodeEncoder(config)
        self.edge_encoder = HierarchicalEdgeEncoder(config)
        self.precision = nn.Embedding(config.num_dtypes, 8)
        self.optimizer = nn.Embedding(config.num_optimizers, 16)
        self.global_encoder = mlp(config.global_continuous_dim + config.quality_dim + 8 + 16 + 2,
                                  width, width, dropout)
        self.layers = nn.ModuleList(DirectedEdgeResidual(width, dropout) for _ in range(2))
        self.interaction_encoder = mlp(7 * width, width, width, dropout)

    def forward(self, batch):
        nodes = self.node_encoder(
            batch.x_cont,
            batch.op_exact_id,
            batch.op_family_id,
            batch.op_hash_id,
            batch.op_overload_hash_id,
            batch.phase_id,
            batch.input_dtype_id,
            batch.dtype_id,
            batch.accumulation_dtype_id,
            batch.backend_id,
            batch.feature_quality_id,
            batch.layout_id,
            batch.rank_id,
            batch.node_flags,
        )
        edges = self.edge_encoder(
            batch.edge_cont,
            batch.edge_flags,
            batch.edge_role_id,
            batch.edge_source_slot_id,
            batch.edge_destination_slot_id,
            batch.edge_dtype_id,
            batch.edge_layout_id,
            batch.edge_rank_id,
            batch.edge_alias_id,
            batch.edge_dynamic_quality_id,
            batch.edge_phase_transition_id,
        )
        graph_count = batch.u_cont.size(0)
        node_count = scatter_sum(nodes.new_ones((nodes.size(0), 1)), batch.batch, graph_count)
        if batch.edge_index.size(1):
            edge_graph = batch.batch[batch.edge_index[0]]
        else:
            edge_graph = batch.batch.new_empty((0,))
        edge_count = scatter_sum(nodes.new_ones((edges.size(0), 1)), edge_graph, graph_count)
        size_features = torch.cat((torch.log1p(node_count), torch.log1p(edge_count)), dim=-1)
        global_context = self.global_encoder(
            torch.cat((batch.u_cont.float(), batch.quality.float(), self.precision(batch.precision_id),
                       self.optimizer(batch.optimizer_id), size_features), dim=-1)
        )
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, global_context, batch.edge_index, batch.batch)
        node_pool = (
            scatter_sum(nodes, batch.batch, graph_count),
            scatter_mean(nodes, batch.batch, graph_count),
            scatter_max(nodes, batch.batch, graph_count),
        )
        edge_pool = (
            scatter_sum(edges, edge_graph, graph_count),
            scatter_mean(edges, edge_graph, graph_count),
            scatter_max(edges, edge_graph, graph_count),
        )
        interaction = self.interaction_encoder(torch.cat((global_context, *node_pool, *edge_pool), dim=-1))
        return nodes, edges, interaction


class AdditiveTimingBranch(nn.Module):
    def __init__(self, base_config, target_medians, mean_node_count):
        super().__init__()
        self.width = 192
        self.core = GraphCore(base_config, self.width, .03)
        config = base_config
        self.hardware = HardwareConditioner(config, self.width, .03)
        self.cost_head = mlp(2 * self.width, 96, 2, .03)
        self.overhead_head = mlp(2 * self.width, 96, 2, .03)
        self.correction = nn.Sequential(
            nn.Linear(self.width, 96), nn.SiLU(), nn.Dropout(.03), nn.Linear(96, 2)
        )
        self.log_calibration = nn.Parameter(torch.zeros(2))
        medians = torch.as_tensor(target_medians, dtype=torch.float32).clone()
        self.register_buffer("per_node_scale", .8 * medians / float(mean_node_count))
        self.register_buffer("overhead_scale", .2 * medians)
        for module in (self.cost_head[-1], self.overhead_head[-1], self.correction[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch):
        nodes, _, interaction = self.core(batch)
        hardware = self.hardware(batch)
        normalizer = F.softplus(nodes.new_zeros(())).detach()
        node_raw = self.cost_head(torch.cat((nodes, hardware[batch.batch]), dim=-1)).float()
        node_cost = (F.softplus(node_raw) + 1e-6) / normalizer * self.per_node_scale
        base_cost = scatter_sum(node_cost, batch.batch, interaction.size(0))
        overhead_raw = self.overhead_head(torch.cat((interaction, hardware), dim=-1)).float()
        overhead = (F.softplus(overhead_raw) + 1e-6) / normalizer * self.overhead_scale
        log_correction = .5 * torch.tanh(self.correction(interaction).float())
        prediction = (base_cost + overhead) * torch.exp(log_correction + self.log_calibration.float())
        modulation_penalty = hardware.float().square().mean() + self.log_calibration.float().square().mean()
        return prediction.clamp_min(1e-6), log_correction, modulation_penalty


class SMBranch(nn.Module):
    def __init__(self, base_config, positive_fraction, positive_median):
        super().__init__()
        self.width = 128
        self.core = GraphCore(base_config, self.width, .03)
        self.hardware = HardwareConditioner(base_config, self.width, .03)
        self.head = mlp(2 * self.width, 64, 2, .03)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            probability = torch.as_tensor(positive_fraction).float().clamp(.01, .99)
            magnitude = (torch.as_tensor(positive_median).float() / 100).clamp(.01, .99)
            self.head[-1].bias.copy_(torch.stack((torch.logit(probability), torch.logit(magnitude))))

    def forward(self, batch):
        _, _, interaction = self.core(batch)
        hardware = self.hardware(batch)
        logits = self.head(torch.cat((interaction, hardware), dim=-1)).float()
        positive = 100 * torch.sigmoid(logits[:, 1])
        prediction = torch.where(logits[:, 0] >= 0, positive, torch.zeros_like(positive))
        return prediction, logits[:, 0], positive, hardware.float().square().mean()


class AdditiveTeacher(nn.Module):
    def __init__(self, vram_config, target_medians, mean_node_count, sm_positive_fraction,
                 sm_positive_median):
        super().__init__()
        from perfseer_v4.seerblock_model import SeerBlockConfig, SeerBlockPredictor

        config = SeerBlockConfig(**dict(vram_config))
        self.vram = SeerBlockPredictor(config, target_medians)
        self.vram.prediction_heads[0].requires_grad_(False)
        self.vram.prediction_heads[1].requires_grad_(False)
        self.timing = AdditiveTimingBranch(config, torch.as_tensor(target_medians)[:2], mean_node_count)
        self.sm = SMBranch(config, sm_positive_fraction, sm_positive_median)
        self.vram_config = config.to_dict()
        self.initialization = {
            "target_medians": torch.as_tensor(target_medians, dtype=torch.float32).tolist(),
            "mean_node_count": float(mean_node_count),
            "sm_positive_fraction": float(sm_positive_fraction),
            "sm_positive_median": float(sm_positive_median),
        }
        self._verify_disjoint_parameters()

    def _verify_disjoint_parameters(self):
        groups = [{id(parameter) for parameter in getattr(self, name).parameters()}
                  for name in ("vram", "timing", "sm")]
        if any(groups[left] & groups[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("branch parameters must not alias")

    def adaptation_parameter_names(self):
        result = {}
        for branch in ("vram", "timing", "sm"):
            names = [name for name, _ in getattr(self, branch).named_parameters()]
            if branch == "vram":
                head = [name for name in names if name.startswith("prediction_heads.2.")]
                adapter = [name for name in names if name.startswith(
                    ("phase_encoder.", "phase_fusion.", "phase_fusion_norm."))]
                full = [name for name in names if not name.startswith(
                    ("prediction_heads.0.", "prediction_heads.1."))]
            elif branch == "timing":
                head = [name for name in names if name.startswith(("cost_head.", "overhead_head."))
                        or name.startswith("correction.3.")]
                adapter = [name for name in names if name.startswith("hardware.")
                           or name == "log_calibration"]
                full = names
            else:
                head = [name for name in names if name.startswith("head.")]
                adapter = [name for name in names if name.startswith("hardware.")]
                full = names
            result[branch] = {"head": head, "adapter": adapter, "full": full}
        if hasattr(self, "physics_config"):
            names = [name for name, _ in self.timing.named_parameters()]
            result["timing"] = {"head": [n for n in names if n.startswith(("host.", "correction.", "experts.", "workload_head."))],
                                "adapter": [n for n in names if n.startswith(("legacy.hardware.", "adapter.")) or n == "mix"],
                                "full": names}
            if self.physics_config["mode"] == "horus":
                names = [name for name, _ in self.sm.named_parameters()]
                result["sm"] = {"head": [n for n in names if n.startswith(("head.", "legacy.head."))],
                                "adapter": [n for n in names if n.startswith(("adapter.", "legacy.hardware."))],
                                "full": names}
        return result

    def adaptation_parameters(self, branch, mode="head"):
        names = self.adaptation_parameter_names()[branch][mode]
        owned = dict(getattr(self, branch).named_parameters())
        return [owned[name] for name in names]

    def forward(self, batch, raw_steps, protocol="measured_single_step", return_aux=False):
        from perfseer_v4.timing_model import epoch_time

        batch = sanitize_batch(batch)
        timing, correction, timing_hardware_penalty = self.timing(batch)
        memory = self.vram(batch).prediction[:, 4:6].float()
        sm, sm_logit, sm_positive, sm_hardware_penalty = self.sm(batch)
        epoch = epoch_time(timing[:, 0], raw_steps, protocol)
        prediction = torch.cat((timing, epoch[:, None], sm[:, None], memory), dim=-1)
        if return_aux:
            return prediction, correction, timing_hardware_penalty, sm_logit, sm_positive, sm_hardware_penalty
        return prediction


def enable_soft_busy_export(model):
    if getattr(model, "physics_config", {}).get("mode") != "horus":
        return model

    def forward(branch, batch):
        _, logit, positive, penalty = branch.legacy(batch)
        residual = branch.head(branch.adapter(branch.features(batch))).float()
        logit = logit + residual[:, 0]
        odds = torch.exp(4 * torch.tanh(residual[:, 1]))
        positive = positive * odds / (1 + (positive / 100) * (odds - 1))
        positive = positive.clamp(0, 100)
        prediction = torch.sigmoid(logit) * positive
        return prediction, logit, positive, penalty + 1e-5 * residual.square().mean()

    model.sm.forward = MethodType(forward, model.sm)
    return model


def model_payload(model, optimizer_steps, epoch=0, cursor=0, best_score=None):
    return {
        "version": VERSION,
        "target_names": list(TARGET_NAMES),
        "architecture_sha256": ARCHITECTURE_SHA256,
        "vram_config": model.vram_config,
        "initialization": model.initialization,
        "optimizer_steps": int(optimizer_steps),
        "epoch": int(epoch),
        "cursor": int(cursor),
        "best_score": None if best_score is None else float(best_score),
        "adaptation_parameter_groups": model.adaptation_parameter_names(),
        "state": {name: tensor.detach() for name, tensor in model.state_dict().items()},
        "physics_config": getattr(model, "physics_config", None),
        "inherited_optimizer_steps": RUN.get("inherited_steps", 0),
        "new_optimizer_steps": int(optimizer_steps)-RUN.get("inherited_steps", 0),
        "initialization_receipt": RUN.get("initialization", {}),
        "v5_objective": OBJECTIVE,
        "dataset_version": VERSION,
    }


def migrate_pinned_v4_seed(payload, path, initialization):
    """Weights-only migration of the explicitly pinned seed, never a resume bypass."""
    if sha256(path) != initialization["checkpoint_sha256"]:
        raise ValueError("pinned A10 seed changed")
    if payload.get("version") != "perfseer_aba2_additive_operation_cost_v1":
        raise ValueError("unexpected source version for pinned seed migration")
    return {**payload, "version": VERSION, "migrated_from_version": payload["version"]}


def restore_model(payload, device="cpu"):
    if (payload.get("version") != VERSION or payload.get("target_names") != list(TARGET_NAMES)
            or payload.get("architecture_sha256") != ARCHITECTURE_SHA256):
        raise ValueError("unexpected additive teacher checkpoint")
    init = payload["initialization"]
    model = AdditiveTeacher(
        payload["vram_config"],
        init["target_medians"],
        init["mean_node_count"],
        init["sm_positive_fraction"],
        init["sm_positive_median"],
    )
    if payload.get("physics_config"):
        attach_physics(model, payload["physics_config"])
    model.load_state_dict(payload["state"], strict=True)
    enable_soft_busy_export(model)
    return model.to(device)


def to_batch(features, indices, device):
    from perfseer_v3.features import batch_graph_features

    batch = batch_graph_features([features[int(index)] for index in indices])
    return replace(batch, **{
        field.name: getattr(batch, field.name).to(device)
        for field in fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    })


def load_normalized_inputs(path, normalizer):
    from perfseer_v3.features import apply_normalization
    from perfseer_v4.timing_model import epoch_time

    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {"features", "sample_ids", "raw_steps_per_epoch", "epoch_protocol"}
    if not required.issubset(data) or len(data["features"]) != len(data["sample_ids"]):
        raise ValueError("invalid input-only graph bundle")
    if len(set(data["sample_ids"])) != len(data["sample_ids"]):
        raise ValueError("sample IDs must be unique")
    count = torch.as_tensor(data["raw_steps_per_epoch"])
    epoch_time(torch.ones(len(data["features"])), count, data["epoch_protocol"])
    supplied = data.get("normalization")
    if supplied is not None and supplied.sha256 != normalizer.sha256:
        raise ValueError("input normalization state differs from saved source normalization")
    prepared = []
    for feature in data["features"]:
        state = feature.metadata.get("normalization_sha256")
        if state == normalizer.sha256:
            prepared.append(feature)
        elif state:
            raise ValueError("feature was normalized with a different normalizer")
        elif supplied is not None:
            prepared.append(feature)
        else:
            prepared.append(apply_normalization(feature, normalizer))
    data["features"] = prepared
    data["raw_steps_per_epoch"] = count
    return data


def resolve_assets(model_dir):
    model_dir = Path(model_dir)
    base = model_dir if (model_dir / "teacher.pt").is_file() else model_dir / "model"
    if not (base / "teacher.pt").is_file():
        raise FileNotFoundError("teacher.pt not found in model directory")
    manifest_path = model_dir / "candidate_manifest.json"
    if not manifest_path.is_file():
        manifest_path = base / "candidate_manifest.json"
    return model_dir, base, manifest_path


def verify_artifacts(model_dir, base, manifest):
    for name, expected in manifest["artifact_sha256"].items():
        candidates = (base / name, model_dir / name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None or sha256(path) != expected:
            raise ValueError("deployment artifact changed: " + name)


@torch.inference_mode()
def predict(model, data, indices, device, batch_size=8):
    model.eval()
    predictions = []
    start = 0
    indices = torch.as_tensor(indices, dtype=torch.long)
    while start < len(indices):
        selected = indices[start:start + batch_size]
        batch = None
        try:
            batch = to_batch(data["features"], selected, device)
            raw_steps = data["raw_steps_per_epoch"][selected].to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                output = model(batch, raw_steps, data["epoch_protocol"])
            predictions.append(output.float().cpu())
            start += len(selected)
        except torch.cuda.OutOfMemoryError:
            batch = None
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
    return torch.from_numpy(quantize_predictions(torch.cat(predictions).numpy())).to(torch.float64)


def run_inference(model_dir, input_path, output_path):
    model_dir, base, manifest_path = resolve_assets(model_dir)
    runtime_root = base if (base / "runtime").is_dir() else model_dir
    bootstrap(runtime_root)
    manifest = json.loads(manifest_path.read_text())
    verify_artifacts(model_dir, base, manifest)
    normalizer = torch.load(base / "normalization.pt", map_location="cpu", weights_only=False)
    if normalizer.sha256 != manifest["normalization_sha256"]:
        raise ValueError("saved source normalizer semantic hash differs")
    data = load_normalized_inputs(input_path, normalizer)
    if "targets" in data:
        raise ValueError("inference accepts input-only graph bundles")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(base / "teacher.pt", map_location="cpu", weights_only=False, mmap=True)
    model = restore_model(payload, device)
    prediction = predict(model, data, torch.arange(len(data["features"])), device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values=prediction.numpy()
    np.savez_compressed(output_path, sample_ids=np.asarray(data["sample_ids"]),
        predictions=values, dataset_version=np.asarray(VERSION),
        vram_100mb_buckets=np.rint(values[:,4:6] * 1048576 / 100000000).astype(np.int64),
        vram_decimal_gb=np.round(values[:,4:6] * 1048576 / 1000000000, 1),
        step_cost_units=values[:,0], cost_unit=np.asarray("A10_reference_millisecond_v1"),
        nvml_busy_proxy_percent=values[:,3], uncertainty_status=np.asarray("NOT_YET_CALIBRATED"))


def timing_loss(prediction, target, correction, hardware_penalty):
    residual = prediction.float().log() - target.float().log()
    huber = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.08, reduction="none")
    weighted_huber = (2 * huber[:, 0] + huber[:, 1]).mean() / 3
    return weighted_huber + .05 * residual.square().mean() + .002 * correction.float().square().mean() \
        + .001 * hardware_penalty.float()


def vram_loss(prediction, target):
    if OBJECTIVE == "rank":
        return memory_interval_loss(prediction, target)
    residual = prediction.float().log() - target.float().log()
    huber = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.08, reduction="mean")
    return huber + .05 * residual.square().mean()


def balanced_binary_weights(positive_fraction, device=None):
    positive_fraction = torch.as_tensor(positive_fraction, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    weights = torch.stack((.5 / (1 - positive_fraction), .5 / positive_fraction))
    return weights.to(device) if device is not None else weights


def sm_loss(sm_logit, sm_positive, target, hardware_penalty, class_weights):
    target = target.float()
    positive = target > 0
    hurdle_loss = F.binary_cross_entropy_with_logits(
        sm_logit.float(), positive.float(), reduction="none"
    )
    example_weights = torch.where(positive, class_weights[1], class_weights[0])
    objective = (hurdle_loss * example_weights).mean()
    if positive.any():
        residual = sm_positive[positive].float().clamp_min(1e-6).log() - target[positive].log()
        objective = objective + F.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=.1, reduction="mean"
        ) + .1 * (sm_positive[positive].float() - target[positive]).abs().mean() / 100
    return objective + .001 * hardware_penalty.float()


def decay_groups(module):
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step, base_lr):
    step = max(1, step)
    if step <= 200:
        return step / 200
    return max(math.sqrt(200 / step), 2e-5 / base_lr)


def optimizer_state(model):
    learning_rates = {"vram": 1e-4, "timing": 3e-4, "sm": 2e-4}
    optimizers = {
        name: torch.optim.AdamW(decay_groups(getattr(model, name)), lr=learning_rates[name])
        for name in ("vram", "timing", "sm")
    }
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(
            optimizers[name], lambda step, lr=learning_rates[name]: lr_factor(step, lr)
        )
        for name in optimizers
    }
    return optimizers, schedulers


def verify_vram_architecture(model, reference):
    expected_config = reference["model_config"]
    if model.vram_config != expected_config:
        raise ValueError("VRAM configuration changed from the preserved node9 architecture")
    state = model.vram.state_dict()
    expected_shapes = {name: tuple(shape) for name, shape in reference["state_shapes"].items()}
    actual_shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    if actual_shapes != expected_shapes:
        missing = sorted(set(expected_shapes) ^ set(actual_shapes))
        wrong = sorted(name for name in set(expected_shapes) & set(actual_shapes)
                       if expected_shapes[name] != actual_shapes[name])
        raise ValueError(f"VRAM state shape contract changed: missing={missing[:5]} wrong={wrong[:5]}")


def score_report(prediction, target, raw_target=None):
    return v5_metrics(prediction.numpy(), target.numpy(), raw_targets=None if raw_target is None else raw_target.numpy())



def train_diagnostic(report, validation_report):
    return {
        "train_rows": report["rows"],
        "train_hit5": report["hit5"],
        "validation_minus_train_hit5": {
            name: validation_report["hit5"][name] - report["hit5"][name]
            for name in TARGET_NAMES
        },
        "train_median_relative_error": report["median_relative_error"],
        "train_p90_relative_error": report["p90_relative_error"],
    }


def save_recovery(path, model, optimizers, schedulers, steps, epoch, cursor, best_score):
    payload = model_payload(model, steps, epoch, cursor, best_score)
    payload["optimizers"] = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
    payload["schedulers"] = {name: scheduler.state_dict() for name, scheduler in schedulers.items()}
    payload["training_stream"] = RUN["stream"].state_dict()
    payload["rng"] = capture_rng()
    atomic_torch_save(payload, path)


def build_manifest(output, experiment, optimizer_steps, normalizer, extra=None):
    deploy_files = ["teacher.pt", "normalization.pt", "inference.py", "adaptation.json", "aba3_training_support.py", "v5_objectives.py"]
    deploy_files.extend(
        str(path.relative_to(output))
        for path in sorted((output / "runtime").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    manifest = {
        "version": experiment["version"],
        "candidate_version": VERSION,
        "dataset_version": VERSION,
        "v5_objective": OBJECTIVE,
        "vram_units": "MiB transport; decimal100MB ceiling buckets, min100MB",
        "source": "A10",
        "optimizer_steps": int(optimizer_steps),
        "architecture_sha256": experiment["architecture_sha256"],
        "normalization_sha256": normalizer.sha256,
        "branches": ["vram", "timing", "sm"],
        "trainable_branches": ["vram", "timing", "sm"],
        "artifact_sha256": {name: sha256(output / name) for name in deploy_files},
        "raw_epoch_semantics": "epoch_ms = wall_step_ms * raw_steps_per_epoch; measured_single_step requires one",
        "predictive_feature_policy": "input tensors only; family/hash categorical channels zeroed; IDs/groups/families/filenames/hashes excluded",
        "transfer_interface": "adaptation_parameter_groups in teacher.pt and adaptation.json",
        "test_status": "NOT_EVALUATED",
        "transfer_status": "NOT_EVALUATED",
        "distillation_status": "NOT_EVALUATED",
        "teacher_accepted": False,
        "replay_verified": False,
    }
    if extra:
        manifest.update(extra)
    return manifest


def adaptation_description(model):
    groups = model.adaptation_parameter_names()
    return {
        "normalizer": "Freeze and apply the saved A10 source normalizer exactly once.",
        "minimal_target_path": {
            "vram": "Tune memory head only; freeze phase pooling/fusion and trunk.",
            "timing": "Tune hardware conditioner and two-channel log calibration; freeze workload graph core.",
            "sm": "Tune hurdle/magnitude head and hardware conditioner; freeze workload graph core.",
        },
        "groups": groups,
        "branch_ownership": {
            "vram": "Preserved node9 SeerBlock trunk, phase pooling/fusion, and prediction_heads.2 memory head.",
            "timing": "Independent width-192 additive node cost, overhead, correction, and hardware modules.",
            "sm": "Independent width-128 graph core, hardware module, and zero-hurdle/magnitude head.",
        },
        "target_transfer_status": "NOT_EVALUATED",
        "inference_requires_source_labels": False,
        "distillation_status": "NOT_EVALUATED",
    }


def audit_public_files(data_dir, output, smoke=False):
    manifest = json.loads((data_dir / "file-manifest.json").read_text())
    relative = ["README.md", "experiment.json", "architecture-reference.json", "normalization.pt",
                "data/A10/smoke-inputs.pt" if smoke else "data/A10/dataset.pt",
                "data/A10/validation-inputs.pt"]
    observed = {}
    for name in relative:
        observed[name] = sha256(data_dir / name)
        if observed[name] != manifest[name]:
            raise ValueError("public file hash mismatch: " + name)
    notes = {
        "checked_sha256": observed,
        "readme_manifest_mismatch": False,
        "readme_expected_sha256": manifest["README.md"],
        "readme_actual_sha256": observed["README.md"],
        "public_files_modified": False,
    }
    atomic_json(notes, output / "setup_notes.json")


def train_full(data_dir, output):
    if not torch.cuda.is_available():
        raise RuntimeError("full fitting must run in the serialized GPU training unit")
    if (output / "teacher.pt").exists():
        raise ValueError("fresh candidate output required")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.95)
    bootstrap(data_dir)
    started = time.monotonic()
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    experiment = json.loads((data_dir / "experiment.json").read_text())
    now = dt.datetime.now(dt.timezone.utc)
    scheduled, deadline = campaign_window(data_dir, now)
    if now < scheduled:
        raise RuntimeError("training cannot begin before the fixed campaign window")
    if now >= deadline:
        raise RuntimeError("absolute campaign deadline passed")
    fit_end = min(started + FIT_SECONDS, started + (deadline - now).total_seconds())
    audit_public_files(data_dir, output, smoke=False)
    data = torch.load(data_dir / "data" / "A10" / "dataset.pt", map_location="cpu",
                      weights_only=False, mmap=True)
    required = {"features", "sample_ids", "configuration_groups", "families", "splits",
                "normalization", "manifest", "targets", "raw_steps_per_epoch", "epoch_protocol"}
    if not required.issubset(data) or data["manifest"].get("source") != "A10":
        raise ValueError("expected fixed A10 source dataset")
    if data["manifest"].get("dataset_version") != VERSION:
        raise ValueError("v5 dataset required")
    if data["normalization"].sha256 != experiment["normalization_sha256"]:
        raise ValueError("native A10 train-only normalizer changed")
    splits = torch.as_tensor(data["splits"], dtype=torch.long)
    targets = torch.as_tensor(data["targets"], dtype=torch.float64)
    raw_targets = torch.as_tensor(data["raw_targets_v4"], dtype=torch.float64)
    train_indices = torch.nonzero(splits == 0).flatten()
    validation_indices = torch.nonzero(splits == 1).flatten()
    if (len(train_indices), len(validation_indices)) != (39294, 2152):
        raise ValueError("fixed source split sizes changed")
    memberships = {}
    for group, split in zip(data["configuration_groups"], splits.tolist(), strict=True):
        if group in memberships and memberships[group] != split:
            raise ValueError("configuration group leaked across source splits")
        memberships[group] = split
    from perfseer_v4.timing_model import epoch_time
    epoch_time(torch.ones(len(targets)), data["raw_steps_per_epoch"], data["epoch_protocol"])
    if data["epoch_protocol"] != "measured_single_step":
        raise ValueError("source epoch protocol changed")
    train_target = targets[train_indices]
    medians = train_target.median(0).values
    mean_node_count = sum(len(data["features"][int(index)].x_cont) for index in train_indices) / len(train_indices)
    sm_positive = train_target[:, 3] > 0
    sm_positive_fraction = sm_positive.float().mean()
    sm_positive_median = train_target[sm_positive, 3].median() if sm_positive.any() else torch.tensor(50.)
    sm_class_weights = balanced_binary_weights(sm_positive_fraction, "cuda")
    reference = json.loads((data_dir / "architecture-reference.json").read_text())
    initial_path = data_dir / "seed" / "resume.pt"
    if CONTINUE_FROM is not None and CONTINUE_FROM.resolve() != initial_path.resolve():
        raise ValueError("fresh node91 campaign forbids historical checkpoint ancestry")
    initial = torch.load(initial_path, weights_only=False, map_location="cpu", mmap=True)
    initialization = json.loads((data_dir / "initialization.json").read_text())
    if sha256(initial_path) != initialization["resume_sha256"]:
        raise ValueError("pinned A10 seed changed")
    if True:  # explicit, fixed node91 resume seed
        if initial.get("dataset_version") != VERSION or initial.get("v5_objective") != OBJECTIVE:
            raise ValueError("continuation dataset/objective differs; explicit migration required")
        parent = json.loads((initial_path.parent / "candidate_manifest.json").read_text())
        parent_metrics = json.loads((initial_path.parent / "metrics.json").read_text())
        if initial_path.name != "resume.pt" or parent.get("resume_sha256") != sha256(initial_path):
            raise ValueError("continuation requires a hashed selected resume.pt")
        if parent.get("source") != "A10" or not parent_metrics.get("replay_verified"):
            raise ValueError("continuation parent must be a replayed A10 candidate")
        if sha256(initial_path.parent / "candidate_manifest.json") != initialization["source_manifest_sha256"]:
            raise ValueError("continuation ancestry differs from verified A10 seed")
        if not {"optimizers", "schedulers", "rng", "training_stream"}.issubset(initial):
            raise ValueError("matching complete continuation state missing")
    # Already v5: no v4 migration or ancestral lookup.
    model = restore_model(initial, "cpu")
    if MODE != "control" and not initial.get("physics_config"):
        attach_physics(model, physics_config(data_dir, MODE, model))
    if initial.get("physics_config") and MODE != initial["physics_config"]["mode"]:
        raise ValueError("continuation mode differs; explicit branch migration required")
    model = model.cuda()
    verify_vram_architecture(model, reference)
    optimizers, schedulers = optimizer_state(model)
    import aibuildai
    aibuildai.init()
    best_score = -float("inf")
    steps = int(initial["optimizer_steps"])
    starting_steps = steps
    stream = TrainingStream(train_indices, initial.get("training_stream"))
    resume_optimizer = bool(initial.get("optimizers"))
    if resume_optimizer:
        if bool(initial.get("physics_config")) != hasattr(model, "physics_config"):
            raise ValueError("optimizer cannot resume after branch migration")
        for name in optimizers:
            optimizers[name].load_state_dict(initial["optimizers"][name])
            schedulers[name].load_state_dict(initial["schedulers"][name])
        restore_rng(initial["rng"])
    RUN.update(stream=stream, inherited_steps=starting_steps, initialization={
        "parent_path": str(initial_path), "parent_sha256": sha256(initial_path),
        "optimizer_state_restored": resume_optimizer, "seed": initialization})
    del initial
    epoch, cursor = stream.epoch, stream.cursor
    last_save = time.monotonic()
    last_eval_step = -1
    seed_predictions = predict(model, data, validation_indices, torch.device("cuda"), batch_size=8)
    incumbent = score_report(seed_predictions, targets[validation_indices], raw_targets[validation_indices])
    memory_floor = np.asarray(incumbent["vram"]["within_100MB"]) - .01
    atomic_json(incumbent, output / "initial_v5_metrics.json")
    best_eligible = False
    branch_best = {"timing": -1., "vram": -1., "busy": -1.}
    del seed_predictions

    def evaluate_and_select(final=False):
        nonlocal best_score, last_eval_step, best_eligible
        prediction = predict(model, data, validation_indices, torch.device("cuda"), batch_size=8)
        report = score_report(prediction, targets[validation_indices], raw_targets[validation_indices])
        event = {
            "event": "validation",
            "optimizer_steps": steps,
            "epoch": epoch,
            "cursor": cursor,
            "elapsed_seconds": time.monotonic() - started,
            "final_completed_step": bool(final),
            "validation": report,
        }
        append_jsonl(event, output / "training_progress.jsonl")
        print(json.dumps(event), flush=True)
        eligible = bool((np.asarray(report["vram"]["within_100MB"]) >= memory_floor).all())
        report["memory_incumbent_guard_passed"] = eligible
        branch_values = {"timing": report["timing"]["wall"]["material"]["accuracy"] or 0.,
                         "vram": min(report["vram"]["within_100MB"]),
                         "busy": report["nvml_busy"]["material"]["accuracy"] or 0.}
        for branch, value in branch_values.items():
            branch_best[branch] = max(branch_best[branch], value)
        atomic_json({"best_values": branch_best, "selection_memory_floor": memory_floor.tolist(),
                     "note": "Branch metrics, not separately composed checkpoints."}, output / "branch_best_metrics.json")
        if (not (output / "teacher.pt").exists() or (eligible and not best_eligible)
                or (eligible == best_eligible and report["score"] > best_score + 1e-6)):
            best_eligible = eligible
            best_score = report["score"]
            atomic_torch_save(model_payload(model, steps, epoch, cursor, best_score), output / "teacher.pt")
            save_recovery(output / "resume.pt", model, optimizers, schedulers, steps, epoch, cursor, best_score)
        last_eval_step = steps
        model.train()
        return prediction, report

    try:
        while not stopped[0] and time.monotonic() < fit_end and (MAX_NEW_STEPS is None or steps-starting_steps < MAX_NEW_STEPS):
            selected = stream.next(EFFECTIVE_BATCH)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            loss_totals = {"timing": 0., "vram": 0., "sm": 0.}
            model.train()
            timing_pairs, busy_pairs, pair_targets = [], [], []
            for index in selected:
                batch = to_batch(data["features"], index.reshape(1), "cuda")
                batch = sanitize_batch(batch)
                target = targets[index:index + 1].cuda()
                scale = 1 / len(selected)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    timing, correction, timing_hw = model.timing(batch)
                    loss_timing = timing_loss(timing, target[:, :2], correction, timing_hw) * scale
                loss_timing.backward(retain_graph=OBJECTIVE == "rank")
                if OBJECTIVE == "rank":
                    timing_pairs.append(timing.float().clamp_min(1e-8).log())
                    pair_targets.append(target.detach())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    memory = model.vram(batch).prediction[:, 4:6]
                    loss_memory = vram_loss(memory, target[:, 4:6]) * scale
                loss_memory.backward()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, sm_logit, sm_positive_value, sm_hw = model.sm(batch)
                    loss_sm = sm_loss(
                        sm_logit, sm_positive_value, target[:, 3], sm_hw, sm_class_weights
                    ) * scale
                loss_sm.backward(retain_graph=OBJECTIVE == "rank")
                if OBJECTIVE == "rank":
                    busy_pairs.append(torch.sigmoid(sm_logit.float()) * sm_positive_value.float() / 10.)
                loss_totals["timing"] += float(loss_timing.detach())
                loss_totals["vram"] += float(loss_memory.detach())
                loss_totals["sm"] += float(loss_sm.detach())
                batch = target = timing = memory = loss_timing = loss_memory = loss_sm = None
            if OBJECTIVE == "rank":
                ranking_targets = torch.cat(pair_targets)
                timing_scores = torch.cat(timing_pairs)
                rank_time = sum(pairwise_loss(timing_scores[:,c], ranking_targets[:,c]) for c in (0,1)) / 2
                rank_busy = pairwise_loss(torch.cat(busy_pairs), ranking_targets[:,3], "busy")
                (rank_time + .5*rank_busy).backward()
                loss_totals["timing"] += float(rank_time.detach())
                loss_totals["sm"] += .5*float(rank_busy.detach())
                timing_pairs = busy_pairs = pair_targets = ranking_targets = timing_scores = rank_time = rank_busy = None
            grad_norms = {}
            for name in ("timing", "vram", "sm"):
                grad_norms[name] = float(torch.nn.utils.clip_grad_norm_(
                    getattr(model, name).parameters(), 1., error_if_nonfinite=True
                ))
                optimizers[name].step()
                schedulers[name].step()
            stream.commit(len(selected))
            epoch, cursor = stream.epoch, stream.cursor
            steps += 1
            log_value = {
                "timing_loss": loss_totals["timing"],
                "vram_loss": loss_totals["vram"],
                "sm_loss": loss_totals["sm"],
                "timing_lr": optimizers["timing"].param_groups[0]["lr"],
                "vram_lr": optimizers["vram"].param_groups[0]["lr"],
                "sm_lr": optimizers["sm"].param_groups[0]["lr"],
                "timing_grad_norm": grad_norms["timing"],
                "vram_grad_norm": grad_norms["vram"],
                "sm_grad_norm": grad_norms["sm"],
                "optimizer_step": steps,
                "new_optimizer_steps": steps-starting_steps,
                "sample_presentations": stream.presentations,
                "effective_passes": stream.presentations/len(train_indices),
            }
            aibuildai.log(log_value)
            if steps % 10 == 0:
                event = {"event": "training", "optimizer_steps": steps, "epoch": epoch,
                         "cursor": cursor, "elapsed_seconds": time.monotonic() - started, **log_value}
                append_jsonl(event, output / "training_progress.jsonl")
                print(json.dumps(event), flush=True)
            if time.monotonic() - last_save >= 120:
                save_recovery(output / "latest.pt", model, optimizers, schedulers,
                              steps, epoch, cursor, best_score)
                last_save = time.monotonic()
                print(json.dumps({"event": "recovery_checkpoint", "optimizer_steps": steps}), flush=True)
            if steps % EVAL_STEPS == 0:
                evaluate_and_select(final=False)
        if steps <= starting_steps:
            raise RuntimeError("no optimizer update completed")
        if not stopped[0] and dt.datetime.now(dt.timezone.utc) < deadline:
            evaluate_and_select(final=True)
    finally:
        aibuildai.finish()
    if stopped[0] or dt.datetime.now(dt.timezone.utc) >= deadline:
        save_recovery(output / "latest.pt", model, optimizers, schedulers,
                      steps, epoch, cursor, best_score)
        raise InterruptedError("interrupted candidate retained only as recovery evidence")
    del optimizers, schedulers, model
    torch.cuda.empty_cache()
    shutil.copytree(data_dir / "runtime", output / "runtime", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(data_dir / "normalization.pt", output / "normalization.pt")
    shutil.copy2(Path(__file__), output / "inference.py")
    shutil.copy2(data_dir / "aba3_training_support.py", output / "aba3_training_support.py")
    shutil.copy2(data_dir / "v5_objectives.py", output / "v5_objectives.py")
    selected = torch.load(output / "teacher.pt", map_location="cpu", weights_only=False, mmap=True)
    selected_model = restore_model(selected, "cpu")
    verify_vram_architecture(selected_model, reference)
    atomic_json(adaptation_description(selected_model), output / "adaptation.json")
    manifest = build_manifest(output, experiment, selected["optimizer_steps"], data["normalization"], {
        "selected_score": selected["best_score"],
        "checkpoint_selection": "single whole checkpoint; score > prior best + 1e-6; earlier on tie",
    })
    atomic_json(manifest, output / "candidate_manifest.json")
    replay_path = output / "validation_replay.npz"
    run_inference(output, data_dir / "data" / "A10" / "validation-inputs.pt", replay_path)
    with np.load(replay_path, allow_pickle=False) as replay:
        expected_ids = np.asarray(data["sample_ids"])[validation_indices.numpy()]
        np.testing.assert_array_equal(replay["sample_ids"], expected_ids)
        replay_prediction = torch.from_numpy(replay["predictions"].copy())
    torch.cuda.empty_cache()
    selected_model = selected_model.cuda()
    selected_prediction = predict(selected_model, data, validation_indices, torch.device("cuda"), batch_size=8)
    torch.testing.assert_close(replay_prediction, selected_prediction, rtol=3e-4, atol=1e-5)
    validation_report = score_report(replay_prediction, targets[validation_indices], raw_targets[validation_indices])
    if validation_report["counts"] != score_report(selected_prediction, targets[validation_indices], raw_targets[validation_indices])["counts"]:
        raise ValueError("saved checkpoint replay changed exact hit counts")
    if (replay_prediction[:, [0, 1, 2, 4, 5]] <= 0).any() or not (
            (replay_prediction[:, 3] >= 0) & (replay_prediction[:, 3] <= 100)).all():
        raise ValueError("replay produced physically invalid predictions")
    expected_epoch = replay_prediction[:, 0] * data["raw_steps_per_epoch"][validation_indices]
    torch.testing.assert_close(replay_prediction[:, 2], expected_epoch, rtol=0, atol=0)
    permutation = torch.tensor([2, 0, 1])
    permuted = predict(selected_model, data, validation_indices[permutation], torch.device("cuda"), batch_size=3)
    torch.testing.assert_close(permuted, selected_prediction[permutation], rtol=3e-4, atol=1e-5)
    diagnostic_indices = train_indices[torch.randperm(len(train_indices))[:512]]
    train_prediction = predict(selected_model, data, diagnostic_indices, torch.device("cuda"), batch_size=8)
    train_report = score_report(train_prediction, targets[diagnostic_indices], raw_targets[diagnostic_indices])
    validation_report.update(
        replay_verified=True,
        arbitrary_input_id_order_verified=True,
        positivity_sm_bounds_verified=True,
        raw_epoch_semantics_verified=True,
        VRAM_branch_trainable=True,
        train_validation_gap=train_diagnostic(train_report, validation_report),
        stopping_reason="matched_new_step_cap" if MAX_NEW_STEPS is not None and steps-starting_steps >= MAX_NEW_STEPS else "fit_wall_clock_allowance",
        test_status="NOT_EVALUATED",
        transfer_status="NOT_EVALUATED",
        distillation_status="NOT_EVALUATED",
        teacher_accepted=False,
    )
    np.savez_compressed(output / "search_predictions.npz",
                        A10_validation_sample_ids=np.asarray(data["sample_ids"])[validation_indices.numpy()],
                        A10_validation_predictions=replay_prediction.numpy())
    atomic_json(validation_report, output / "metrics.json")
    manifest["memory_incumbent_guard_passed"] = bool((np.asarray(validation_report["vram"]["within_100MB"]) >= memory_floor).all())
    manifest["new_optimizer_steps"] = selected["new_optimizer_steps"]
    manifest["initialization_receipt"] = selected["initialization_receipt"]
    manifest["resume_sha256"] = sha256(output / "resume.pt")
    manifest["replay_verified"] = True
    manifest["quality_gates"] = {
        "vram_guard_passed": validation_report["vram_guard_passed"],
        "v5_validation_gate_passed": validation_report["v5_validation_gate_passed"],
    }
    atomic_json(manifest, output / "candidate_manifest.json")
    (output / "latest.pt").unlink(missing_ok=True)
    del selected_model
    torch.cuda.empty_cache()
    delivery_layout_check(output, data_dir / "data" / "A10" / "smoke-inputs.pt")
    print(json.dumps({"event": "complete", "optimizer_steps": selected["optimizer_steps"],
                      "score": validation_report["score"], "replay_verified": True}), flush=True)


def delivery_layout_check(output, smoke_input):
    scratch = output / "delivery_layout_check"
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "model").mkdir(parents=True)
    shutil.copy2(output / "inference.py", scratch / "inference.py")
    shutil.copy2(output / "aba3_training_support.py", scratch / "aba3_training_support.py")
    shutil.copy2(output / "v5_objectives.py", scratch / "v5_objectives.py")
    shutil.copy2(output / "candidate_manifest.json", scratch / "candidate_manifest.json")
    for name in ("teacher.pt", "normalization.pt", "adaptation.json"):
        shutil.copy2(output / name, scratch / "model" / name)
    shutil.copytree(output / "runtime", scratch / "model" / "runtime")
    command = [sys.executable, str(scratch / "inference.py"), "--model-dir", str(scratch),
               "--input", str(smoke_input), "--output", str(scratch / "smoke-output.npz")]
    subprocess.run(command, check=True, timeout=600)
    with np.load(scratch / "smoke-output.npz", allow_pickle=False) as result:
        if result["predictions"].shape != (8, 6):
            raise ValueError("delivered inference smoke output shape changed")
    atomic_json({"verified": True, "layout": "inference.py plus model/ assets",
                 "rows": 8}, output / "delivery_check.json")
    shutil.rmtree(scratch)


def smoke_check(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    bootstrap(data_dir)
    audit_public_files(data_dir, output, smoke=True)
    normalizer = torch.load(data_dir / "normalization.pt", map_location="cpu", weights_only=False)
    data = load_normalized_inputs(data_dir / "data" / "A10" / "smoke-inputs.pt", normalizer)
    reference = json.loads((data_dir / "architecture-reference.json").read_text())
    reduced = dict(reference["model_config"])
    reduced.update(hidden=32, num_blocks=1, checkpoint_blocks=False,
                   exact_embedding_dim=8, family_embedding_dim=4, hash_embedding_dim=4,
                   overload_hash_embedding_dim=4, phase_embedding_dim=4,
                   input_dtype_embedding_dim=4, dtype_embedding_dim=4,
                   accumulation_dtype_embedding_dim=4, backend_embedding_dim=4,
                   feature_quality_embedding_dim=2, layout_embedding_dim=2, rank_embedding_dim=2)
    synthetic_targets = torch.tensor([10., 8., 10., 50., 1000., 1200.])
    model = AdditiveTeacher(reduced, synthetic_targets, 10., .75, 50.)
    if MODE != "control":
        attach_physics(model, physics_config(data_dir, MODE, model))
    branch_ids = [{id(parameter) for parameter in getattr(model, name).parameters()}
                  for name in ("vram", "timing", "sm")]
    if any(branch_ids[i] & branch_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("smoke detected aliased branch parameters")
    start = time.perf_counter()
    batch = sanitize_batch(to_batch(data["features"], torch.arange(2), "cpu"))
    target = synthetic_targets[None].repeat(2, 1)
    timing, correction, timing_hw = model.timing(batch)
    timing_loss(timing, target[:, :2], correction, timing_hw).backward()
    model.zero_grad(set_to_none=True)
    memory = model.vram(batch).prediction[:, 4:6]
    vram_loss(memory, target[:, 4:6]).backward()
    model.zero_grad(set_to_none=True)
    _, sm_logit, sm_positive, sm_hw = model.sm(batch)
    sm_class_weights = balanced_binary_weights(.75)
    sm_loss(sm_logit, sm_positive, target[:, 3], sm_hw, sm_class_weights).backward()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        output_value = model(batch, data["raw_steps_per_epoch"][:2], data["epoch_protocol"])
    feature = data["features"][0]
    empty = replace(
        feature,
        edge_index=feature.edge_index[:, :0],
        edge_cont=feature.edge_cont[:0],
        edge_role_id=feature.edge_role_id[:0],
        edge_source_slot_id=feature.edge_source_slot_id[:0],
        edge_destination_slot_id=feature.edge_destination_slot_id[:0],
        edge_dtype_id=feature.edge_dtype_id[:0],
        edge_layout_id=feature.edge_layout_id[:0],
        edge_rank_id=feature.edge_rank_id[:0],
        edge_alias_id=feature.edge_alias_id[:0],
        edge_dynamic_quality_id=feature.edge_dynamic_quality_id[:0],
        edge_phase_transition_id=feature.edge_phase_transition_id[:0],
        edge_flags=feature.edge_flags[:0],
    )
    empty.validate()
    empty_data = {**data, "features": [empty], "raw_steps_per_epoch": data["raw_steps_per_epoch"][:1]}
    empty_batch = to_batch(empty_data["features"], torch.tensor([0]), "cpu")
    empty_prediction = model(empty_batch, empty_data["raw_steps_per_epoch"], data["epoch_protocol"])
    if output_value.shape != (2, 6) or empty_prediction.shape != (1, 6):
        raise ValueError("smoke output shape changed")
    if not torch.isfinite(output_value).all() or (output_value[:, [0, 1, 2, 4, 5]] <= 0).any():
        raise ValueError("smoke prediction is not finite and positive")
    if not torch.equal(output_value[:, 2], output_value[:, 0]):
        raise ValueError("smoke epoch protocol changed")
    elapsed = time.perf_counter() - start
    atomic_torch_save(model_payload(model, 0), output / "teacher.pt")
    shutil.copytree(data_dir / "runtime", output / "runtime", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(data_dir / "normalization.pt", output / "normalization.pt")
    shutil.copy2(Path(__file__), output / "inference.py")
    shutil.copy2(data_dir / "aba3_training_support.py", output / "aba3_training_support.py")
    shutil.copy2(data_dir / "v5_objectives.py", output / "v5_objectives.py")
    atomic_json(adaptation_description(model), output / "adaptation.json")
    experiment = json.loads((data_dir / "experiment.json").read_text())
    manifest = build_manifest(output, experiment, 0, normalizer, {
        "smoke_only": True,
        "gpu_training_completed": False,
        "replay_verified": False,
    })
    atomic_json(manifest, output / "candidate_manifest.json")
    delivery_layout_check(output, data_dir / "data" / "A10" / "smoke-inputs.pt")
    result = {
        "cpu_only": True,
        "rows": 2,
        "empty_edge_graph_checked": True,
        "branch_parameter_disjointness_checked": True,
        "packaged_inference_rows": 8,
        "packaged_model_layout_verified": True,
        "forward_backward_seconds": elapsed,
        "gpu_runtime_estimate": "UNVERIFIED",
        "fixed_training_work_unit_cap_minutes": 180,
    }
    atomic_json(result, output / "smoke_metrics.json")
    print(json.dumps(result), flush=True)


def main():
    global MODE, CONTINUE_FROM, FIT_SECONDS, OBJECTIVE, MAX_NEW_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", choices=["control", "physics", "physics_graph", "horus"], default="horus")
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--objective", choices=["rank", "control"], default="rank")
    parser.add_argument("--max-new-steps", type=int)
    parser.add_argument("--fit-minutes", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.fit_minutes <= 120:
        parser.error("fit minutes must be in (0,120]")
    MODE, CONTINUE_FROM, FIT_SECONDS = args.mode, args.continue_from, args.fit_minutes*60
    OBJECTIVE, MAX_NEW_STEPS = args.objective, args.max_new_steps
    if MAX_NEW_STEPS is not None and MAX_NEW_STEPS <= 0:
        parser.error("max new steps must be positive")
    torch.set_num_threads(4)
    if args.model_dir:
        if args.input is None or args.output is None:
            parser.error("inference requires --input and --output")
        run_inference(args.model_dir, args.input, args.output)
        return
    data_dir = Path(os.environ["DATA_DIR"])
    output = Path(os.environ["OUTPUT_DIR"])
    if args.smoke:
        smoke_check(data_dir, output)
    else:
        train_full(data_dir, output)


if __name__ == "__main__":
    main()
