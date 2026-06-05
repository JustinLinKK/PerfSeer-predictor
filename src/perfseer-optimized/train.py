"""Training CLI for optimized PerfSeer models."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from .calibration import fit_linear_calibration
from .data import (
    FeatureConfig,
    LABEL_DOMAIN_VOCAB,
    NUM_TARGETS,
    SOURCE_UNKNOWN_PRECISION_CONFIG,
    TARGET_NAMES,
    PerfSeerOptimizedDataset,
    compute_norm_stats,
    feature_config_for_pair,
    feature_layout,
    invert_targets,
    is_log_ratio_target,
    label_domain_for_pair,
    norm_stats_to_serializable,
    precision_hardware_config,
    precision_from_label_path,
    normalize_precision_config,
    split_dataset,
    split_hash,
    supported_precision_hardware_summary,
)
from .losses import build_loss, weighted_metric_loss
from .model import SeerNet, SeerNetConfig, SeerNetMulti, count_parameters
from .pcgrad import pcgrad_backward


METRIC_NAMES = TARGET_NAMES


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {"run_id": None, "out_dir": "runs/optimized", "results_path": "runs/results.jsonl", "notes": ""},
    "seed": 42,
    "data": {
        "root": "dataset",
        "limit": 0,
        "num_workers": 0,
        "split_unit": "pair",
        "source_precision_provenance": "",
        "source_precision_confirmed": False,
    },
    "features": FeatureConfig().to_dict(),
    "model": {
        "name": "seernet",
        "hidden": 256,
        "num_blocks": 1,
        "activation": "relu",
        "dropout": 0.0,
        "encoder_norm": "none",
        "block_norm": "none",
        "residual": "direct",
        "residual_gate_init": 0.1,
        "residual_gate_mode": "scalar_per_stream",
        "use_synmm": True,
        "global_agg": "synmm",
        "attention_pool": False,
        "use_gnpb": True,
        "include_u_in_edge_update": True,
        "mlp_z_num_linear_layers": 3,
        "softmax_agg_mode": "learned_score",
        "metric_heads": "separate",
    },
    "train": {
        "metric": "all",
        "epochs": 500,
        "batch_size": 128,
        "lr": 1e-3,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "patience": 30,
        "scheduler_patience": 5,
        "min_lr": 1e-6,
        "loss": "mse_logstd",
        "huber_delta": 1.0,
        "grad_clip_norm": 0.0,
        "ema_decay": 0.0,
        "init_checkpoint": None,
        "init_strict": True,
        "device": "auto",
        "threads": 0,
        "interop_threads": 0,
    },
    "multi_task": {
        "enabled": False,
        "loss_reduction": "plain_sum",
        "loss_weights": {},
    },
    "distillation": {
        "enabled": False,
        "teacher_ckpt_dir": None,
        "alpha": 0.5,
        "source_hard_alpha": 0.5,
        "precision_hard_alpha": 1.0,
        "pseudo_hard_alpha": 0.0,
    },
    "calibration": {"enabled": False, "method": "linear"},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r") as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg = deep_update(cfg, loaded)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_threads(cfg: dict[str, Any]) -> None:
    threads = int(cfg["train"].get("threads", 0) or 0)
    interop = int(cfg["train"].get("interop_threads", 0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    if interop > 0:
        torch.set_num_interop_threads(max(1, interop))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")


def resolve_metrics(metric_arg: str) -> list[int]:
    if str(metric_arg).lower() == "all":
        return list(range(NUM_TARGETS))
    idx = int(metric_arg)
    if not 0 <= idx < NUM_TARGETS:
        raise ValueError(f"metric must be 0..{NUM_TARGETS - 1} or all")
    return [idx]


def make_run_id(cfg: dict[str, Any]) -> str:
    explicit = cfg["run"].get("run_id")
    if explicit:
        return str(explicit)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = cfg["model"].get("name", "seernet")
    return f"{stamp}_{model_name}"


def build_datasets(cfg: dict[str, Any]):
    seed = int(cfg.get("seed", 42))
    data_root = cfg["data"].get("root", "dataset")
    feature_cfg = FeatureConfig.from_dict(cfg.get("features"))
    split_unit = str(cfg["data"].get("split_unit", "pair") or "pair")
    train_files, val_files, test_files = split_dataset(data_root, seed=seed, split_unit=split_unit)
    limit = int(cfg["data"].get("limit", 0) or 0)
    if limit > 0:
        train_files = train_files[:limit]
        val_files = val_files[: max(1, limit // 2)]
        test_files = test_files[: max(1, limit // 2)]
    print(f"split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test", flush=True)
    norm_stats = compute_norm_stats(train_files, feature_cfg)
    common = dict(root=data_root, norm_stats=norm_stats, feature_config=feature_cfg)
    train_ds = PerfSeerOptimizedDataset(file_list=train_files, split="train", **common)
    val_ds = PerfSeerOptimizedDataset(file_list=val_files, split="val", **common)
    test_ds = PerfSeerOptimizedDataset(file_list=test_files, split="test", **common)
    def pair_ids(pairs):
        return [
            {
                "graph_stem": Path(gp).stem,
                "label_stem": Path(lp).stem,
            }
            for gp, lp in pairs
        ]

    def label_domain_counts(pairs):
        counts: dict[str, int] = {}
        for gp, lp in pairs:
            domain = label_domain_for_pair(gp, lp)
            counts[domain] = counts.get(domain, 0) + 1
        return dict(sorted(counts.items()))

    def precision_config_counts(pairs):
        counts: dict[str, int] = {}
        fallback = normalize_precision_config(str(feature_cfg.precision_config or "fp32_ieee"))
        for gp, lp in pairs:
            precision = precision_from_label_path(gp, lp) or fallback
            counts[precision] = counts.get(precision, 0) + 1
        return dict(sorted(counts.items()))

    def hardware_id_counts(pairs):
        counts: dict[str, int] = {}
        for gp, lp in pairs:
            pair_cfg = feature_config_for_pair(feature_cfg, gp, lp)
            hardware_id = str(pair_cfg.hardware_id or "unknown")
            counts[hardware_id] = counts.get(hardware_id, 0) + 1
        return dict(sorted(counts.items()))

    split_meta = {
        "seed": seed,
        "split_unit": split_unit,
        "train_hash": split_hash(train_files),
        "val_hash": split_hash(val_files),
        "test_hash": split_hash(test_files),
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "train_stems": [Path(gp).stem for gp, _ in train_files],
        "val_stems": [Path(gp).stem for gp, _ in val_files],
        "test_stems": [Path(gp).stem for gp, _ in test_files],
        "train_pair_ids": pair_ids(train_files),
        "val_pair_ids": pair_ids(val_files),
        "test_pair_ids": pair_ids(test_files),
        "label_domain_counts": {
            "train": label_domain_counts(train_files),
            "val": label_domain_counts(val_files),
            "test": label_domain_counts(test_files),
        },
        "precision_config_counts": {
            "train": precision_config_counts(train_files),
            "val": precision_config_counts(val_files),
            "test": precision_config_counts(test_files),
        },
        "hardware_id_counts": {
            "train": hardware_id_counts(train_files),
            "val": hardware_id_counts(val_files),
            "test": hardware_id_counts(test_files),
        },
        "supported_precision_hardware": supported_precision_hardware_summary(train_files + val_files + test_files, feature_cfg),
    }
    return train_ds, val_ds, test_ds, norm_stats, feature_cfg, split_meta


def make_model_config(cfg: dict[str, Any], feature_cfg: FeatureConfig, num_outputs: int) -> SeerNetConfig:
    layout = feature_layout(feature_cfg)
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg.update(
        {
            "node_dim": layout.node_dim,
            "edge_dim": layout.edge_dim,
            "global_dim": layout.global_dim,
            "num_outputs": num_outputs,
        }
    )
    model_cfg.pop("name", None)
    return SeerNetConfig.from_dict(model_cfg)


def make_model(cfg: dict[str, Any], feature_cfg: FeatureConfig, num_outputs: int, multi: bool) -> torch.nn.Module:
    model_cfg = make_model_config(cfg, feature_cfg, num_outputs)
    return SeerNetMulti(model_cfg) if multi else SeerNet(model_cfg)


def make_optimizer(model: torch.nn.Module, cfg: dict[str, Any]):
    train_cfg = cfg["train"]
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    kwargs = {"lr": float(train_cfg.get("lr", 1e-3)), "weight_decay": float(train_cfg.get("weight_decay", 0.0))}
    if opt_name == "adamw":
        return AdamW(model.parameters(), **kwargs)
    if opt_name == "adam":
        return Adam(model.parameters(), **kwargs)
    raise ValueError(f"unknown optimizer {opt_name!r}")


def checkpoint_is_multi(ckpt: dict[str, Any]) -> bool:
    model_name = str(ckpt.get("model_name", "")).lower()
    if model_name == "seernet_multi":
        return True
    if "metric_idx" in ckpt:
        return False
    model_cfg = ckpt.get("model_config") or {}
    return int(model_cfg.get("num_outputs", 1) or 1) == NUM_TARGETS


def checkpoint_model(ckpt: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = SeerNetConfig.from_dict(ckpt["model_config"])
    model = SeerNetMulti(model_cfg) if checkpoint_is_multi(ckpt) else SeerNet(model_cfg)
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def resolve_init_checkpoint(raw_path: str | None, metric_idx: int | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_dir():
        if metric_idx is None:
            path = path / "seernet_multi.pt"
        else:
            matches = sorted(path.glob(f"seernet_metric{metric_idx}_*.pt"))
            if not matches:
                raise FileNotFoundError(f"no seernet_metric{metric_idx}_*.pt checkpoint found under {path}")
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"initialization checkpoint not found: {path}")
    return path


def load_initial_weights(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    metric_idx: int | None,
) -> dict[str, Any] | None:
    train_cfg = cfg.get("train", {})
    path = resolve_init_checkpoint(train_cfg.get("init_checkpoint"), metric_idx)
    if path is None:
        return None
    strict = bool(train_cfg.get("init_strict", True))
    ckpt = safe_torch_load(str(path), device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    source_meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    info = {
        "path": str(path),
        "strict": strict,
        "source_run_id": source_meta.get("run_id"),
        "source_epoch": ckpt.get("epoch"),
        "source_val_loss": ckpt.get("val_loss"),
        "source_metric_idx": ckpt.get("metric_idx"),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
    }
    print(f"loaded initialization checkpoint: {path} (strict={strict})", flush=True)
    return info


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for k, v in self.shadow.items():
            self.backup[k] = state[k].detach().clone()
            state[k].copy_(v)

    def restore(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.backup.items():
            state[k].copy_(v)
        self.backup = {}


def select_target(batch, metric_idx: int) -> torch.Tensor:
    return batch.y.view(-1, NUM_TARGETS)[:, metric_idx : metric_idx + 1]


def full_target(batch) -> torch.Tensor:
    return batch.y.view(-1, NUM_TARGETS)


def sample_weights(batch, target: torch.Tensor) -> torch.Tensor:
    weights = getattr(batch, "sample_weight", None)
    if weights is None:
        return target.new_ones(target.size(0))
    return weights.to(device=target.device, dtype=target.dtype).view(-1).clamp_min(0.0)


def elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, huber_delta: float) -> torch.Tensor:
    key = (loss_name or "mse_logstd").lower()
    diff = pred - target
    if key in {"mse", "mse_logstd"}:
        return diff * diff
    if key in {"huber", "huber_logstd", "smooth_l1"}:
        abs_diff = diff.abs()
        delta = float(huber_delta)
        return torch.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
    if key in {"logcosh", "log_cosh", "logcosh_logstd"}:
        return diff + torch.nn.functional.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, device=diff.device, dtype=diff.dtype))
    raise ValueError(f"unknown loss {loss_name!r}")


def weighted_sample_metric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    huber_delta: float,
    metric_weights: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    losses = elementwise_loss(pred, target, loss_name, huber_delta)
    return reduce_weighted_losses(losses, metric_weights, sample_weight)


def reduce_weighted_losses(
    losses: torch.Tensor,
    metric_weights: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if sample_weight is None:
        sample_weight = losses.new_ones(losses.size(0))
    denom = sample_weight.sum().clamp_min(1e-8)
    task_losses = [(losses[:, idx] * sample_weight).sum() / denom for idx in range(losses.size(1))]
    if metric_weights is None:
        total = torch.stack(task_losses).sum()
    else:
        total = torch.stack([task_losses[i] * metric_weights[i] for i in range(len(task_losses))]).sum()
    return total, task_losses


def label_domain_ids(batch, target: torch.Tensor) -> torch.Tensor:
    ids = getattr(batch, "label_domain_idx", None)
    if ids is None:
        source_idx = LABEL_DOMAIN_VOCAB.index("source")
        return torch.full((target.size(0),), source_idx, dtype=torch.long, device=target.device)
    return ids.to(device=target.device).view(-1).long()


def distillation_hard_alphas(batch, target: torch.Tensor, distill_cfg: dict[str, Any]) -> torch.Tensor:
    default_alpha = float(distill_cfg.get("alpha", 0.5))
    source_alpha = float(distill_cfg.get("source_hard_alpha", default_alpha))
    precision_alpha = float(distill_cfg.get("precision_hard_alpha", 1.0))
    pseudo_alpha = float(distill_cfg.get("pseudo_hard_alpha", 0.0))
    ids = label_domain_ids(batch, target)
    alphas = target.new_full((target.size(0),), default_alpha)
    source_idx = LABEL_DOMAIN_VOCAB.index("source")
    precision_idx = LABEL_DOMAIN_VOCAB.index("precision_profile")
    pseudo_idx = LABEL_DOMAIN_VOCAB.index("pseudo")
    alphas = torch.where(ids == source_idx, target.new_tensor(source_alpha), alphas)
    alphas = torch.where(ids == precision_idx, target.new_tensor(precision_alpha), alphas)
    alphas = torch.where(ids == pseudo_idx, target.new_tensor(pseudo_alpha), alphas)
    return alphas.clamp(0.0, 1.0)


def weighted_sample_distillation_loss(
    pred: torch.Tensor,
    hard_target: torch.Tensor,
    teacher_target: torch.Tensor,
    loss_name: str,
    huber_delta: float,
    metric_weights: torch.Tensor | None,
    sample_weight: torch.Tensor,
    hard_alpha: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    hard_losses = elementwise_loss(pred, hard_target, loss_name, huber_delta)
    soft_losses = elementwise_loss(pred, teacher_target, loss_name, huber_delta)
    alpha = hard_alpha.to(device=pred.device, dtype=pred.dtype).view(-1, 1).clamp(0.0, 1.0)
    return reduce_weighted_losses(alpha * hard_losses + (1.0 - alpha) * soft_losses, metric_weights, sample_weight)


def evaluate_loss(model, loader, device, criterion, metric_idx: int | None = None, weights: torch.Tensor | None = None) -> float:
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            if metric_idx is None:
                target = full_target(batch)
                loss, _ = weighted_metric_loss(pred, target, criterion, weights)
                bs = target.size(0)
            else:
                target = select_target(batch, metric_idx)
                loss = criterion(pred, target)
                bs = target.size(0)
            loss_sum += float(loss.item()) * bs
            n += bs
    return loss_sum / max(n, 1)


def collect_std_predictions(model, loader, device, metric_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y = full_target(batch)
            if metric_idx is not None:
                pred = pred[:, :1]
                y = y[:, metric_idx : metric_idx + 1]
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def base_metadata(
    cfg: dict[str, Any],
    run_id: str,
    feature_cfg: FeatureConfig,
    norm_stats: dict[str, np.ndarray],
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    layout = feature_layout(feature_cfg)
    source_precision_provenance = str(cfg.get("data", {}).get("source_precision_provenance") or "").strip()
    source_precision_confirmed = (
        bool(cfg.get("data", {}).get("source_precision_confirmed"))
        and feature_cfg.precision_config != SOURCE_UNKNOWN_PRECISION_CONFIG
    )
    return {
        "run_id": run_id,
        "config": cfg,
        "feature_config": feature_cfg.to_dict(),
        "precision_hardware_config": precision_hardware_config(feature_cfg),
        "source_precision": {
            "precision_config": feature_cfg.precision_config,
            "hardware_id": feature_cfg.hardware_id,
            "provenance": source_precision_provenance,
            "confirmed": source_precision_confirmed,
        },
        "feature_layout": {
            "node_dim": layout.node_dim,
            "edge_dim": layout.edge_dim,
            "global_dim": layout.global_dim,
            "node_names": list(layout.node_names),
            "edge_names": list(layout.edge_names),
            "global_names": list(layout.global_names),
        },
        "norm_stats": norm_stats_to_serializable(norm_stats),
        "split": split_meta,
        "supported_precision_hardware": split_meta.get("supported_precision_hardware", {}),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def checkpoint_payload(
    *,
    model,
    cfg: dict[str, Any],
    model_cfg: SeerNetConfig,
    metadata: dict[str, Any],
    epoch: int,
    val_loss: float,
    metric_idx: int | None,
    norm_stats: dict[str, np.ndarray],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "model_name": cfg["model"].get("name", "seernet"),
        "epoch": epoch,
        "val_loss": val_loss,
        "num_targets": NUM_TARGETS,
        "metadata": metadata,
        "calibration": calibration,
    }
    if metric_idx is not None:
        payload.update(
            {
                "metric_idx": metric_idx,
                "metric_name": METRIC_NAMES[metric_idx],
                "target_stats": {
                    "log_mean": float(norm_stats["y_mean"][metric_idx]),
                    "log_std": float(norm_stats["y_std"][metric_idx]),
                },
            }
        )
    return payload


def checkpoint_metadata_summary(path: str) -> dict[str, Any]:
    ckpt_path = Path(path)
    summary: dict[str, Any] = {
        "path": str(ckpt_path),
        "exists": ckpt_path.is_file(),
    }
    if not ckpt_path.is_file():
        return summary
    try:
        ckpt = safe_torch_load(str(ckpt_path), torch.device("cpu"))
    except Exception as exc:
        summary["load_error"] = str(exc)
        return summary
    if not isinstance(ckpt, dict):
        summary["load_error"] = f"unexpected checkpoint payload type: {type(ckpt).__name__}"
        return summary
    meta = ckpt.get("metadata") if isinstance(ckpt.get("metadata"), dict) else {}
    split = meta.get("split") if isinstance(meta.get("split"), dict) else {}
    summary.update(
        {
            "model_name": ckpt.get("model_name"),
            "metric_idx": ckpt.get("metric_idx"),
            "metric_name": ckpt.get("metric_name"),
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "source_precision": meta.get("source_precision", {}),
            "precision_hardware_config": meta.get("precision_hardware_config", {}),
            "initialization": meta.get("initialization"),
            "distillation_teacher": meta.get("distillation_teacher"),
            "distillation_policy": meta.get("distillation_policy"),
            "supported_precision_hardware": meta.get("supported_precision_hardware", {}),
            "split": {
                "split_unit": split.get("split_unit"),
                "train_count": split.get("train_count"),
                "val_count": split.get("val_count"),
                "test_hash": split.get("test_hash"),
                "test_count": split.get("test_count"),
                "label_domain_counts": split.get("label_domain_counts", {}),
                "precision_config_counts": split.get("precision_config_counts", {}),
                "hardware_id_counts": split.get("hardware_id_counts", {}),
                "supported_precision_hardware": split.get("supported_precision_hardware", {}),
            },
        }
    )
    return summary


def checkpoint_metadata_summaries(paths: list[str]) -> list[dict[str, Any]]:
    return [checkpoint_metadata_summary(path) for path in paths]


@dataclass
class TeacherBundle:
    kind: str
    models: list[torch.nn.Module]
    paths: list[str]
    norm_stats: list[dict[str, np.ndarray] | None] | None = None
    feature_configs: list[FeatureConfig | None] | None = None
    metric_indices: list[int | None] | None = None

    def to_metadata(self) -> dict[str, Any]:
        feature_configs = self.feature_configs or []
        return {
            "kind": self.kind,
            "count": len(self.models),
            "paths": list(self.paths),
            "has_norm_stats": [item is not None for item in (self.norm_stats or [])],
            "target_modes": [getattr(item, "target_mode", None) for item in feature_configs],
            "time_target_modes": [getattr(item, "time_target_mode", None) for item in feature_configs],
            "metric_indices": list(self.metric_indices or []),
        }


def train_one_metric(
    metric_idx: int,
    train_ds,
    val_ds,
    norm_stats,
    cfg: dict[str, Any],
    run_id: str,
    out_dir: str,
    metadata: dict[str, Any],
    feature_cfg: FeatureConfig,
    device: torch.device,
) -> str:
    name = METRIC_NAMES[metric_idx]
    print(f"\n=== Training metric [{metric_idx}] {name} ===", flush=True)
    train_cfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(cfg["data"].get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    model = make_model(cfg, feature_cfg, 1, multi=False).to(device)
    init_info = load_initial_weights(model, cfg, device, metric_idx)
    model_metadata = copy.deepcopy(metadata)
    model_metadata["initialization"] = init_info
    model_cfg = model.cfg
    print(f"model parameters: {count_parameters(model):,}", flush=True)
    optimizer = make_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(train_cfg.get("scheduler_patience", 5)), min_lr=float(train_cfg.get("min_lr", 1e-6)))
    loss_name = str(train_cfg.get("loss", "mse_logstd"))
    huber_delta = float(train_cfg.get("huber_delta", 1.0))
    criterion = build_loss(loss_name, huber_delta)
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    ckpt_path = os.path.join(out_dir, f"seernet_metric{metric_idx}_{name}.pt")
    curve_path = os.path.join(out_dir, f"seernet_metric{metric_idx}_{name}.curve.json")
    best_val = float("inf")
    best_epoch = -1
    best_source = "raw"
    patience = int(train_cfg.get("patience", 30))
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(train_cfg.get("epochs", 500)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            target = select_target(batch, metric_idx)
            loss, _ = weighted_sample_metric_loss(pred, target, loss_name, huber_delta, sample_weight=sample_weights(batch, target))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            bs = target.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n += bs
        train_loss = train_loss_sum / max(train_n, 1)

        raw_val = evaluate_loss(model, val_loader, device, criterion, metric_idx=metric_idx)
        val_loss = raw_val
        source = "raw"
        if ema is not None:
            ema.apply(model)
            ema_val = evaluate_loss(model, val_loader, device, criterion, metric_idx=metric_idx)
            ema.restore(model)
            if ema_val < raw_val:
                val_loss = ema_val
                source = "ema"
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "raw_val_loss": raw_val, "source": source, "lr": lr})
        print(f"[{name}] epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} ({source}) | lr {lr:.2e}", flush=True)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            best_source = source
            epochs_no_improve = 0
            if source == "ema" and ema is not None:
                ema.apply(model)
            torch.save(
                checkpoint_payload(
                    model=model,
                    cfg=cfg,
                    model_cfg=model_cfg,
                    metadata=model_metadata,
                    epoch=epoch,
                    val_loss=val_loss,
                    metric_idx=metric_idx,
                    norm_stats=norm_stats,
                ),
                ckpt_path,
            )
            if source == "ema" and ema is not None:
                ema.restore(model)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[{name}] early stop at epoch {epoch}; best epoch {best_epoch} ({best_source})", flush=True)
            break

    with open(curve_path, "w") as fh:
        json.dump({"best_epoch": best_epoch, "best_val": best_val, "best_source": best_source, "history": history}, fh, indent=2)

    if cfg.get("calibration", {}).get("enabled"):
        ckpt = safe_torch_load(ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        pred_std, true_std = collect_std_predictions(model, val_loader, device, metric_idx=metric_idx)
        cal = fit_linear_calibration(pred_std, true_std).to_dict()
        ckpt["calibration"] = cal
        torch.save(ckpt, ckpt_path)
    print(f"[{name}] done. best val {best_val:.6f} -> {ckpt_path}", flush=True)
    return ckpt_path


def metric_index_from_checkpoint(ckpt: dict[str, Any], path: Path) -> int:
    if "metric_idx" in ckpt:
        idx = int(ckpt["metric_idx"])
    else:
        match = re.search(r"seernet_metric(\d+)_", path.name)
        if not match:
            raise ValueError(f"cannot infer metric index from teacher checkpoint {path}")
        idx = int(match.group(1))
    if not 0 <= idx < NUM_TARGETS:
        raise ValueError(f"teacher checkpoint {path} has invalid metric_idx {idx}")
    return idx


def checkpoint_norm_stats(ckpt: dict[str, Any], metric_idx: int | None = None) -> dict[str, np.ndarray] | None:
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    raw = meta.get("norm_stats") or ckpt.get("norm_stats")
    if raw:
        stats = {str(key): np.asarray(value, dtype=np.float32) for key, value in dict(raw).items()}
        if "y_mean" in stats and "y_std" in stats:
            return stats
    target_stats = ckpt.get("target_stats", {}) or {}
    if metric_idx is not None and {"log_mean", "log_std"} <= set(target_stats):
        return {
            "y_mean": np.asarray([float(target_stats["log_mean"])], dtype=np.float32),
            "y_std": np.asarray([float(target_stats["log_std"])], dtype=np.float32),
        }
    return None


def checkpoint_feature_config(ckpt: dict[str, Any]) -> FeatureConfig | None:
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    raw = meta.get("feature_config") or (meta.get("config", {}) or {}).get("features")
    if raw is None:
        return None
    return FeatureConfig.from_dict(raw)


def target_stats_for_prediction(
    stats: dict[str, np.ndarray],
    width: int,
    metric_idx: int | None = None,
) -> dict[str, np.ndarray]:
    out = dict(stats)
    for key in ("y_mean", "y_std"):
        arr = np.asarray(stats[key], dtype=np.float32).reshape(-1)
        if metric_idx is not None:
            if arr.size == NUM_TARGETS:
                arr = arr[metric_idx : metric_idx + 1]
            elif arr.size != 1:
                raise ValueError(f"teacher target stats {key} has {arr.size} entries; expected 1 or {NUM_TARGETS}")
        elif arr.size != width:
            raise ValueError(f"teacher target stats {key} has {arr.size} entries; expected {width}")
        out[key] = arr
    return out


def batch_base_raw(batch, target: torch.Tensor, metric_idx: int | None = None) -> torch.Tensor | None:
    raw = getattr(batch, "y_base_raw", None)
    if raw is None:
        return None
    base = raw.to(device=target.device, dtype=target.dtype).view(-1, NUM_TARGETS)
    if metric_idx is not None:
        return base[:, metric_idx : metric_idx + 1]
    return base


def standardize_absolute_targets(
    y_abs: torch.Tensor,
    stats: dict[str, np.ndarray],
    cfg: FeatureConfig,
    base_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    ym = torch.as_tensor(stats["y_mean"], dtype=y_abs.dtype, device=y_abs.device)
    ys = torch.as_tensor(stats["y_std"], dtype=y_abs.dtype, device=y_abs.device)
    if is_log_ratio_target(cfg):
        if base_raw is None:
            raise ValueError("base_raw is required to standardize absolute targets for log-ratio target mode")
        eps = y_abs.new_tensor(1e-9)
        base = base_raw.to(device=y_abs.device, dtype=y_abs.dtype)
        target_value = torch.log(torch.clamp(y_abs, min=eps)) - torch.log(torch.clamp(base, min=eps))
    else:
        target_value = torch.log1p(torch.clamp(y_abs, min=0.0))
    return (target_value - ym) / ys


def convert_teacher_prediction_to_student_space(
    pred: torch.Tensor,
    batch,
    teacher_stats: dict[str, np.ndarray] | None,
    teacher_cfg: FeatureConfig | None,
    student_stats: dict[str, np.ndarray] | None,
    student_cfg: FeatureConfig | None,
    metric_idx: int | None = None,
) -> torch.Tensor:
    if teacher_stats is None or teacher_cfg is None or student_stats is None or student_cfg is None:
        return pred
    if teacher_cfg.time_target_mode != student_cfg.time_target_mode:
        raise ValueError(
            "teacher/student time_target_mode mismatch: "
            f"{teacher_cfg.time_target_mode!r} vs {student_cfg.time_target_mode!r}"
        )
    width = int(pred.size(-1))
    teacher_target_stats = target_stats_for_prediction(teacher_stats, width, metric_idx)
    student_target_stats = target_stats_for_prediction(student_stats, width, metric_idx)
    teacher_base = batch_base_raw(batch, pred, metric_idx) if is_log_ratio_target(teacher_cfg) else None
    absolute = invert_targets(pred, teacher_target_stats, teacher_cfg, teacher_base)
    student_base = batch_base_raw(batch, pred, metric_idx) if is_log_ratio_target(student_cfg) else None
    return standardize_absolute_targets(absolute, student_target_stats, student_cfg, student_base)


def load_teacher_models(path: str | None, device: torch.device) -> TeacherBundle:
    if not path:
        return TeacherBundle(kind="none", models=[], paths=[])
    root = Path(str(path)).expanduser()
    if root.is_file():
        ckpt = safe_torch_load(str(root), device)
        if not checkpoint_is_multi(ckpt):
            raise ValueError(f"single teacher checkpoint {root} is not enough for multi-output distillation")
        model = checkpoint_model(ckpt, device)
        if getattr(model.cfg, "num_outputs", 0) != NUM_TARGETS:
            raise ValueError(f"multi-output teacher {root} has {model.cfg.num_outputs} outputs, expected {NUM_TARGETS}")
        return TeacherBundle(
            kind="multi",
            models=[model],
            paths=[str(root)],
            norm_stats=[checkpoint_norm_stats(ckpt)],
            feature_configs=[checkpoint_feature_config(ckpt)],
            metric_indices=[None],
        )
    if not root.is_dir():
        raise FileNotFoundError(f"teacher checkpoint path not found: {root}")

    multi_path = root / "seernet_multi.pt"
    if multi_path.exists():
        ckpt = safe_torch_load(str(multi_path), device)
        if not checkpoint_is_multi(ckpt):
            raise ValueError(f"teacher checkpoint {multi_path} is not a multi-output checkpoint")
        model = checkpoint_model(ckpt, device)
        if getattr(model.cfg, "num_outputs", 0) != NUM_TARGETS:
            raise ValueError(f"multi-output teacher {multi_path} has {model.cfg.num_outputs} outputs, expected {NUM_TARGETS}")
        return TeacherBundle(
            kind="multi",
            models=[model],
            paths=[str(multi_path)],
            norm_stats=[checkpoint_norm_stats(ckpt)],
            feature_configs=[checkpoint_feature_config(ckpt)],
            metric_indices=[None],
        )

    ckpts = sorted(root.glob("seernet_metric*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no seernet_multi.pt or seernet_metric*.pt teacher checkpoints found under {root}")
    teachers_by_metric: dict[int, torch.nn.Module] = {}
    paths_by_metric: dict[int, str] = {}
    stats_by_metric: dict[int, dict[str, np.ndarray] | None] = {}
    feature_cfg_by_metric: dict[int, FeatureConfig | None] = {}
    for ckpt_path in ckpts:
        ckpt = safe_torch_load(str(ckpt_path), device)
        if checkpoint_is_multi(ckpt):
            continue
        idx = metric_index_from_checkpoint(ckpt, ckpt_path)
        if idx in teachers_by_metric:
            raise ValueError(f"duplicate teacher checkpoint for metric {idx}: {paths_by_metric[idx]} and {ckpt_path}")
        teachers_by_metric[idx] = checkpoint_model(ckpt, device)
        paths_by_metric[idx] = str(ckpt_path)
        stats_by_metric[idx] = checkpoint_norm_stats(ckpt, idx)
        feature_cfg_by_metric[idx] = checkpoint_feature_config(ckpt)
    missing = [idx for idx in range(NUM_TARGETS) if idx not in teachers_by_metric]
    if missing:
        raise ValueError(f"teacher checkpoint directory {root} is missing metric checkpoints: {missing}")
    teachers = [teachers_by_metric[idx] for idx in range(NUM_TARGETS)]
    paths = [paths_by_metric[idx] for idx in range(NUM_TARGETS)]
    return TeacherBundle(
        kind="metric_ensemble",
        models=teachers,
        paths=paths,
        norm_stats=[stats_by_metric[idx] for idx in range(NUM_TARGETS)],
        feature_configs=[feature_cfg_by_metric[idx] for idx in range(NUM_TARGETS)],
        metric_indices=list(range(NUM_TARGETS)),
    )


def teacher_predictions(
    teachers: TeacherBundle | list[torch.nn.Module],
    batch,
    student_norm_stats: dict[str, np.ndarray] | None = None,
    student_feature_cfg: FeatureConfig | None = None,
) -> torch.Tensor | None:
    if isinstance(teachers, list):
        if len(teachers) != NUM_TARGETS:
            return None
        with torch.no_grad():
            return torch.cat([teacher(batch) for teacher in teachers], dim=-1)
    if teachers.kind == "multi" and len(teachers.models) == 1:
        with torch.no_grad():
            pred = teachers.models[0](batch)
            return convert_teacher_prediction_to_student_space(
                pred,
                batch,
                (teachers.norm_stats or [None])[0],
                (teachers.feature_configs or [None])[0],
                student_norm_stats,
                student_feature_cfg,
            )
    if teachers.kind != "metric_ensemble" or len(teachers.models) != NUM_TARGETS:
        return None
    with torch.no_grad():
        preds = []
        norm_stats = teachers.norm_stats or [None] * NUM_TARGETS
        feature_configs = teachers.feature_configs or [None] * NUM_TARGETS
        metric_indices = teachers.metric_indices or list(range(NUM_TARGETS))
        for idx, teacher in enumerate(teachers.models):
            metric_idx = metric_indices[idx]
            pred = teacher(batch)
            preds.append(
                convert_teacher_prediction_to_student_space(
                    pred,
                    batch,
                    norm_stats[idx],
                    feature_configs[idx],
                    student_norm_stats,
                    student_feature_cfg,
                    metric_idx,
                )
            )
        return torch.cat(preds, dim=-1)


def train_multi(
    train_ds,
    val_ds,
    norm_stats,
    cfg: dict[str, Any],
    run_id: str,
    out_dir: str,
    metadata: dict[str, Any],
    feature_cfg: FeatureConfig,
    device: torch.device,
) -> str:
    print("\n=== Training multi-output SeerNet ===", flush=True)
    train_cfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(cfg["data"].get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    model = make_model(cfg, feature_cfg, NUM_TARGETS, multi=True).to(device)
    init_info = load_initial_weights(model, cfg, device, metric_idx=None)
    model_metadata = copy.deepcopy(metadata)
    model_metadata["initialization"] = init_info
    model_cfg = model.cfg
    print(f"model parameters: {count_parameters(model):,}", flush=True)
    optimizer = make_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(train_cfg.get("scheduler_patience", 5)), min_lr=float(train_cfg.get("min_lr", 1e-6)))
    loss_name = str(train_cfg.get("loss", "mse_logstd"))
    huber_delta = float(train_cfg.get("huber_delta", 1.0))
    criterion = build_loss(loss_name, huber_delta)
    weights_cfg = cfg.get("multi_task", {}).get("loss_weights", {}) or {}
    weights = torch.tensor([float(weights_cfg.get(name, 1.0)) for name in METRIC_NAMES], dtype=torch.float32, device=device)
    use_pcgrad = str(cfg.get("multi_task", {}).get("loss_reduction", "plain_sum")).lower() == "pcgrad"
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    distill_cfg = cfg.get("distillation", {})
    teachers = load_teacher_models(distill_cfg.get("teacher_ckpt_dir"), device) if distill_cfg.get("enabled") else TeacherBundle(kind="none", models=[], paths=[])
    if distill_cfg.get("enabled") and teachers.kind == "none":
        print("distillation enabled but no teacher_ckpt_dir was provided; using hard labels only", flush=True)
    elif teachers.kind != "none":
        print(f"distillation teacher: {teachers.kind} from {len(teachers.paths)} checkpoint(s)", flush=True)
    model_metadata["distillation_teacher"] = teachers.to_metadata()
    model_metadata["distillation_policy"] = {
        "alpha": float(distill_cfg.get("alpha", 0.5)),
        "source_hard_alpha": float(distill_cfg.get("source_hard_alpha", distill_cfg.get("alpha", 0.5))),
        "precision_hard_alpha": float(distill_cfg.get("precision_hard_alpha", 1.0)),
        "pseudo_hard_alpha": float(distill_cfg.get("pseudo_hard_alpha", 0.0)),
        "label_domains": list(LABEL_DOMAIN_VOCAB),
    }

    ckpt_path = os.path.join(out_dir, "seernet_multi.pt")
    curve_path = os.path.join(out_dir, "seernet_multi.curve.json")
    best_val = float("inf")
    best_epoch = -1
    patience = int(train_cfg.get("patience", 30))
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(train_cfg.get("epochs", 500)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            target = full_target(batch)
            sw = sample_weights(batch, target)
            label_loss, task_losses = weighted_sample_metric_loss(pred, target, loss_name, huber_delta, weights, sw)
            teacher_pred = teacher_predictions(teachers, batch, norm_stats, feature_cfg)
            if teacher_pred is not None:
                hard_alpha = distillation_hard_alphas(batch, target, distill_cfg)
                total_loss, _ = weighted_sample_distillation_loss(pred, target, teacher_pred, loss_name, huber_delta, weights, sw, hard_alpha)
            else:
                total_loss = label_loss
            if use_pcgrad and teacher_pred is None:
                pcgrad_backward(task_losses, [p for p in model.parameters() if p.requires_grad], weights)
            else:
                total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            bs = target.size(0)
            train_loss_sum += float(total_loss.item()) * bs
            train_n += bs
        train_loss = train_loss_sum / max(train_n, 1)

        raw_val = evaluate_loss(model, val_loader, device, criterion, metric_idx=None, weights=weights)
        val_loss = raw_val
        source = "raw"
        if ema is not None:
            ema.apply(model)
            ema_val = evaluate_loss(model, val_loader, device, criterion, metric_idx=None, weights=weights)
            ema.restore(model)
            if ema_val < raw_val:
                val_loss = ema_val
                source = "ema"
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "raw_val_loss": raw_val, "source": source, "lr": lr})
        print(f"[multi] epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} ({source}) | lr {lr:.2e}", flush=True)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if source == "ema" and ema is not None:
                ema.apply(model)
            torch.save(
                checkpoint_payload(
                    model=model,
                    cfg=cfg,
                    model_cfg=model_cfg,
                    metadata=model_metadata,
                    epoch=epoch,
                    val_loss=val_loss,
                    metric_idx=None,
                    norm_stats=norm_stats,
                ),
                ckpt_path,
            )
            if source == "ema" and ema is not None:
                ema.restore(model)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[multi] early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    with open(curve_path, "w") as fh:
        json.dump({"best_epoch": best_epoch, "best_val": best_val, "history": history}, fh, indent=2)
    if cfg.get("calibration", {}).get("enabled"):
        ckpt = safe_torch_load(ckpt_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        pred_std, true_std = collect_std_predictions(model, val_loader, device, metric_idx=None)
        ckpt["calibration"] = fit_linear_calibration(pred_std, true_std).to_dict()
        torch.save(ckpt, ckpt_path)
    print(f"[multi] done. best val {best_val:.6f} -> {ckpt_path}", flush=True)
    return ckpt_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train optimized PerfSeer models.")
    p.add_argument("--config", help="YAML config path")
    p.add_argument("--metric", help="override train.metric")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="override train.batch_size")
    p.add_argument("--lr", type=float, help="override train.lr")
    p.add_argument("--out", help="override run.out_dir")
    p.add_argument("--results-path", dest="results_path", help="override run.results_path")
    p.add_argument("--run-id", dest="run_id", help="override run.run_id")
    p.add_argument("--data-root", dest="data_root", help="override data.root")
    p.add_argument("--split-unit", choices=("pair", "graph", "graph_signature", "graph_family"), dest="split_unit", help="override data.split_unit")
    p.add_argument("--precision-config", dest="precision_config", help="override features.precision_config")
    p.add_argument("--hardware-id", dest="hardware_id", help="override features.hardware_id")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--limit", type=int, help="override data.limit")
    p.add_argument("--num-workers", type=int, dest="num_workers", help="override data.num_workers")
    p.add_argument("--threads", type=int, help="override train.threads")
    p.add_argument("--init-checkpoint", dest="init_checkpoint", help="override train.init_checkpoint")
    p.add_argument("--teacher-ckpt-dir", dest="teacher_ckpt_dir", help="override distillation.teacher_ckpt_dir")
    p.add_argument("--source-precision-provenance", default=None, help="Record provenance for source-domain labels used by this run.")
    p.add_argument("--require-source-precision-provenance", action="store_true", help="Fail if source precision provenance is not recorded.")
    return p.parse_args(argv)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.metric is not None:
        cfg["train"]["metric"] = args.metric
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.out is not None:
        cfg["run"]["out_dir"] = args.out
    if args.results_path is not None:
        cfg["run"]["results_path"] = args.results_path
    if args.run_id is not None:
        cfg["run"]["run_id"] = args.run_id
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.split_unit is not None:
        cfg["data"]["split_unit"] = args.split_unit
    if args.precision_config is not None:
        cfg.setdefault("features", {})
        cfg["features"]["precision_config"] = normalize_precision_config(args.precision_config)
    if args.hardware_id is not None:
        cfg.setdefault("features", {})
        cfg["features"]["hardware_id"] = str(args.hardware_id)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.threads is not None:
        cfg["train"]["threads"] = args.threads
    if args.init_checkpoint is not None:
        cfg["train"]["init_checkpoint"] = args.init_checkpoint
    if args.teacher_ckpt_dir is not None:
        cfg.setdefault("distillation", {})
        cfg["distillation"]["enabled"] = True
        cfg["distillation"]["teacher_ckpt_dir"] = args.teacher_ckpt_dir
    if args.source_precision_provenance is not None:
        cfg.setdefault("data", {})
        cfg["data"]["source_precision_provenance"] = str(args.source_precision_provenance or "").strip()
        precision_config = normalize_precision_config(str(cfg.get("features", {}).get("precision_config", "fp32_ieee")))
        cfg["data"]["source_precision_confirmed"] = bool(cfg["data"]["source_precision_provenance"]) and precision_config != SOURCE_UNKNOWN_PRECISION_CONFIG
    if args.require_source_precision_provenance:
        cfg.setdefault("data", {})
        provenance_recorded = bool(str(cfg["data"].get("source_precision_provenance") or "").strip())
        precision_config = normalize_precision_config(str(cfg.get("features", {}).get("precision_config", "fp32_ieee")))
        cfg["data"]["source_precision_confirmed"] = provenance_recorded and precision_config != SOURCE_UNKNOWN_PRECISION_CONFIG
        if not provenance_recorded:
            raise ValueError("--require-source-precision-provenance was set but source precision provenance is empty")
    return cfg


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))
    configure_threads(cfg)
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    run_id = make_run_id(cfg)
    out_dir = os.path.join(str(cfg["run"].get("out_dir", "runs/optimized")), run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.resolved.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"run_id: {run_id}", flush=True)
    print(f"out_dir: {out_dir}", flush=True)
    print(f"device: {device} | torch threads: {torch.get_num_threads()}", flush=True)

    train_ds, val_ds, _test_ds, norm_stats, feature_cfg, split_meta = build_datasets(cfg)
    metadata = base_metadata(cfg, run_id, feature_cfg, norm_stats, split_meta)
    ckpts: list[str] = []
    t0 = time.time()
    model_name = str(cfg["model"].get("name", "seernet")).lower()
    multi_enabled = bool(cfg.get("multi_task", {}).get("enabled")) or model_name == "seernet_multi"
    if multi_enabled:
        cfg["model"]["name"] = "seernet_multi"
        ckpts.append(train_multi(train_ds, val_ds, norm_stats, cfg, run_id, out_dir, metadata, feature_cfg, device))
    else:
        for metric_idx in resolve_metrics(str(cfg["train"].get("metric", "all"))):
            ckpts.append(train_one_metric(metric_idx, train_ds, val_ds, norm_stats, cfg, run_id, out_dir, metadata, feature_cfg, device))

    elapsed = time.time() - t0
    checkpoint_metadata = checkpoint_metadata_summaries(ckpts)
    append_jsonl(
        str(cfg["run"].get("results_path", "runs/results.jsonl")),
        {
            "event": "train_complete",
            "run_id": run_id,
            "out_dir": out_dir,
            "checkpoints": ckpts,
            "checkpoint_metadata": checkpoint_metadata,
            "split": split_meta,
            "elapsed_sec": elapsed,
            "config": cfg,
        },
    )
    print(f"\nall done in {elapsed:.1f}s. checkpoints:", flush=True)
    for ckpt in ckpts:
        print(f"  {ckpt}", flush=True)


if __name__ == "__main__":
    main()
