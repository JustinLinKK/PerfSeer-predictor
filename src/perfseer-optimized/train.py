"""Training CLI for optimized PerfSeer models."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time
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
    NUM_TARGETS,
    TARGET_NAMES,
    PerfSeerOptimizedDataset,
    compute_norm_stats,
    feature_layout,
    norm_stats_to_serializable,
    split_dataset,
    split_hash,
)
from .losses import build_loss, weighted_metric_loss
from .model import SeerNet, SeerNetConfig, SeerNetMulti, count_parameters
from .pcgrad import pcgrad_backward


METRIC_NAMES = TARGET_NAMES


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {"run_id": None, "out_dir": "runs/optimized", "results_path": "runs/results.jsonl", "notes": ""},
    "seed": 42,
    "data": {"root": "dataset", "limit": 0, "num_workers": 0},
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
    train_files, val_files, test_files = split_dataset(data_root, seed=seed)
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
    split_meta = {
        "seed": seed,
        "train_hash": split_hash(train_files),
        "val_hash": split_hash(val_files),
        "test_hash": split_hash(test_files),
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "train_stems": [Path(gp).stem for gp, _ in train_files],
        "val_stems": [Path(gp).stem for gp, _ in val_files],
        "test_stems": [Path(gp).stem for gp, _ in test_files],
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
    return {
        "run_id": run_id,
        "config": cfg,
        "feature_config": feature_cfg.to_dict(),
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
    model_cfg = model.cfg
    print(f"model parameters: {count_parameters(model):,}", flush=True)
    optimizer = make_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(train_cfg.get("scheduler_patience", 5)), min_lr=float(train_cfg.get("min_lr", 1e-6)))
    criterion = build_loss(str(train_cfg.get("loss", "mse_logstd")), float(train_cfg.get("huber_delta", 1.0)))
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
            loss = criterion(pred, target)
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
                    metadata=metadata,
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
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        pred_std, true_std = collect_std_predictions(model, val_loader, device, metric_idx=metric_idx)
        cal = fit_linear_calibration(pred_std, true_std).to_dict()
        ckpt["calibration"] = cal
        torch.save(ckpt, ckpt_path)
    print(f"[{name}] done. best val {best_val:.6f} -> {ckpt_path}", flush=True)
    return ckpt_path


def load_teacher_models(path: str | None, device: torch.device) -> list[torch.nn.Module]:
    if not path:
        return []
    ckpts = sorted(Path(path).glob("seernet_metric*.pt"))
    teachers: list[torch.nn.Module] = []
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_cfg = SeerNetConfig.from_dict(ckpt["model_config"])
        model = SeerNet(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        teachers.append(model)
    return teachers


def teacher_predictions(teachers: list[torch.nn.Module], batch) -> torch.Tensor | None:
    if len(teachers) != NUM_TARGETS:
        return None
    with torch.no_grad():
        return torch.cat([teacher(batch) for teacher in teachers], dim=-1)


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
    model_cfg = model.cfg
    print(f"model parameters: {count_parameters(model):,}", flush=True)
    optimizer = make_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(train_cfg.get("scheduler_patience", 5)), min_lr=float(train_cfg.get("min_lr", 1e-6)))
    criterion = build_loss(str(train_cfg.get("loss", "mse_logstd")), float(train_cfg.get("huber_delta", 1.0)))
    weights_cfg = cfg.get("multi_task", {}).get("loss_weights", {}) or {}
    weights = torch.tensor([float(weights_cfg.get(name, 1.0)) for name in METRIC_NAMES], dtype=torch.float32, device=device)
    use_pcgrad = str(cfg.get("multi_task", {}).get("loss_reduction", "plain_sum")).lower() == "pcgrad"
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    teachers = load_teacher_models(cfg.get("distillation", {}).get("teacher_ckpt_dir"), device) if cfg.get("distillation", {}).get("enabled") else []
    distill_alpha = float(cfg.get("distillation", {}).get("alpha", 0.5))

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
            label_loss, task_losses = weighted_metric_loss(pred, target, criterion, weights)
            teacher_pred = teacher_predictions(teachers, batch)
            if teacher_pred is not None:
                distill_loss, _ = weighted_metric_loss(pred, teacher_pred, criterion, weights)
                total_loss = distill_alpha * label_loss + (1.0 - distill_alpha) * distill_loss
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
                    metadata=metadata,
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
        ckpt = torch.load(ckpt_path, map_location=device)
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
    p.add_argument("--data-root", dest="data_root", help="override data.root")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--limit", type=int, help="override data.limit")
    p.add_argument("--num-workers", type=int, dest="num_workers", help="override data.num_workers")
    p.add_argument("--threads", type=int, help="override train.threads")
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
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.threads is not None:
        cfg["train"]["threads"] = args.threads
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
    append_jsonl(
        str(cfg["run"].get("results_path", "runs/results.jsonl")),
        {
            "event": "train_complete",
            "run_id": run_id,
            "out_dir": out_dir,
            "checkpoints": ckpts,
            "elapsed_sec": elapsed,
            "config": cfg,
        },
    )
    print(f"\nall done in {elapsed:.1f}s. checkpoints:", flush=True)
    for ckpt in ckpts:
        print(f"  {ckpt}", flush=True)


if __name__ == "__main__":
    main()
