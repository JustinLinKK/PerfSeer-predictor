"""Canonical, extensible optimizer and learning-schedule semantics for v3."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping, Sequence

from .baseline import canonical_json


OPTIMIZERS: tuple[str, ...] = (
    "none",
    "sgd",
    "asgd",
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamw",
    "lbfgs",
    "muon",
    "nadam",
    "radam",
    "rmsprop",
    "rprop",
    "sparse_adam",
    "lamb",
    "lars",
    "lion",
    "other",
)
OPTIMIZER_FAMILIES: tuple[str, ...] = (
    "none",
    "sgd_momentum",
    "adaptive_moment",
    "adaptive_factor",
    "orthogonalized",
    "second_order",
    "resilient",
    "composite",
    "custom",
)
SCHEDULERS: tuple[str, ...] = (
    "none",
    "constant",
    "constant_with_warmup",
    "linear",
    "linear_with_warmup",
    "step",
    "multi_step",
    "exponential",
    "cosine",
    "cosine_warm_restarts",
    "cosine_with_warmup",
    "polynomial",
    "one_cycle",
    "cyclic",
    "reduce_on_plateau",
    "inverse_sqrt",
    "warmup_stable_decay",
    "other",
)
SCHEDULER_FAMILIES: tuple[str, ...] = (
    "none",
    "constant",
    "warmup",
    "step_decay",
    "exponential",
    "cosine",
    "cyclic",
    "metric_adaptive",
    "inverse_power",
    "composite",
    "custom",
)
OPTIMIZER_HASH_BUCKETS = 256
SCHEDULER_HASH_BUCKETS = 128


_OPTIMIZER_ALIASES = {
    "meuon": "muon",
    "sparseadam": "sparse_adam",
    "adam_w": "adamw",
    "l_bfgs": "lbfgs",
}
_OPTIMIZER_FAMILY = {
    "none": "none",
    "sgd": "sgd_momentum",
    "asgd": "sgd_momentum",
    "lars": "sgd_momentum",
    "adam": "adaptive_moment",
    "adamax": "adaptive_moment",
    "adamw": "adaptive_moment",
    "lamb": "adaptive_moment",
    "lion": "adaptive_moment",
    "nadam": "adaptive_moment",
    "radam": "adaptive_moment",
    "sparse_adam": "adaptive_moment",
    "adadelta": "adaptive_factor",
    "adafactor": "adaptive_factor",
    "adagrad": "adaptive_factor",
    "rmsprop": "adaptive_factor",
    "muon": "orthogonalized",
    "lbfgs": "second_order",
    "rprop": "resilient",
}
_SCHEDULER_ALIASES = {
    "steplr": "step",
    "multisteplr": "multi_step",
    "exponentiallr": "exponential",
    "linearlr": "linear",
    "constantlr": "constant",
    "cosineannealinglr": "cosine",
    "cosineannealingwarmrestarts": "cosine_warm_restarts",
    "onecyclelr": "one_cycle",
    "cycliclr": "cyclic",
    "reducelronplateau": "reduce_on_plateau",
    "polynomiallr": "polynomial",
    "lambdalr": "other",
    "sequentiallr": "other",
    "chainedscheduler": "other",
    "warmup_stable_decay_schedule": "warmup_stable_decay",
}
_SCHEDULER_FAMILY = {
    "none": "none",
    "constant": "constant",
    "constant_with_warmup": "warmup",
    "linear": "step_decay",
    "linear_with_warmup": "warmup",
    "step": "step_decay",
    "multi_step": "step_decay",
    "exponential": "exponential",
    "cosine": "cosine",
    "cosine_warm_restarts": "cosine",
    "cosine_with_warmup": "warmup",
    "polynomial": "inverse_power",
    "one_cycle": "cyclic",
    "cyclic": "cyclic",
    "reduce_on_plateau": "metric_adaptive",
    "inverse_sqrt": "inverse_power",
    "warmup_stable_decay": "warmup",
}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.removeprefix("torch.optim.").removeprefix("torch.optim.lr_scheduler.")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "none"


def canonical_optimizer_name(value: Any) -> str:
    name = _slug(value)
    return _OPTIMIZER_ALIASES.get(name, name)


def optimizer_components(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("components", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    names = []
    for component in raw:
        if isinstance(component, Mapping):
            names.append(canonical_optimizer_name(component.get("name", "other")))
        else:
            names.append(canonical_optimizer_name(component))
    return tuple(sorted(set(names)))


def optimizer_identity(config: Mapping[str, Any]) -> tuple[str, str, str]:
    name = canonical_optimizer_name(config.get("name", "none"))
    components = optimizer_components(config)
    if len(components) > 1:
        family = "composite"
        signature = "+".join(components)
    else:
        family = _OPTIMIZER_FAMILY.get(name)
        if family is None:
            if any(token in name for token in ("adam", "lion", "lamb")):
                family = "adaptive_moment"
            elif any(token in name for token in ("sgd", "momentum", "lars")):
                family = "sgd_momentum"
            elif "muon" in name or "orthogonal" in name:
                family = "orthogonalized"
            else:
                family = "custom"
        signature = name
    exact = name if name in OPTIMIZERS else "other"
    return exact, family, signature


def scheduler_config(training_config: Mapping[str, Any]) -> dict[str, Any]:
    raw = training_config.get("scheduler")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        return {"name": raw}
    name = training_config.get(
        "scheduler_name",
        training_config.get("lr_scheduler_type", "none"),
    )
    extra = training_config.get("scheduler_config", {})
    result = dict(extra) if isinstance(extra, Mapping) else {}
    result.setdefault("name", name)
    return result


def canonical_scheduler_name(value: Any) -> str:
    name = _slug(value)
    return _SCHEDULER_ALIASES.get(name, name)


def scheduler_identity(training_config: Mapping[str, Any]) -> tuple[str, str, str]:
    config = scheduler_config(training_config)
    name = canonical_scheduler_name(config.get("name", "none"))
    chained = config.get("components", ())
    if isinstance(chained, Sequence) and not isinstance(chained, (str, bytes)):
        components = sorted(
            {
                canonical_scheduler_name(
                    component.get("name", "other")
                    if isinstance(component, Mapping)
                    else component
                )
                for component in chained
            }
        )
    else:
        components = []
    if len(components) > 1:
        family = "composite"
        signature = "+".join(components)
    else:
        family = _SCHEDULER_FAMILY.get(name)
        if family is None:
            if "warmup" in name:
                family = "warmup"
            elif "cos" in name:
                family = "cosine"
            elif "cycle" in name:
                family = "cyclic"
            elif "plateau" in name:
                family = "metric_adaptive"
            else:
                family = "custom"
        signature = name
    exact = name if name in SCHEDULERS else "other"
    return exact, family, signature


def stable_category_bucket(value: str, buckets: int) -> int:
    if not value or value == "none":
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (buckets - 1)


def optimizer_state_multiplier(name: str, config: Mapping[str, Any]) -> float:
    components = config.get("components", ())
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)) and components:
        parsed = [component for component in components if isinstance(component, Mapping)]
        if parsed:
            explicit = [float(component.get("parameter_fraction", 0.0)) for component in parsed]
            if sum(explicit) <= 0:
                explicit = [1.0 / len(parsed)] * len(parsed)
            else:
                total = sum(explicit)
                explicit = [value / total for value in explicit]
            return sum(
                weight
                * optimizer_state_multiplier(
                    canonical_optimizer_name(component.get("name", "other")),
                    component,
                )
                for weight, component in zip(explicit, parsed)
            )
    name = canonical_optimizer_name(name)
    if name in {"adam", "adamw", "nadam", "radam", "lamb", "sparse_adam"}:
        return 3.0 if bool(config.get("amsgrad", False)) else 2.0
    if name == "adamax":
        return 2.0
    if name in {"sgd", "lars"}:
        return 1.0 if float(config.get("momentum", 0.0)) > 0 else 0.0
    if name in {"asgd", "adagrad", "adafactor", "muon", "lion"}:
        return 1.0
    if name == "adadelta":
        return 2.0
    if name == "rmsprop":
        return 1.0 + float(float(config.get("momentum", 0.0)) > 0) + float(
            bool(config.get("centered", False))
        )
    if name == "rprop":
        return 2.0
    if name == "lbfgs":
        return float(max(1, int(config.get("history_size", 100))) + 1)
    return float(max(0.0, float(config.get("state_multiplier", 0.0))))


def optimizer_flops_per_parameter(name: str, config: Mapping[str, Any]) -> float:
    if "flops_per_parameter" in config:
        return _finite_number(config["flops_per_parameter"], "flops_per_parameter")
    name = canonical_optimizer_name(name)
    if name in {"adam", "adamax", "adamw", "lamb", "nadam", "radam", "sparse_adam"}:
        return 8.0
    if name == "muon":
        return 4.0 + 2.0 * max(1.0, float(config.get("ns_steps", 5)))
    if name in {"adadelta", "adafactor", "adagrad", "rmsprop", "rprop"}:
        return 6.0
    if name == "lbfgs":
        return 10.0
    return 2.0


def _finite_number(value: Any, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _first(sources: Sequence[Mapping[str, Any]], keys: Sequence[str], default: Any = 0.0) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _numeric_summary(values: Sequence[float], fallback: float) -> tuple[float, float, float, float]:
    materialized = [float(value) for value in values] or [float(fallback)]
    mean = sum(materialized) / len(materialized)
    variance = sum((value - mean) ** 2 for value in materialized) / len(materialized)
    return min(materialized), max(materialized), mean, math.sqrt(variance)


def training_hyperparameter_values(
    optimizer_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, float]:
    """Return finite scalar summaries without restricting custom configurations."""

    optimizer = dict(optimizer_config)
    training = dict(training_config)
    scheduler = scheduler_config(training)
    sources = (training, optimizer)
    betas = optimizer.get("betas", ())
    if not isinstance(betas, Sequence) or isinstance(betas, (str, bytes)):
        betas = ()
    base_lr = _finite_number(
        _first(sources, ("learning_rate", "lr"), 0.0),
        "learning_rate",
    )
    initial_lr = _finite_number(
        _first((training, scheduler, optimizer), ("initial_learning_rate", "initial_lr"), base_lr),
        "initial_learning_rate",
    )
    current_lr = _finite_number(
        _first((training, scheduler, optimizer), ("current_learning_rate", "current_lr", "learning_rate", "lr"), base_lr),
        "current_learning_rate",
    )
    groups = optimizer.get("parameter_groups", optimizer.get("param_groups", ()))
    group_rows = (
        [row for row in groups if isinstance(row, Mapping)]
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes))
        else []
    )
    group_lrs = [
        _finite_number(_first((row,), ("learning_rate", "lr"), current_lr), "parameter_group.lr")
        for row in group_rows
    ]
    group_weight_decays = [
        _finite_number(row.get("weight_decay", optimizer.get("weight_decay", 0.0)), "parameter_group.weight_decay")
        for row in group_rows
    ]
    lr_min, lr_max, lr_mean, lr_std = _numeric_summary(group_lrs, current_lr)
    wd = _finite_number(optimizer.get("weight_decay", 0.0), "weight_decay")
    wd_min, wd_max, wd_mean, wd_std = _numeric_summary(group_weight_decays, wd)
    name, _, _ = optimizer_identity(optimizer)
    components = optimizer_components(optimizer)

    values = {
        "total_epochs": _finite_number(_first((training,), ("total_epochs", "epochs", "num_train_epochs"), 0.0), "total_epochs"),
        "current_epoch": _finite_number(training.get("current_epoch", 0.0), "current_epoch"),
        "steps_per_epoch": _finite_number(training.get("steps_per_epoch", 0.0), "steps_per_epoch"),
        "total_training_steps": _finite_number(_first((training, scheduler), ("total_training_steps", "total_steps"), 0.0), "total_training_steps"),
        "current_training_step": _finite_number(_first((training,), ("current_training_step", "current_step", "global_step"), 0.0), "current_training_step"),
        "learning_rate_initial": initial_lr,
        "learning_rate_current": current_lr,
        "learning_rate_min": _finite_number(_first((training, scheduler), ("min_learning_rate", "min_lr", "eta_min"), lr_min), "learning_rate_min"),
        "learning_rate_max": _finite_number(_first((training, scheduler), ("max_learning_rate", "max_lr"), lr_max), "learning_rate_max"),
        "parameter_group_learning_rate_min": lr_min,
        "parameter_group_learning_rate_max": lr_max,
        "parameter_group_learning_rate_mean": lr_mean,
        "parameter_group_learning_rate_std": lr_std,
        "weight_decay": wd,
        "parameter_group_weight_decay_min": wd_min,
        "parameter_group_weight_decay_max": wd_max,
        "parameter_group_weight_decay_mean": wd_mean,
        "parameter_group_weight_decay_std": wd_std,
        "optimizer_momentum": _finite_number(optimizer.get("momentum", 0.0), "momentum"),
        "optimizer_dampening": _finite_number(optimizer.get("dampening", 0.0), "dampening"),
        "optimizer_beta1": _finite_number(optimizer.get("beta1", betas[0] if len(betas) > 0 else 0.0), "beta1"),
        "optimizer_beta2": _finite_number(optimizer.get("beta2", betas[1] if len(betas) > 1 else 0.0), "beta2"),
        "optimizer_beta3": _finite_number(optimizer.get("beta3", betas[2] if len(betas) > 2 else 0.0), "beta3"),
        "optimizer_epsilon": _finite_number(_first((optimizer,), ("epsilon", "eps"), 0.0), "epsilon"),
        "optimizer_rho": _finite_number(optimizer.get("rho", 0.0), "rho"),
        "optimizer_alpha": _finite_number(optimizer.get("alpha", 0.0), "alpha"),
        "optimizer_trust_coefficient": _finite_number(optimizer.get("trust_coefficient", 0.0), "trust_coefficient"),
        "optimizer_clip_threshold": _finite_number(optimizer.get("clip_threshold", 0.0), "clip_threshold"),
        "optimizer_decay_rate": _finite_number(optimizer.get("decay_rate", 0.0), "optimizer.decay_rate"),
        "optimizer_ns_steps": _finite_number(optimizer.get("ns_steps", 0.0), "ns_steps"),
        "optimizer_parameter_group_count": float(len(group_rows) or (name != "none")),
        "optimizer_component_count": float(len(components) or (name != "none")),
        "scheduler_warmup_steps": _finite_number(_first((training, scheduler), ("warmup_steps", "num_warmup_steps"), 0.0), "warmup_steps"),
        "scheduler_warmup_epochs": _finite_number(_first((training, scheduler), ("warmup_epochs", "num_warmup_epochs"), 0.0), "warmup_epochs"),
        "scheduler_warmup_ratio": _finite_number(_first((training, scheduler), ("warmup_ratio", "pct_start"), 0.0), "warmup_ratio"),
        "scheduler_decay_rate": _finite_number(_first((scheduler, training), ("decay_rate", "gamma"), 0.0), "scheduler.decay_rate"),
        "scheduler_decay_steps": _finite_number(_first((scheduler, training), ("decay_steps", "step_size", "total_iters"), 0.0), "scheduler.decay_steps"),
        "scheduler_patience": _finite_number(scheduler.get("patience", 0.0), "scheduler.patience"),
        "scheduler_threshold": _finite_number(scheduler.get("threshold", 0.0), "scheduler.threshold"),
        "scheduler_cosine_cycles": _finite_number(_first((scheduler, training), ("num_cycles", "cycles"), 0.0), "scheduler.cosine_cycles"),
        "scheduler_polynomial_power": _finite_number(scheduler.get("power", 0.0), "scheduler.power"),
        "scheduler_cooldown_steps": _finite_number(_first((scheduler,), ("cooldown_steps", "cooldown"), 0.0), "scheduler.cooldown_steps"),
        "optimizer_nesterov": float(bool(optimizer.get("nesterov", False))),
        "optimizer_amsgrad": float(bool(optimizer.get("amsgrad", False))),
        "optimizer_maximize": float(bool(optimizer.get("maximize", False))),
        "optimizer_capturable": float(bool(optimizer.get("capturable", False))),
        "optimizer_differentiable": float(bool(optimizer.get("differentiable", False))),
        "optimizer_decoupled_weight_decay": float(bool(optimizer.get("decoupled_weight_decay", name in {"adamw", "lamb", "muon"}))),
        "optimizer_relative_step": float(bool(optimizer.get("relative_step", False))),
        "optimizer_scale_parameter": float(bool(optimizer.get("scale_parameter", False))),
        "optimizer_warmup_init": float(bool(optimizer.get("warmup_init", False))),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("training hyperparameter summaries must be finite")
    return values


def training_semantics_payload() -> dict[str, Any]:
    return {
        "optimizers": list(OPTIMIZERS),
        "optimizer_families": list(OPTIMIZER_FAMILIES),
        "optimizer_hash_buckets": OPTIMIZER_HASH_BUCKETS,
        "schedulers": list(SCHEDULERS),
        "scheduler_families": list(SCHEDULER_FAMILIES),
        "scheduler_hash_buckets": SCHEDULER_HASH_BUCKETS,
    }


def training_semantics_sha256() -> str:
    return hashlib.sha256(canonical_json(training_semantics_payload()).encode()).hexdigest()


__all__ = [
    "OPTIMIZERS",
    "OPTIMIZER_FAMILIES",
    "OPTIMIZER_HASH_BUCKETS",
    "SCHEDULERS",
    "SCHEDULER_FAMILIES",
    "SCHEDULER_HASH_BUCKETS",
    "canonical_optimizer_name",
    "canonical_scheduler_name",
    "optimizer_components",
    "optimizer_flops_per_parameter",
    "optimizer_identity",
    "optimizer_state_multiplier",
    "scheduler_config",
    "scheduler_identity",
    "stable_category_bucket",
    "training_hyperparameter_values",
    "training_semantics_payload",
    "training_semantics_sha256",
]
