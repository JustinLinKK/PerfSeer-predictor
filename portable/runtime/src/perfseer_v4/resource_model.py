"""Static training-resource descriptors and a compact hardware-conditioned MLP."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from perfseer_v3.hardware import HARDWARE_CONTINUOUS_FIELDS

from .capture import graph_from_design
from .model import SeerOutputV4
from .version import TARGET_NAMES, HEAD_GROUPS, SM_INDICES, POSITIVE_INDICES


RESOURCE_NAMES = (
    "log_flops", "log_operations", "log_tensor_bytes", "log_microbatch", "log_accumulation",
    "amp", "tf32", "compiled", "log_critical_path", "checkpointing", "log_parameter_bytes",
    "log_optimizer_bytes", "log_live_activation_bytes", "log_saved_bytes", "alias_fraction",
    "log_steps_per_epoch", "epoch_length_unknown",
)


def resource_values(design):
    graph = graph_from_design(design)
    global_features, training = graph.global_features, graph.training_config
    precision = training["precision"]

    def log(value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid static training resource")
        return math.log1p(value)

    steps = training.get("steps_per_epoch")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("known epoch length must be a positive integer")
    values = {
        "log_flops": log(global_features.total_flops), "log_operations": log(global_features.operation_nodes),
        "log_tensor_bytes": log(sum(edge.tensor_bytes or 0 for edge in graph.tensor_edges)),
        "log_microbatch": log(training["microbatch_size"]),
        "log_accumulation": log(training["gradient_accumulation_steps"]),
        "amp": float(precision in {"bf16_amp", "fp16_amp", "bf16", "fp16_grad_scaler", "mixed_structured"}),
        "tf32": float(precision in {"tf32", "fp32_tf32"}),
        "compiled": float(training.get("backend", "unknown") in {"torch_compile", "torch_compile_inductor", "inductor", "inductor_cuda"}),
        "log_critical_path": log(global_features.critical_path_length),
        "checkpointing": float(bool(training.get("checkpointing", training.get("activation_checkpointing", False)))),
        "log_parameter_bytes": log(global_features.total_parameter_bytes),
        "log_optimizer_bytes": log(global_features.total_optimizer_state_bytes),
        "log_live_activation_bytes": log(global_features.peak_live_activation_bytes),
        "log_saved_bytes": log(global_features.total_saved_for_backward_bytes),
        "alias_fraction": sum(edge.is_view for edge in graph.tensor_edges) / max(1, len(graph.tensor_edges)),
        "log_steps_per_epoch": 0. if steps is None else log(steps),
        "epoch_length_unknown": float(steps is None),
    }
    return {name: values[name] for name in RESOURCE_NAMES}


@dataclass(frozen=True)
class ResourceMLPConfig:
    resource_dim: int = len(RESOURCE_NAMES)
    hardware_dim: int = 2 * len(HARDWARE_CONTINUOUS_FIELDS)
    hidden: int = 128
    num_outputs: int = 7

    def __post_init__(self):
        if self.resource_dim != len(RESOURCE_NAMES) or self.num_outputs != len(TARGET_NAMES):
            raise ValueError("resource MLP requires seventeen descriptors and seven outputs")
        if self.hidden < 1 or self.hardware_dim < 2 or self.hardware_dim % 2:
            raise ValueError("invalid resource MLP dimensions")

    def to_dict(self):
        return asdict(self)


class ResourceMLP(nn.Module):
    model_variant = "v4.3"

    def __init__(self, config, target_scales, resource_mean, resource_scale, target_medians=None):
        super().__init__()
        self.config = config
        scales = torch.as_tensor(target_scales, dtype=torch.float32).detach().clone()
        mean = torch.as_tensor(resource_mean, dtype=torch.float32).detach().clone()
        scale = torch.as_tensor(resource_scale, dtype=torch.float32).detach().clone()
        if scales.shape != (len(TARGET_NAMES),) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("target scales must contain seven finite positive values")
        if mean.shape != (config.resource_dim,) or scale.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("invalid fitted resource normalization")
        self.register_buffer("target_scales", scales)
        self.register_buffer("resource_mean", mean)
        self.register_buffer("resource_scale", scale)
        self.layers = nn.Sequential(nn.Linear(config.resource_dim + config.hardware_dim, config.hidden), nn.GELU(),
                                    nn.Linear(config.hidden, config.hidden), nn.GELU())
        self.prediction_heads = nn.ModuleList(nn.Linear(config.hidden, len(indices)) for indices in HEAD_GROUPS)
        medians = scales.clone() if target_medians is None else torch.as_tensor(target_medians, dtype=torch.float32)
        if medians.shape != scales.shape or not torch.isfinite(medians).all() or (medians[list(POSITIVE_INDICES)] <= 0).any() or ((medians[list(SM_INDICES)] < 0) | (medians[list(SM_INDICES)] > 100)).any():
            raise ValueError("invalid training target medians")
        with torch.no_grad():
            for head, indices in zip(self.prediction_heads, HEAD_GROUPS, strict=True):
                head.weight.zero_()
                for local, index in enumerate(indices):
                    if index in SM_INDICES:
                        value = torch.logit((medians[index] / 100).clamp(1e-6, 1 - 1e-6))
                    else:
                        ratio = medians[index] / scales[index]
                        value = ratio + torch.log(-torch.expm1(-ratio))
                    head.bias[local].copy_(value)

    def forward(self, resources, hardware_cont, hardware_missing_mask):
        if resources.ndim != 2 or resources.size(1) != self.config.resource_dim:
            raise ValueError("resource inputs must have shape [batch, 17]")
        if hardware_cont.shape != (resources.size(0), self.config.hardware_dim // 2) or hardware_missing_mask.shape != hardware_cont.shape:
            raise ValueError("resource and hardware batch dimensions differ")
        if any(not torch.isfinite(value).all() for value in (resources, hardware_cont, hardware_missing_mask)):
            raise ValueError("nonfinite resource or hardware input")
        normalized = (resources.float() - self.resource_mean) / self.resource_scale
        graph = self.layers(torch.cat((normalized, hardware_cont.float(), hardware_missing_mask.float()), dim=-1))
        with torch.autocast(resources.device.type, enabled=False):
            raw = torch.cat([head(graph.float()) for head in self.prediction_heads], dim=-1)
            prediction = torch.stack([100 * torch.sigmoid(raw[:, index]) if index in SM_INDICES else
                                      F.softplus(raw[:, index]).clamp_min(1e-6) * self.target_scales[index]
                                      for index in range(len(TARGET_NAMES))], dim=-1)
        phases = graph.new_empty((graph.size(0), 1, 0, graph.size(1)))
        presence = torch.empty((graph.size(0), 1, 0), dtype=torch.bool, device=graph.device)
        return SeerOutputV4(prediction, graph[:, None], phases, presence)

    def predict_batch(self, batch):
        if getattr(batch, "resources", None) is None:
            raise ValueError("resource MLP requires raw resource descriptors")
        return self(batch.resources, batch.training.hardware_cont, batch.training.hardware_missing_mask)
