"""Six native training targets on the unchanged v3.2 SeerBlock graph trunk."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from perfseer_v3.model import (HierarchicalNodeEncoder, HierarchicalEdgeEncoder,
                               WorkloadGlobalEncoderV3, SeerBlockV3, _mlp, graph_batch_tensors)
from perfseer_v31.model import SeerNetV31Config
from perfseer_v31.training import T1, S1
from perfseer_v32.model import SeerNetV32

from .native_data import HEAD_GROUPS, TARGET_NAMES

VERSION = "perfseer_v4_native_seerblock_six_v2"
LEGACY_VERSION = "perfseer_v4_native_seerblock_six_v1"


class SeerBlockOutput(NamedTuple):
    prediction: torch.Tensor
    graph_embedding: torch.Tensor
    phase_embedding: torch.Tensor
    phase_presence: torch.Tensor


@dataclass(frozen=True)
class SeerBlockConfig(SeerNetV31Config):
    num_outputs: int = 6
    pooling_mode: str = "phase_aware"
    sm_hurdle: bool = False

    def __post_init__(self):
        if self.hidden <= 0 or self.num_blocks <= 0 or self.num_outputs != len(TARGET_NAMES):
            raise ValueError("positive SeerBlock dimensions and exactly six outputs required")
        if self.node_identity_fusion not in {"additive", "concatenation"}:
            raise ValueError("unsupported node identity fusion")
        if self.pooling_mode not in {"existing", "phase_aware"}:
            raise ValueError("unsupported pooling mode")


def model_config(registry, layout, role="teacher", **overrides):
    if role not in {"teacher", "student"}:
        raise ValueError("expected teacher or student")
    return SeerBlockConfig.from_registry(registry, layout, **{**(T1 if role == "teacher" else S1), **overrides})


class SeerBlockPredictor(nn.Module):
    _encode = SeerNetV32._encode

    def __init__(self, config, target_medians, target_scales=None):
        super().__init__()
        self.config = config
        medians = torch.as_tensor(target_medians, dtype=torch.float32).detach().clone()
        if (medians.shape != (6,) or not torch.isfinite(medians).all()
                or (medians[[0, 1, 2, 4, 5]] <= 0).any() or not 0 <= medians[3] <= 100):
            raise ValueError("invalid six-target training medians")
        if config.sm_hurdle:
            self.register_buffer("target_medians", medians)
        scales = medians.clone() if target_scales is None else torch.as_tensor(target_scales, dtype=torch.float32).detach().clone()
        if target_scales is None:
            scales[3] = max(float(scales[3]), 1.) if config.sm_hurdle else 100.
        if scales.shape != (6,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("six finite positive target scales required")
        self.register_buffer("target_scales", scales)
        self.use_phase_aware_pooling = config.pooling_mode == "phase_aware"
        self.num_phases, self.hidden = config.num_phases, config.hidden
        self.node_encoder = HierarchicalNodeEncoder(config)
        self.edge_encoder = HierarchicalEdgeEncoder(config)
        self.global_encoder = WorkloadGlobalEncoderV3(config)
        self.blocks = nn.ModuleList(SeerBlockV3(config) for _ in range(config.num_blocks))
        self.phase_encoder = nn.ModuleList([_mlp(2 * config.hidden, config.hidden, config.hidden, config.dropout)])
        self.phase_fusion = nn.ModuleList([_mlp((config.num_phases + 1) * config.hidden, config.hidden, config.hidden, config.dropout)])
        self.phase_fusion_norm = nn.ModuleList([nn.LayerNorm(config.hidden)])
        self.prediction_heads = nn.ModuleList(_mlp(config.hidden, size, config.hidden, config.dropout)
                                             for size in (3, 2 if config.sm_hurdle else 1, 2))
        with torch.no_grad():
            for head, indices in zip(self.prediction_heads, HEAD_GROUPS, strict=True):
                head[-1].weight.zero_()
                if indices == (3,):
                    if config.sm_hurdle:
                        head[-1].bias[0] = 2.
                        head[-1].bias[1] = torch.logit((medians[3] / 100).clamp(.01, .99))
                    else:
                        head[-1].bias[0] = torch.logit((medians[3] / 100).clamp(1e-6, 1 - 1e-6))
                else:
                    ratio = medians[list(indices)] / scales[list(indices)]
                    head[-1].bias.copy_(ratio + torch.log(-torch.expm1(-ratio)))

    def forward(self, training, *, return_aux=False):
        graph, phases, presence = self._encode(0, *graph_batch_tensors(training))
        with torch.autocast(training.x_cont.device.type, enabled=False):
            timing, sm, memory = [head(graph.float()) for head in self.prediction_heads]
            positive_sm = 100 * torch.sigmoid(sm[:, 1 if self.config.sm_hurdle else 0])
            sm_logit = sm[:, 0] if self.config.sm_hurdle else None
            sm_value = torch.where(sm_logit >= 0, positive_sm, 0.) if sm_logit is not None else positive_sm
            prediction = torch.cat((F.softplus(timing).clamp_min(1e-6) * self.target_scales[:3],
                                    sm_value[:, None],
                                    F.softplus(memory).clamp_min(1e-6) * self.target_scales[4:]), dim=-1)
        output = SeerBlockOutput(prediction, graph[:, None], phases[:, None], presence[:, None])
        return (output, sm_logit, positive_sm) if return_aux else output

    def predict_batch(self, batch):
        return self(batch.training if hasattr(batch, "training") else batch)


def restore(payload, device="cpu"):
    if payload.get("version") not in {VERSION, LEGACY_VERSION} or payload.get("target_names") != list(TARGET_NAMES):
        raise ValueError("expected a six-target SeerBlock checkpoint")
    state = payload["model_state_dict"]
    config = {**payload["model_config"]}
    if payload["version"] == LEGACY_VERSION:
        config["sm_hurdle"] = True
    initial = state.get("target_medians", state["target_scales"]).detach().clone()
    if "target_medians" not in state:
        initial[3] = 50.
    model = SeerBlockPredictor(SeerBlockConfig(**config), initial, state["target_scales"])
    model.load_state_dict(state, strict=True)
    return model.to(device)


def convert_v32_checkpoint(payload):
    """Verification conversion retaining the original first six training predictions.

    New source experiments initialize fresh weights; this conversion establishes
    structural and output parity with the supplied reference artifact only.
    """
    if (payload.get("version") != "perfseer_v32_checkpoint_v2"
            or payload.get("input_schema") != "perfseer_v32_paired_graphs_v1"
            or tuple(payload.get("target_names", ())[:6]) != TARGET_NAMES
            or payload["model_config"].get("num_outputs") != 12):
        raise ValueError("expected the original twelve-output v3.2 checkpoint")
    removed = ("phase_encoder.1.", "phase_fusion.1.", "phase_fusion_norm.1.",
               "prediction_heads.3.", "prediction_heads.4.", "prediction_heads.5.")
    state = {key: value for key, value in payload["model_state_dict"].items() if not key.startswith(removed)}
    state["target_scales"] = state["target_scales"][:6].clone()
    for key in ("prediction_heads.2.4.weight", "prediction_heads.2.4.bias"):
        state[key] = state[key][:2].clone()
    return {"version": VERSION, "target_names": list(TARGET_NAMES),
            "model_config": {**payload["model_config"], "num_outputs": 6, "sm_hurdle": False},
            "model_state_dict": state, "conversion": {"verification_only": True, "source_epoch": payload.get("epoch"),
                                                        "source_dataset_fingerprint": payload.get("dataset_fingerprint")}}
