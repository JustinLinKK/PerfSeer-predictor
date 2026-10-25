"""Six-output native-description backbones for the four v4 proposals."""

import torch
from torch import nn
from torch.nn import functional as F

from .native_data import FEATURE_DIM, HEAD_GROUPS, TARGET_NAMES

VERSION = "perfseer_v4_native_six_checkpoint_v2"
PROPOSALS = ("v4.0", "v4.1", "v4.2", "v4.3")


class NativePredictor(nn.Module):
    def __init__(self, config, mean, scale, medians):
        super().__init__()
        self.config = dict(config)
        if config["proposal"] not in PROPOSALS or config["role"] not in {"teacher", "student"}:
            raise ValueError("invalid native proposal/role")
        self.register_buffer("feature_mean", mean.detach().float().clone())
        self.register_buffer("feature_scale", scale.detach().float().clone())
        target_scales = medians.detach().float().clone()
        target_scales[3] = max(float(target_scales[3]), 1.)
        if mean.shape != (FEATURE_DIM,) or scale.shape != mean.shape or medians.shape != (6,):
            raise ValueError("native feature/output dimensions differ")
        if not torch.isfinite(scale).all() or (scale <= 0).any() or not torch.isfinite(medians).all():
            raise ValueError("invalid native normalization")
        self.register_buffer("target_scales", target_scales)
        width, layers = config["hidden"], config["layers"]
        self.revised = config.get("revision", 1) == 2
        self.dense = config.get("backbone", "mlp" if config["proposal"] == "v4.3" else "transformer") == "mlp"
        if self.dense:
            modules = [nn.Linear(FEATURE_DIM, width), nn.GELU()]
            for _ in range(layers - 1):
                modules.extend((nn.Linear(width, width), nn.GELU()))
            self.backbone = nn.Sequential(*modules)
        else:
            self.projection = nn.Linear(64, width)
            self.position = nn.Parameter(torch.empty(1, FEATURE_DIM // 64, width))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(width, config["heads"], 4 * width, dropout=.1,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.backbone = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
            # TransformerEncoder clones its layer; initialize each clone independently.
            for block in self.backbone.layers:
                for parameter in block.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        self.heads = nn.ModuleList(nn.Linear(width, 2 if self.revised and i == 1 else len(indices))
                                   for i, indices in enumerate(HEAD_GROUPS))
        with torch.no_grad():
            for head, indices in zip(self.heads, HEAD_GROUPS):
                head.weight.zero_()
                if self.revised and indices == (3,):
                    head.bias[0] = 2.
                    head.bias[1] = torch.logit((medians[3] / 100).clamp(.01, .99))
                    continue
                for local, index in enumerate(indices):
                    ratio = medians[index] / target_scales[index]
                    head.bias[local] = ratio if index == 3 else ratio + torch.log(-torch.expm1(-ratio))

    def forward(self, features, *, return_aux=False):
        value = (features.float() - self.feature_mean) / self.feature_scale
        if self.dense:
            hidden = self.backbone(value)
        else:
            value = self.projection(value.reshape(-1, FEATURE_DIM // 64, 64)) + self.position
            hidden = self.backbone(value).mean(1)
        with torch.autocast(features.device.type, enabled=False):
            if self.revised:
                timing, sm, memory = [head(hidden.float()) for head in self.heads]
                positive_sm = 100 * torch.sigmoid(sm[:, 1])
                prediction = torch.cat((F.softplus(timing).clamp_min(1e-6) * self.target_scales[:3],
                                        torch.where(sm[:, 0] >= 0, positive_sm, 0.)[:, None],
                                        F.softplus(memory).clamp_min(1e-6) * self.target_scales[4:]), dim=-1)
                return (prediction, sm[:, 0], positive_sm) if return_aux else prediction
            raw = torch.cat([head(hidden.float()) for head in self.heads], dim=-1)
            return torch.stack([torch.clamp(raw[:, i] * self.target_scales[i], 0, 100) if i == 3 else
                                F.softplus(raw[:, i]).clamp_min(1e-6) * self.target_scales[i] for i in range(6)], dim=-1)


def model_config(proposal, role, *, smoke=False, revision=2):
    resource = proposal == "v4.3"
    if revision == 2:
        transformer = proposal == "v4.0"
        return {"proposal": proposal, "role": role, "revision": revision, "objective": "balanced_v2",
                "backbone": "transformer" if transformer else "mlp",
                "hidden": 32 if smoke else (128 if transformer or resource else 256) if role == "teacher" else 64,
                "layers": 1 if smoke else (3 if proposal in {"v4.0", "v4.1"} else 4 if proposal == "v4.2" else 2),
                "heads": 4}
    return {"proposal": proposal, "role": role,
            "hidden": (16 if smoke else 128 if role == "teacher" else 64) if resource else
                      (32 if smoke else 1280 if role == "teacher" else 224),
            "layers": 1 if smoke else 2 if resource or role == "student" else 10,
            "heads": 4 if smoke or role == "student" else 8}


def restore(payload, device="cpu"):
    if payload.get("version") not in {VERSION, "perfseer_v4_native_six_checkpoint_v1"} or payload.get("target_names") != list(TARGET_NAMES):
        raise ValueError("expected a native six-target checkpoint; legacy graph artifacts are incompatible")
    state = payload["model_state_dict"]
    model = NativePredictor(payload["model_config"], state["feature_mean"], state["feature_scale"], payload["target_medians"])
    model.load_state_dict(state, strict=True)
    return model.to(device)
