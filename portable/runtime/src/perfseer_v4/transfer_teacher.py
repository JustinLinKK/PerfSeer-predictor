"""A trainable three-branch teacher with explicit later adaptation controls.

The memory branch retains the proven SeerBlock architecture. Timing and SM
own their representations; neither can backpropagate into the memory branch.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .seerblock_model import SeerBlockConfig, SeerBlockPredictor
from .three_branch import combine_branches

VERSION = "perfseer_a10_trainable_three_branch_v1"


class ContextBranch(nn.Module):
    def __init__(self, context_dim, outputs, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(context_dim, hidden), nn.LayerNorm(hidden),
                                     nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.hardware = nn.Sequential(nn.Linear(124, 64), nn.SiLU(), nn.Linear(64, hidden))
        self.head = nn.Linear(hidden, outputs)

    def forward(self, context, batch):
        # Unknown specifications stay missing. The host GPU is never a feature.
        mask = batch.hardware_missing_mask.float()
        hardware = torch.cat((batch.hardware_cont.float() * (1-mask), mask), dim=-1)
        return self.head(self.encoder(context) + self.hardware(hardware))


class TransferTeacher(nn.Module):
    def __init__(self, vram_config, medians, context_mean, context_std, hidden=256):
        super().__init__()
        self.vram = SeerBlockPredictor(SeerBlockConfig(**vram_config), medians)
        # These two legacy heads are retained only for architecture compatibility.
        self.vram.prediction_heads[0].requires_grad_(False)
        self.vram.prediction_heads[1].requires_grad_(False)
        self.timing = ContextBranch(len(context_mean), 2, hidden)
        self.sm = ContextBranch(len(context_mean), 2, hidden)
        self.register_buffer("context_mean", torch.as_tensor(context_mean).float().clone())
        self.register_buffer("context_std", torch.as_tensor(context_std).float().clone().clamp_min(.01))
        self.register_buffer("timing_scale", torch.as_tensor(medians[:2]).float().clone())
        self.hidden = hidden

    def forward(self, batch, context, raw_steps, protocol="measured_single_step", *, return_aux=False):
        context = ((context.float()-self.context_mean)/self.context_std).clamp(-20,20)
        vram = self.vram(batch).prediction[:, 4:6]
        timing = F.softplus(self.timing(context, batch).float()).clamp_min(1e-6) * self.timing_scale
        sm_logits = self.sm(context, batch).float()
        positive_sm = 100 * torch.sigmoid(sm_logits[:, 1])
        sm = torch.where(sm_logits[:, 0] >= 0, positive_sm, 0.)[:, None]
        prediction = combine_branches(timing, sm, vram, raw_steps, protocol)
        return (prediction, sm_logits[:, 0], positive_sm) if return_aux else prediction

    def adaptation_parameters(self, branch, mode="head"):
        """Explicit parameter groups; never freezes or trains anything implicitly."""
        if branch not in {"vram", "timing", "sm"} or mode not in {"head", "full", "hardware"}:
            raise ValueError("unknown branch/adaptation mode")
        module = getattr(self, branch)
        if mode == "full":
            return [p for name,p in module.named_parameters()
                    if branch != "vram" or not name.startswith(("prediction_heads.0.", "prediction_heads.1."))]
        if mode == "head":
            return list((module.prediction_heads[2] if branch == "vram" else module.head).parameters())
        if branch == "vram":
            # Preserve the VRAM architecture: head fine-tuning is its minimal path.
            raise ValueError("VRAM supports head or full adaptation; no new adapter is inserted")
        return list(module.hardware.parameters())


def objective(prediction, sm_logit, sm_positive, target):
    timing = (prediction[:, :2].log()-target[:, :2].log()).square().mean()
    vram = (prediction[:, 4:6].log()-target[:, 4:6].log()).square().mean()
    positive = (target[:, 3] > 0).float()
    sm = F.binary_cross_entropy_with_logits(sm_logit, positive)
    sm = sm + (((sm_positive-target[:, 3])/100).square()*positive).sum()/positive.sum().clamp_min(1)
    return .70*timing + .25*vram + .05*sm


def payload(model):
    return {"version": VERSION, "vram_config": model.vram.config.to_dict(),
            "medians": model.vram.target_scales.detach().cpu().clone().index_fill(0, torch.tensor([3]), 50.),
            "context_mean": model.context_mean.detach().cpu(), "context_std": model.context_std.detach().cpu(),
            "hidden": model.hidden, "state": {k:v.detach().cpu() for k,v in model.state_dict().items()}}


def restore(value, device="cpu"):
    if value.get("version") != VERSION:
        raise ValueError("unexpected three-branch checkpoint")
    model = TransferTeacher(value["vram_config"], value["medians"], value["context_mean"], value["context_std"], value["hidden"])
    model.load_state_dict(value["state"], strict=True)
    return model.to(device)
