"""Selected teacher loss functions and parameter groups; no historical trainer imports."""
import torch
from torch.nn import functional as F
from v5_objectives import BUCKET_MIB, metrics, pairwise_loss, memory_interval_loss


def _add_group(groups, name, params, lr):
    params = list(params)
    if params:
        for param in params:
            param.requires_grad_(True)
        groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.})


def pretrain_parameter_groups(model):
    """Full-data target continuation groups used for P_A10G and P_A10."""
    model.requires_grad_(False)
    p1, groups = model.p1, []
    _add_group(groups, "timing", p1.timing.parameters(), 2e-4)
    _add_group(groups, "busy", p1.sm.parameters(), 4e-4)
    _add_group(groups, "vram_head", p1.vram9.memory_head.parameters(), 1e-5)
    _add_group(groups, "vram_trunk", p1.vram9.seer.parameters(), 1e-6)
    return groups


def t7_parameter_groups(model, calibrate=False):
    """Only the selected T7 optimizer groups; no other transfer variants."""
    model.requires_grad_(False)
    p1, groups = model.p1, []
    if calibrate:
        _add_group(groups, "timing_calibration", [p for n, p in p1.timing.named_parameters()
                    if "log_calibration" in n or n.startswith("workload_head.")], 3e-4)
    else:
        _add_group(groups, "timing", p1.timing.parameters(), 1e-4)
    _add_group(groups, "busy", p1.sm.parameters(), 1e-4 if calibrate else 2e-4)
    memory = p1.vram9
    _add_group(groups, "offset", [memory.offset], 2e-3)
    _add_group(groups, "memory_head", memory.base.memory_head.parameters(), 2e-5)
    _add_group(groups, "adapter", [p for n, p in memory.base.seer.named_parameters() if "lora_" in n], 2e-4)
    return groups

def log_interval_loss(prediction, target):
    """Relative version of the v5 100 MB bucket interval: each job counts by its log error.

    Diagnosed 2026-10-22: the absolute-bucket memory_interval_loss (2x under-prediction penalty) is dominated by a
    few large jobs missed by many buckets; even at lr 1e-6/1e-7 it shifted all A10 VRAM predictions +3% in 60
    steps and cut 100 MB coverage 70.7% -> 62.5%. Symmetric log-space penalties remove that large-job dominance.
    """
    p = prediction.float().clamp_min(1.0).log()
    t = target.float() / BUCKET_MIB
    low, high = ((t - 1).clamp_min(0.5) * BUCKET_MIB).log(), (t * BUCKET_MIB).log()
    zero = torch.zeros_like(p)
    return (F.smooth_l1_loss((low - p).clamp_min(0), zero, beta=.05) + F.smooth_l1_loss((p - high).clamp_min(0), zero, beta=.05)
            + .05 * F.smooth_l1_loss(p, ((t - .5).clamp_min(.5) * BUCKET_MIB).log(), beta=.05))

def memory_loss(prediction, quantized_target, raw_target):
    """Symmetric, bounded-gradient surrogate for quantized-prediction 100 MB accuracy.

    If the raw target is inside bucket k, predictions in buckets k-1 and k are
    within 100 MB. A small center term avoids boundary ties. All inputs are MiB.
    Unlike the historical absolute loss, underprediction has no doubled penalty.
    """
    bucket = BUCKET_MIB
    raw = raw_target.float()
    k = torch.ceil(raw/bucket).clamp_min(1.)
    low = ((k-2)*bucket).clamp_min(1.) + 1e-4
    high = k*bucket - 1e-4
    pred = prediction.float()
    outside = (low-pred).clamp_min(0) + (pred-high).clamp_min(0)
    gate = F.smooth_l1_loss(outside/bucket, torch.zeros_like(outside), beta=.5)
    center = F.smooth_l1_loss((pred-(raw-.5*bucket).clamp_min(1.))/bucket,
                            torch.zeros_like(pred), beta=1.)
    return gate + .05*center + .25*log_interval_loss(pred, quantized_target.float())
