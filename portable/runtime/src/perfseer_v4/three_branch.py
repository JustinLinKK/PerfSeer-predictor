"""Independent VRAM, SM and timing branches over common workload inputs."""
import torch
from torch import nn
from .timing_model import epoch_time

VERSION = "perfseer_three_branch_a100_v1"


def combine_branches(timing, sm, vram, raw_steps, protocol):
    n = len(timing)
    if timing.shape != (n, 2) or sm.shape != (n, 1) or vram.shape != (n, 2):
        raise ValueError("expected timing(2), SM(1), VRAM(2)")
    if (not torch.isfinite(timing).all() or not torch.isfinite(sm).all()
            or not torch.isfinite(vram).all() or (timing <= 0).any()
            or (vram <= 0).any() or (sm < 0).any() or (sm > 100).any()):
        raise ValueError("branch outputs violate physical bounds")
    epoch = epoch_time(timing[:, 0], raw_steps, protocol)
    return torch.cat((timing, epoch[:, None], sm, vram), dim=-1)


class ThreeBranchPredictor(nn.Module):
    """Only the protected teacher is shared, and it is always frozen.

    Trainable branches receive the same input batch plus optional frozen context.
    They may own separate graph encoders. No trainable trunk feeds VRAM.
    """
    def __init__(self, vram_teacher, sm_branch, timing_branch):
        super().__init__()
        self.vram_teacher = vram_teacher.requires_grad_(False).eval()
        self.sm_branch = sm_branch
        self.timing_branch = timing_branch
        protected = {id(p) for p in self.vram_teacher.parameters()}
        if any(id(p) in protected for branch in (sm_branch, timing_branch) for p in branch.parameters()):
            raise ValueError("trainable branches must not alias protected teacher parameters")
        if {id(p) for p in sm_branch.parameters()} & {id(p) for p in timing_branch.parameters()}:
            raise ValueError("SM and timing branches must own separate parameters")

    def train(self, mode=True):
        super().train(mode)
        self.vram_teacher.eval()
        return self

    def forward(self, batch, raw_steps, protocol="measured_single_step"):
        with torch.no_grad():
            frozen = self.vram_teacher(batch)
        base = frozen.prediction
        protected_vram = base[:, 4:6].detach().clone()
        timing = self.timing_branch(batch, frozen)
        sm = self.sm_branch(batch, frozen)
        return combine_branches(timing, sm, protected_vram, raw_steps, protocol)


class PreservedSMBranch(nn.Module):
    def forward(self, batch, frozen):
        sm = frozen.prediction[:, 3:4].detach()
        return torch.where(sm < 1, 0, sm)
