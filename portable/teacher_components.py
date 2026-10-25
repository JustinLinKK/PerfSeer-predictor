"""Exact P1/SM0 teacher components; no historical trainer imports."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
BLOCKS = (("x_cont", "node"), ("edge_cont", "edge"), ("u_cont", "global_features"))

class Renormalize(nn.Module):
    """A10-normalised features -> node 9's A100 normalisation (exact; clip only where A100 bounds bind)."""

    def __init__(self, source, target):
        super().__init__()
        for attr, block in BLOCKS:
            s, t = getattr(source, block), getattr(target, block)
            for name, value in (("s_mean", s.mean), ("s_std", s.std), ("t_mean", t.mean), ("t_std", t.std), ("low", t.clip_low), ("high", t.clip_high)):
                self.register_buffer(f"{attr}_{name}", torch.tensor(value, dtype=torch.float32))

    def forward(self, batch):
        values = {}
        for attr, _ in BLOCKS:
            g = lambda n: getattr(self, f"{attr}_{n}")  # noqa: E731
            raw = getattr(batch, attr).float() * g("s_std") + g("s_mean")
            values[attr] = (torch.maximum(torch.minimum(raw, g("high")), g("low")) - g("t_mean")) / g("t_std")
        return replace(batch, **values)

class VramSeerBlock(nn.Module):
    """Node 9's SeerBlock reduced to its VRAM output.

    The timing (prediction_heads[0]) and SM (prediction_heads[1]) heads are deleted; the encoders, SeerBlocks,
    phase-aware pooling and the memory head (prediction_heads[2]) are kept, so avg/peak VRAM is unchanged.
    """

    def __init__(self, seer):
        super().__init__()
        self.memory_head = seer.prediction_heads[2]
        self.register_buffer("memory_scale", seer.target_scales[4:6].detach().clone())
        del seer.prediction_heads, seer.target_scales  # _encode never reads either
        self.seer = seer

    def forward(self, batch):
        from perfseer_v3.model import graph_batch_tensors
        graph, _, _ = self.seer._encode(0, *graph_batch_tensors(batch))
        with torch.autocast(batch.x_cont.device.type, enabled=False):
            return F.softplus(self.memory_head(graph.float())).clamp_min(1e-6) * self.memory_scale

class P1Model(nn.Module):
    """timing + SM branches of the current teacher design (no VRAM SeerBlock) + VRAM-only node-9 SeerBlock."""

    def __init__(self, base, timing, sm, vram9, renorm):
        super().__init__()
        self.base, self.timing, self.sm, self.vram9, self.renorm = base, timing, sm, vram9, renorm

    def forward(self, batch, raw_steps, protocol="measured_single_step", return_aux=False):
        from perfseer_v4.timing_model import epoch_time
        clean = self.base.sanitize_batch(batch)
        timing, correction, timing_penalty = self.timing(clean)
        _, sm_logit, sm_positive, sm_penalty = self.sm(clean)
        busy = torch.sigmoid(sm_logit.float()) * sm_positive.float()
        memory = self.vram9(self.renorm(batch)).float()
        epoch = epoch_time(timing[:, 0], raw_steps, protocol)
        prediction = torch.cat((timing.float(), epoch[:, None].float(), busy[:, None], memory), -1)
        return (prediction, correction, timing_penalty, sm_logit, sm_positive, sm_penalty) if return_aux else prediction

class Adapted(nn.Module):
 """Per-target copy. Map identity to source token; target is represented by its own learned weights."""
 def __init__(self,p1,offset=True):
  super().__init__();self.p1=p1
  self.offset=nn.Parameter(torch.zeros(2)) if offset else None
 def forward(self,batch,steps,protocol='measured_single_step',return_aux=False):
  batch=replace(batch,hardware_id=torch.full_like(batch.hardware_id,190))
  out=self.p1(batch,steps,protocol,return_aux)
  pred=out[0] if return_aux else out
  if self.offset is not None:
   pred=torch.cat((pred[:,:4],(pred[:,4:]+1000*self.offset).clamp_min(1.)),dim=-1)
  return (pred,*out[1:]) if return_aux else pred
