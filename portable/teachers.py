"""Portable factories for the selected P1/SM0 and T7 teacher designs only."""
from __future__ import annotations
import copy
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def bootstrap():
    """Register feature and normalization dataclasses before any torch.load."""
    for path in reversed((ROOT, ROOT / "runtime/src", ROOT / "runtime/aliases")):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


bootstrap()
from teacher_components import Adapted, P1Model, Renormalize, VramSeerBlock


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=1)
def load_base():
    from base_loader import load_base as restore_base, deterministic_evaluation
    base = restore_base(ROOT / "model/starter.py")
    base.bootstrap(ROOT)
    base.deterministic_evaluation = deterministic_evaluation
    return base


class IdentityTeacher(nn.Module):
    """Uniform teacher wrapper for distillation taps; identity=None preserves P1."""
    def __init__(self, p1, identity, offset=False):
        super().__init__()
        self.p1, self.identity = p1, identity
        self.offset = nn.Parameter(torch.zeros(2)) if offset else None

    def forward(self, batch, steps, protocol="measured_single_step", return_aux=False):
        from dataclasses import replace
        if self.identity is not None:
            batch = replace(batch, hardware_id=torch.full_like(batch.hardware_id, self.identity))
        output = self.p1(batch, steps, protocol, return_aux)
        if self.offset is None:
            return output
        pred = output[0] if return_aux else output
        pred = torch.cat((pred[:, :4], (pred[:, 4:] + 1000 * self.offset).clamp_min(1.)), -1)
        return (pred, *output[1:]) if return_aux else pred


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_a), self.lora_b)


def inject_lora(module):
    names = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child))
            names.append(name)
        else:
            names.extend(name + "." + n for n in inject_lora(child))
    return names


class T7MemoryAdapter(nn.Module):
    """Selected rank-eight memory-trunk LoRA with the original state-key layout."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.candidate = "T7"
        self.mode = "lora"
        self.offset = nn.Parameter(torch.zeros(2))
        self.lora_sites = inject_lora(self.base.seer.blocks)
        if not self.lora_sites:
            raise ValueError("T7 requires memory-trunk linear layers")

    def forward(self, batch):
        from perfseer_v3.model import graph_batch_tensors
        train_encoder = any(p.requires_grad for p in self.base.seer.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and train_encoder):
            graph, _, _ = self.base.seer._encode(0, *graph_batch_tensors(batch))
        graph = graph.float()
        with torch.autocast(batch.x_cont.device.type, enabled=False):
            memory = F.softplus(self.base.memory_head(graph)).clamp_min(1e-6) * self.base.memory_scale
            return (memory + 1000 * self.offset).clamp_min(1.)


def build(kind, architecture_path=None, checkpoint_path=None, device="cpu"):
    """Return (base, model, architecture), strictly restoring a supplied checkpoint.

    P1 consumes raw unprefixed P1 state. Adapted and T7 consume p1.* state and
    map input hardware identity to token190. No source checkpoint is loaded as
    an implicit initialization or a dependency of checkpoint restoration.
    """
    if kind not in ("P1", "Adapted", "T7"):
        raise ValueError("kind must be P1, Adapted, or T7")
    base = load_base()
    architecture_path = Path(architecture_path or ROOT / "architecture.pt")
    metadata = torch.load(architecture_path, weights_only=False, map_location="cpu")
    from perfseer_v4.seerblock_model import SeerBlockConfig, SeerBlockPredictor
    from aba3_training_support import PhysicalTiming, HorusSM
    conf = SeerBlockConfig(**metadata["vram_config"])
    init = metadata["initialization"]
    timing = PhysicalTiming(base.AdditiveTimingBranch(conf, torch.tensor(init["target_medians"])[:2],
                            init["mean_node_count"]), copy.deepcopy(metadata["physics_config"]))
    busy = HorusSM(base.SMBranch(conf, init["sm_positive_fraction"], init["sm_positive_median"]), metadata["physics_config"])
    memory = VramSeerBlock(SeerBlockPredictor(SeerBlockConfig(**metadata["node9_config"]), init["target_medians"]))
    p1 = P1Model(base, timing, busy, memory, Renormalize(metadata["source_normalization"], metadata["target_normalization"]))
    if kind == "T7":
        p1.vram9 = T7MemoryAdapter(p1.vram9)
    model = p1 if kind == "P1" else Adapted(p1, offset=False)
    if checkpoint_path is not None:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        if "state" not in saved:
            raise ValueError("expected a checkpoint containing state")
        config = saved.get("config", {})
        if kind == "T7" and config.get("candidate") != "T7":
            raise ValueError("T7 loader requires config.candidate=T7")
        if kind != "T7" and config.get("candidate") == "T7":
            raise ValueError("T7 checkpoint requires the T7 architecture")
        if config.get("architecture_sha256") and config["architecture_sha256"] != sha(architecture_path):
            raise ValueError("checkpoint architecture hash differs")
        override = saved.get("hardware_identity_override", config.get("hardware_identity_override"))
        if override is not None and override != (None if kind == "P1" else 190):
            raise ValueError("checkpoint hardware-identity override differs from selected kind")
        model.load_state_dict(saved["state"], strict=True)
    return base, model.to(device).eval(), metadata


def load_teacher(spec, architecture_path=None, device="cpu"):
    """Load a named teacher specification with a uniform .p1 wrapper for taps."""
    base, model, meta = build(spec["kind"], architecture_path, spec["checkpoint"], device)
    if spec["kind"] == "P1":
        identity = spec.get("identity_override")
        if identity not in (None, 190):
            raise ValueError("P1 supports its original identity or explicit source token190")
        model = IdentityTeacher(model, identity=identity)
    elif spec.get("identity_override", 190) != 190:
        raise ValueError("Adapted and T7 require identity_override=190")
    return base, model.eval().requires_grad_(False), meta


def initialize_target(source_checkpoint, architecture_path=None, *, transfer=False, device="cpu"):
    """Explicit A100 P1 warm start for target continuation or new T7 training.

    This initializer is separate from strict restoration: it does not pretend
    that raw P1 weights already contain trained target-specific/LoRA parameters.
    """
    base, p1, metadata = build("P1", architecture_path, source_checkpoint, device="cpu")
    if transfer:
        p1.vram9 = T7MemoryAdapter(p1.vram9)
    return base, Adapted(p1, offset=False).to(device).eval(), metadata


@torch.inference_mode()
def predict(base, model, data, rows=None, device="cpu", batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = list(range(len(data["features"]))) if rows is None else list(rows)
    values = []
    model.eval()
    with base.deterministic_evaluation():
        for start in range(0, len(rows), batch_size):
            selected = torch.tensor(rows[start:start + batch_size], dtype=torch.long)
            batch = base.to_batch(data["features"], selected, device)
            steps = torch.as_tensor(data["raw_steps_per_epoch"])[selected].to(device)
            with torch.autocast(torch.device(device).type, enabled=False):
                values.append(model(batch, steps, data["epoch_protocol"]).double().cpu())
    result = torch.cat(values).numpy() if values else __import__("numpy").empty((0, 6), dtype="float64")
    if not __import__("numpy").isfinite(result).all():
        raise ValueError("nonfinite prediction")
    if len(result) and ((result[:, :3] <= 0).any() or (result[:, 4:] <= 0).any() or
                        (result[:, 3] < 0).any() or (result[:, 3] > 100).any()):
        raise ValueError("prediction outside physical output ranges")
    return result
