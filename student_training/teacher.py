"""Portable selected teacher restoration and TRAIN-cache embedding taps."""
import torch
from support import legacy
from portable.teachers import load_teacher as portable_load_teacher

def load_teacher(config, device="cpu"):
    spec = dict(config["teacher"])
    saved = torch.load(spec["checkpoint"], weights_only=False, mmap=True, map_location="cpu")
    declared = saved.get("config", {}).get("hardware")
    if declared and declared != spec["hardware"]:
        raise ValueError("Checkpoint hardware differs from requested teacher")
    identity = saved.get("hardware_identity_override", saved.get("config", {}).get(
        "hardware_identity_override", 190 if spec["kind"] != "P1" else spec["identity_override"]))
    if identity != spec["identity_override"]:
        raise ValueError("Checkpoint hardware identity differs from requested override")
    if spec["kind"] == "transfer_T7":
        spec["kind"] = "T7"
    del saved
    return portable_load_teacher(spec, config["architecture"], device)

class Tap:

    def __init__(self, model):
        self.model, self.values, self.hooks = (model, {}, [])
        core = model.p1
        for name, module in (('timing', core.timing.legacy.core), ('busy', core.sm.legacy.core)):
            self.hooks.append(module.register_forward_hook(self._hook(name)))
        memory = core.vram9.base if hasattr(core.vram9, 'base') else core.vram9
        self.hooks.append(memory.memory_head.register_forward_pre_hook(self._memory))

    def _hook(self, name):

        def capture(module, inputs, output):
            self.values[name] = output[2]
        return capture

    def _memory(self, module, inputs):
        self.values['memory'] = inputs[0]

    @torch.no_grad()
    def __call__(self, batch, steps, protocol):
        self.values.clear()
        pred, _, _, logit, positive, _ = self.model(batch, steps, protocol, True)
        return {'prediction': pred.float(), 'busy_logit': logit.float().reshape(-1), 'busy_positive': positive.float().reshape(-1), 'embedding': torch.cat([self.values[k] for k in ('timing', 'busy', 'memory')], -1).float()}

    def close(self):
        for hook in self.hooks:
            hook.remove()
