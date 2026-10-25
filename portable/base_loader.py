"""Pinned inference loader extracted from the selected node-164 package.

Only the inference loader and deterministic context are retained here.
"""
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def trace_postflight(stage):
    if os.environ.get("PERFSEER_POSTFLIGHT_TRACE") == "1":
        os.write(2, ("postflight:" + stage + "\n").encode("ascii"))

@contextlib.contextmanager
def deterministic_evaluation():
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)

def load_base(path):
    root = Path(path).parent
    for directory in (root / "runtime" / "aliases", root / "runtime" / "src", root):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("perfseer_candidate_starter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_init = module.AdditiveTeacher.__init__

    def init_with_numeric_adapter(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        width = self.vram.config.hardware_continuous_dim + self.vram.config.hardware_missing_mask_dim
        self.vram.numeric_hardware_adapter = nn.Linear(width, 2)
        nn.init.zeros_(self.vram.numeric_hardware_adapter.weight)
        nn.init.zeros_(self.vram.numeric_hardware_adapter.bias)

    module.AdditiveTeacher.__init__ = init_with_numeric_adapter

    def restore_model(payload, device="cpu"):
        if (payload.get("version") != module.VERSION or
                payload.get("architecture_sha256") != module.ARCHITECTURE_SHA256 or
                payload.get("target_names") != list(module.TARGET_NAMES)):
            raise ValueError("incompatible teacher checkpoint")
        init = payload["initialization"]
        trace_postflight("restore-construction-start")
        model = module.AdditiveTeacher(payload["vram_config"], init["target_medians"],
                                       init["mean_node_count"], init["sm_positive_fraction"],
                                       init["sm_positive_median"])
        trace_postflight("restore-construction-complete")
        if payload.get("physics_config"):
            module.attach_physics(model, payload["physics_config"])
        trace_postflight("restore-physics-complete")
        has_adapter = "vram.numeric_hardware_adapter.weight" in payload["state"]
        trace_postflight("restore-state-start")
        result = model.load_state_dict(payload["state"], strict=has_adapter)
        trace_postflight("restore-state-complete")
        if not has_adapter and (set(result.missing_keys) != {
                "vram.numeric_hardware_adapter.weight", "vram.numeric_hardware_adapter.bias"}
                or result.unexpected_keys):
            raise ValueError("seed migration added more than zero-output VRAM adapter")
        module.enable_soft_busy_export(model)
        trace_postflight("restore-busy-export-complete")
        model = model.to(device)
        trace_postflight("restore-device-complete")
        return model

    module.restore_model = restore_model
    old_adaptation = module.AdditiveTeacher.adaptation_parameter_names

    def adaptation_names(self):
        groups = old_adaptation(self)
        for mode in ("adapter", "full"):
            for name in ("numeric_hardware_adapter.weight", "numeric_hardware_adapter.bias"):
                if name not in groups["vram"][mode]:
                    groups["vram"][mode].append(name)
        return groups

    module.AdditiveTeacher.adaptation_parameter_names = adaptation_names
    old_forward = module.AdditiveTeacher.forward

    def forward(self, batch, raw_steps, protocol="measured_single_step", return_aux=False):
        result = old_forward(self, batch, raw_steps, protocol, return_aux)
        if return_aux:
            prediction, *aux = result
        else:
            prediction = result
        clean = module.sanitize_batch(batch)
        mask = clean.hardware_missing_mask.float()
        values = clean.hardware_cont.float() * (1 - mask)
        adjustment = self.vram.numeric_hardware_adapter(torch.cat((values, mask), -1)).float()
        memory = prediction[:, 4:6] * torch.exp(.1 * torch.tanh(adjustment))
        prediction = torch.cat((prediction[:, :4], memory), -1)
        return (prediction, *aux) if return_aux else prediction

    module.AdditiveTeacher.original_seed_forward = old_forward
    module.AdditiveTeacher.forward = forward

    def sm_loss(sm_logit, sm_positive, target, hardware_penalty, class_weights):
        y = target.float()
        expected = torch.sigmoid(sm_logit.float()) * sm_positive.float()
        label = (y > 0).float()
        weights = torch.where(label.bool(), class_weights[1], class_weights[0])
        huber = F.smooth_l1_loss((expected - y) / 10., torch.zeros_like(y), beta=.5)
        hurdle = (weights * F.binary_cross_entropy_with_logits(
            sm_logit.float(), label, reduction="none")).mean()
        return huber + .1 * hurdle + .001 * hardware_penalty.float()

    module.sm_loss = sm_loss

    def busy_pair_loss(scores, targets):
        score, y = scores.float().reshape(-1), targets.float().reshape(-1)
        i, j = torch.triu_indices(len(score), len(score), 1, device=score.device)
        if not len(i):
            return score.sum() * 0.
        delta = y[i] - y[j]
        material = delta.abs() >= 5.
        if not material.any():
            return score.sum() * 0.
        errors = F.softplus(-delta.sign() * (score[i] - score[j]))
        close = material & (delta.abs() < 15.)
        return .5 * errors[material].mean() + (.5 * errors[close].mean() if close.any() else 0.)

    module.busy_pair_loss = busy_pair_loss

    @torch.inference_mode()
    def predict(model, data, indices, device, batch_size=8):
        model.eval()
        values = []
        indices = torch.as_tensor(indices, dtype=torch.long)
        for selected in indices.split(batch_size):
            batch = module.to_batch(data["features"], selected, device)
            steps = data["raw_steps_per_epoch"][selected].to(device)
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                values.append(model(batch, steps, data["epoch_protocol"]).float().cpu())
        return torch.cat(values).double()

    module.predict = predict

    def run_inference(model_dir, input_path, output_path, device="cpu"):
        model_dir = Path(model_dir)
        base = model_dir / "model"
        if not (base / "best_model.pth").is_file():
            base = model_dir
        manifest_path = base / "candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        trace_postflight("manifest-loaded")
        for name, expected_hash in manifest["artifact_sha256"].items():
            candidates = (base / name, model_dir / name, model_dir / "model" / name)
            if not any(p.is_file() and module.sha256(p) == expected_hash for p in candidates):
                raise ValueError("deployment artifact changed: " + name)
        trace_postflight("hashes-verified")
        module.bootstrap(base)
        trace_postflight("runtime-bootstrapped")
        normalizer = torch.load(base / "normalization.pt", map_location="cpu", weights_only=False)
        if normalizer.sha256 != manifest["normalization_sha256"]:
            raise ValueError("normalization provenance mismatch")
        data = module.load_normalized_inputs(input_path, normalizer)
        if any(key in data for key in ("targets", "raw_targets_v4", "families", "splits")):
            raise ValueError("inference requires input-only data")
        trace_postflight("inputs-normalized")
        payload = torch.load(base / "best_model.pth", map_location="cpu", weights_only=False, mmap=False)
        trace_postflight("checkpoint-loaded-unmapped")
        device = torch.device(device)
        model = module.restore_model(payload, device)
        trace_postflight("model-restored")
        with deterministic_evaluation():
            predictions = predict(model, data, torch.arange(len(data["features"])), device).numpy()
        trace_postflight("forward-complete")
        if not np.isfinite(predictions).all():
            raise ValueError("invalid prediction")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, sample_ids=np.asarray(data["sample_ids"]),
            predictions=predictions, dataset_version=np.asarray(module.VERSION),
            vram_100mb_buckets=np.maximum(1, np.ceil(predictions[:, 4:6] * 1048576 / 100000000)).astype(np.int64),
            vram_decimal_gb=np.round(np.maximum(1, np.ceil(predictions[:, 4:6] * 1048576 / 100000000)) / 10, 1),
            step_cost_units=predictions[:, 0], cost_unit=np.asarray("A10_reference_millisecond_v1"),
            nvml_busy_proxy_percent=predictions[:, 3],
            uncertainty_status=np.asarray("NOT_YET_CALIBRATED"))
        trace_postflight("predictions-saved")

    module.run_inference = run_inference
    old_verify = module.verify_vram_architecture

    def verify_vram_architecture(model, reference):
        adapter = model.vram.numeric_hardware_adapter
        del model.vram.numeric_hardware_adapter
        try:
            old_verify(model, reference)
        finally:
            model.vram.numeric_hardware_adapter = adapter

    module.verify_vram_architecture = verify_vram_architecture
    return module
