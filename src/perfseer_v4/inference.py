"""Export and predict seven training metrics from a training graph only."""

import torch

from perfseer_v31.io import atomic_write, fingerprint

from .capture import graph_from_design
from .features import build_features, normalize_features
from .runner import normalization_from_dict
from .training import restore_model, to_batch
from .version import TARGET_NAMES


def export_model(checkpoint, path):
    restore_model(checkpoint)
    normalization_from_dict(checkpoint["normalization"])
    keys = ("version", "output_contract", "target_names", "model_variant", "role", "epoch", "model_config",
            "model_state_dict", "target_scales", "normalization", "normalization_sha256", "dataset_fingerprint",
            "input_schema", "feature_version", "metric_version", "loss_version", "selection_version")
    payload = {key: checkpoint[key] for key in keys}
    for key in ("prediction_hardware", "label_policy", "conversion", "transfer"):
        if key in checkpoint:
            payload[key] = checkpoint[key]
    payload["evaluation"] = (checkpoint.get("validation") or {}).get("evaluation", checkpoint.get("evaluation", {}))
    atomic_write(path, payload, checkpoint=True)


@torch.no_grad()
def predict(artifact, designs, device="cpu", *, amp=None, microbatch=None):
    model = restore_model(artifact, device=device).eval()
    normalization = normalization_from_dict(artifact["normalization"])
    if normalization.split_fingerprint != artifact["dataset_fingerprint"]:
        raise ValueError("normalization dataset fingerprint differs")
    if artifact.get("normalization_sha256") != fingerprint(artifact["normalization"]):
        raise ValueError("exported normalization hash differs")
    hardware = artifact.get("prediction_hardware")
    if not hardware:
        raise ValueError("prediction artifact must declare its hardware")
    if any(graph_from_design(design).metadata.get("target_hardware_id") != hardware for design in designs):
        raise ValueError("design hardware differs from prediction hardware")
    configuration = artifact.get("evaluation", {})
    amp = configuration.get("amp", False) if amp is None else amp
    microbatch = configuration.get("requested_microbatch", 1) if microbatch is None else microbatch
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("prediction microbatch must be positive")
    predictions = []
    for start in range(0, len(designs), microbatch):
        batch = to_batch([{"features": normalize_features(build_features(design), normalization)}
                          for design in designs[start:start + microbatch]], device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                            enabled=amp and torch.device(device).type == "cuda"):
            predictions.extend(model.predict_batch(batch).prediction.float().cpu().tolist())
    return [dict(zip(TARGET_NAMES, row, strict=True)) for row in predictions]
