"""Export and consume paired-input, twelve-output A10 predictor artifacts."""

import torch

from .features import build_features, normalize_features
from perfseer_v31.io import atomic_write, fingerprint
from .runner import normalization_from_dict
from .training import restore_model, to_batch
from .version import HARDWARE_ID, INPUT_SCHEMA_VERSION, TARGET_NAMES


def export_model(checkpoint, path):
    restore_model(checkpoint)
    normalization_from_dict(checkpoint["normalization"])
    keys = ("version", "output_contract", "target_names", "role", "epoch", "model_config",
            "model_state_dict", "target_scales", "normalization", "dataset_fingerprint",
            "input_schema", "feature_version", "metric_version", "loss_version", "selection_version")
    payload = {key: checkpoint[key] for key in keys}
    payload.update(prediction_hardware=HARDWARE_ID, normalization_sha256=fingerprint(payload["normalization"]),
                   label_policy=checkpoint.get("label_policy"),
                   evaluation=checkpoint.get("validation", {}).get("evaluation", {}))
    atomic_write(path, payload, checkpoint=True)


@torch.no_grad()
def predict(artifact, designs, device="cpu", *, amp=None, microbatch=None):
    if artifact.get("input_schema") != INPUT_SCHEMA_VERSION or artifact.get("prediction_hardware") != HARDWARE_ID:
        raise ValueError("not a v3.2 paired-input A10 prediction artifact")
    if artifact.get("normalization_sha256") != fingerprint(artifact["normalization"]):
        raise ValueError("exported normalization hash differs")
    model = restore_model(artifact, device=device).eval()
    normalization = normalization_from_dict(artifact["normalization"])
    if normalization.split_fingerprint != artifact["dataset_fingerprint"]:
        raise ValueError("normalization dataset fingerprint differs")
    configuration = artifact.get("evaluation", {})
    amp = configuration.get("amp", False) if amp is None else amp
    microbatch = configuration.get("requested_microbatch", 1) if microbatch is None else microbatch
    if microbatch < 1:
        raise ValueError("prediction microbatch must be positive")
    predictions = []
    for start in range(0, len(designs), microbatch):
        batch = to_batch([{"features": normalize_features(build_features(design), normalization)}
                          for design in designs[start:start + microbatch]], device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp and torch.device(device).type == "cuda"):
            predictions.extend(model.predict_batch(batch).prediction.float().cpu().tolist())
    return [dict(zip(TARGET_NAMES, row, strict=True)) for row in predictions]
