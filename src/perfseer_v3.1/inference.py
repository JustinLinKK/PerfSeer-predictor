"""Export and consume three-output, A10-bound predictor artifacts."""

import torch

from .features import build_features
from .io import atomic_write
from .runner import normalization_from_dict
from .training import restore_model, to_batch
from .version import HARDWARE_ID, INPUT_SCHEMA_VERSION, TARGET_NAMES
from perfseer_v3.features import apply_normalization


def export_model(checkpoint, path):
    restore_model(checkpoint)
    keys = ("version", "output_contract", "target_names", "role", "epoch", "model_config",
            "model_state_dict", "target_scales", "normalization", "dataset_fingerprint")
    payload = {key: checkpoint[key] for key in keys}
    payload.update({"input_schema": INPUT_SCHEMA_VERSION, "prediction_hardware": HARDWARE_ID})
    atomic_write(path, payload, checkpoint=True)


@torch.no_grad()
def predict(artifact, designs, device="cpu"):
    if artifact.get("input_schema") != INPUT_SCHEMA_VERSION or artifact.get("prediction_hardware") != HARDWARE_ID:
        raise ValueError("not a v3.1 A10 prediction artifact")
    model = restore_model(artifact, device=device).eval()
    normalization = normalization_from_dict(artifact["normalization"])
    batch = to_batch([{"features": apply_normalization(build_features(design), normalization)} for design in designs], device)
    predictions = model.predict_batch(batch).prediction.cpu().tolist()
    return [dict(zip(TARGET_NAMES, row, strict=True)) for row in predictions]
