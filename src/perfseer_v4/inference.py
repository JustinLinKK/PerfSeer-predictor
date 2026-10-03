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
    for key in ("prediction_hardware", "label_policy", "conversion", "transfer", "adapter_config"):
        if key in checkpoint:
            payload[key] = checkpoint[key]
    payload["evaluation"] = (checkpoint.get("validation") or {}).get("evaluation", checkpoint.get("evaluation", {}))
    atomic_write(path, payload, checkpoint=True)


def predict(artifact, designs, device="cpu", *, amp=None, microbatch=None, calibration=None, environment=None):
    if calibration is not None:
        return predict_calibrated(artifact, designs, calibration, environment, device=device,
                                  amp=amp, microbatch=microbatch)["predictions"]
    if artifact.get("transfer"):
        from perfseer_v32.calibration_contracts import environment_identity
        if environment is None or environment_identity(environment) != artifact["transfer"]["domain_fingerprint"]:
            raise ValueError("adapted prediction requires its bound target environment")
    return _predict(artifact, designs, device, amp=amp, microbatch=microbatch)


@torch.no_grad()
def _predict(artifact, designs, device="cpu", *, amp=None, microbatch=None, target_hardware=None):
    model = restore_model(artifact, device=device).eval()
    normalization = normalization_from_dict(artifact["normalization"])
    if normalization.split_fingerprint != artifact["dataset_fingerprint"]:
        raise ValueError("normalization dataset fingerprint differs")
    if artifact.get("normalization_sha256") != fingerprint(artifact["normalization"]):
        raise ValueError("exported normalization hash differs")
    hardware = target_hardware or artifact.get("prediction_hardware")
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
        batch = to_batch([{"features": normalize_features(build_features(design, include_resources=artifact["model_variant"] == "v4.3"), normalization)}
                          for design in designs[start:start + microbatch]], device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                            enabled=amp and torch.device(device).type == "cuda"):
            predictions.extend(model.predict_batch(batch).prediction.float().cpu().tolist())
    return [dict(zip(TARGET_NAMES, row, strict=True)) for row in predictions]


def predict_calibrated(artifact, designs, calibration, environment, *, device="cpu", amp=None,
                       microbatch=None, unknown_domain="reject"):
    from perfseer_v32.calibration_contracts import environment_identity
    from .residuals import apply_adapter, validate_adapter
    from .resource_model import resource_values
    from .transfer import source_identity
    if str(device) != "cpu" or amp not in (None, False) or microbatch not in (None, 1):
        raise ValueError("residual calibration is bound to CPU FP32 microbatch-one source predictions")
    if environment is None:
        raise ValueError("calibrated prediction requires a target environment")
    environment_identity(environment)
    validate_adapter(calibration)
    source = source_identity(artifact)
    if source != calibration["source_identity"]:
        raise ValueError("calibration source checkpoint or preprocessing differs")
    predictions = _predict(artifact, designs, device="cpu", amp=False, microbatch=1,
                           target_hardware=environment["hardware_id"])
    rows = [{"source_prediction": prediction, "features": resource_values(design)}
            for design, prediction in zip(designs, predictions, strict=True)]
    return apply_adapter(calibration, rows, source, environment, unknown_domain=unknown_domain)
