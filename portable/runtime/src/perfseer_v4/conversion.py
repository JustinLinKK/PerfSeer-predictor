"""Explicit v3.2-to-v4 initialization; never resume a twelve-output run."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint
from perfseer_v31 import features_core, capture_training
from perfseer_v3 import features as graph_features, model as graph_model
from perfseer_v32 import features as source_features, model as source_model, capture as source_capture
from perfseer_v32.runner import normalization_from_dict as v32_normalization_from_dict
from perfseer_v32.training import restore_model as restore_v32_model

from .model import SeerNetV4, SeerNetV4Config
from .version import (CHECKPOINT_VERSION, OUTPUT_CONTRACT_VERSION, INPUT_SCHEMA_VERSION,
                      FEATURE_VERSION, NORMALIZATION_VERSION, LOSS_VERSION, METRIC_VERSION,
                      SELECTION_VERSION, TARGET_NAMES, HARDWARE_ID)


def convert_v32_checkpoint(payload, *, source_sha256):
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("conversion requires the source checkpoint SHA-256")
    source = restore_v32_model(payload)
    normalization = v32_normalization_from_dict(payload["normalization"])
    if normalization.split_fingerprint != payload["dataset_fingerprint"]:
        raise ValueError("source normalization dataset fingerprint differs")
    if not torch.equal(source.target_scales, torch.as_tensor(payload["target_scales"])):
        raise ValueError("source target scale metadata differs from weights")
    if any(not torch.isfinite(value).all() for value in source.state_dict().values()):
        raise ValueError("source checkpoint contains nonfinite weights")
    config = SeerNetV4Config(**{**source.config.to_dict(), "num_outputs": len(TARGET_NAMES)})
    scales = source.target_scales[:len(TARGET_NAMES)].clone()
    model = SeerNetV4(config, scales)
    source_state = source.state_dict()
    state = {name: source_state[name].detach().cpu().clone() for name in model.state_dict() if name != "target_scales"}
    state["target_scales"] = scales
    model.load_state_dict(state, strict=True)
    normalization = {"training": asdict(replace(normalization.training, normalization_version=NORMALIZATION_VERSION)),
                     "version": NORMALIZATION_VERSION}
    preprocessing = {"input_schema": payload["input_schema"], "feature_version": payload["feature_version"],
                     "model_config_sha256": fingerprint(payload["model_config"]),
                     "normalization_sha256": payload["normalization_sha256"],
                     "checkpoint_code_fingerprint": payload.get("code_fingerprint"),
                     "conversion_runtime_files": {module.__name__: file_sha256(module.__file__) for module in
                         (features_core, capture_training, graph_features, graph_model, source_features, source_model, source_capture)},
                     "conversion_torch_version": str(torch.__version__), "historical_runtime_verified": False}
    return {"version": CHECKPOINT_VERSION, "output_contract": OUTPUT_CONTRACT_VERSION,
            "input_schema": INPUT_SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "metric_version": METRIC_VERSION, "loss_version": LOSS_VERSION, "selection_version": SELECTION_VERSION,
            "target_names": TARGET_NAMES, "model_variant": "v4.0", "role": payload["role"], "epoch": 0,
            "model_config": config.to_dict(), "model_state_dict": state, "target_scales": scales,
            "normalization": normalization, "normalization_sha256": fingerprint(normalization),
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "prediction_hardware": payload.get("prediction_hardware", HARDWARE_ID),
            "label_policy": payload.get("label_policy"),
            "conversion": {"version": "perfseer_v4_from_v32_v1", "source_checkpoint_sha256": source_sha256,
                           "source_checkpoint_version": payload["version"], "source_epoch": payload["epoch"],
                           "source_target_names": list(payload["target_names"]),
                           "source_normalization_sha256": payload["normalization_sha256"],
                           "source_dataset_fingerprint": payload["dataset_fingerprint"],
                           "source_preprocessing": preprocessing, "source_preprocessing_sha256": fingerprint(preprocessing),
                           "retained_target_indices": list(range(len(TARGET_NAMES))), "initialization_only": True}}


def convert_file(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("conversion requires a new output path")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    converted = convert_v32_checkpoint(payload, source_sha256=file_sha256(source))
    atomic_write(output, converted, checkpoint=True)
    restored = torch.load(output, map_location="cpu", weights_only=False)
    if restored["conversion"] != converted["conversion"] or restored["normalization_sha256"] != fingerprint(restored["normalization"]):
        raise ValueError("converted artifact verification failed")
    model = SeerNetV4(SeerNetV4Config(**restored["model_config"]), restored["target_scales"])
    model.load_state_dict(restored["model_state_dict"], strict=True)
    if any(not torch.equal(value, converted["model_state_dict"][name]) for name, value in model.state_dict().items()):
        raise ValueError("converted weights differ after serialization")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = convert_file(args.checkpoint, args.output)
    print(f"Converted {result['conversion']['source_checkpoint_sha256']} to {args.output}; fresh training state required.")


if __name__ == "__main__":
    main()
