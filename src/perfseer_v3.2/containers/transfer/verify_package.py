#!/usr/bin/env python3
"""Verify the portable file inventory; optionally check the exact teacher runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "PACKAGE-MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or unsafe package file: {relative}")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"package checksum differs: {relative}")
    dataset = json.loads((root / "dataset/dataset_manifest.json").read_text())
    if dataset["fingerprint"] != manifest["dataset_fingerprint"] or dataset["prediction_hardware"] != manifest["prediction_hardware"]:
        raise ValueError("package dataset identity differs")
    for variable, value in (("PERFSEER_EXPECTED_DATASET", manifest["dataset_fingerprint"]),
                            ("PERFSEER_EXPECTED_TEACHER", manifest["base_teacher_sha256"])):
        if os.environ.get(variable, value) != value:
            raise ValueError(f"container image differs from submitted package: {variable}")
    if args.runtime:
        import sys
        sys.path.insert(0, str(root / "scripts"))
        import run_perfseer_v32_transfer_training
        import torch
        from perfseer_v31.io import read_json
        from perfseer_v32.runner import Samples
        from perfseer_v32.training import restore_model, to_batch
        from perfseer_v32.transfer_training import transfer_normalization
        torch.set_num_threads(1)
        base = torch.load(root / "weights/base-teacher.pt", map_location="cpu", weights_only=False)
        if base["role"] != "teacher" or base["epoch"] != manifest["base_teacher_epoch"]:
            raise ValueError("base teacher identity differs")
        model = restore_model(base).eval()
        normalization = transfer_normalization(base, dataset["fingerprint"])
        rows = read_json(root / "dataset" / dataset["split_files"]["train"]["path"])
        row = min(rows, key=lambda item: (root / "dataset" / item["input_path"]).stat().st_size)
        sample = Samples(root / "dataset", [row], normalization)[0]
        with torch.no_grad():
            prediction = model.predict_batch(to_batch([sample], "cpu")).prediction
        if prediction.shape != (1, 12) or not torch.isfinite(prediction).all():
            raise ValueError("teacher forward failed on a real normalized RTX 5090 graph pair")
    print(json.dumps({"status": "ok", "files": len(manifest["files"]), "dataset_fingerprint": dataset["fingerprint"],
                      "splits": {key: value["rows"] for key, value in dataset["split_files"].items()},
                      "prediction_hardware": manifest["prediction_hardware"], "runtime_checked": args.runtime}, indent=2))


if __name__ == "__main__":
    main()
