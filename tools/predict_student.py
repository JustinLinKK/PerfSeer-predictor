#!/usr/bin/env python3
"""CPU inference for a released TorchScript student and an input-only bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="released student.ts TorchScript file")
    parser.add_argument("--input", required=True, help="input-only .pt bundle; never a labeled dataset")
    parser.add_argument("--output", required=True, help="output .npz; refuses to overwrite another result")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("Batch size and CPU threads must be positive")
    root = Path(__file__).resolve().parent.parent
    if not (root / "student_training/runtime/student.py").is_file() or not (root / "portable/teachers.py").is_file():
        parser.error("Place this script in tools/ beside student_training/ and portable/")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from portable.teachers import load_base
    from student_training.runtime.student import pack_batch

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    base = load_base()  # Registers the packaged GraphFeatures class before input deserialization.
    paths = {name: Path(value).resolve() for name, value in (("model", args.model), ("input", args.input))}
    paths["architecture"] = root / "portable/architecture.pt"
    output = Path(args.output).resolve()
    if output in paths.values():
        raise ValueError("Output must differ from input and model")
    hashes = {name: sha(path) for name, path in paths.items()}
    architecture = torch.load(paths["architecture"], weights_only=False, map_location="cpu")
    data = base.load_normalized_inputs(paths["input"], architecture["source_normalization"])
    if not isinstance(data, dict) or not {"features", "sample_ids", "raw_steps_per_epoch", "epoch_protocol"} <= data.keys():
        raise ValueError("Expected features, sample_ids, raw_steps_per_epoch and epoch_protocol")
    if {"targets", "raw_targets_v4", "labels", "splits", "families", "configuration_groups", "release_partition"} & data.keys():
        raise ValueError("Use an input-only bundle, not a labeled training dataset")
    if data["epoch_protocol"] != "measured_single_step":
        raise ValueError("Expected measured_single_step epoch protocol")
    count = len(data["features"])
    ids = np.asarray(list(map(str, data["sample_ids"])), dtype=str)
    steps = torch.as_tensor(data["raw_steps_per_epoch"]).float()
    if not count or len(ids) != count or steps.shape != (count,) or len(set(ids.tolist())) != count:
        raise ValueError("Input lengths must match, be nonempty, and have unique sample IDs")
    if not torch.isfinite(steps).all() or not (steps > 0).all():
        raise ValueError("Steps per epoch must be positive and finite")
    extra = {"student_config.json": ""}
    model = torch.jit.load(str(paths["model"]), map_location="cpu", _extra_files=extra).eval()
    metadata = json.loads(extra["student_config.json"]) if extra["student_config.json"] else {}
    hardware = metadata.get("hardware")
    if hardware and data.get("hardware") and hardware != data["hardware"]:
        raise ValueError("Student and input hardware identities differ")
    values = []
    with torch.no_grad(), base.deterministic_evaluation():
        for rows in torch.arange(count).split(args.batch_size):
            prediction = model(*pack_batch(base.to_batch(data["features"], rows, "cpu"), steps[rows]))["prediction"]
            if prediction.shape != (len(rows), 6) or not torch.isfinite(prediction).all():
                raise FloatingPointError("Student returned invalid six-target predictions")
            if not (prediction[:, [0, 1, 2, 4, 5]] > 0).all() or not ((prediction[:, 3] >= 0) & (prediction[:, 3] <= 100)).all():
                raise FloatingPointError("Student returned physically invalid predictions")
            torch.testing.assert_close(prediction[:, 2], prediction[:, 0] * steps[rows], rtol=1e-5, atol=1e-5)
            values.append(prediction.cpu().numpy())
    if torch.cuda.is_initialized():
        raise AssertionError("CPU inference initialized CUDA")
    if hashes != {name: sha(path) for name, path in paths.items()}:
        raise ValueError("Input or model changed during inference")
    result = {
        "predictions": np.concatenate(values), "sample_ids": ids,
        "target_names": np.asarray(base.TARGET_NAMES),
        "units": np.asarray(["ms", "ms", "ms", "percent", "MiB", "MiB"]),
        "hardware": np.asarray(hardware or data.get("hardware", "unspecified")),
        "input_sha256": np.asarray(hashes["input"]), "model_sha256": np.asarray(hashes["model"]),
        "architecture_sha256": np.asarray(hashes["architecture"]),
        "normalization_sha256": np.asarray(architecture["source_normalization"].sha256),
    }
    if output.exists():
        with np.load(output, allow_pickle=False) as previous:
            if set(previous.files) != set(result) or any(not np.array_equal(previous[k], v) for k, v in result.items()):
                raise ValueError("Another prediction result already occupies output")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".predictions-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, **result)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(output)
    print(json.dumps({"rows": count, "targets": 6, "device": "cpu", "hardware": str(result["hardware"]),
                      "input_sha256": hashes["input"], "model_sha256": hashes["model"]}, sort_keys=True))


if __name__ == "__main__":
    main()
