#!/usr/bin/env python3
"""Verify portable inference in python -I outside the checkout with no PYTHONPATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("P1", "Adapted", "T7"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="perfseer-portable-cpu-") as temporary:
        temporary = Path(temporary)
        clone = temporary / "portable"
        shutil.copytree(package, clone, ignore=shutil.ignore_patterns("verification", "__pycache__", "*.pyc"))
        output = temporary / "predictions.npz"
        command = [sys.executable, "-I", str(clone / "infer.py"), "--kind", args.kind,
                   "--checkpoint", str(args.checkpoint.resolve()), "--architecture", str(clone / "architecture.pt"),
                   "--input", str(args.input.resolve()), "--output", str(output), "--batch-size", "2", "--threads", "1"]
        executed = subprocess.run(command, cwd=temporary, env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""},
                                  text=True, capture_output=True, check=True)
        actual, expected = np.load(output, allow_pickle=False), np.load(args.reference, allow_pickle=False)
        if not np.array_equal(actual["sample_ids"], expected["sample_ids"]):
            raise ValueError("identity order differs")
        difference = np.max(np.abs(actual["predictions"] - expected["predictions"]))
        if not np.array_equal(actual["predictions"], expected["predictions"]):
            raise ValueError(f"CPU outputs are not bit-identical: maximum difference {difference}")
        result = {"passed": True, "kind": args.kind, "rows": len(actual["sample_ids"]),
                  "prediction_max_absolute_difference": float(difference), "bit_identical": True,
                  "python_isolated_mode": True, "working_directory_outside_repository": True,
                  "PYTHONPATH_empty": True, "CUDA_VISIBLE_DEVICES_empty": True,
                  "checkpoint_copied": False, "stdout": json.loads(executed.stdout),
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "predictions": actual["predictions"].tolist()}
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}))


if __name__ == "__main__":
    main()
