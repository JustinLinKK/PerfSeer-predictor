#!/usr/bin/env python3
"""Create the immutable source/dependency identity embedded in the image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perfseer_v3.dataset_pack.fingerprints import file_sha256
from perfseer_v3.dataset_pack.source_identity import source_tree_sha256
from perfseer_v3.dataset_pack.task_registry import MLEBENCH_METADATA_REVISION


BASE_IMAGE = (
    "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@"
    "sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "containers/v100-labeler/build-manifest.json",
    )
    parser.add_argument("--image-identity", default="perfseer-v3-v100-labeler:local-candidate")
    arguments = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    lock = ROOT / "containers/v100-labeler/requirements.lock"
    payload = {
        "version": "perfseer_v3_v100_image_build_manifest_v1",
        "base_image": BASE_IMAGE,
        "source_revision": revision,
        "source_tree_sha256": source_tree_sha256(ROOT),
        "dependency_lock_sha256": file_sha256(lock),
        "mle_bench_revision": MLEBENCH_METADATA_REVISION,
        "image_identity": arguments.image_identity,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
