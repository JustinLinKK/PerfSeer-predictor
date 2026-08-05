#!/usr/bin/env python3
"""Build deterministic transfer-learning JSON schema assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.artifact import artifact_metadata_json_schema
from perfseer_v3.dataset_pack.transfer_labeling import target_training_manifest_json_schema
from perfseer_v3.hardware import hardware_profile_json_schema
from perfseer_v3.hardware_transfer import transfer_manifest_json_schema


def main() -> int:
    output = SRC / "perfseer_v3" / "schemas"
    output.mkdir(parents=True, exist_ok=True)
    schemas = {
        "perfseer_hardware_profile_v1.json": hardware_profile_json_schema(),
        "perfseer_transfer_subset_manifest_v1.json": transfer_manifest_json_schema(),
        "perfseer_target_training_manifest_v1.json": target_training_manifest_json_schema(),
        "perfseer_transfer_artifact_metadata_v1.json": artifact_metadata_json_schema(),
    }
    for name, payload in schemas.items():
        path = output / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
