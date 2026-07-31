#!/usr/bin/env python3
"""Strictly merge nlp/vision/rest A10G shard workspaces and finalize once."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
existing_pythonpath = tuple(
    value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
)
if str(SOURCE_ROOT) not in existing_pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), *existing_pythonpath))

from perfseer_v3.dataset_pack.sharding import merge_shard_workspaces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp-workspace", type=Path, required=True)
    parser.add_argument("--vision-workspace", type=Path, required=True)
    parser.add_argument("--rest-workspace", type=Path, required=True)
    parser.add_argument("--output-workspace", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    merge_shard_workspaces(
        {
            "nlp": arguments.nlp_workspace,
            "vision": arguments.vision_workspace,
            "rest": arguments.rest_workspace,
        },
        arguments.output_workspace,
        REPOSITORY_ROOT,
        verify_only=arguments.verify_only,
    )


if __name__ == "__main__":
    main()
