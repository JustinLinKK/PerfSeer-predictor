#!/usr/bin/env python3
"""Recompute or verify the completed PerfSeer V3 A10G label pack."""

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

from perfseer_v3.dataset_pack.finalization import finalize_workspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    finalize_workspace(arguments.workspace, verify_only=arguments.verify_only)


if __name__ == "__main__":
    main()
