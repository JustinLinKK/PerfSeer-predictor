#!/usr/bin/env python
"""Bootstrap CPU-only frozen-source time and VRAM calibration from this checkout."""

import importlib.util
import os
from pathlib import Path
import sys

if importlib.util.find_spec("torch") is None:
    candidate = Path(os.environ.get("PERFSEER_PYTHON", Path.home() / "miniconda3/envs/perfseer/bin/python")).expanduser()
    if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
        os.execv(candidate, [str(candidate), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
    raise SystemExit("PyTorch is required; set PERFSEER_PYTHON to the prepared interpreter")

import run_perfseer_v32_transfer_labeling
from perfseer_v32.calibration_runner import main


if __name__ == "__main__":
    main()
