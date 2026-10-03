#!/usr/bin/env python
"""Prepare/verify on CPU; label explicitly after the RTX 5090 is available.

python scripts/run_perfseer_v32_transfer_labeling.py prepare
python scripts/run_perfseer_v32_transfer_labeling.py prepare-data
python scripts/run_perfseer_v32_transfer_labeling.py verify
python scripts/run_perfseer_v32_transfer_labeling.py label --resume
python scripts/run_perfseer_v32_transfer_labeling.py label --campaign-profile 24h --resume

prepare-data acquires and verifies all nine original A10 datasets on CPU.
The default source root is record/perfseer-v32/transfer-source-data.
No GPU pilot or labeling is performed by prepare, prepare-data or verify.
"""

import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path.home() / "miniconda3/envs/perfseer/bin/python"


def ensure_runtime():
    if importlib.util.find_spec("torch") is not None:
        return
    candidate = Path(os.environ.get("PERFSEER_PYTHON", DEFAULT_PYTHON)).expanduser()
    if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
        os.execv(candidate, [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(
        "This Python environment lacks PyTorch. Set PERFSEER_PYTHON to the prepared "
        "PerfSeer interpreter or run /home/justin/miniconda3/envs/perfseer/bin/python."
    )


def load_local_package(name, directory):
    package = ROOT / directory
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load local package {name} from {package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


ensure_runtime()
for package_name, package_directory in (
    ("perfseer_v3", "src/perfseer_v3"),
    ("perfseer_v31", "src/perfseer_v3.1"),
    ("perfseer_v32", "src/perfseer_v3.2"),
):
    load_local_package(package_name, package_directory)

from perfseer_v32.transfer_labeling import main


if __name__ == "__main__":
    raise SystemExit(main())
