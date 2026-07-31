#!/usr/bin/env python3
"""Resume the exact one-task-at-a-time PerfSeer A10G label workflow."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
existing_pythonpath = tuple(
    value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
)
if str(SOURCE_ROOT) not in existing_pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join((str(SOURCE_ROOT), *existing_pythonpath))

from perfseer_v3.dataset_pack.mlebench_bridge import verify_frozen_preparers
from perfseer_v3.dataset_pack.kaggle import (
    KaggleCliClient,
    KaggleMaterializationError,
    validate_external_credentials,
)
from perfseer_v3.dataset_pack.task_registry import load_task_registry
REQUIRED_RUNTIME_MODULES = {
    "appdirs": "appdirs",
    "kaggle": "kaggle",
    "networkx": "networkx",
    "numpy": "numpy",
    "nvidia-ml-py": "pynvml",
    "ogb": "ogb",
    "pandas": "pandas",
    "Pillow": "PIL",
    "py7zr": "py7zr",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "SciPy": "scipy",
    "soundfile": "soundfile",
    "TensorFlow": "tensorflow",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torch-geometric": "torch_geometric",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "triton": "triton",
}

FROZEN_RUNTIME_VERSIONS = {
    "torch": "2.11.0",
    "torch-geometric": "2.7.0",
    "torchaudio": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.7.0",
    "triton": "3.6.0",
}

_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "CREDENTIAL",
    "KAGGLE_CONFIG",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def _probe_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if name not in {"KAGGLE_USERNAME", "KAGGLE_KEY"}
        and not any(
            fragment in name.upper()
            for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS
        )
    }


def run_task_workflow(**kwargs) -> None:
    """Load the CUDA-heavy workflow only after runtime preflight succeeds."""

    from perfseer_v3.dataset_pack.workflow import run_task_workflow as implementation

    implementation(**kwargs)


def verify_runtime_dependencies() -> None:
    if not (sys.version_info >= (3, 11) and sys.version_info < (3, 14)):
        raise RuntimeError("the pinned A10G dataset workflow requires Python 3.11–3.13")
    missing = tuple(
        distribution
        for distribution, module in REQUIRED_RUNTIME_MODULES.items()
        if importlib.util.find_spec(module) is None
    )
    if missing:
        raise RuntimeError(
            "A10G dataset runtime dependencies are missing: "
            f"{', '.join(missing)}; install .[a10g-dataset-pack] in the "
            "PyTorch CUDA environment"
        )
    mismatched = tuple(
        f"{distribution}=={actual} (expected {expected})"
        for distribution, expected in FROZEN_RUNTIME_VERSIONS.items()
        if (actual := importlib.metadata.version(distribution)) != expected
    )
    if mismatched:
        raise RuntimeError(
            "A10G dataset runtime differs from the frozen environment: "
            + ", ".join(mismatched)
        )
    with tempfile.TemporaryDirectory(prefix=".perfseer-runtime-probe-") as sandbox:
        os.chmod(sandbox, 0o700)
        environment = _probe_environment()
        environment["KAGGLE_CONFIG_DIR"] = sandbox
        environment["KAGGLE_USERNAME"] = "runtime-probe"
        environment["KAGGLE_KEY"] = "runtime-probe"
        for distribution, module in REQUIRED_RUNTIME_MODULES.items():
            try:
                probe = (
                    "import importlib, json, sys; "
                    "module = importlib.import_module(sys.argv[1]); "
                    "print(json.dumps({'cuda': module.version.cuda, "
                    "'available': module.cuda.is_available()}))"
                    if module == "torch"
                    else "import importlib, sys; importlib.import_module(sys.argv[1])"
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        probe,
                        module,
                    ],
                    check=False,
                    capture_output=True,
                    timeout=120,
                    env=environment,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(
                    f"A10G dataset runtime dependency {distribution} cannot be imported"
                ) from error
            if result.returncode != 0:
                raise RuntimeError(
                    f"A10G dataset runtime dependency {distribution} cannot be imported"
                )
            if module == "torch":
                try:
                    torch_probe = json.loads(result.stdout)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise RuntimeError("A10G dataset torch probe returned invalid output") from error
                if set(torch_probe) != {"cuda", "available"}:
                    raise RuntimeError("A10G dataset torch probe returned invalid output")
                if not isinstance(torch_probe["cuda"], str) or not torch_probe["cuda"]:
                    raise RuntimeError("A10G dataset runtime requires a CUDA PyTorch build")
                if torch_probe["available"] is not True:
                    raise RuntimeError("A10G dataset runtime requires a working CUDA PyTorch build")


def verify_all_kaggle_access(kaggle_executable: str) -> None:
    """Fail before collection unless the account can inventory every frozen task."""

    validate_external_credentials(REPOSITORY_ROOT)
    client = KaggleCliClient(executable=kaggle_executable)
    client.authenticate()
    for entry in load_task_registry().entries:
        try:
            client.probe_competition(entry.kaggle_slug)
        except KaggleMaterializationError as error:
            raise RuntimeError(
                f"Kaggle access preflight failed for {entry.kaggle_slug!r}"
            ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mlebench-checkout", type=Path, required=True)
    parser.add_argument("--kaggle-executable", default="kaggle")
    parser.add_argument("--materialize-only", action="store_true")
    arguments = parser.parse_args()
    validate_external_credentials(REPOSITORY_ROOT)
    verify_runtime_dependencies()
    verify_frozen_preparers(
        arguments.mlebench_checkout,
        load_task_registry().entries,
    )
    verify_all_kaggle_access(arguments.kaggle_executable)
    run_task_workflow(
        workspace=arguments.workspace,
        repository_root=REPOSITORY_ROOT,
        mlebench_checkout=arguments.mlebench_checkout,
        kaggle_executable=arguments.kaggle_executable,
        materialize_only=arguments.materialize_only,
    )


if __name__ == "__main__":
    main()
