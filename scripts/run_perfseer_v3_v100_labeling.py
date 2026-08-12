#!/usr/bin/env python3
"""Single entry point for the immutable V100 labeling image."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from perfseer_v3.dataset_pack.campaign import (
    analysis_summary,
    freeze_campaign_contract,
    run_campaign,
    verify_campaign,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256, file_sha256
from perfseer_v3.dataset_pack.kaggle import KaggleCliClient, validate_external_credentials
from perfseer_v3.dataset_pack.local_smoke import (
    LOCAL_SMOKE_MODELS,
    run_family_matrix_smoke,
    run_local_smoke,
    verify_local_smoke,
)
from perfseer_v3.dataset_pack.mlebench_bridge import validate_mlebench_checkout
from perfseer_v3.dataset_pack.source_identity import source_tree_sha256
from perfseer_v3.dataset_pack.task_registry import MLEBENCH_METADATA_REVISION, load_task_registry


PINNED_IMPORTS = {
    "appdirs": "appdirs",
    "networkx": "networkx",
    "numpy": "numpy",
    "nvidia-ml-py": "pynvml",
    "ogb": "ogb",
    "pandas": "pandas",
    "Pillow": "PIL",
    "py7zr": "py7zr",
    "PyYAML": "yaml",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "torch-geometric": "torch_geometric",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
}
PINNED_VERSIONS = {
    "torch": "2.10.0+cu128",
    "torchaudio": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "torch-geometric": "2.7.0",
    "kaggle": "2.2.2",
}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _image_preflight(arguments: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(arguments.build_manifest)
    required = {
        "version",
        "base_image",
        "source_revision",
        "source_tree_sha256",
        "dependency_lock_sha256",
        "mle_bench_revision",
        "image_identity",
    }
    if set(manifest) != required:
        raise RuntimeError("image build manifest schema differs")
    if manifest["mle_bench_revision"] != MLEBENCH_METADATA_REVISION:
        raise RuntimeError("image contains another MLE-bench revision")
    if file_sha256(arguments.dependency_lock) != manifest["dependency_lock_sha256"]:
        raise RuntimeError("image dependency lock differs from its build manifest")
    if source_tree_sha256(arguments.repository_root) != manifest["source_tree_sha256"]:
        raise RuntimeError("image source tree differs from its build manifest")
    validate_mlebench_checkout(arguments.mlebench_checkout)
    import torch

    versions = {"torch": importlib.metadata.version("torch")}
    for distribution, module in PINNED_IMPORTS.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    for distribution in PINNED_VERSIONS:
        versions.setdefault(distribution, importlib.metadata.version(distribution))
    mismatched = {
        name: {"actual": versions.get(name), "expected": expected}
        for name, expected in PINNED_VERSIONS.items()
        if versions.get(name) != expected
    }
    if mismatched:
        raise RuntimeError(f"pinned dependency versions differ: {mismatched}")
    architectures = tuple(torch.cuda.get_arch_list())
    if not architectures:
        architectures = tuple(str(torch._C._cuda_getArchFlags()).split())
    if "sm_70" not in architectures or "sm_120" not in architectures:
        raise RuntimeError("PyTorch must embed both sm_70 and sm_120 support")
    forbidden = tuple(
        path
        for path in arguments.repository_root.rglob("*")
        if path.is_file()
        and (
            path.name == "kaggle.json"
            or path.name.endswith(".pem")
            or path.name.endswith(".key")
        )
    )
    if forbidden:
        raise RuntimeError("credential-like files are present in the image source")
    cuda_probe = None
    if arguments.require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("local image preflight requires exactly one visible CUDA GPU")
        left = torch.randn((256, 256), device="cuda")
        right = torch.randn((256, 256), device="cuda")
        value = left @ right
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("CUDA matrix probe is non-finite")
        properties = torch.cuda.get_device_properties(0)
        cuda_probe = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "finite_matrix": True,
        }
    return {
        "status": "passed",
        "manifest_sha256": canonical_sha256(manifest),
        "versions": versions,
        "cuda_architectures": architectures,
        "cuda_probe": cuda_probe,
        "credential_files_found": 0,
    }


def _preflight_kaggle() -> tuple[Mapping[str, Any], ...]:
    validate_external_credentials(REPOSITORY_ROOT)
    client = KaggleCliClient()
    client.authenticate()
    entries = {
        row.task_id: row
        for row in load_task_registry().entries
        if row.task_id in analysis_summary()["task_ids"]
    }
    return tuple(
        {
            "task_id": task_id,
            **client.download_smallest_file(entries[task_id].kaggle_slug).to_dict(),
        }
        for task_id in analysis_summary()["task_ids"]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("analyze")

    preflight = commands.add_parser("image-preflight")
    preflight.add_argument(
        "--build-manifest",
        type=Path,
        default=Path("/opt/perfseer/build-manifest.json"),
    )
    preflight.add_argument(
        "--dependency-lock",
        type=Path,
        default=Path("/opt/perfseer/repository/containers/v100-labeler/requirements.lock"),
    )
    preflight.add_argument("--mlebench-checkout", type=Path, default=Path("/opt/mle-bench"))
    preflight.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    preflight.add_argument("--require-cuda", action="store_true")
    preflight.add_argument("--verify-kaggle-access", action="store_true")

    smoke = commands.add_parser("smoke-local")
    smoke.add_argument("--workspace", type=Path, required=True)
    smoke.add_argument("--mlebench-checkout", type=Path, default=Path("/opt/mle-bench"))
    smoke.add_argument(
        "--dependency-lock",
        type=Path,
        default=REPOSITORY_ROOT / "containers/v100-labeler/requirements.lock",
    )
    smoke.add_argument("--build-manifest", type=Path, default=Path("/opt/perfseer/build-manifest.json"))
    smoke.add_argument("--model", action="append", choices=LOCAL_SMOKE_MODELS)
    smoke.add_argument("--all-families", action="store_true")
    smoke.add_argument("--kaggle-executable", default="kaggle")

    run = commands.add_parser("run-campaign")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--mlebench-checkout", type=Path, default=Path("/opt/mle-bench"))
    run.add_argument("--repository-revision", required=True)
    run.add_argument("--image-digest", required=True)
    run.add_argument("--kaggle-executable", default="kaggle")
    run.add_argument(
        "--build-manifest", type=Path, default=Path("/opt/perfseer/build-manifest.json")
    )
    run.add_argument("--skip-kaggle-preflight", action="store_true")
    run.add_argument("--pilot", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("--kind", choices=("campaign", "local"), required=True)
    verify.add_argument("--workspace", type=Path, required=True)
    verify.add_argument("--repository-revision")
    verify.add_argument("--image-digest")
    verify.add_argument("--model", action="append", choices=LOCAL_SMOKE_MODELS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "analyze":
        print(json.dumps(analysis_summary(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "image-preflight":
        result = dict(_image_preflight(arguments))
        if arguments.verify_kaggle_access:
            result["kaggle_download_probes"] = _preflight_kaggle()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "smoke-local":
        if arguments.all_families:
            if arguments.model:
                raise ValueError("--all-families cannot be combined with --model")
            print(
                json.dumps(
                    run_family_matrix_smoke(arguments.workspace),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        manifest = _load_json(arguments.build_manifest)
        models = tuple(arguments.model or LOCAL_SMOKE_MODELS)
        records = run_local_smoke(
            workspace=arguments.workspace,
            repository_root=REPOSITORY_ROOT,
            mlebench_checkout=arguments.mlebench_checkout,
            dependency_lock=arguments.dependency_lock,
            source_revision=str(manifest["source_revision"]),
            source_tree_sha256=str(manifest["source_tree_sha256"]),
            image_identity=str(manifest["image_identity"]),
            models=models,
            kaggle_executable=arguments.kaggle_executable,
        )
        print(json.dumps({"status": "passed", "records": records}, indent=2, sort_keys=True))
        return 0
    if arguments.command == "run-campaign":
        manifest = _load_json(arguments.build_manifest)
        if manifest.get("source_revision") != arguments.repository_revision:
            raise RuntimeError("Job source revision differs from the embedded build manifest")
        if file_sha256(
            REPOSITORY_ROOT / "containers/v100-labeler/requirements.lock"
        ) != manifest.get("dependency_lock_sha256"):
            raise RuntimeError("Job dependency lock differs from the embedded build manifest")
        freeze_campaign_contract(arguments.workspace)
        if not arguments.skip_kaggle_preflight:
            _preflight_kaggle()
        os.environ["PERFSEER_V100_LABELING"] = "1"
        os.environ["PERFSEER_CONTAINER_DIGEST"] = arguments.image_digest
        os.environ["PERFSEER_BUILD_SOURCE_REVISION"] = str(manifest["source_revision"])
        os.environ["PERFSEER_BUILD_SOURCE_TREE_SHA256"] = str(
            manifest["source_tree_sha256"]
        )
        os.environ["PERFSEER_BUILD_DEPENDENCY_LOCK_SHA256"] = str(
            manifest["dependency_lock_sha256"]
        )
        os.environ["PERFSEER_BUILD_IMAGE_IDENTITY"] = str(manifest["image_identity"])
        run_campaign(
            workspace=arguments.workspace,
            repository_root=REPOSITORY_ROOT,
            mlebench_checkout=arguments.mlebench_checkout,
            repository_revision=arguments.repository_revision,
            image_digest=arguments.image_digest,
            kaggle_executable=arguments.kaggle_executable,
            pilot=arguments.pilot,
        )
        return 0
    if arguments.kind == "local":
        records = verify_local_smoke(
            arguments.workspace,
            expected_models=tuple(arguments.model or LOCAL_SMOKE_MODELS),
        )
        print(json.dumps({"status": "passed", "records": records}, indent=2, sort_keys=True))
        return 0
    if not arguments.repository_revision or not arguments.image_digest:
        raise ValueError("campaign verification requires revision and image digest")
    receipt = verify_campaign(
        arguments.workspace,
        repository_revision=arguments.repository_revision,
        image_digest=arguments.image_digest,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
