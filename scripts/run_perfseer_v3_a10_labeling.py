#!/usr/bin/env python3
"""Single entry point for the immutable native Nautilus A10 image."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


EXPECTED_PROFILE = os.environ.get("PERFSEER_A10_IMAGE_PROFILE", "native_a10")
if EXPECTED_PROFILE not in {"native_a10", "native_a10_speech_v2"}:
    raise RuntimeError("A10 image declares an unsupported baked profile")
os.environ.setdefault("PERFSEER_LABELER_PROFILE", EXPECTED_PROFILE)
if os.environ["PERFSEER_LABELER_PROFILE"] != EXPECTED_PROFILE:
    raise RuntimeError("A10 image refuses a labeler profile other than its baked identity")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from perfseer_v3.dataset_pack.a10_campaign import (
    analysis_summary,
    run_campaign,
    verify_campaign,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256, file_sha256
from perfseer_v3.dataset_pack.kaggle import KaggleCliClient, validate_external_credentials
from perfseer_v3.dataset_pack.local_smoke import (
    CAMPAIGN_FAMILIES,
    LOCAL_SMOKE_MODELS,
    run_family_matrix_smoke,
    run_local_smoke,
    run_speech_precision_matrix_smoke,
    verify_local_smoke,
)
from perfseer_v3.dataset_pack.mlebench_bridge import validate_mlebench_checkout
from perfseer_v3.dataset_pack.source_identity import source_tree_sha256
from perfseer_v3.dataset_pack.task_registry import MLEBENCH_METADATA_REVISION, load_task_registry


PINNED_IMPORTS = {
    "appdirs": "appdirs",
    # Kaggle 2.x authenticates while importing its top-level CLI package.
    # Verify the distribution here; exercise the CLI only in the explicit
    # download gate so this image check remains genuinely offline.
    "kaggle": None,
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
    "tensorflow-cpu": "tensorflow",
    "torch-geometric": "torch_geometric",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "tqdm": "tqdm",
}
PINNED_VERSIONS = {
    "torch": "2.10.0+cu128",
    "torchaudio": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "torch-geometric": "2.7.0",
    "kaggle": "2.2.2",
    "tensorflow-cpu": "2.20.0",
    "transformers": "5.7.0",
}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _image_preflight(arguments: argparse.Namespace) -> Mapping[str, Any]:
    manifest = _load_json(arguments.build_manifest)
    expected_keys = {
        "version",
        "base_image",
        "source_revision",
        "source_tree_sha256",
        "dependency_lock_sha256",
        "mle_bench_revision",
        "image_identity",
        "target_hardware_id",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError("A10 image build manifest schema differs")
    if (
        manifest["version"]
        != (
            "perfseer_v3_nrp_a10_speech_image_build_manifest_v2"
            if EXPECTED_PROFILE == "native_a10_speech_v2"
            else "perfseer_v3_nrp_a10_image_build_manifest_v1"
        )
        or manifest["target_hardware_id"] != "nvidia_a10_24gb_nrp"
        or manifest["mle_bench_revision"] != MLEBENCH_METADATA_REVISION
    ):
        raise RuntimeError("A10 image build identity differs")
    if file_sha256(arguments.dependency_lock) != manifest["dependency_lock_sha256"]:
        raise RuntimeError("A10 dependency lock differs from its build manifest")
    if source_tree_sha256(arguments.repository_root) != manifest["source_tree_sha256"]:
        raise RuntimeError("A10 source tree differs from its build manifest")
    validate_mlebench_checkout(arguments.mlebench_checkout)
    import torch

    versions = {"torch": importlib.metadata.version("torch")}
    for distribution, module in PINNED_IMPORTS.items():
        if module is not None:
            importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    mismatched = {
        name: {"actual": versions.get(name), "expected": expected}
        for name, expected in PINNED_VERSIONS.items()
        if versions.get(name) != expected
    }
    if mismatched:
        raise RuntimeError(f"pinned dependency versions differ: {mismatched}")
    dependency_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if dependency_check.returncode != 0:
        raise RuntimeError("installed Python dependency graph fails pip check")
    architectures = tuple(torch.cuda.get_arch_list())
    if not architectures:
        architectures = tuple(str(torch._C._cuda_getArchFlags()).split())
    if "sm_86" not in architectures or "sm_120" not in architectures:
        raise RuntimeError("PyTorch must embed both sm_86 and sm_120 support")
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
            raise RuntimeError("local preflight requires exactly one visible CUDA GPU")
        left = torch.randn((256, 256), device="cuda")
        value = left @ left
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("CUDA matrix probe is non-finite")
        properties = torch.cuda.get_device_properties(0)
        cuda_probe = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "finite_matrix": True,
            "production_eligible": False,
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
    client = KaggleCliClient(maximum_attempts=4, initial_backoff_seconds=1.0)
    client.authenticate()
    entries = load_task_registry().entries
    if len(entries) != 22:
        raise RuntimeError("native campaign must gate exactly 22 Kaggle competitions")
    ordered_entries = (
        tuple(
            entry
            for entry in entries
            if entry.task_id == "tensorflow-speech-yes-no"
        )
        + tuple(
            entry
            for entry in entries
            if entry.task_id != "tensorflow-speech-yes-no"
        )
        if EXPECTED_PROFILE == "native_a10_speech_v2"
        else entries
    )
    results = tuple(
        {
            "task_id": entry.task_id,
            "rules_url": entry.license_or_rules_url,
            **client.download_smallest_file(entry.kaggle_slug).to_dict(),
        }
        for entry in ordered_entries
    )
    if EXPECTED_PROFILE == "native_a10_speech_v2":
        speech = results[0]
        if (
            speech.get("task_id") != "tensorflow-speech-yes-no"
            or speech.get("remote_name") != "link_to_gcp_credits_form.txt"
            or speech.get("advertised_bytes") != 50
            or speech.get("downloaded_bytes") != 50
        ):
            raise RuntimeError(
                "speech rules gate did not download the advertised 50-byte file"
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("analyze")

    preflight = commands.add_parser("image-preflight")
    preflight.add_argument("--build-manifest", type=Path, default=Path("/opt/perfseer/build-manifest.json"))
    preflight.add_argument(
        "--dependency-lock",
        type=Path,
        default=Path(
            "/opt/perfseer/repository/containers/a10-labeler/requirements.lock"
        ),
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
        default=REPOSITORY_ROOT / "containers/a10-labeler/requirements.lock",
    )
    smoke.add_argument("--build-manifest", type=Path, default=Path("/opt/perfseer/build-manifest.json"))
    smoke.add_argument("--model", action="append", choices=LOCAL_SMOKE_MODELS)
    smoke.add_argument("--all-families", action="store_true")
    smoke.add_argument("--speech-precision-matrix", action="store_true")
    smoke.add_argument("--kaggle-executable", default="kaggle")

    run = commands.add_parser("run-campaign")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--mlebench-checkout", type=Path, default=Path("/opt/mle-bench"))
    run.add_argument("--repository-revision", required=True)
    run.add_argument("--image-digest", required=True)
    run.add_argument("--kaggle-executable", default="kaggle")
    run.add_argument("--build-manifest", type=Path, default=Path("/opt/perfseer/build-manifest.json"))
    run.add_argument("--skip-kaggle-preflight", action="store_true")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--chunk-index", type=int)
    run.add_argument("--max-new-accepted", type=int, default=256)

    verify = commands.add_parser("verify")
    verify.add_argument("--workspace", type=Path, required=True)
    completeness = verify.add_mutually_exclusive_group(required=True)
    completeness.add_argument("--partial", action="store_true")
    completeness.add_argument("--complete", action="store_true")
    verify.add_argument("--local", action="store_true")
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
        if arguments.all_families and arguments.speech_precision_matrix:
            raise ValueError("select only one fixture matrix")
        if arguments.all_families:
            if arguments.model:
                raise ValueError("--all-families cannot be combined with --model")
            result = run_family_matrix_smoke(arguments.workspace)
        elif arguments.speech_precision_matrix:
            if arguments.model:
                raise ValueError(
                    "--speech-precision-matrix cannot be combined with --model"
                )
            result = run_speech_precision_matrix_smoke(
                arguments.workspace,
                repository_root=REPOSITORY_ROOT,
                mlebench_checkout=arguments.mlebench_checkout,
                kaggle_executable=arguments.kaggle_executable,
            )
        else:
            manifest = _load_json(arguments.build_manifest)
            result = run_local_smoke(
                workspace=arguments.workspace,
                repository_root=REPOSITORY_ROOT,
                mlebench_checkout=arguments.mlebench_checkout,
                dependency_lock=arguments.dependency_lock,
                source_revision=str(manifest["source_revision"]),
                source_tree_sha256=str(manifest["source_tree_sha256"]),
                image_identity=str(manifest["image_identity"]),
                models=tuple(arguments.model or LOCAL_SMOKE_MODELS),
                kaggle_executable=arguments.kaggle_executable,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "run-campaign":
        manifest = _load_json(arguments.build_manifest)
        if manifest.get("source_revision") != arguments.repository_revision:
            raise RuntimeError("Job revision differs from the embedded build manifest")
        lock_path = REPOSITORY_ROOT / "containers/a10-labeler/requirements.lock"
        if file_sha256(lock_path) != manifest.get("dependency_lock_sha256"):
            raise RuntimeError("Job dependency lock differs from its build manifest")
        if not arguments.skip_kaggle_preflight:
            _preflight_kaggle()
        os.environ["PERFSEER_CONTAINER_DIGEST"] = arguments.image_digest
        os.environ["PERFSEER_BUILD_SOURCE_REVISION"] = str(manifest["source_revision"])
        os.environ["PERFSEER_BUILD_SOURCE_TREE_SHA256"] = str(manifest["source_tree_sha256"])
        os.environ["PERFSEER_BUILD_DEPENDENCY_LOCK_SHA256"] = str(manifest["dependency_lock_sha256"])
        os.environ["PERFSEER_BUILD_IMAGE_IDENTITY"] = str(manifest["image_identity"])
        receipt = run_campaign(
            workspace=arguments.workspace,
            repository_root=REPOSITORY_ROOT,
            mlebench_checkout=arguments.mlebench_checkout,
            repository_revision=arguments.repository_revision,
            image_digest=arguments.image_digest,
            kaggle_executable=arguments.kaggle_executable,
            pilot=arguments.pilot,
            chunk_index=arguments.chunk_index,
            max_new_accepted=arguments.max_new_accepted,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if arguments.local:
        records = verify_local_smoke(
            arguments.workspace,
            expected_models=tuple(arguments.model or LOCAL_SMOKE_MODELS),
        )
        result: Any = {"status": "passed", "records": records}
    else:
        result = verify_campaign(arguments.workspace, complete=arguments.complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
