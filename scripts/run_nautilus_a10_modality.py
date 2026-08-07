#!/usr/bin/env python3
"""Analyze, label, verify, or merge Nautilus A10 modality workspaces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from perfseer_v3.dataset_pack.kaggle import KaggleCliClient, validate_external_credentials
from perfseer_v3.dataset_pack.family_sharding import (
    SUPPORTED_FAMILIES,
    build_family_contract,
    family_analysis_summary,
    freeze_family_contract,
    verify_family_workspace,
)
from perfseer_v3.dataset_pack.mlebench_bridge import verify_frozen_preparers
from perfseer_v3.dataset_pack.modality_sharding import (
    MODALITIES,
    analysis_summary,
    build_modality_contract,
    freeze_modality_contract,
    freeze_run_identity,
    merge_modality_workspaces,
    verify_modality_workspace,
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _verify_clean_revision(revision: str) -> None:
    if _git("rev-parse", "HEAD") != revision:
        raise RuntimeError("repository checkout differs from the requested immutable revision")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("repository checkout must be clean before labeling")


def _verify_runtime_dependencies() -> None:
    # Keep the original frozen CUDA/package gate as the single runtime policy.
    scripts_root = str(REPOSITORY_ROOT / "scripts")
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from run_a10g_18k_pack import verify_runtime_dependencies

    verify_runtime_dependencies()


def _selection_contract(modality: str, family_id: str | None):
    contract = (
        build_modality_contract(modality)
        if family_id is None
        else build_family_contract(family_id)
    )
    if contract.modality != modality:
        raise ValueError(
            f"family {family_id!r} belongs to {contract.modality!r}, not {modality!r}"
        )
    return contract


def _preflight_external_data(
    modality: str,
    family_id: str | None,
    checkout: Path,
    kaggle: str,
) -> None:
    validate_external_credentials(REPOSITORY_ROOT)
    contract = _selection_contract(modality, family_id)
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entries = tuple(
        entry for entry in load_task_registry().entries if entry.task_id in contract.task_ids
    )
    verify_frozen_preparers(checkout, entries)
    client = KaggleCliClient(executable=kaggle)
    client.authenticate()
    for entry in entries:
        client.probe_competition(entry.kaggle_slug)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--modality", choices=MODALITIES)
    analyze.add_argument("--family-id", choices=SUPPORTED_FAMILIES)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("--modality", choices=MODALITIES, required=True)
    initialize.add_argument("--family-id", choices=SUPPORTED_FAMILIES)
    initialize.add_argument("--workspace", type=Path, required=True)
    initialize.add_argument("--repository-revision", required=True)
    initialize.add_argument("--image-digest", required=True)

    for name in ("label", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--modality", choices=MODALITIES, required=True)
        command.add_argument("--family-id", choices=SUPPORTED_FAMILIES)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--repository-revision", required=True)
        command.add_argument("--image-digest", required=True)
        if name == "label":
            command.add_argument("--mlebench-checkout", type=Path, required=True)
            command.add_argument("--kaggle-executable", default="kaggle")
            command.add_argument("--max-new-accepted", type=int)

    merge = commands.add_parser("merge")
    for modality in MODALITIES:
        merge.add_argument(f"--{modality}-workspace", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "analyze":
        if arguments.family_id is not None:
            if arguments.modality is None:
                raise ValueError("--family-id requires --modality")
            _selection_contract(arguments.modality, arguments.family_id)
            summary = family_analysis_summary(arguments.family_id)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        summary = analysis_summary()
        if arguments.modality is not None:
            summary = {
                "hardware_family_id": summary["hardware_family_id"],
                "logical_target_hardware_id": summary["logical_target_hardware_id"],
                "target_manifest_sha256": summary["target_manifest_sha256"],
                "rest_candidate_count": summary["rest_candidate_count"],
                "rest_measured_epoch_count": summary["rest_measured_epoch_count"],
                "modality": arguments.modality,
                **summary["modalities"][arguments.modality],
            }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if arguments.command == "initialize":
        contract = (
            freeze_modality_contract(arguments.workspace, arguments.modality)
            if arguments.family_id is None
            else freeze_family_contract(arguments.workspace, arguments.family_id)
        )
        if contract.modality != arguments.modality:
            raise ValueError("family and modality do not match")
        identity = freeze_run_identity(
            arguments.workspace,
            repository_revision=arguments.repository_revision,
            image_digest=arguments.image_digest,
        )
        print(
            json.dumps(
                {
                    "modality": contract.modality,
                    **(
                        {"family_id": arguments.family_id}
                        if arguments.family_id is not None
                        else {}
                    ),
                    "candidate_count": contract.candidate_count,
                    "contract_sha256": contract.contract_sha256,
                    "identity_sha256": identity["identity_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "verify":
        _selection_contract(arguments.modality, arguments.family_id)
        completion = (
            verify_modality_workspace(
                arguments.workspace,
                arguments.modality,
                repository_revision=arguments.repository_revision,
                image_digest=arguments.image_digest,
            )
            if arguments.family_id is None
            else verify_family_workspace(
                arguments.workspace,
                arguments.family_id,
                repository_revision=arguments.repository_revision,
                image_digest=arguments.image_digest,
            )
        )
        print(json.dumps(completion.to_dict(), indent=2, sort_keys=True))
        return 0
    if arguments.command == "merge":
        receipt = merge_modality_workspaces(
            {
                modality: getattr(arguments, f"{modality}_workspace")
                for modality in MODALITIES
            },
            arguments.output,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    _verify_clean_revision(arguments.repository_revision)
    _verify_runtime_dependencies()
    _preflight_external_data(
        arguments.modality,
        arguments.family_id,
        arguments.mlebench_checkout,
        arguments.kaggle_executable,
    )
    os.environ["PERFSEER_ALLOW_A10_FAMILY"] = "1"
    os.environ["PERFSEER_CONTAINER_DIGEST"] = arguments.image_digest
    from perfseer_v3.dataset_pack.modality_workflow import (
        run_family_workflow,
        run_modality_workflow,
    )

    workflow = (
        run_modality_workflow
        if arguments.family_id is None
        else run_family_workflow
    )
    workflow(
        workspace=arguments.workspace,
        repository_root=REPOSITORY_ROOT,
        mlebench_checkout=arguments.mlebench_checkout,
        **(
            {"modality": arguments.modality}
            if arguments.family_id is None
            else {"family_id": arguments.family_id}
        ),
        repository_revision=arguments.repository_revision,
        image_digest=arguments.image_digest,
        kaggle_executable=arguments.kaggle_executable,
        max_new_accepted=arguments.max_new_accepted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
