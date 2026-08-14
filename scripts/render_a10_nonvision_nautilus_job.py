#!/usr/bin/env python3
"""Render and offline-verify a digest-only four-A10 non-vision Job."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


IMAGE_RE = re.compile(
    r"^gitlab-registry\.nrp-nautilus\.io/[a-zA-Z0-9._/-]+@sha256:[0-9a-f]{64}$"
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
ZERO_DIGEST = "sha256:" + "0" * 64
WORKSPACE = "/workspace/perfseer-v3-native-a10-nonvision-11200-v1"
PROFILE = "native_a10_nonvision_4gpu_v1"
CHUNK_COUNT = 44


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Job template must be one YAML object")
    return value


def _replace_argument(arguments: list[str], option: str, value: str) -> None:
    try:
        index = arguments.index(option)
    except ValueError as error:
        raise ValueError(f"Job template has no {option} argument") from error
    arguments[index + 1] = value


def verify_job(
    value: Mapping[str, Any], *, mode: str, chunk_index: int | None = None
) -> None:
    if value.get("apiVersion") != "batch/v1" or value.get("kind") != "Job":
        raise ValueError("rendered object is not a batch/v1 Job")
    job_spec = value["spec"]
    pod_spec = job_spec["template"]["spec"]
    if (
        job_spec.get("backoffLimit") != 0
        or job_spec.get("activeDeadlineSeconds") != 172800
        or pod_spec.get("restartPolicy") != "Never"
    ):
        raise ValueError("Job retry/deadline policy differs from the contract")
    containers = pod_spec.get("containers", [])
    if len(containers) != 1:
        raise ValueError("four-A10 Job must contain one controller container")
    container = containers[0]
    image = str(container.get("image", ""))
    if not IMAGE_RE.fullmatch(image) or ZERO_DIGEST in image:
        raise ValueError("image must be a non-placeholder NRP registry digest")
    arguments = container.get("args", [])
    if arguments[:1] != ["run-campaign"]:
        raise ValueError("Job must execute the unified campaign command")
    if arguments[arguments.index("--workspace") + 1] != WORKSPACE:
        raise ValueError("Job uses another workspace generation")
    if mode == "pilot":
        if "--pilot" not in arguments or "--chunk-index" in arguments:
            raise ValueError("pilot Job mode differs")
    else:
        if "--pilot" in arguments or "--chunk-index" not in arguments:
            raise ValueError("chunk Job mode differs")
        actual_index = int(arguments[arguments.index("--chunk-index") + 1])
        if actual_index != chunk_index:
            raise ValueError("rendered chunk index differs")
        cap = arguments[arguments.index("--max-new-accepted") + 1]
        if cap != "256":
            raise ValueError("production chunk cap must remain 256")
    revision = arguments[arguments.index("--repository-revision") + 1]
    digest_argument = arguments[arguments.index("--image-digest") + 1]
    if not REVISION_RE.fullmatch(revision) or revision == "0" * 40:
        raise ValueError("source revision is a placeholder")
    if digest_argument != image:
        raise ValueError("result provenance digest differs from the image")
    if value["metadata"].get("annotations", {}).get(
        "perfseer.ai/source-revision"
    ) != revision:
        raise ValueError("Job annotation differs from the source revision")
    expected_resources = {
        "cpu": "32",
        "memory": "128Gi",
        "ephemeral-storage": "32Gi",
        "nvidia.com/gpu": "4",
    }
    resources = container["resources"]
    if resources["requests"] != expected_resources or resources["limits"] != expected_resources:
        raise ValueError("equal request/limit resources differ from the NRP contract")
    terms = pod_spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    expected_terms = [
        {
            "matchExpressions": [
                {
                    "key": "nvidia.com/gpu.product",
                    "operator": "In",
                    "values": ["NVIDIA-A10"],
                }
            ]
        }
    ]
    if terms != expected_terms:
        raise ValueError("required NVIDIA-A10 affinity differs")
    profiles = {
        row["name"]: row["value"]
        for row in container.get("env", [])
        if row.get("name") in {"PERFSEER_A10_IMAGE_PROFILE", "PERFSEER_LABELER_PROFILE"}
    }
    if profiles != {
        "PERFSEER_A10_IMAGE_PROFILE": PROFILE,
        "PERFSEER_LABELER_PROFILE": PROFILE,
    }:
        raise ValueError("Job does not bake/select the non-vision profile")
    if container.get("securityContext", {}).get("readOnlyRootFilesystem") is not True:
        raise ValueError("container root filesystem must be read-only")
    volumes = {row["name"]: row for row in pod_spec["volumes"]}
    if volumes["shared-memory"]["emptyDir"] != {
        "medium": "Memory",
        "sizeLimit": "32Gi",
    }:
        raise ValueError("/dev/shm differs from 32 GiB")
    if volumes["temporary-files"]["emptyDir"] != {"sizeLimit": "32Gi"}:
        raise ValueError("temporary storage volume differs from 32 GiB")
    if volumes["kaggle-credential"]["secret"].get("defaultMode") != 0o400:
        raise ValueError("Kaggle Secret mode is not 0400")


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    if not IMAGE_RE.fullmatch(arguments.image) or ZERO_DIGEST in arguments.image:
        raise ValueError("--image must be a non-placeholder NRP registry digest")
    if not REVISION_RE.fullmatch(arguments.source_revision) or arguments.source_revision == "0" * 40:
        raise ValueError("--source-revision must be a non-placeholder Git SHA")
    for label, value in (
        ("namespace", arguments.namespace),
        ("PVC", arguments.pvc),
        ("Secret", arguments.secret),
    ):
        if not DNS_LABEL_RE.fullmatch(value) or len(value) > 63:
            raise ValueError(f"{label} is not a Kubernetes DNS label")
    if arguments.mode == "chunk" and (
        arguments.chunk_index is None
        or not 0 <= arguments.chunk_index < CHUNK_COUNT
    ):
        raise ValueError("chunk mode requires --chunk-index in [0, 43]")
    if arguments.mode == "pilot" and arguments.chunk_index is not None:
        raise ValueError("pilot mode does not accept --chunk-index")
    result = _load(arguments.template)
    suffix = "pilot" if arguments.mode == "pilot" else f"chunk-{arguments.chunk_index:02d}"
    result["metadata"]["name"] = f"perfseer-v3-a10-nonvision-{suffix}"
    result["metadata"]["namespace"] = arguments.namespace
    result["metadata"]["annotations"][
        "perfseer.ai/source-revision"
    ] = arguments.source_revision
    pod_spec = result["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    container["image"] = arguments.image
    _replace_argument(container["args"], "--repository-revision", arguments.source_revision)
    _replace_argument(container["args"], "--image-digest", arguments.image)
    if arguments.mode == "chunk":
        container["args"].remove("--pilot")
        container["args"].extend(
            ["--chunk-index", str(arguments.chunk_index), "--max-new-accepted", "256"]
        )
    for volume in pod_spec["volumes"]:
        if volume["name"] == "workspace":
            volume["persistentVolumeClaim"]["claimName"] = arguments.pvc
        elif volume["name"] == "kaggle-credential":
            volume["secret"]["secretName"] = arguments.secret
    verify_job(result, mode=arguments.mode, chunk_index=arguments.chunk_index)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("k8s/a10-nonvision-4gpu-labeler-job.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--pvc", default="perfseer-v3-a10-nonvision-labels")
    parser.add_argument("--secret", default="perfseer-kaggle-nonvision")
    parser.add_argument("--mode", choices=("pilot", "chunk"), required=True)
    parser.add_argument("--chunk-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = render(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    verify_job(
        _load(arguments.output),
        mode=arguments.mode,
        chunk_index=arguments.chunk_index,
    )
    print(f"verified {arguments.mode} non-vision four-A10 Job: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
