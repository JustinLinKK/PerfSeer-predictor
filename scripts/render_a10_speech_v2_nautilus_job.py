#!/usr/bin/env python3
"""Render and strictly verify a digest-only speech-V2 Nautilus A10 Job."""

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
WORKSPACE = "/workspace/perfseer-v3-native-a10-18k-speech-v2"


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
    metadata = value["metadata"]
    pod_spec = value["spec"]["template"]["spec"]
    if value["spec"].get("backoffLimit") != 0 or pod_spec.get("restartPolicy") != "Never":
        raise ValueError("Job retry policy differs from the frozen contract")
    containers = pod_spec.get("containers", [])
    if len(containers) != 1:
        raise ValueError("speech V2 Job must contain exactly one container")
    container = containers[0]
    image = str(container.get("image", ""))
    if not IMAGE_RE.fullmatch(image) or ZERO_DIGEST in image:
        raise ValueError("image must be a non-placeholder NRP registry digest")
    arguments = container.get("args", [])
    if arguments[:1] != ["run-campaign"]:
        raise ValueError("Job must execute the unified run-campaign command")
    workspace = arguments[arguments.index("--workspace") + 1]
    if workspace != WORKSPACE:
        raise ValueError("speech V2 Job uses another workspace generation")
    if mode == "pilot":
        if "--pilot" not in arguments or "--chunk-index" in arguments:
            raise ValueError("pilot Job mode differs")
    else:
        if "--pilot" in arguments or "--chunk-index" not in arguments:
            raise ValueError("chunk Job mode differs")
        index = int(arguments[arguments.index("--chunk-index") + 1])
        if index != chunk_index or "--max-new-accepted" not in arguments:
            raise ValueError("chunk index or acceptance cap differs")
        if arguments[arguments.index("--max-new-accepted") + 1] != "256":
            raise ValueError("chunk acceptance cap must be exactly 256")
    revision = arguments[arguments.index("--repository-revision") + 1]
    digest_argument = arguments[arguments.index("--image-digest") + 1]
    if not REVISION_RE.fullmatch(revision) or revision == "0" * 40:
        raise ValueError("source revision must be a non-placeholder Git SHA")
    if digest_argument != image:
        raise ValueError("result provenance digest differs from the image")
    if metadata.get("annotations", {}).get("perfseer.ai/source-revision") != revision:
        raise ValueError("Job annotation differs from its source revision")
    expected_resources = {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"}
    resources = container["resources"]
    if resources["requests"] != expected_resources or resources["limits"] != expected_resources:
        raise ValueError("requests/limits differ from the A10 NRP policy contract")
    terms = pod_spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    expected_terms = [{"matchExpressions": [{"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A10"]}]}]
    if terms != expected_terms:
        raise ValueError("required exact NVIDIA-A10 product affinity differs")
    profiles = {
        row["name"]: row["value"]
        for row in container.get("env", [])
        if row.get("name") in {"PERFSEER_A10_IMAGE_PROFILE", "PERFSEER_LABELER_PROFILE"}
    }
    if profiles != {
        "PERFSEER_A10_IMAGE_PROFILE": "native_a10_speech_v2",
        "PERFSEER_LABELER_PROFILE": "native_a10_speech_v2",
    }:
        raise ValueError("Job does not bake and select the speech V2 profile")
    volumes = {row["name"]: row for row in pod_spec["volumes"]}
    if volumes["shared-memory"]["emptyDir"] != {"medium": "Memory", "sizeLimit": "16Gi"}:
        raise ValueError("/dev/shm differs from 16 GiB")
    if volumes["kaggle-credential"]["secret"].get("defaultMode") != 0o400:
        raise ValueError("Kaggle Secret mode is not 0400")


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    if not IMAGE_RE.fullmatch(arguments.image) or ZERO_DIGEST in arguments.image:
        raise ValueError("--image must be a non-placeholder NRP registry digest")
    if not REVISION_RE.fullmatch(arguments.source_revision) or arguments.source_revision == "0" * 40:
        raise ValueError("--source-revision must be a non-placeholder Git SHA")
    for label, value in (("namespace", arguments.namespace), ("PVC", arguments.pvc), ("Secret", arguments.secret)):
        if not DNS_LABEL_RE.fullmatch(value) or len(value) > 63:
            raise ValueError(f"{label} is not a Kubernetes DNS label")
    if arguments.mode == "chunk" and (
        arguments.chunk_index is None or not 0 <= arguments.chunk_index < 70
    ):
        raise ValueError("chunk mode requires --chunk-index in [0, 69]")
    if arguments.mode == "pilot" and arguments.chunk_index is not None:
        raise ValueError("pilot mode does not accept --chunk-index")
    result = _load(arguments.template)
    suffix = "pilot" if arguments.mode == "pilot" else f"chunk-{arguments.chunk_index:02d}"
    result["metadata"]["name"] = f"perfseer-v3-a10-speech-v2-{suffix}"
    result["metadata"]["namespace"] = arguments.namespace
    result["metadata"]["annotations"]["perfseer.ai/source-revision"] = arguments.source_revision
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
        default=Path("k8s/a10-speech-v2-labeler-job.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--pvc", default="perfseer-v3-a10-speech-v2-labels")
    parser.add_argument("--secret", default="perfseer-kaggle")
    parser.add_argument("--mode", choices=("pilot", "chunk"), required=True)
    parser.add_argument("--chunk-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = render(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        yaml.safe_dump(result, sort_keys=False), encoding="utf-8"
    )
    verify_job(
        _load(arguments.output),
        mode=arguments.mode,
        chunk_index=arguments.chunk_index,
    )
    print(f"verified {arguments.mode} speech V2 Job: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
