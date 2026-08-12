#!/usr/bin/env python3
"""Render and strictly verify a digest-only PerfSeer Nautilus Job."""

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
    if index + 1 >= len(arguments):
        raise ValueError(f"Job template has no value after {option}")
    arguments[index + 1] = value


def verify_job(value: Mapping[str, Any], *, mode: str) -> None:
    if value.get("apiVersion") != "batch/v1" or value.get("kind") != "Job":
        raise ValueError("rendered object is not a batch/v1 Job")
    metadata = value["metadata"]
    pod_spec = value["spec"]["template"]["spec"]
    if value["spec"].get("backoffLimit") != 0 or pod_spec.get("restartPolicy") != "Never":
        raise ValueError("Job retry policy differs from the frozen contract")
    containers = pod_spec.get("containers", [])
    if len(containers) != 1:
        raise ValueError("Job must contain one four-worker container")
    container = containers[0]
    image = str(container.get("image", ""))
    if not IMAGE_RE.fullmatch(image) or ZERO_DIGEST in image:
        raise ValueError("image must be a non-placeholder NRP GitLab registry digest")
    arguments = container.get("args", [])
    if mode == "pilot" and "--pilot" not in arguments:
        raise ValueError("pilot Job omits --pilot")
    if mode == "production" and "--pilot" in arguments:
        raise ValueError("production Job unexpectedly enables --pilot")
    revision = arguments[arguments.index("--repository-revision") + 1]
    digest_argument = arguments[arguments.index("--image-digest") + 1]
    if not REVISION_RE.fullmatch(revision) or revision == "0" * 40:
        raise ValueError("source revision must be a non-placeholder 40-character Git SHA")
    if digest_argument != image:
        raise ValueError("result provenance digest differs from the container image")
    if metadata.get("annotations", {}).get("perfseer.ai/source-revision") != revision:
        raise ValueError("Job annotation differs from its source revision argument")
    resources = container["resources"]
    if resources["requests"] != {"cpu": "16", "memory": "64Gi", "nvidia.com/gpu": "4"}:
        raise ValueError("resource requests differ from the V100 contract")
    if resources["limits"] != {"cpu": "32", "memory": "128Gi", "nvidia.com/gpu": "4"}:
        raise ValueError("resource limits differ from the V100 contract")
    terms = pod_spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    if terms != [{"matchExpressions": [{
        "key": "nvidia.com/gpu.product",
        "operator": "In",
        "values": ["Tesla-V100-SXM2-32GB"],
    }]}]:
        raise ValueError("required exact V100 product affinity differs")
    volumes = {row["name"]: row for row in pod_spec["volumes"]}
    if volumes["shared-memory"]["emptyDir"] != {"medium": "Memory", "sizeLimit": "16Gi"}:
        raise ValueError("/dev/shm differs from the 16 GiB contract")
    if volumes["kaggle-credential"]["secret"].get("defaultMode") != 0o400:
        raise ValueError("Kaggle Secret mode is not 0400")


def render(arguments: argparse.Namespace) -> dict[str, Any]:
    if not IMAGE_RE.fullmatch(arguments.image) or ZERO_DIGEST in arguments.image:
        raise ValueError("--image must be a non-placeholder NRP registry digest reference")
    if not REVISION_RE.fullmatch(arguments.source_revision) or arguments.source_revision == "0" * 40:
        raise ValueError("--source-revision must be a non-placeholder Git SHA")
    for label, value in (
        ("namespace", arguments.namespace),
        ("PVC", arguments.pvc),
        ("Secret", arguments.secret),
    ):
        if not DNS_LABEL_RE.fullmatch(value) or len(value) > 63:
            raise ValueError(f"{label} is not a Kubernetes DNS label")
    result = _load(arguments.template)
    name = f"perfseer-v3-v100-{arguments.mode}"
    result["metadata"]["name"] = name
    result["metadata"]["namespace"] = arguments.namespace
    result["metadata"]["annotations"]["perfseer.ai/source-revision"] = arguments.source_revision
    pod_spec = result["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    container["image"] = arguments.image
    _replace_argument(container["args"], "--repository-revision", arguments.source_revision)
    _replace_argument(container["args"], "--image-digest", arguments.image)
    if arguments.mode == "production":
        container["args"] = [row for row in container["args"] if row != "--pilot"]
    for volume in pod_spec["volumes"]:
        if volume["name"] == "workspace":
            volume["persistentVolumeClaim"]["claimName"] = arguments.pvc
        elif volume["name"] == "kaggle-credential":
            volume["secret"]["secretName"] = arguments.secret
    verify_job(result, mode=arguments.mode)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("k8s/v100-labeler-job.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--pvc", default="perfseer-v3-v100-labels")
    parser.add_argument("--secret", default="perfseer-kaggle")
    parser.add_argument("--mode", choices=("pilot", "production"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = render(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    verify_job(_load(arguments.output), mode=arguments.mode)
    print(f"verified {arguments.mode} Job: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
