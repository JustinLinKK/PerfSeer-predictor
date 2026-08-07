#!/usr/bin/env python3
"""Deterministically render one Nautilus A10 Pod or Job manifest."""

from __future__ import annotations

import argparse
from pathlib import PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import yaml


MODALITIES = ("audio", "tabular", "graph", "generated")
MODES = ("pod", "pilot-job", "production-job")
FAMILY_MODALITIES = {"panns_cnn14": "audio"}
MLEBENCH_REVISION = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class RenderError(ValueError):
    pass


def _dns(value: str, context: str) -> str:
    if len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise RenderError(f"{context} must be a Kubernetes DNS label")
    return value


def _repository_url(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise RenderError("repository URL must be a non-empty value without whitespace")
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https", "ssh"} and (
        parsed.password is not None or (parsed.username and parsed.scheme != "ssh")
    ):
        raise RenderError("repository URL must not contain embedded credentials")
    return value


def _image(value: str) -> tuple[str, str]:
    if "@" not in value:
        raise RenderError("image must include an immutable @sha256 digest")
    name, digest = value.rsplit("@", 1)
    if not name or not SHA256.fullmatch(digest):
        raise RenderError("image must include an immutable @sha256 digest")
    return value, digest


def _labels(
    modality: str,
    run_id: str,
    mode: str,
    family_id: str | None,
) -> Mapping[str, str]:
    labels = {
        "app.kubernetes.io/name": "perfseer-v3-labeling",
        "app.kubernetes.io/component": family_id or modality,
        "perfseer.ai/run-id": run_id,
        "perfseer.ai/mode": mode,
    }
    if family_id is not None:
        labels["perfseer.ai/family-id"] = family_id
    return labels


def _environment(arguments: argparse.Namespace, image_digest: str) -> list[Mapping[str, str]]:
    selection = arguments.family_id or arguments.modality
    if arguments.mode == "pod":
        run_root = str(
            PurePosixPath("/pvc/perfseer-v3")
            / "debug"
            / selection
            / arguments.run_id
        )
    elif arguments.family_id is not None:
        run_root = str(
            PurePosixPath("/pvc/perfseer-v3")
            / "families"
            / selection
            / arguments.run_id
        )
    else:
        run_root = str(
            PurePosixPath("/pvc/perfseer-v3")
            / arguments.modality
            / arguments.run_id
        )
    environment = [
        {"name": "PERFSEER_MODALITY", "value": arguments.modality},
        {"name": "PERFSEER_RUN_ID", "value": arguments.run_id},
        {"name": "PERFSEER_RUN_ROOT", "value": run_root},
        {"name": "PERFSEER_REPOSITORY_URL", "value": arguments.repository_url},
        {"name": "PERFSEER_REPOSITORY_REVISION", "value": arguments.revision},
        {"name": "PERFSEER_CONTAINER_DIGEST", "value": image_digest},
        {"name": "PERFSEER_ALLOW_A10_FAMILY", "value": "1"},
        {"name": "PIP_CACHE_DIR", "value": "/tmp/pip-cache"},
        {"name": "TMPDIR", "value": "/tmp"},
    ]
    if arguments.family_id is not None:
        environment.insert(
            1,
            {"name": "PERFSEER_FAMILY_ID", "value": arguments.family_id},
        )
    return environment


def _job_script(arguments: argparse.Namespace, image_digest: str) -> str:
    pilot = (
        f" --max-new-accepted {arguments.pilot_labels}"
        if arguments.mode == "pilot-job"
        else ""
    )
    family = (
        f" --family-id {arguments.family_id}"
        if arguments.family_id is not None
        else ""
    )
    return f"""set -euo pipefail
umask 077
mkdir -p "$PERFSEER_RUN_ROOT"
source_root="$PERFSEER_RUN_ROOT/source"
mlebench_root="$PERFSEER_RUN_ROOT/mle-bench"
workspace_root="$PERFSEER_RUN_ROOT/workspace"
ephemeral_source="/tmp/perfseer-source-$PERFSEER_RUN_ID"
if [ ! -d "$source_root/.git" ]; then
  git clone --filter=blob:none "$PERFSEER_REPOSITORY_URL" "$source_root"
fi
git -C "$source_root" remote set-url origin "$PERFSEER_REPOSITORY_URL"
git -C "$source_root" fetch --no-tags origin "$PERFSEER_REPOSITORY_REVISION"
git -C "$source_root" checkout --detach "$PERFSEER_REPOSITORY_REVISION"
test "$(git -C "$source_root" rev-parse HEAD)" = "$PERFSEER_REPOSITORY_REVISION"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=all)"
if [ ! -d "$mlebench_root/.git" ]; then
  git clone --filter=blob:none https://github.com/openai/mle-bench.git "$mlebench_root"
fi
git -C "$mlebench_root" fetch --no-tags origin {MLEBENCH_REVISION}
git -C "$mlebench_root" checkout --detach {MLEBENCH_REVISION}
test "$(git -C "$mlebench_root" rev-parse HEAD)" = "{MLEBENCH_REVISION}"
test -z "$(git -C "$mlebench_root" status --porcelain --untracked-files=all)"
test ! -e "$ephemeral_source"
cp -a "$source_root" "$ephemeral_source"
python -m pip install --no-cache-dir "$ephemeral_source[a10g-dataset-pack]"
cd "$ephemeral_source"
python scripts/run_nautilus_a10_modality.py initialize \
  --modality "$PERFSEER_MODALITY"{family} \
  --workspace "$workspace_root" \
  --repository-revision "$PERFSEER_REPOSITORY_REVISION" \
  --image-digest "$PERFSEER_CONTAINER_DIGEST"
python scripts/run_nautilus_a10_modality.py label \
  --modality "$PERFSEER_MODALITY"{family} \
  --workspace "$workspace_root" \
  --mlebench-checkout "$mlebench_root" \
  --repository-revision "$PERFSEER_REPOSITORY_REVISION" \
  --image-digest "$PERFSEER_CONTAINER_DIGEST"{pilot}
"""


def render(arguments: argparse.Namespace) -> Mapping[str, Any]:
    _dns(arguments.namespace, "namespace")
    _dns(arguments.pvc, "PVC")
    _dns(arguments.kaggle_secret, "Kaggle Secret")
    _dns(arguments.run_id, "run ID")
    _repository_url(arguments.repository_url)
    if not REVISION.fullmatch(arguments.revision):
        raise RenderError("revision must be a full lowercase 40-character Git commit")
    image, image_digest = _image(arguments.image)
    if (
        arguments.family_id is not None
        and FAMILY_MODALITIES.get(arguments.family_id) != arguments.modality
    ):
        raise RenderError("family ID and modality do not match")
    name_mode = {
        "pod": "debug",
        "pilot-job": "pilot",
        "production-job": "production",
    }[arguments.mode]
    resource_segment = (arguments.family_id or arguments.modality).replace("_", "-")
    resource_name = _dns(
        f"perfseer-a10-{resource_segment}-{name_mode}-{arguments.run_id}",
        "rendered resource name",
    )
    labels = _labels(
        arguments.modality,
        arguments.run_id,
        name_mode,
        arguments.family_id,
    )
    container: dict[str, Any] = {
        "name": "labeler",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/bash", "-lc"],
        "args": ["exec sleep infinity" if arguments.mode == "pod" else _job_script(arguments, image_digest)],
        "env": _environment(arguments, image_digest),
        "envFrom": [{"secretRef": {"name": arguments.kaggle_secret}}],
        "resources": {
            "requests": {"cpu": arguments.cpu, "memory": arguments.memory, "nvidia.com/gpu": 1},
            "limits": {"cpu": arguments.cpu, "memory": arguments.memory, "nvidia.com/gpu": 1},
        },
        "volumeMounts": [{"name": "workspace", "mountPath": "/pvc"}],
    }
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/gpu.product",
                                    "operator": "In",
                                    "values": [arguments.gpu_product],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "containers": [container],
        "volumes": [
            {"name": "workspace", "persistentVolumeClaim": {"claimName": arguments.pvc}}
        ],
    }
    if arguments.mode == "pod":
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": resource_name, "namespace": arguments.namespace, "labels": labels},
            "spec": pod_spec,
        }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": resource_name, "namespace": arguments.namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "parallelism": 1,
            "completions": 1,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--family-id", choices=tuple(FAMILY_MODALITIES))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pvc", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--kaggle-secret", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-product", default="NVIDIA-A10")
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32Gi")
    parser.add_argument("--pilot-labels", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.pilot_labels < 1:
        raise RenderError("pilot label count must be positive")
    yaml.safe_dump(
        render(arguments),
        stream=sys.stdout,
        sort_keys=False,
        explicit_start=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RenderError as error:
        raise SystemExit(f"render error: {error}") from None
