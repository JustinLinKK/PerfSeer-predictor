"""Generate a tracked ConfigMap+Job manifest containing the local v3 source."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "record" / "perfseer_v3_cuda_verifier_job.yaml"


def source_bundle() -> bytes:
    tar_buffer = io.BytesIO()
    included = (
        ROOT / "src" / "perfseer_v3",
        ROOT / "scripts" / "run_perfseer_v3_cuda_verifier.py",
    )
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for path in included:
            if path.is_dir():
                files = sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and "__pycache__" not in child.parts
                )
            else:
                files = [path]
            for file_path in files:
                archive_info = archive.gettarinfo(
                    file_path,
                    arcname=file_path.relative_to(ROOT).as_posix(),
                )
                archive_info.uid = 0
                archive_info.gid = 0
                archive_info.uname = ""
                archive_info.gname = ""
                archive_info.mtime = 0
                with file_path.open("rb") as source:
                    archive.addfile(archive_info, source)
    return gzip.compress(tar_buffer.getvalue(), compresslevel=9, mtime=0)


def build_manifest(namespace: str) -> tuple[dict, dict]:
    bundle = source_bundle()
    digest = hashlib.sha256(bundle).hexdigest()
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "perfseer-v3-cuda-source",
            "namespace": namespace,
            "annotations": {"perfseer.openai/source-bundle-sha256": digest},
        },
        "binaryData": {
            "perfseer-v3-source.tgz": base64.b64encode(bundle).decode("ascii"),
        },
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "perfseer-v3-cuda-verifier",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "perfseer-v3-cuda-verifier",
                "app.kubernetes.io/component": "verification",
            },
            "annotations": {"perfseer.openai/source-bundle-sha256": digest},
        },
        "spec": {
            "backoffLimit": 1,
            "activeDeadlineSeconds": 900,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "perfseer-v3-cuda-verifier",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "initContainers": [
                        {
                            "name": "prepare-source",
                            "image": "alpine/git:2.47.2",
                            "command": ["/bin/sh", "-lc"],
                            "args": [
                                "git clone --branch v2 --single-branch "
                                "https://github.com/JustinLinKK/PerfSeer-predictor.git "
                                "/workspace/repo && "
                                "git -C /workspace/repo checkout "
                                "f1f28e358c134e9acfaab819bddd64fb6360f6a3 && "
                                "tar -xzf /bundle/perfseer-v3-source.tgz -C /workspace/repo"
                            ],
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "source-bundle", "mountPath": "/bundle", "readOnly": True},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "128Mi",
                                    "ephemeral-storage": "256Mi",
                                },
                                "limits": {
                                    "cpu": "1",
                                    "memory": "1Gi",
                                    "ephemeral-storage": "1Gi",
                                },
                            },
                        }
                    ],
                    "containers": [
                        {
                            "name": "verifier",
                            "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
                            "command": ["/bin/bash", "-lc"],
                            "args": [
                                "python -m pip install --quiet pyyaml && "
                                "cd /workspace/repo && "
                                "python scripts/run_perfseer_v3_cuda_verifier.py "
                                "2>&1 | tee /workspace/perfseer_v3_cuda_verifier.log"
                            ],
                            "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}],
                            "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}],
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "8Gi",
                                    "ephemeral-storage": "1Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "4",
                                    "memory": "12Gi",
                                    "ephemeral-storage": "4Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                        }
                    ],
                    "volumes": [
                        {"name": "workspace", "emptyDir": {"sizeLimit": "4Gi"}},
                        {
                            "name": "source-bundle",
                            "configMap": {"name": "perfseer-v3-cuda-source"},
                        },
                    ],
                },
            },
        },
    }
    return config_map, job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="ecepxie")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    documents = build_manifest(args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump_all(documents, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
