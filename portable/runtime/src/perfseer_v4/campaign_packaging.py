"""Verified local build contexts and fail-closed one-A100 manifests."""

import math
import json
import os
from pathlib import Path
import re
import shutil

import yaml

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .campaign import validate_config, source_normalization
from .campaign_data import verify_dataset

BASE_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
OUTPUT_SETUP = """import os
from pathlib import Path
os.umask(0o007)
for name in ('prepared', 'campaign'):
    path = Path('/outputs') / name
    if path.is_symlink():
        raise RuntimeError(f'output directory must not be a symlink: {path}')
    path.mkdir(mode=0o770, exist_ok=True)
    info = path.stat()
    print(f'{path}: uid={info.st_uid} gid={info.st_gid} mode={oct(info.st_mode & 0o7777)}', flush=True)
"""
OUTPUT_CHECK = """import os
from pathlib import Path
from tempfile import TemporaryDirectory
assert os.getuid() == os.getgid() == 1000
for name in ('prepared', 'campaign'):
    with TemporaryDirectory(prefix='.write-check-', dir=Path('/outputs') / name) as directory:
        path = Path(directory) / 'probe'
        path.write_bytes(b'perfseer-output-check')
        path = path.rename(path.with_name('renamed'))
        assert path.read_bytes() == b'perfseer-output-check'
    print(f'/outputs/{name}: UID/GID 1000 write/read/rename/delete passed', flush=True)
"""


def _snapshot(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def package(repo, config, output, *, prepare_on_cluster=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise ValueError("build context must be empty")
    data = Path(config["data"]).resolve()
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("build context and data must be separate")
    native_files = []
    if prepare_on_cluster:
        if config["smoke"]:
            raise ValueError("cluster preparation requires the full native corpus")
        from .campaign_data import native_rows, family_splits
        from .dataset import checked_path, verify
        configurations_path = repo / "labels/A10/model_config.jsonl"
        configurations = [json.loads(line) for line in configurations_path.open()]
        configs = {row["profile_point_id"]: row for row in configurations}
        if len(configs) != 48000 or len(configurations) != len(configs):
            raise ValueError("expected all 48,000 source configurations")
        splits = family_splits(configurations)
        native_files.append(configurations_path)
        for gpu in ("A100", "A10"):
            native_rows(repo, gpu, configs, splits)
            native_files.append(repo / f"labels/{gpu}/labels/labels.jsonl")
        for row in configurations:
            path = checked_path(repo / "labels/A10", row["model_file"])
            if file_sha256(path) != row["source_sha256"]:
                raise ValueError("native workload source hash differs")
            native_files.append(path)
        target = repo / "labels/RTX5090/ready_for_train"
        if verify(target, workers=8)["total_rows"] != 4128:
            raise ValueError("expected the full RTX 5090 transfer corpus")
        native_files.extend(path for path in target.rglob("*") if path.is_file())
    else:
        for gpu in ("A100", "A10", "RTX5090"):
            manifest = verify_dataset(data / gpu)
            if bool(manifest["smoke_rows_per_split"]) != config["smoke"]:
                raise ValueError("production package cannot contain smoke data")
            if gpu != "RTX5090":
                source_normalization(data / gpu)
    output.mkdir(parents=True, exist_ok=True)
    for package_name in ("perfseer_v3", "perfseer_v3.1", "perfseer_v3.2", "perfseer_v4"):
        folder = repo / "src" / package_name
        for path in folder.rglob("*"):
            if not path.is_file() or any(part in {"__pycache__", "dataset_with_label", "checkpoints"} for part in path.parts):
                continue
            if path.suffix in {".py", ".json", ".yaml", ".md"}:
                destination = output / "code" / path.relative_to(repo)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    for path in (repo / "scripts").glob("*.py"):
        destination = output / "code/scripts" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for path in (repo / "tests").glob("test_perfseer_v4*.py"):
        destination = output / "code/tests" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    if prepare_on_cluster:
        for path in native_files:
            _snapshot(path, output / "data" / path.relative_to(repo))
    else:
        for gpu in ("A100", "A10", "RTX5090"):
            for path in (data / gpu).rglob("*"):
                if path.is_file() and "preparation-failure" not in path.name:
                    _snapshot(path, output / "data" / path.relative_to(data))
    dockerfile = repo / "src/perfseer_v4/containers/training/Dockerfile"
    shutil.copy2(dockerfile, output / "Dockerfile")
    packaged_config = {**config, "data": "/outputs/prepared" if prepare_on_cluster else "/opt/campaign/data", "output": "/outputs/campaign"}
    atomic_write(output / "campaign.json", packaged_config)
    files = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    data_bytes = sum(p.stat().st_size for p in (output / "data").rglob("*") if p.is_file())
    # Bound 32 selected models and 32 resumable states using conservative T1
    # parameter/optimizer storage; renderer includes all retained artifacts.
    checkpoint_budget = 64 * 2 ** 30
    preparation_budget = 128 * 2 ** 30 if prepare_on_cluster else 0
    pvc_gib = max(100, math.ceil(1.25 * (data_bytes + preparation_budget + checkpoint_budget) / 2 ** 30))
    scratch_gib = max(16, math.ceil(1.25 * data_bytes / 2 ** 30))
    manifest = {"version": "perfseer_v4_campaign_package_v1", "smoke": config["smoke"], "base_image": BASE_IMAGE,
                "files": files, "data_bytes": data_bytes, "checkpoint_budget_bytes": checkpoint_budget,
                "pvc_gib": pvc_gib, "scratch_gib": scratch_gib}
    if prepare_on_cluster:
        manifest.update(preparation="on_cluster", preparation_budget_bytes=preparation_budget)
    manifest["fingerprint"] = fingerprint(manifest)
    atomic_write(output / "package.json", manifest)
    verify_package(output)
    return {"status": "smoke_package" if config["smoke"] else "verified", "context": str(output),
            "fingerprint": manifest["fingerprint"], "files": len(files), "pvc_gib": pvc_gib, "scratch_gib": scratch_gib}


def verify_package(root):
    root = Path(root)
    manifest = read_json(root / "package.json")
    if manifest["fingerprint"] != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("package manifest changed")
    from .dataset import checked_path
    for relative, sha in manifest["files"].items():
        if file_sha256(checked_path(root, relative)) != sha:
            raise ValueError("build context differs from verified snapshot")
    return manifest


def job_documents(package_manifest, image, *, namespace="ecepxie", name="perfseer-v4-eight-pair-a100"):
    if package_manifest["smoke"]:
        raise ValueError("a smoke package cannot be submitted as the full campaign")
    if not re.fullmatch(r"[a-zA-Z0-9._:/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("published immutable image digest required; tags/local image IDs are not deployable")
    if any(not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", value) or len(value) > 50 for value in (name, namespace)):
        raise ValueError("invalid Kubernetes name/namespace")
    pvc = name + "-output"
    request = {"cpu": "8", "memory": "64Gi", "ephemeral-storage": f'{package_manifest["scratch_gib"]}Gi', "nvidia.com/a100": 1}
    limits = dict(request)
    volumes = [{"name": "outputs", "persistentVolumeClaim": {"claimName": pvc}},
               {"name": "scratch", "emptyDir": {"sizeLimit": f'{package_manifest["scratch_gib"]}Gi'}},
               {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}]
    pod = {"restartPolicy": "Never", "terminationGracePeriodSeconds": 300,
           "nodeSelector": {"kubernetes.io/arch": "amd64"},
           "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [
               {"key": "nvidia.com/gpu.product", "operator": "In", "values": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"]},
               {"key": "nvidia.com/cuda.driver.major", "operator": "Gt", "values": ["569"]}]}]}}},
           "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch"},
           "containers": [{"name": "trainer", "image": image, "imagePullPolicy": "IfNotPresent",
                           "command": ["python", "-u", "/opt/campaign/code/scripts/run_perfseer_v4.py"],
                           "args": ["campaign", "--config", "/opt/campaign/campaign.json", "--resume"],
                           "resources": {"requests": request, "limits": limits},
                           "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}, {"name": "OMP_NUM_THREADS", "value": "2"},
                                   {"name": "MKL_NUM_THREADS", "value": "2"}, {"name": "TMPDIR", "value": "/scratch"},
                                   {"name": "CUDA_CACHE_PATH", "value": "/scratch/cuda-cache"}],
                           "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
                           "volumeMounts": [{"name": "outputs", "mountPath": "/outputs"}, {"name": "scratch", "mountPath": "/scratch"},
                                            {"name": "shm", "mountPath": "/dev/shm"}]}], "volumes": volumes}
    from copy import deepcopy
    setup = {"name": "prepare-output-directories", "image": image, "imagePullPolicy": "IfNotPresent",
             "command": ["python", "-u", "-c", OUTPUT_SETUP],
             "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "100m", "memory": "128Mi"}},
             "securityContext": {"runAsNonRoot": False, "runAsUser": 0, "runAsGroup": 1000,
                                 "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
             "volumeMounts": [{"name": "outputs", "mountPath": "/outputs"}]}
    check = deepcopy(setup)
    check["name"] = "verify-output-access"
    check["command"] = ["python", "-u", "-c", OUTPUT_CHECK]
    check["securityContext"].update(runAsNonRoot=True, runAsUser=1000)
    pod["initContainers"] = [setup, check]
    if package_manifest.get("preparation") == "on_cluster":
        preparation = deepcopy(pod["containers"][0])
        preparation["name"] = "prepare"
        preparation["args"] = ["prepare-campaign-data", "--repo", "/opt/campaign/data", "--output", "/outputs/prepared",
                               "--device", "cuda", "--feature-workers", "2"]
        pod["initContainers"].append(preparation)
    return [{"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": pvc, "namespace": namespace},
             "spec": {"accessModes": ["ReadWriteMany"], "storageClassName": "rook-cephfs",
                      "resources": {"requests": {"storage": f'{package_manifest["pvc_gib"]}Gi'}}}},
            {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": name, "namespace": namespace,
             "annotations": {"perfseer.ai/package-fingerprint": package_manifest["fingerprint"]}},
             "spec": {"backoffLimit": 2, "activeDeadlineSeconds": 604800,
                      "template": {"metadata": {"labels": {"app.kubernetes.io/name": "perfseer-v4-campaign"}}, "spec": pod}}}]


def verify_job(documents):
    pvc, job = documents
    pod = job["spec"]["template"]["spec"]
    trainer = pod["containers"][0]
    if not re.fullmatch(r"[a-zA-Z0-9._:/-]+@sha256:[0-9a-f]{64}", trainer["image"]):
        raise ValueError("published immutable image digest required")
    if job["kind"] != "Job" or job["spec"]["activeDeadlineSeconds"] != 604800 or pod["restartPolicy"] != "Never":
        raise ValueError("finite batch Job required")
    for direction in ("requests", "limits"):
        if trainer["resources"][direction]["nvidia.com/a100"] != 1 or "nvidia.com/gpu" in trainer["resources"][direction]:
            raise ValueError("exactly one dedicated A100 required")
    if trainer["resources"]["requests"] != trainer["resources"]["limits"]:
        raise ValueError("campaign resource requests and limits must match")
    initializers = pod.get("initContainers", [])
    if len(initializers) not in (2, 3):
        raise ValueError("output directory setup and non-root write verification required")
    for initializer, name, uid, code in zip(initializers[:2],
            ("prepare-output-directories", "verify-output-access"), (0, 1000), (OUTPUT_SETUP, OUTPUT_CHECK)):
        expected_security = {"runAsNonRoot": uid != 0, "runAsUser": uid, "runAsGroup": 1000,
                             "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}}
        if (initializer["name"] != name or initializer["image"] != trainer["image"] or
                initializer["command"] != ["python", "-u", "-c", code] or initializer.get("args") or
                initializer["securityContext"] != expected_security or
                initializer["volumeMounts"] != [{"name": "outputs", "mountPath": "/outputs"}] or
                initializer["resources"] != {"requests": {"cpu": "100m", "memory": "128Mi"},
                                            "limits": {"cpu": "100m", "memory": "128Mi"}}):
            raise ValueError("output directory setup and non-root write verification differ")
    for preparation in initializers[2:]:
        if (preparation["image"] != trainer["image"] or preparation["resources"] != trainer["resources"] or
                preparation["args"] != ["prepare-campaign-data", "--repo", "/opt/campaign/data", "--output", "/outputs/prepared",
                                        "--device", "cuda", "--feature-workers", "2"]):
            raise ValueError("full CUDA preparation must use the same verified image and single GPU")
    values = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    if set(values) != {"NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"}:
        raise ValueError("full 80GB A100 affinity required")
    if pvc["metadata"]["namespace"] != job["metadata"]["namespace"] or pod["volumes"][0]["persistentVolumeClaim"]["claimName"] != pvc["metadata"]["name"]:
        raise ValueError("persistent output binding differs")
    if trainer["args"][:1] != ["campaign"] or "--resume" not in trainer["args"]:
        raise ValueError("finite resumable campaign entrypoint required")
    if job["spec"]["backoffLimit"] != 2 or pod["terminationGracePeriodSeconds"] != 300:
        raise ValueError("replacement and graceful checkpoint limits differ")
    if pod["securityContext"].get("runAsNonRoot") is not True or not trainer["securityContext"].get("readOnlyRootFilesystem"):
        raise ValueError("non-root, read-only container required")
    return {"status": "passed", "gpu": "one full 80GB A100", "namespace": job["metadata"]["namespace"]}


def render_job(package_root, image, output, *, namespace="ecepxie", name="perfseer-v4-eight-pair-a100"):
    manifest = verify_package(package_root)
    documents = job_documents(manifest, image, namespace=namespace, name=name)
    result = verify_job(documents)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump_all(documents, sort_keys=False))
    verify_job(list(yaml.safe_load_all(output.read_text())))
    return {**result, "manifest": str(output), "submitted": False}
