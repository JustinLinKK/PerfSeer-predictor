#!/usr/bin/env python3
"""Run the source-first Nautilus workflow and download finished dataset packages."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from run_nautilus_label_sampling_e2e import (
        GPU_PRESETS,
        README_GPU_KEYS,
        classify_failure,
        clean_name,
        pod_is_unschedulable,
    )
except ModuleNotFoundError:
    from scripts.run_nautilus_label_sampling_e2e import (
        GPU_PRESETS,
        README_GPU_KEYS,
        classify_failure,
        clean_name,
        pod_is_unschedulable,
    )


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = ROOT / "nrp_calibration_pack" / "submit_nrp_source_workflow.sh"
K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def run(
    cmd: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        message = f"command timed out after {timeout_seconds} second(s): {' '.join(cmd)}\n{stdout}"
        if check:
            raise RuntimeError(message) from exc
        return subprocess.CompletedProcess(cmd, 124, stdout=message, stderr=None)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def kubectl(
    args: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    request_timeout = os.environ.get("PERFSEER_KUBECTL_REQUEST_TIMEOUT", "30s")
    hard_timeout = float(os.environ.get("PERFSEER_KUBECTL_HARD_TIMEOUT_SECONDS", "180"))
    command = ["kubectl", f"--request-timeout={request_timeout}", *args]
    return run(command, input_text=input_text, check=check, timeout_seconds=hard_timeout)


def validate_kubernetes_name_prefix(value: str, field: str) -> None:
    if not K8S_NAME_RE.match(value):
        raise ValueError(f"{field} must be a lowercase Kubernetes name prefix using only letters, digits, and hyphens")
    if len(value) > 42:
        raise ValueError(f"{field} must be at most 42 characters so workflow suffixes fit Kubernetes names")


def validate_profile_image_reference(image: str, allow_mutable_tag: bool) -> None:
    if "@sha256:" in image:
        return
    if allow_mutable_tag:
        print(
            json.dumps(
                {
                    "event": "mutable_image_tag_allowed",
                    "image": image,
                    "risk": "tag can move between runs; digest pinning is required for byte-identical image reuse",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    raise ValueError(
        "--image must be pinned by digest, for example registry/repo:tag@sha256:<digest>. "
        "Pass --allow-mutable-image-tag only for exploratory runs."
    )


def create_repo_archive(repo_dir: Path, archive_path: Path) -> dict[str, int]:
    include_roots = [
        "AGENTS.md",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "nrp_calibration_pack",
        "scripts",
        "src",
        "doc",
    ]
    excluded_names = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "record",
        "labels",
        "dataset",
        "transfer-learning",
        "nrp_downloads",
    }
    excluded_suffixes = (".pyc", ".pyo", ".tar", ".tar.gz", ".zip")
    file_count = 0
    byte_count = 0

    def should_skip(path: Path) -> bool:
        relative = path.relative_to(repo_dir)
        if relative.parts[:2] == ("nrp_calibration_pack", "models"):
            return True
        if any(part in excluded_names for part in relative.parts):
            return True
        return path.is_file() and path.name.endswith(excluded_suffixes)

    with tarfile.open(archive_path, "w:gz") as tar:
        for root_name in include_roots:
            root_path = repo_dir / root_name
            if not root_path.exists():
                continue
            paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
            for path in paths:
                if should_skip(path) or path.is_dir():
                    continue
                tar.add(path, arcname=str(path.relative_to(repo_dir)))
                file_count += 1
                byte_count += path.stat().st_size
    if file_count <= 0:
        raise RuntimeError(f"repo archive is empty: {repo_dir}")
    return {"files": file_count, "bytes": byte_count}


@dataclass
class ProfileShardAttempt:
    shard_index: int
    gpu_key: str
    job_name: str
    attempt_number: int
    submitted_at: float
    status: str = "pending"


@dataclass(frozen=True)
class ProfileGpuPartition:
    gpu_key: str
    shard_start: int
    shard_count: int


@dataclass
class ProfileSwitchState:
    gpu_keys: list[str]
    active_limit: int
    pending_timeout_seconds: int
    max_retries_per_shard: int
    next_gpu_index: int = 0
    retry_counts: dict[int, int] = field(default_factory=dict)
    blocked_nodes_by_gpu: dict[str, set[str]] = field(default_factory=dict)
    warmed_nodes_by_gpu: dict[str, set[str]] = field(default_factory=dict)

    def next_gpu(self) -> str:
        gpu_key = self.gpu_keys[self.next_gpu_index % len(self.gpu_keys)]
        self.next_gpu_index += 1
        return gpu_key

    def next_gpu_excluding(self, excluded_gpu_key: str) -> str:
        if len(self.gpu_keys) <= 1:
            return excluded_gpu_key
        for _ in range(len(self.gpu_keys)):
            gpu_key = self.next_gpu()
            if gpu_key != excluded_gpu_key:
                return gpu_key
        return excluded_gpu_key

    def can_retry(self, shard_index: int) -> bool:
        return self.retry_counts.get(shard_index, 0) < self.max_retries_per_shard

    def count_retry(self, shard_index: int) -> int:
        next_count = self.retry_counts.get(shard_index, 0) + 1
        self.retry_counts[shard_index] = next_count
        return next_count

    def block_node_for_gpu(self, gpu_key: str, node_name: str) -> None:
        if node_name:
            self.blocked_nodes_by_gpu.setdefault(gpu_key, set()).add(node_name)


def parse_gpus(value: str) -> list[str]:
    if value.strip().lower() == "all-readme":
        return list(README_GPU_KEYS)
    gpu_keys = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
    if not gpu_keys:
        raise ValueError("--gpus must contain at least one GPU preset")
    if len(set(gpu_keys)) != len(gpu_keys):
        raise ValueError("--gpus must contain distinct GPU presets")
    unknown = [key for key in gpu_keys if key not in GPU_PRESETS]
    if unknown:
        raise ValueError(f"unknown GPU preset(s): {', '.join(unknown)}")
    return gpu_keys


def partition_profile_shards(total_shards: int, gpu_keys: list[str]) -> list[ProfileGpuPartition]:
    if len(gpu_keys) != 4:
        raise ValueError("--gpus must specify exactly four GPU presets for partitioned profiling")
    if total_shards < len(gpu_keys):
        raise ValueError("--completions must be at least 4 so each GPU preset receives one shard")
    base = total_shards // len(gpu_keys)
    remainder = total_shards % len(gpu_keys)
    partitions: list[ProfileGpuPartition] = []
    offset = 0
    for index, gpu_key in enumerate(gpu_keys):
        count = base + (1 if index < remainder else 0)
        partitions.append(ProfileGpuPartition(gpu_key=gpu_key, shard_start=offset, shard_count=count))
        offset += count
    return partitions


def gpu_affinity(
    products: list[str],
    blocked_nodes: set[str] | None = None,
    allowed_nodes: set[str] | None = None,
) -> str:
    product_values = "\n".join(f"                      - {product}" for product in products)
    blocked_nodes = blocked_nodes or set()
    blocked_expression = ""
    if blocked_nodes:
        blocked_values = "\n".join(f"                      - {node}" for node in sorted(blocked_nodes))
        blocked_expression = f"""
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
{blocked_values}"""
    allowed_expression = ""
    allowed_nodes = allowed_nodes or set()
    if allowed_nodes:
        allowed_values = "\n".join(f"                      - {node}" for node in sorted(allowed_nodes))
        allowed_expression = f"""
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
{allowed_values}"""
    return f"""
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
{product_values}{blocked_expression}{allowed_expression}"""


def profile_shard_job_name(args: argparse.Namespace, shard_index: int, gpu_key: str, attempt_number: int) -> str:
    return clean_name(f"{args.job_prefix}-profile-s{shard_index:04d}-{gpu_key}-a{attempt_number}")


def profile_partition_job_name(args: argparse.Namespace, partition: ProfileGpuPartition) -> str:
    return clean_name(f"{args.job_prefix}-profile-{partition.gpu_key}")


def profile_partition_job_yaml(args: argparse.Namespace, partition: ProfileGpuPartition) -> str:
    preset = GPU_PRESETS[partition.gpu_key]
    name = profile_partition_job_name(args, partition)
    pack_dir = f"{args.workflow_dir.rstrip('/')}/pack"
    results_dir = f"{args.workflow_dir.rstrip('/')}/results/{args.hardware_id}"
    profile_dataset_dir = f"{pack_dir}/profile_datasets"
    partition_hardware_id = f"{args.hardware_id}_{partition.gpu_key}"
    bootstrap = f"{args.bootstrap_command} && " if args.bootstrap_command else ""
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: profile-labels
    perfseer-gpu-key: {partition.gpu_key}
    perfseer-shard-start: "{partition.shard_start}"
    perfseer-shard-count: "{partition.shard_count}"
spec:
  completionMode: Indexed
  completions: {partition.shard_count}
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: profile-labels
        perfseer-gpu-key: {partition.gpu_key}
    spec:{gpu_affinity(preset["products"])}
      restartPolicy: Never
      containers:
      - name: profile-labels
        image: {args.image}
        imagePullPolicy: IfNotPresent
        workingDir: {args.repo_dir}
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        command: ["/bin/bash", "-lc"]
        args:
        - >
          set -euo pipefail &&
          GLOBAL_SHARD_INDEX=$(({partition.shard_start} + ${{JOB_COMPLETION_INDEX:-0}})) &&
          {bootstrap}
          python nrp_calibration_pack/profile/run_profile.py
          --manifest {pack_dir}/manifest/subset_manifest.jsonl
          --models-dir {pack_dir}/models
          --output-dir {results_dir}
          --hardware-id {partition_hardware_id}
          --precision-sweep {args.profile_precision_sweep}
          --profile-dataset-dir {profile_dataset_dir}
          --device cuda
          --warmup {args.warmup}
          --infer-repeats {args.infer_repeats}
          --train-repeats {args.train_repeats}
          --sample-interval {args.sample_interval}
          --optimizer {args.optimizer}
          --sm-occupancy-source {args.sm_occupancy_source}
          --num-shards {args.completions}
          --shard-index ${{GLOBAL_SHARD_INDEX}}
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            {preset["resource"]}: "1"
          limits:
            cpu: "4"
            memory: "16Gi"
            {preset["resource"]}: "1"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: {args.pvc}
"""


def profile_shard_job_yaml(
    args: argparse.Namespace,
    shard_index: int,
    gpu_key: str,
    attempt_number: int,
    blocked_nodes: set[str] | None = None,
    warmed_nodes: set[str] | None = None,
) -> str:
    preset = GPU_PRESETS[gpu_key]
    name = profile_shard_job_name(args, shard_index, gpu_key, attempt_number)
    pack_dir = f"{args.workflow_dir.rstrip('/')}/pack"
    results_dir = f"{args.workflow_dir.rstrip('/')}/results/{args.hardware_id}"
    profile_dataset_dir = f"{pack_dir}/profile_datasets"
    bootstrap = f"{args.bootstrap_command} && " if args.bootstrap_command else ""
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: profile-labels
    perfseer-shard-index: "{shard_index}"
    perfseer-gpu-key: {gpu_key}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: profile-labels
        perfseer-shard-index: "{shard_index}"
        perfseer-gpu-key: {gpu_key}
    spec:{gpu_affinity(preset["products"], blocked_nodes, warmed_nodes)}
      restartPolicy: Never
      containers:
      - name: profile-labels
        image: {args.image}
        imagePullPolicy: IfNotPresent
        workingDir: {args.repo_dir}
        command: ["/bin/bash", "-lc"]
        args:
        - >
          set -euo pipefail &&
          {bootstrap}
          python nrp_calibration_pack/profile/run_profile.py
          --manifest {pack_dir}/manifest/subset_manifest.jsonl
          --models-dir {pack_dir}/models
          --output-dir {results_dir}
          --hardware-id {args.hardware_id}
          --precision-sweep {args.profile_precision_sweep}
          --profile-dataset-dir {profile_dataset_dir}
          --device cuda
          --warmup {args.warmup}
          --infer-repeats {args.infer_repeats}
          --train-repeats {args.train_repeats}
          --sample-interval {args.sample_interval}
          --optimizer {args.optimizer}
          --sm-occupancy-source {args.sm_occupancy_source}
          --num-shards {args.completions}
          --shard-index {shard_index}
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            {preset["resource"]}: "1"
          limits:
            cpu: "4"
            memory: "16Gi"
            {preset["resource"]}: "1"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: {args.pvc}
"""


def retry_gpu_for_profile_failure(
    state: ProfileSwitchState,
    attempt: ProfileShardAttempt,
    reason: str,
    node_name: str,
) -> str | None:
    if reason in {"cuda_initialization_failure", "nvidia_smi_failure", "hardware_failure"}:
        state.block_node_for_gpu(attempt.gpu_key, node_name)
        return attempt.gpu_key
    if reason in {"unschedulable", "pending_timeout"}:
        return state.next_gpu_excluding(attempt.gpu_key)
    return None


def image_warmup_job_name(args: argparse.Namespace, gpu_key: str, index: int) -> str:
    return clean_name(f"{args.job_prefix}-image-warm-{gpu_key}-{index}")


def image_warmup_job_yaml(args: argparse.Namespace, gpu_key: str, index: int) -> str:
    preset = GPU_PRESETS[gpu_key]
    name = image_warmup_job_name(args, gpu_key, index)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: image-warmup
    perfseer-gpu-key: {gpu_key}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: image-warmup
        perfseer-gpu-key: {gpu_key}
    spec:{gpu_affinity(preset["products"])}
      restartPolicy: Never
      containers:
      - name: image-warmup
        image: {args.image}
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-lc"]
        args:
        - >
          set -euo pipefail &&
          nvidia-smi -L &&
          python -c "import json, torch; print(json.dumps({{'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}}), flush=True)"
        resources:
          requests:
            cpu: "1"
            memory: "4Gi"
            {preset["resource"]}: "1"
          limits:
            cpu: "1"
            memory: "4Gi"
            {preset["resource"]}: "1"
"""


def wait_for_named_job(namespace: str, name: str, timeout_seconds: int, poll_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        data = job_json(namespace, name)
        if data is None:
            time.sleep(max(poll_seconds, 1))
            continue
        status = data.get("status", {})
        if int(status.get("succeeded", 0)) >= int(data.get("spec", {}).get("completions", 1)):
            print_diagnostics(namespace, name)
            return
        if int(status.get("failed", 0)) > 0:
            print_diagnostics(namespace, name)
            raise RuntimeError(f"job failed: {name}")
        time.sleep(max(poll_seconds, 1))
    print_diagnostics(namespace, name)
    raise TimeoutError(f"timed out waiting for {name}")


def completed_pod_for_job(namespace: str, job: str) -> dict[str, Any]:
    pods = pod_json(namespace, job).get("items", [])
    if not pods:
        raise RuntimeError(f"no pod found for job {job}")
    pods.sort(key=lambda pod: pod.get("metadata", {}).get("creationTimestamp", ""))
    return pods[-1]


def warm_profile_image_cache(args: argparse.Namespace) -> dict[str, set[str]]:
    if args.skip_image_warmup:
        return {}
    gpu_keys = parse_gpus(args.gpus)
    warmed_nodes: dict[str, set[str]] = {gpu_key: set() for gpu_key in gpu_keys}
    submitted: list[tuple[str, str]] = []
    for gpu_key in gpu_keys:
        for index in range(max(args.image_warmup_per_gpu, 1)):
            name = image_warmup_job_name(args, gpu_key, index)
            kubectl(["delete", "job", name, "-n", args.namespace, "--ignore-not-found=true"], check=False)
            kubectl(["apply", "-f", "-"], input_text=image_warmup_job_yaml(args, gpu_key, index))
            print_diagnostics(args.namespace, name)
            submitted.append((gpu_key, name))
    for gpu_key, name in submitted:
        wait_for_named_job(args.namespace, name, args.timeout_seconds, args.poll_seconds)
        pod = completed_pod_for_job(args.namespace, name)
        node_name = str(pod.get("spec", {}).get("nodeName") or "")
        statuses = pod.get("status", {}).get("containerStatuses", [])
        image_id = statuses[0].get("imageID", "") if statuses else ""
        if node_name:
            warmed_nodes.setdefault(gpu_key, set()).add(node_name)
        print(
            json.dumps(
                {
                    "event": "image_warmup_complete",
                    "gpu_key": gpu_key,
                    "image": args.image,
                    "image_id": image_id,
                    "job": name,
                    "node": node_name,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return warmed_nodes


def pvc_utility_pod_manifest(args: argparse.Namespace, pod_name: str, stage: str) -> str:
    image = args.utility_image
    return f"""
apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: {stage}
spec:
  restartPolicy: Never
  containers:
  - name: {stage}
    image: {image}
    imagePullPolicy: IfNotPresent
    command: ["/bin/sh", "-lc"]
    args:
    - |
      set -euo pipefail
      sleep infinity
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    volumeMounts:
    - name: output
      mountPath: /mnt/output
  volumes:
  - name: output
    persistentVolumeClaim:
      claimName: {args.pvc}
""".strip()


def wait_for_pod_running(namespace: str, pod_name: str, timeout_seconds: int, poll_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = kubectl(["get", "pod", pod_name, "-n", namespace, "-o", "json"], check=False)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            phase = data.get("status", {}).get("phase", "")
            if phase == "Running":
                return
            if phase in {"Failed", "Succeeded"}:
                raise RuntimeError(f"staging pod ended before repository copy: {pod_name} phase={phase}")
        time.sleep(max(poll_seconds, 1))
    raise TimeoutError(f"timed out waiting for staging pod: {pod_name}")


def print_pod_diagnostics(namespace: str, pod_name: str) -> None:
    for command in (
        ["get", "pod", pod_name, "-n", namespace, "-o", "wide"],
        ["describe", "pod", pod_name, "-n", namespace],
        ["logs", pod_name, "-n", namespace, "--all-containers=true", "--tail=120"],
        ["get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ):
        proc = kubectl(command, check=False)
        print(f"\n$ kubectl {' '.join(command)}\n{proc.stdout}")


def stage_local_repo(args: argparse.Namespace) -> None:
    repo_dir = Path(args.local_repo_dir).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"local repo does not exist: {repo_dir}")
    pod_name = args.staging_pod_name or f"{args.job_prefix}-repo-stage"
    staged_repo_dir = args.staged_repo_dir or f"{args.workflow_dir.rstrip('/')}/repo"
    args.repo_dir = staged_repo_dir

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "perfseer_repo.tar.gz"
        archive_report = create_repo_archive(repo_dir, archive_path)
        kubectl(["delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found=true", "--wait=true"], check=False)
        manifest_path = Path(tmp) / "repo_stage_pod.yaml"
        manifest_path.write_text(pvc_utility_pod_manifest(args, pod_name, "repo-stage") + "\n", encoding="utf-8")
        print(kubectl(["apply", "-f", str(manifest_path)]).stdout, end="")
        print_pod_diagnostics(args.namespace, pod_name)
        wait_for_pod_running(args.namespace, pod_name, args.stage_timeout_seconds, args.poll_seconds)
        run(["kubectl", "cp", str(archive_path), f"{args.namespace}/{pod_name}:/tmp/perfseer_repo.tar.gz"])
        parent_dir = str(Path(staged_repo_dir).parent)
        extract_cmd = (
            "set -eu; "
            f"rm -rf {shell_quote(staged_repo_dir)}; "
            f"mkdir -p {shell_quote(parent_dir)} {shell_quote(staged_repo_dir)}; "
            f"tar -xzf /tmp/perfseer_repo.tar.gz -C {shell_quote(staged_repo_dir)}; "
            f"test -f {shell_quote(staged_repo_dir + '/nrp_calibration_pack/submit_nrp_source_workflow.sh')}; "
            f"test -f {shell_quote(staged_repo_dir + '/scripts/rebuild_source_tar_dataset.py')}; "
            f"find {shell_quote(staged_repo_dir)} -maxdepth 2 -type f | sort | head -40"
        )
        print(run(["kubectl", "exec", "-n", args.namespace, pod_name, "--", "/bin/sh", "-lc", extract_cmd]).stdout)
        print(json.dumps({"staged_repo_dir": staged_repo_dir, "repo_archive": archive_report}, sort_keys=True))
        if not args.keep_staging_pod:
            kubectl(["delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found=true", "--wait=false"], check=False)


def start_monitor(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, Path | None]:
    if args.no_monitor:
        return None, None
    record_dir = ROOT / "record"
    record_dir.mkdir(exist_ok=True)
    log_path = Path(args.monitor_log) if args.monitor_log else record_dir / (
        f"nrp_source_workflow_{args.job_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_monitor.log"
    )
    local_output = str(Path(args.local_output_dir).resolve())
    monitor_timeout = max(int(args.kubectl_hard_timeout_seconds), 1)
    request_timeout = shell_quote(args.kubectl_request_timeout)
    command = f"""
run_with_timeout() {{
  local seconds="$1"
  shift
  "$@" &
  local child="$!"
  (
    sleep "$seconds"
    kill "$child" >/dev/null 2>&1 || true
  ) &
  local watchdog="$!"
  wait "$child"
  local status="$?"
  kill "$watchdog" >/dev/null 2>&1 || true
  wait "$watchdog" 2>/dev/null || true
  return "$status"
}}
while true; do
  date
  run_with_timeout {monitor_timeout} kubectl --request-timeout={request_timeout} get jobs -n {shell_quote(args.namespace)} -l app={shell_quote(args.job_prefix)} -o wide || true
  run_with_timeout {monitor_timeout} kubectl --request-timeout={request_timeout} get pods -n {shell_quote(args.namespace)} -l app={shell_quote(args.job_prefix)} -o wide || true
  run_with_timeout {monitor_timeout} kubectl --request-timeout={request_timeout} get events -n {shell_quote(args.namespace)} --sort-by=.lastTimestamp | tail -80 || true
  run_with_timeout {monitor_timeout} kubectl --request-timeout={request_timeout} logs -n {shell_quote(args.namespace)} -l app={shell_quote(args.job_prefix)} --all-containers=true --tail=120 || true
  find {shell_quote(local_output)} -maxdepth 2 -type f -print -exec ls -lh {{}} \\; 2>/dev/null || true
  sleep {max(args.monitor_interval_seconds, 1)}
done
""".strip()
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(["nohup", "bash", "-lc", command], stdout=handle, stderr=subprocess.STDOUT, text=True)
    print(f"monitor_log={log_path}", flush=True)
    return proc, log_path


def submit_stage(args: argparse.Namespace, stage: str) -> None:
    cmd = [
        str(SUBMIT_SCRIPT),
        "--namespace",
        args.namespace,
        "--image",
        args.image,
        "--pvc",
        args.pvc,
        "--job-prefix",
        args.job_prefix,
        "--stage",
        stage,
        "--workflow-dir",
        args.workflow_dir,
        "--hardware-id",
        args.hardware_id,
        "--subset-size",
        str(args.subset_size),
        "--seed",
        str(args.seed),
        "--generation-workers",
        str(args.generation_workers),
        "--low-precision-focus",
        args.low_precision_focus,
        "--parallelism",
        str(args.parallelism),
        "--completions",
        str(args.completions),
        "--warmup",
        str(args.warmup),
        "--infer-repeats",
        str(args.infer_repeats),
        "--train-repeats",
        str(args.train_repeats),
        "--sample-interval",
        str(args.sample_interval),
        "--optimizer",
        args.optimizer,
        "--sm-occupancy-source",
        args.sm_occupancy_source,
    ]
    if args.gpu_product:
        cmd.extend(["--gpu-product", args.gpu_product])
    if args.gpu_resource:
        cmd.extend(["--gpu-resource", args.gpu_resource])
    if args.repo_dir:
        cmd.extend(["--repo-dir", args.repo_dir])
    if args.bootstrap_command:
        cmd.extend(["--bootstrap-command", args.bootstrap_command])
    print(run(cmd).stdout, end="")


def job_name(args: argparse.Namespace, stage: str) -> str:
    suffix = {
        "prepare": "prepare-sources",
        "profile": "profile-labels",
        "package": "package-results",
    }[stage]
    return f"{args.job_prefix}-{suffix}"


def job_json(namespace: str, name: str) -> dict[str, Any] | None:
    proc = kubectl(["get", "job", name, "-n", namespace, "-o", "json"], check=False)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def pod_json(namespace: str, job: str) -> dict[str, Any]:
    proc = kubectl(["get", "pods", "-n", namespace, "-l", f"job-name={job}", "-o", "json"])
    return json.loads(proc.stdout)


def first_pod_name(namespace: str, job: str) -> str:
    pods = pod_json(namespace, job).get("items", [])
    if not pods:
        raise RuntimeError(f"no pod found for job {job}")
    pods.sort(key=lambda pod: pod.get("metadata", {}).get("creationTimestamp", ""))
    return str(pods[-1]["metadata"]["name"])


def print_diagnostics(namespace: str, job: str) -> None:
    for command in (
        ["get", "job", job, "-n", namespace, "-o", "wide"],
        ["get", "pods", "-n", namespace, "-l", f"job-name={job}", "-o", "wide"],
        ["describe", "pod", "-n", namespace, "-l", f"job-name={job}"],
        ["logs", "-n", namespace, f"job/{job}", "--all-containers=true", "--tail=120"],
        ["get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ):
        proc = kubectl(command, check=False)
        print(f"\n$ kubectl {' '.join(command)}\n{proc.stdout}")


def profile_jobs_json(args: argparse.Namespace) -> list[dict[str, Any]]:
    proc = kubectl(
        ["get", "jobs", "-n", args.namespace, "-l", f"app={args.job_prefix},stage=profile-labels", "-o", "json"],
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    return list(json.loads(proc.stdout).get("items", []))


def profile_pods_json(args: argparse.Namespace) -> list[dict[str, Any]]:
    proc = kubectl(
        ["get", "pods", "-n", args.namespace, "-l", f"app={args.job_prefix},stage=profile-labels", "-o", "json"],
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    return list(json.loads(proc.stdout).get("items", []))


def pod_logs(namespace: str, job: str) -> str:
    return kubectl(["logs", "-n", namespace, f"job/{job}", "--all-containers=true", "--tail=240"], check=False).stdout


def verify_same_workflow_image(args: argparse.Namespace) -> dict[str, Any]:
    proc = kubectl(["get", "jobs", "-n", args.namespace, "-l", f"app={args.job_prefix}", "-o", "json"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"cannot read workflow jobs for image verification: {proc.stdout}")
    checked_stages = {"image-warmup", "prepare-sources", "profile-labels", "package-results"}
    images: dict[str, str] = {}
    for job in json.loads(proc.stdout).get("items", []):
        metadata = job.get("metadata", {})
        stage = str(metadata.get("labels", {}).get("stage", ""))
        if stage not in checked_stages:
            continue
        containers = job.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        for container in containers:
            images[str(metadata.get("name", ""))] = str(container.get("image", ""))
    bad = {name: image for name, image in images.items() if image != args.image}
    if bad:
        raise RuntimeError(
            "workflow image mismatch: "
            + json.dumps({"expected": args.image, "mismatches": bad}, sort_keys=True)
        )
    result = {"event": "same_image_verified", "image": args.image, "jobs": len(images)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def submit_profile_shard(
    args: argparse.Namespace,
    state: ProfileSwitchState,
    shard_index: int,
    gpu_key: str,
    attempt_number: int,
) -> ProfileShardAttempt:
    job_name = profile_shard_job_name(args, shard_index, gpu_key, attempt_number)
    yaml_text = profile_shard_job_yaml(
        args,
        shard_index,
        gpu_key,
        attempt_number,
        state.blocked_nodes_by_gpu.get(gpu_key, set()),
        state.warmed_nodes_by_gpu.get(gpu_key, set()) if args.reuse_warmed_image_nodes else set(),
    )
    kubectl(["apply", "-f", "-"], input_text=yaml_text)
    print_diagnostics(args.namespace, job_name)
    return ProfileShardAttempt(
        shard_index=shard_index,
        gpu_key=gpu_key,
        job_name=job_name,
        attempt_number=attempt_number,
        submitted_at=time.time(),
    )


def delete_profile_job(args: argparse.Namespace, job_name_value: str) -> None:
    kubectl(["delete", "job", job_name_value, "-n", args.namespace, "--ignore-not-found=true"], check=False)


def submit_profile_partition(args: argparse.Namespace, partition: ProfileGpuPartition) -> str:
    job_name_value = profile_partition_job_name(args, partition)
    kubectl(["delete", "job", job_name_value, "-n", args.namespace, "--ignore-not-found=true"], check=False)
    kubectl(["apply", "-f", "-"], input_text=profile_partition_job_yaml(args, partition))
    print_diagnostics(args.namespace, job_name_value)
    return job_name_value


def run_partitioned_profile_jobs(args: argparse.Namespace) -> dict[str, Any]:
    gpu_keys = parse_gpus(args.gpus)
    partitions = partition_profile_shards(args.completions, gpu_keys)
    jobs = {submit_profile_partition(args, partition): partition for partition in partitions}
    started_at = time.time()
    print(
        json.dumps(
            {
                "event": "profile_partition_start",
                "gpus": gpu_keys,
                "jobs": len(jobs),
                "partitions": [
                    {
                        "gpu_key": partition.gpu_key,
                        "shard_count": partition.shard_count,
                        "shard_start": partition.shard_start,
                    }
                    for partition in partitions
                ],
                "total_shards": args.completions,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    completed: set[str] = set()
    deadline = started_at + args.timeout_seconds
    while time.time() < deadline:
        for job_name_value, partition in jobs.items():
            if job_name_value in completed:
                continue
            data = job_json(args.namespace, job_name_value)
            if data is None:
                continue
            status = data.get("status", {})
            if int(status.get("succeeded", 0)) >= partition.shard_count:
                print_diagnostics(args.namespace, job_name_value)
                completed.add(job_name_value)
                continue
            if int(status.get("failed", 0)) > 0:
                print_diagnostics(args.namespace, job_name_value)
                raise RuntimeError(f"profile partition job failed: {job_name_value}")
        elapsed = max(time.time() - started_at, 1e-9)
        summary = {
            "event": "profile_partition_progress",
            "completed_jobs": len(completed),
            "elapsed_seconds": round(elapsed, 2),
            "total_jobs": len(jobs),
            "total_shards": args.completions,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if len(completed) == len(jobs):
            return summary
        time.sleep(max(args.poll_seconds, 1))
    raise TimeoutError(f"timed out waiting for {len(jobs)} profile partition jobs after {args.timeout_seconds} seconds")


def run_profile_switcher(args: argparse.Namespace) -> dict[str, Any]:
    gpu_keys = parse_gpus(args.gpus)
    state = ProfileSwitchState(
        gpu_keys=gpu_keys,
        active_limit=max(1, min(args.active_gpus, args.parallelism, args.completions)),
        pending_timeout_seconds=args.pending_timeout_seconds,
        max_retries_per_shard=args.max_retries_per_shard,
        warmed_nodes_by_gpu=getattr(args, "warmed_nodes_by_gpu", {}) or {},
    )
    pending_shards = list(range(args.completions))
    running: dict[int, ProfileShardAttempt] = {}
    succeeded: set[int] = set()
    started_at = time.time()
    deadline = started_at + args.timeout_seconds
    print(
        json.dumps(
            {
                "event": "profile_switcher_start",
                "active_gpus": state.active_limit,
                "gpus": gpu_keys,
                "shards": args.completions,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    while time.time() < deadline:
        while pending_shards and len(running) < state.active_limit:
            shard_index = pending_shards.pop(0)
            gpu_key = state.next_gpu()
            attempt_number = state.retry_counts.get(shard_index, 0)
            running[shard_index] = submit_profile_shard(args, state, shard_index, gpu_key, attempt_number)

        jobs_by_name = {job.get("metadata", {}).get("name", ""): job for job in profile_jobs_json(args)}
        pods_by_job: dict[str, list[dict[str, Any]]] = {}
        for pod in profile_pods_json(args):
            for owner in pod.get("metadata", {}).get("ownerReferences", []):
                if owner.get("kind") == "Job":
                    pods_by_job.setdefault(owner.get("name", ""), []).append(pod)

        now = time.time()
        for shard_index, attempt in list(running.items()):
            job = jobs_by_name.get(attempt.job_name)
            pods = pods_by_job.get(attempt.job_name, [])
            pod_phase = pods[0].get("status", {}).get("phase", "Pending") if pods else "Pending"
            unschedulable = any(pod_is_unschedulable(pod) for pod in pods)
            if pod_phase in {"Running", "Succeeded"}:
                attempt.status = "running"
            if job and int(job.get("status", {}).get("succeeded", 0)) >= 1:
                succeeded.add(shard_index)
                running.pop(shard_index, None)
                continue

            failed = job and int(job.get("status", {}).get("failed", 0)) > 0
            pending_too_long = now - attempt.submitted_at > state.pending_timeout_seconds
            should_switch = bool(failed or (pending_too_long and (pod_phase == "Pending" or unschedulable)))
            if not should_switch:
                continue

            logs = pod_logs(args.namespace, attempt.job_name)
            reason = classify_failure(logs) if failed else ("unschedulable" if unschedulable else "pending_timeout")
            node_name = pods[0].get("spec", {}).get("nodeName", "") if pods else ""
            delete_profile_job(args, attempt.job_name)
            running.pop(shard_index, None)
            retry_gpu = retry_gpu_for_profile_failure(state, attempt, reason, node_name)
            if retry_gpu is None:
                print_diagnostics(args.namespace, attempt.job_name)
                raise RuntimeError(f"profile shard {shard_index} failed with non-retryable reason: {reason}")
            if not state.can_retry(shard_index):
                print_diagnostics(args.namespace, attempt.job_name)
                raise RuntimeError(
                    f"profile shard {shard_index} failed after {state.max_retries_per_shard} retries: {reason}"
                )
            retry_number = state.count_retry(shard_index)
            print(
                json.dumps(
                    {
                        "event": "profile_shard_retry",
                        "failed_gpu": attempt.gpu_key,
                        "node": node_name,
                        "reason": reason,
                        "retry": retry_number,
                        "retry_gpu": retry_gpu,
                        "shard_index": shard_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            running[shard_index] = submit_profile_shard(args, state, shard_index, retry_gpu, retry_number)

        elapsed = max(time.time() - started_at, 1e-9)
        summary = {
            "event": "profile_switcher_progress",
            "active": len(running),
            "completed_shards": len(succeeded),
            "elapsed_seconds": round(elapsed, 2),
            "pending_shards": len(pending_shards),
            "shards_per_hour": round(len(succeeded) / elapsed * 3600, 2),
            "total_shards": args.completions,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if len(succeeded) >= args.completions:
            return summary
        time.sleep(max(args.poll_seconds, 1))

    raise TimeoutError(f"timed out waiting for profile switcher after {args.timeout_seconds} seconds")


def wait_for_job(args: argparse.Namespace, stage: str) -> None:
    name = job_name(args, stage)
    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        data = job_json(args.namespace, name)
        if data is None:
            time.sleep(args.poll_seconds)
            continue
        status = data.get("status", {})
        if int(status.get("succeeded", 0)) >= int(data.get("spec", {}).get("completions", 1)):
            print_diagnostics(args.namespace, name)
            return
        if int(status.get("failed", 0)) > 0:
            print_diagnostics(args.namespace, name)
            raise RuntimeError(f"job failed: {name}")
        time.sleep(args.poll_seconds)
    print_diagnostics(args.namespace, name)
    raise TimeoutError(f"timed out waiting for {name}")


def ensure_download_pod(args: argparse.Namespace) -> str:
    pod_name = args.download_pod_name or f"{args.job_prefix}-download"
    proc = kubectl(["get", "pod", pod_name, "-n", args.namespace, "-o", "json"], check=False)
    if proc.returncode == 0:
        phase = json.loads(proc.stdout).get("status", {}).get("phase", "")
        if phase != "Running":
            kubectl(["delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found=true", "--wait=true"], check=False)
        else:
            return pod_name
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "download_pod.yaml"
        manifest_path.write_text(pvc_utility_pod_manifest(args, pod_name, "download") + "\n", encoding="utf-8")
        print(kubectl(["apply", "-f", str(manifest_path)]).stdout, end="")
    print_pod_diagnostics(args.namespace, pod_name)
    wait_for_pod_running(args.namespace, pod_name, args.stage_timeout_seconds, args.poll_seconds)
    return pod_name


def copy_from_workflow_pvc(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    pod = ensure_download_pod(args)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run(["kubectl", "cp", f"{args.namespace}/{pod}:{remote_path}", str(local_path)])


def normalize_tar_name(name: str) -> str:
    return name.lstrip("./")


def validate_main_branch_label_text(text: str, label_name: str) -> None:
    try:
        label = ast.literal_eval(text.strip())
    except Exception as exc:
        raise ValueError(f"{label_name} is not a Python literal label dict: {exc}") from exc
    if not isinstance(label, dict):
        raise ValueError(f"{label_name} is not a label dict")
    for phase in ("train", "infer"):
        raw = label.get(phase)
        if not isinstance(raw, str):
            raise ValueError(f"{label_name} missing string field {phase!r}")
        fields = raw.split("|")
        if len(fields) != 7:
            raise ValueError(f"{label_name} {phase!r} has {len(fields)} fields, expected 7")
        for field in fields:
            float(field)


def verify_dataset_tar(path: Path) -> dict[str, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"dataset tarball is missing or empty: {path}")
    graph_count = 0
    label_count = 0
    metadata_count = 0
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            name = normalize_tar_name(member.name)
            if member.isfile() and name.startswith("cg/cg/") and name.endswith(".pkl"):
                graph_count += 1
            elif member.isfile() and name.startswith("label/label/") and name.endswith(".txt"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read label member: {member.name}")
                validate_main_branch_label_text(extracted.read().decode("utf-8"), member.name)
                label_count += 1
            elif member.isfile() and name == "label/precision_metadata.jsonl":
                metadata_count += 1
    if graph_count <= 0:
        raise ValueError(f"dataset tarball has no cg/cg/*.pkl graphs: {path}")
    if label_count <= 0:
        raise ValueError(f"dataset tarball has no label/label/*.txt labels: {path}")
    if metadata_count != 1:
        raise ValueError(f"dataset tarball must contain label/precision_metadata.jsonl exactly once: {path}")
    return {"graphs": graph_count, "labels": label_count, "metadata_files": metadata_count}


def verify_source_tar(path: Path) -> dict[str, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"source tarball is missing or empty: {path}")
    models = 0
    results = 0
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            name = normalize_tar_name(member.name)
            if member.isfile() and name.startswith("pack/models/") and name.endswith(".py") and not name.endswith("/__init__.py"):
                models += 1
            elif member.isfile() and name.startswith("results/") and name.endswith(".jsonl"):
                results += 1
    if models <= 0:
        raise ValueError(f"source tarball has no pack/models/*.py files: {path}")
    if results <= 0:
        raise ValueError(f"source tarball has no results/*.jsonl files: {path}")
    return {"models": models, "result_files": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-first Nautilus workflow and download dataset artifacts.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--allow-mutable-image-tag", action="store_true")
    parser.add_argument("--pvc", required=True)
    parser.add_argument("--gpu-product", default="")
    parser.add_argument("--gpu-resource", default="nvidia.com/gpu")
    parser.add_argument("--gpus", default="a100,a40,l4,rtx_a4000")
    parser.add_argument("--active-gpus", type=int, default=4)
    parser.add_argument("--profile-scheduling-mode", choices=("gpu-partition", "shard-switcher"), default="gpu-partition")
    parser.add_argument("--pending-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-retries-per-shard", type=int, default=3)
    parser.add_argument("--skip-image-warmup", action="store_true")
    parser.add_argument("--image-warmup-per-gpu", type=int, default=1)
    parser.add_argument("--reuse-warmed-image-nodes", action="store_true", default=True)
    parser.add_argument("--no-reuse-warmed-image-nodes", dest="reuse_warmed_image_nodes", action="store_false")
    parser.add_argument("--job-prefix", default="perfseer-nrp-source")
    parser.add_argument("--workflow-dir", default="/mnt/output/perfseer_nrp_source_workflow")
    parser.add_argument("--repo-dir", default="/workspace/PerfSeer-predictor")
    parser.add_argument("--stage-local-repo", action="store_true")
    parser.add_argument("--utility-image", default="python:3.11-slim")
    parser.add_argument("--local-repo-dir", default=str(ROOT))
    parser.add_argument("--staged-repo-dir", default="")
    parser.add_argument("--staging-pod-name", default="")
    parser.add_argument("--download-pod-name", default="")
    parser.add_argument("--stage-timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-staging-pod", action="store_true")
    parser.add_argument("--keep-download-pod", action="store_true")
    parser.add_argument("--hardware-id", default="rtx5090")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--generation-workers", type=int, default=0)
    parser.add_argument("--low-precision-focus", default="none", choices=("none", "te_transformer"))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--completions", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--infer-repeats", type=int, default=50)
    parser.add_argument("--train-repeats", type=int, default=50)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--sm-occupancy-source", default="nvml_proxy", choices=("ncu", "nvml_proxy"))
    parser.add_argument("--profile-precision-sweep", default="auto")
    parser.add_argument("--bootstrap-command", default="")
    parser.add_argument("--local-output-dir", default="nrp_downloads")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    parser.add_argument("--kubectl-request-timeout", default="30s")
    parser.add_argument("--kubectl-hard-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--monitor-log", default="")
    parser.add_argument("--monitor-interval-seconds", type=int, default=60)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--skip-local-verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["PERFSEER_KUBECTL_REQUEST_TIMEOUT"] = args.kubectl_request_timeout
    os.environ["PERFSEER_KUBECTL_HARD_TIMEOUT_SECONDS"] = str(args.kubectl_hard_timeout_seconds)
    validate_kubernetes_name_prefix(args.job_prefix, "--job-prefix")
    if args.staging_pod_name:
        validate_kubernetes_name_prefix(args.staging_pod_name, "--staging-pod-name")
    if args.download_pod_name:
        validate_kubernetes_name_prefix(args.download_pod_name, "--download-pod-name")
    validate_profile_image_reference(args.image, args.allow_mutable_image_tag)
    monitor_proc, _monitor_log = start_monitor(args)
    try:
        if args.stage_local_repo:
            stage_local_repo(args)
        if not args.skip_prepare:
            submit_stage(args, "prepare")
            print_diagnostics(args.namespace, job_name(args, "prepare"))
            wait_for_job(args, "prepare")
        if not args.skip_profile:
            args.warmed_nodes_by_gpu = warm_profile_image_cache(args)
            if args.profile_scheduling_mode == "gpu-partition":
                run_partitioned_profile_jobs(args)
            else:
                run_profile_switcher(args)
        if not args.skip_package:
            submit_stage(args, "package")
            print_diagnostics(args.namespace, job_name(args, "package"))
            wait_for_job(args, "package")
        verify_same_workflow_image(args)

        out_dir = Path(args.local_output_dir).resolve()
        source_name = f"perfseer_{args.hardware_id}_source_labels.tar.gz"
        dataset_name = f"perfseer_{args.hardware_id}_dataset.tar.gz"
        source_tar = out_dir / source_name
        dataset_tar = out_dir / dataset_name
        copy_from_workflow_pvc(args, f"{args.workflow_dir}/{source_name}", source_tar)
        copy_from_workflow_pvc(args, f"{args.workflow_dir}/{dataset_name}", dataset_tar)
        if not args.keep_download_pod:
            pod_name = args.download_pod_name or f"{args.job_prefix}-download"
            kubectl(["delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found=true", "--wait=false"], check=False)
        report: dict[str, Any] = {"source_tar": str(source_tar), "dataset_tar": str(dataset_tar)}
        if not args.skip_local_verify:
            report["source_tar_verification"] = verify_source_tar(source_tar)
            report["dataset_tar_verification"] = verify_dataset_tar(dataset_tar)
        print(json.dumps(report, sort_keys=True))
    finally:
        if monitor_proc is not None:
            monitor_proc.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
