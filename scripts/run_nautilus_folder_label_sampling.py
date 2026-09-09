#!/usr/bin/env python3
"""Sample labels on Nautilus for a local folder of PyTorch model source files."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from run_nautilus_label_sampling_e2e import (
    ControllerState,
    GPU_PRESETS,
    IMAGE,
    README_GPU_KEYS,
    VERIFIER_IMAGE,
    classify_failure,
    clean_name,
    kubectl,
    namespace_default,
    pod_is_unschedulable,
)


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    model_file: str
    input_shape: list[int]


def parse_shape(value: str) -> list[int]:
    shape = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must be comma-separated positive integers")
    return shape


def literal_assignment(tree: ast.Module, name: str) -> Any | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    return ast.literal_eval(node.value)
                except Exception:
                    return None
    return None


def has_make_model(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "make_model" for node in tree.body)


def discover_models(models_dir: Path, default_input_shape: list[int]) -> list[ModelEntry]:
    if not models_dir.is_dir():
        raise SystemExit(f"models dir not found: {models_dir}")
    entries: list[ModelEntry] = []
    for path in sorted(models_dir.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not has_make_model(tree):
            raise SystemExit(f"{path} must define make_model()")
        model_id = literal_assignment(tree, "MODEL_ID") or path.stem
        input_shape = literal_assignment(tree, "INPUT_SHAPE") or default_input_shape
        input_shape = [int(dim) for dim in input_shape]
        if not input_shape or any(dim <= 0 for dim in input_shape):
            raise SystemExit(f"{path} has invalid INPUT_SHAPE")
        entries.append(ModelEntry(str(model_id), path.name, input_shape))
    if not entries:
        raise SystemExit(f"no Python model files found in {models_dir}")
    return entries


def write_manifest(path: Path, entries: list[ModelEntry], precision_config: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            row = {
                "model_id": entry.model_id,
                "graph_id": entry.model_id,
                "profile_point_id": f"{entry.model_id}::{precision_config}",
                "model_file": entry.model_file,
                "input_shape": entry.input_shape,
                "precision_config": precision_config,
                "label_file": f"label/label/{entry.model_id}_{precision_config}.txt",
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stage_pod_yaml(namespace: str, run_id: str, pvc: str, image: str) -> str:
    name = clean_name(f"perfseer-label-stage-{run_id}")
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: perfseer-label-folder
    perfseer-run-id: {run_id}
spec:
  restartPolicy: Never
  containers:
    - name: stage
      image: {image}
      imagePullPolicy: IfNotPresent
      command: ["/bin/bash", "-lc", "sleep 86400"]
      resources:
        requests:
          cpu: "100m"
          memory: "256Mi"
        limits:
          cpu: "500m"
          memory: "1Gi"
      volumeMounts:
        - name: work
          mountPath: /workspace
  volumes:
    - name: work
      persistentVolumeClaim:
        claimName: {pvc}
"""


def affinity(products: list[str], blocked_nodes: set[str] | None = None) -> str:
    values = "\n".join(f"                      - {product}" for product in products)
    blocked_nodes = blocked_nodes or set()
    blocked_expression = ""
    if blocked_nodes:
        blocked_values = "\n".join(f"                      - {node}" for node in sorted(blocked_nodes))
        blocked_expression = f"""
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
{blocked_values}"""
    return f"""
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
{values}{blocked_expression}"""


def gpu_job_yaml(args: argparse.Namespace, gpu_key: str, remote_root: str, blocked_nodes: set[str] | None = None) -> str:
    preset = GPU_PRESETS[gpu_key]
    name = clean_name(f"perfseer-label-{args.run_id}-{gpu_key}")
    output_dir = f"{remote_root}/labels/{gpu_key}"
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {args.namespace}
  labels:
    app: perfseer-label-folder
    perfseer-run-id: {args.run_id}
    perfseer-gpu-key: {gpu_key}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: {args.backoff_limit}
  template:
    metadata:
      labels:
        app: perfseer-label-folder
        perfseer-run-id: {args.run_id}
        perfseer-gpu-key: {gpu_key}
    spec:{affinity(preset["products"], blocked_nodes)}
      restartPolicy: Never
      containers:
        - name: label-sampler
          image: {args.image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - >
              set -euo pipefail &&
              mkdir -p {output_dir} &&
              python {remote_root}/profile/run_profile.py
              --manifest {remote_root}/manifest/manifest.jsonl
              --models-dir {remote_root}/models
              --output-dir {output_dir}
              --device cuda
              --warmup-epochs {args.warmup_epochs}
              --profile-epochs {args.profile_epochs}
              --batches-per-epoch {args.batches_per_epoch}
              --sample-interval {args.sample_interval}
              --optimizer {args.optimizer}
              --sm-occupancy-source {args.sm_occupancy_source}
              --precision-config {args.precision_config}
              &&
              python {remote_root}/profile/verify_sampled_labels.py {output_dir}
          resources:
            requests:
              cpu: "{args.cpu}"
              memory: "{args.memory}"
              {preset["resource"]}: "1"
            limits:
              cpu: "{args.cpu_limit}"
              memory: "{args.memory_limit}"
              {preset["resource"]}: "1"
          volumeMounts:
            - name: work
              mountPath: /workspace
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {args.pvc}
"""


def gpu_job_name(run_id: str, gpu_key: str) -> str:
    return clean_name(f"perfseer-label-{run_id}-{gpu_key}")


def verifier_job_yaml(args: argparse.Namespace, remote_root: str) -> str:
    name = clean_name(f"perfseer-label-{args.run_id}-verify")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {args.namespace}
  labels:
    app: perfseer-label-folder
    perfseer-run-id: {args.run_id}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: perfseer-label-folder
        perfseer-run-id: {args.run_id}
    spec:
      restartPolicy: Never
      containers:
        - name: verifier
          image: {args.verifier_image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - >
              set -euo pipefail &&
              python {remote_root}/profile/verify_sampled_labels.py {remote_root}/labels
          resources:
            requests:
              cpu: "100m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "1Gi"
          volumeMounts:
            - name: work
              mountPath: /workspace
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {args.pvc}
"""


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def wait_pod_running(namespace: str, pod_name: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = kubectl(["get", "pod", pod_name, "-n", namespace, "-o", "json"], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            pod = json.loads(proc.stdout)
            phase = pod.get("status", {}).get("phase")
            if phase == "Running":
                return
            if phase in {"Failed", "Succeeded"}:
                raise SystemExit(f"stage pod ended early: {phase}")
        time.sleep(5)
    raise SystemExit(f"timeout waiting for stage pod: {pod_name}")


def wait_jobs(namespace: str, run_id: str, expected_jobs: int, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = kubectl(["get", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "json"], check=False)
        jobs = json.loads(proc.stdout).get("items", []) if proc.returncode == 0 and proc.stdout.strip() else []
        sample_jobs = [job for job in jobs if not job["metadata"]["name"].endswith("-verify")]
        succeeded = sum(1 for job in sample_jobs if int(job.get("status", {}).get("succeeded", 0)) >= 1)
        failed = [job["metadata"]["name"] for job in sample_jobs if int(job.get("status", {}).get("failed", 0)) > 0]
        print(json.dumps({"sample_jobs": len(sample_jobs), "succeeded": succeeded, "failed": failed}, sort_keys=True), flush=True)
        if failed:
            raise SystemExit(f"sample job failed: {failed}")
        if len(sample_jobs) == expected_jobs and succeeded == expected_jobs:
            return
        time.sleep(60)
    raise SystemExit("timeout waiting for sample jobs")


def append_record(record_file: Path, title: str, body: str) -> None:
    with record_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n{body}\n")


def pod_logs(namespace: str, job_name: str) -> str:
    return kubectl(["logs", "-n", namespace, f"job/{job_name}", "--all-containers=true", "--tail=200"], check=False).stdout


def submit_gpu_job(
    args: argparse.Namespace,
    gpu_key: str,
    remote_root: str,
    record_file: Path,
    state: ControllerState,
    switched_from: str = "",
) -> None:
    kubectl(
        ["apply", "-f", "-"],
        input_text=gpu_job_yaml(args, gpu_key, remote_root, state.blocked_nodes_by_gpu.get(gpu_key, set())),
    )
    append_record(
        record_file,
        "Submitted Job",
        f"- GPU: `{gpu_key}`\n- Job: `{gpu_job_name(args.run_id, gpu_key)}`\n- Switched from: `{switched_from or 'none'}`\n",
    )
    immediate_diagnostics(args.namespace, args.run_id, record_file)


def submit_until_active(args: argparse.Namespace, remote_root: str, record_file: Path, state: ControllerState) -> None:
    while state.active_count() < state.active_limit:
        gpu_key = state.next_replacement_gpu()
        if gpu_key is None:
            return
        submit_gpu_job(args, gpu_key, remote_root, record_file, state)
        state.mark_submitted(gpu_key, gpu_job_name(args.run_id, gpu_key))


def wait_jobs_with_switching(args: argparse.Namespace, remote_root: str, record_file: Path, state: ControllerState) -> None:
    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        submit_until_active(args, remote_root, record_file, state)
        jobs_proc = kubectl(["get", "jobs", "-n", args.namespace, "-l", f"perfseer-run-id={args.run_id}", "-o", "json"], check=False)
        pods_proc = kubectl(["get", "pods", "-n", args.namespace, "-l", f"perfseer-run-id={args.run_id}", "-o", "json"], check=False)
        jobs = json.loads(jobs_proc.stdout).get("items", []) if jobs_proc.returncode == 0 and jobs_proc.stdout.strip() else []
        pods = json.loads(pods_proc.stdout).get("items", []) if pods_proc.returncode == 0 and pods_proc.stdout.strip() else []
        jobs_by_name = {job["metadata"]["name"]: job for job in jobs}
        pods_by_gpu: dict[str, list[dict[str, Any]]] = {}
        for pod in pods:
            gpu_key = pod.get("metadata", {}).get("labels", {}).get("perfseer-gpu-key")
            if gpu_key:
                pods_by_gpu.setdefault(gpu_key, []).append(pod)

        now = time.time()
        for gpu_key, gpu_state in list(state.gpu_states.items()):
            if not gpu_state.job_name or gpu_state.status not in {"pending", "running"}:
                continue
            job = jobs_by_name.get(gpu_state.job_name)
            gpu_pods = pods_by_gpu.get(gpu_key, [])
            pod_phase = gpu_pods[0].get("status", {}).get("phase", "") if gpu_pods else "Pending"
            unschedulable = any(pod_is_unschedulable(pod) for pod in gpu_pods)
            if pod_phase in {"Running", "Succeeded"}:
                state.mark_running(gpu_key)
            if job and int(job.get("status", {}).get("succeeded", 0)) >= 1:
                state.mark_succeeded(gpu_key)
                continue
            if job and int(job.get("status", {}).get("failed", 0)) > 0:
                logs = pod_logs(args.namespace, gpu_state.job_name)
                reason = classify_failure(logs)
                if reason in {"cuda_initialization_failure", "nvidia_smi_failure", "hardware_failure"}:
                    node_name = gpu_pods[0].get("spec", {}).get("nodeName", "") if gpu_pods else ""
                    state.block_node_for_gpu(gpu_key, node_name)
                    kubectl(["delete", "job", gpu_state.job_name, "-n", args.namespace, "--ignore-not-found=true"], check=False)
                    if state.retry_counts.get(gpu_key, 0) >= args.max_retries_per_gpu:
                        state.mark_failed(gpu_key, reason)
                        append_record(record_file, "Failed Job", f"- GPU: `{gpu_key}`\n- Reason: `{reason}`\n\n```text\n{logs[-2000:]}\n```\n")
                        continue
                    state.clear_for_retry(gpu_key, reason)
                    append_record(
                        record_file,
                        "Retry GPU After Node Failure",
                        f"- GPU: `{gpu_key}`\n- Failed node: `{node_name or 'unknown'}`\n- Reason: `{reason}`\n- Retry count: `{state.retry_counts[gpu_key]}`\n\n```text\n{logs[-2000:]}\n```\n",
                    )
                    submit_gpu_job(args, gpu_key, remote_root, record_file, state, switched_from=f"{gpu_key}@{node_name or 'retry'}")
                    state.mark_submitted(gpu_key, gpu_job_name(args.run_id, gpu_key), switched_from=f"{gpu_key}@{node_name or 'retry'}")
                    continue
                state.mark_failed(gpu_key, reason)
                append_record(record_file, "Failed Job", f"- GPU: `{gpu_key}`\n- Job: `{gpu_state.job_name}`\n- Reason: `{reason}`\n\n```text\n{logs[-2000:]}\n```\n")
                continue

            decision = state.plan_pending_switch(gpu_key, now, pod_phase, unschedulable)
            if decision is not None:
                kubectl(["delete", "job", decision.old_job_name, "-n", args.namespace, "--ignore-not-found=true"], check=False)
                append_record(
                    record_file,
                    "GPU Switch",
                    f"- Old GPU: `{decision.old_gpu}`\n- Old Job: `{decision.old_job_name}`\n- Reason: `{decision.reason}`\n- New GPU: `{decision.new_gpu or 'none available'}`\n",
                )
                if decision.new_gpu:
                    submit_gpu_job(args, decision.new_gpu, remote_root, record_file, state, switched_from=decision.old_gpu)
                    state.mark_submitted(decision.new_gpu, gpu_job_name(args.run_id, decision.new_gpu), switched_from=decision.old_gpu)

        summary = {
            "active": state.active_count(),
            "succeeded": state.succeeded_count(),
            "terminal": state.terminal_count(),
            "total": len(state.candidate_gpus),
            "states": {key: item.status for key, item in state.gpu_states.items()},
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if state.succeeded_count() >= args.min_successful_gpus:
            return
        if state.terminal_count() == len(state.candidate_gpus):
            failed = {key: item.failure_reason for key, item in state.gpu_states.items() if item.status != "succeeded"}
            append_record(record_file, "Terminal GPU Failures", f"```json\n{json.dumps(failed, indent=2, sort_keys=True)}\n```\n")
            raise SystemExit(f"not enough GPU jobs succeeded: {state.succeeded_count()} < {args.min_successful_gpus}")
        time.sleep(60)
    raise SystemExit("timeout waiting for sample jobs with switching")


def wait_verifier(namespace: str, run_id: str, timeout_seconds: int) -> str:
    name = clean_name(f"perfseer-label-{run_id}-verify")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = kubectl(["get", "job", name, "-n", namespace, "-o", "json"], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            job = json.loads(proc.stdout)
            if int(job.get("status", {}).get("succeeded", 0)) >= 1:
                return kubectl(["logs", "-n", namespace, f"job/{name}", "--all-containers=true"], check=False).stdout
            if int(job.get("status", {}).get("failed", 0)) > 0:
                raise SystemExit("verifier job failed")
        time.sleep(10)
    raise SystemExit("timeout waiting for verifier job")


def immediate_diagnostics(namespace: str, run_id: str, record_file: Path) -> None:
    commands = [
        ["get", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"],
        ["get", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"],
        ["describe", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}"],
        ["logs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "--all-containers=true", "--tail=100"],
        ["get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ]
    with record_file.open("a", encoding="utf-8") as handle:
        for command in commands:
            proc = kubectl(command, check=False)
            text = proc.stdout
            if command[:2] == ["get", "events"]:
                text = "\n".join(text.splitlines()[-80:])
            handle.write(f"\n$ kubectl {' '.join(command)}\n{text}\n")


def start_monitor(namespace: str, run_id: str, record_file: Path) -> subprocess.Popen[str]:
    monitor_path = record_file.with_suffix(".monitor.log")
    script = (
        f"while true; do date -u +%Y-%m-%dT%H:%M:%SZ; "
        f"kubectl get jobs -n {namespace} -l perfseer-run-id={run_id} -o wide; "
        f"kubectl get pods -n {namespace} -l perfseer-run-id={run_id} -o wide; "
        f"kubectl get events -n {namespace} --sort-by=.lastTimestamp | tail -80; "
        f"sleep 60; done"
    )
    return subprocess.Popen(["nohup", "bash", "-lc", script], stdout=monitor_path.open("a"), stderr=subprocess.STDOUT)


def kubectl_cp(src: str, dst: str) -> None:
    proc = run(["kubectl", "cp", src, dst], check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)


def cleanup(namespace: str, run_id: str, stage_pod: str) -> None:
    kubectl(["delete", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "--ignore-not-found=true"], check=False)
    kubectl(["delete", "pod", stage_pod, "-n", namespace, "--ignore-not-found=true"], check=False)


def parse_gpus(value: str) -> list[str]:
    if value.strip().lower() == "all-readme":
        return list(README_GPU_KEYS)
    keys = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
    if not keys or len(set(keys)) != len(keys):
        raise SystemExit("--gpus must contain distinct GPU presets")
    unknown = [key for key in keys if key not in GPU_PRESETS]
    if unknown:
        raise SystemExit(f"unknown GPU preset(s): {', '.join(unknown)}")
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Nautilus label sampling for a local PyTorch model folder.")
    parser.add_argument("--models-dir", required=True, help="Local folder containing Python files with make_model().")
    parser.add_argument("--local-labels-dir", default="labels", help="Local output folder for copied labels.")
    parser.add_argument("--namespace", default=namespace_default())
    parser.add_argument("--pvc", default="test-pvc")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--verifier-image", default=VERIFIER_IMAGE)
    parser.add_argument("--gpus", default="a100,a40,l4,rtx_a4000")
    parser.add_argument("--active-gpus", type=int, default=4)
    parser.add_argument("--pending-timeout-seconds", type=int, default=300)
    parser.add_argument("--blacklist-ttl-seconds", type=int, default=21600)
    parser.add_argument("--max-retries-per-gpu", type=int, default=3)
    parser.add_argument("--min-successful-gpus", type=int, default=1)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("folder-%Y%m%d%H%M%S"))
    parser.add_argument("--default-input-shape", type=parse_shape, default=[16, 32])
    parser.add_argument("--precision-config", default="fp32_ieee")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--profile-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--sm-occupancy-source", default="nvml_proxy", choices=("ncu", "nvml_proxy"))
    parser.add_argument("--cpu", default="1")
    parser.add_argument("--cpu-limit", default="2")
    parser.add_argument("--memory", default="2Gi")
    parser.add_argument("--memory-limit", default="4Gi")
    parser.add_argument("--backoff-limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/YAML only; do not use kubectl.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.run_id = clean_name(args.run_id)
    models_dir = Path(args.models_dir).resolve()
    local_labels_root = Path(args.local_labels_dir).resolve()
    gpu_keys = parse_gpus(args.gpus)
    remote_root = f"/workspace/perfseer-folder-runs/{args.run_id}"
    record_dir = Path("record")
    record_dir.mkdir(exist_ok=True)
    record_file = record_dir / f"nautilus_folder_labels_{args.run_id}.md"
    yaml_file = record_dir / f"nautilus_folder_labels_{args.run_id}.yaml"
    manifest_file = record_dir / f"nautilus_folder_labels_{args.run_id}.manifest.jsonl"

    entries = discover_models(models_dir, args.default_input_shape)
    write_manifest(manifest_file, entries, args.precision_config)
    yaml_docs = [stage_pod_yaml(args.namespace, args.run_id, args.pvc, args.verifier_image)]
    yaml_docs.extend(gpu_job_yaml(args, key, remote_root) for key in gpu_keys)
    yaml_docs.append(verifier_job_yaml(args, remote_root))
    yaml_file.write_text("---\n".join(yaml_docs), encoding="utf-8")
    record_file.write_text(
        dedent(
            f"""
            # Nautilus Folder Label Sampling {args.run_id}

            - Models dir: `{models_dir}`
            - Local labels dir: `{local_labels_root / args.run_id}`
            - Namespace: `{args.namespace}`
            - Persistent Volume Claim: `{args.pvc}`
            - GPUs: `{', '.join(gpu_keys)}`
            - Manifest: `{manifest_file}`
            - YAML: `{yaml_file}`
            - Model count: `{len(entries)}`
            """
        ).lstrip(),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps({"record": str(record_file), "yaml": str(yaml_file), "manifest": str(manifest_file)}, indent=2))
        return

    stage_pod = clean_name(f"perfseer-label-stage-{args.run_id}")
    monitor: subprocess.Popen[str] | None = None
    try:
        kubectl(["apply", "-f", "-"], input_text=stage_pod_yaml(args.namespace, args.run_id, args.pvc, args.verifier_image))
        immediate_diagnostics(args.namespace, args.run_id, record_file)
        monitor = start_monitor(args.namespace, args.run_id, record_file)
        wait_pod_running(args.namespace, stage_pod, args.timeout_seconds)
        kubectl(["exec", "-n", args.namespace, stage_pod, "--", "bash", "-lc", f"rm -rf {remote_root} && mkdir -p {remote_root}/models {remote_root}/manifest {remote_root}/profile {remote_root}/labels"])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_stage = tmp_path / "models"
            model_stage.mkdir()
            for source in sorted(models_dir.glob("*.py")):
                if not source.name.startswith("_") and source.name != "__init__.py":
                    shutil.copy2(source, model_stage / source.name)
            profile_stage = tmp_path / "profile"
            profile_stage.mkdir()
            shutil.copy2("nrp_calibration_pack/profile/run_profile.py", profile_stage / "run_profile.py")
            shutil.copy2("scripts/verify_sampled_labels.py", profile_stage / "verify_sampled_labels.py")
            manifest_stage = tmp_path / "manifest"
            manifest_stage.mkdir()
            shutil.copy2(manifest_file, manifest_stage / "manifest.jsonl")
            kubectl_cp(str(model_stage), f"{args.namespace}/{stage_pod}:{remote_root}/")
            kubectl_cp(str(profile_stage), f"{args.namespace}/{stage_pod}:{remote_root}/")
            kubectl_cp(str(manifest_stage), f"{args.namespace}/{stage_pod}:{remote_root}/")

        controller_state = ControllerState(
            candidate_gpus=gpu_keys,
            active_limit=min(args.active_gpus, len(gpu_keys)),
            pending_timeout_seconds=args.pending_timeout_seconds,
            blacklist_ttl_seconds=args.blacklist_ttl_seconds,
        )
        wait_jobs_with_switching(args, remote_root, record_file, controller_state)
        kubectl(["apply", "-f", "-"], input_text=verifier_job_yaml(args, remote_root))
        verifier_logs = wait_verifier(args.namespace, args.run_id, 900)
        with record_file.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Controller States\n\n```json\n{json.dumps({key: value.status for key, value in controller_state.gpu_states.items()}, indent=2, sort_keys=True)}\n```\n"
            )
            handle.write(f"\n## Verifier Logs\n\n```text\n{verifier_logs}\n```\n")
        local_run_dir = local_labels_root / args.run_id
        local_run_dir.parent.mkdir(parents=True, exist_ok=True)
        if local_run_dir.exists():
            shutil.rmtree(local_run_dir)
        kubectl_cp(f"{args.namespace}/{stage_pod}:{remote_root}/labels", str(local_run_dir))
        print(json.dumps({"labels": str(local_run_dir), "record": str(record_file), "verifier_logs": verifier_logs}, indent=2, sort_keys=True))
    finally:
        if monitor is not None:
            monitor.terminate()
        if not args.keep:
            cleanup(args.namespace, args.run_id, stage_pod)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
