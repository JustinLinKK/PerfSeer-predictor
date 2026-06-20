#!/usr/bin/env python3
"""Run and verify end-to-end label sampling on Nautilus GPU types."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any


IMAGE = "pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel"
VERIFIER_IMAGE = "python:3.11-slim"
README_GPU_KEYS = (
    "a10",
    "a40",
    "a100",
    "l4",
    "l40",
    "l40s",
    "rtx_a4000",
    "rtx_a5000",
    "rtx_a6000",
    "rtx_4000_ada",
    "rtx_5000_ada",
    "rtx_pro_6000_blackwell",
    "quadro_rtx_6000",
    "quadro_rtx_8000",
    "t4",
    "v100",
)
GPU_PRESETS = {
    "a10": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-A10"]},
    "a40": {"resource": "nvidia.com/a40", "products": ["NVIDIA-A40"]},
    "a100": {"resource": "nvidia.com/a100", "products": ["NVIDIA-A100-PCIE-40GB", "NVIDIA-A100-80GB-PCIe"]},
    "l4": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-L4"]},
    "l40": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-L40"]},
    "l40s": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-L40S"]},
    "rtx_a4000": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-RTX-A4000"]},
    "rtx_a5000": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-RTX-A5000"]},
    "rtx_a6000": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-RTX-A6000"]},
    "rtx_4000_ada": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-RTX-4000-Ada-Generation"]},
    "rtx_5000_ada": {"resource": "nvidia.com/gpu", "products": ["NVIDIA-RTX-5000-Ada-Generation"]},
    "rtx_pro_6000_blackwell": {
        "resource": "nvidia.com/gpu",
        "products": ["NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition"],
    },
    "quadro_rtx_6000": {"resource": "nvidia.com/gpu", "products": ["Quadro-RTX-6000"]},
    "quadro_rtx_8000": {"resource": "nvidia.com/gpu", "products": ["Quadro-RTX-8000"]},
    "t4": {"resource": "nvidia.com/gpu", "products": ["Tesla-T4"]},
    "v100": {
        "resource": "nvidia.com/gpu",
        "products": ["Tesla-V100-PCIE-16GB", "Tesla-V100-SXM2-16GB", "Tesla-V100-SXM2-32GB"],
    },
}
assert set(README_GPU_KEYS) == set(GPU_PRESETS)


@dataclass
class GpuJobState:
    gpu_key: str
    status: str = "pending"
    job_name: str = ""
    submitted_at: float = 0.0
    switched_from: str = ""
    failure_reason: str = ""


@dataclass
class SwitchDecision:
    old_gpu: str
    new_gpu: str
    reason: str
    old_job_name: str


@dataclass
class ControllerState:
    candidate_gpus: list[str]
    active_limit: int
    pending_timeout_seconds: int
    blacklist_ttl_seconds: int
    verified_gpus: set[str] = field(default_factory=set)
    gpu_states: dict[str, GpuJobState] = field(init=False)
    blacklisted_until: dict[str, float] = field(default_factory=dict)
    blocked_nodes_by_gpu: dict[str, set[str]] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gpu_states = {}
        for key in self.candidate_gpus:
            status = "succeeded" if key in self.verified_gpus else "pending"
            self.gpu_states[key] = GpuJobState(gpu_key=key, status=status)

    def active_count(self) -> int:
        return sum(1 for state in self.gpu_states.values() if state.job_name and state.status in {"pending", "running"})

    def succeeded_count(self) -> int:
        return sum(1 for state in self.gpu_states.values() if state.status == "succeeded")

    def terminal_count(self) -> int:
        return sum(1 for state in self.gpu_states.values() if state.status in {"succeeded", "failed", "blacklisted"})

    def next_replacement_gpu(self, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        for key in self.candidate_gpus:
            state = self.gpu_states[key]
            if state.status != "pending" or state.job_name:
                continue
            if self.blacklisted_until.get(key, 0.0) > now:
                continue
            return key
        return None

    def mark_submitted(self, gpu_key: str, job_name: str, now: float | None = None, switched_from: str = "") -> None:
        state = self.gpu_states[gpu_key]
        state.status = "pending"
        state.job_name = job_name
        state.submitted_at = time.time() if now is None else now
        state.switched_from = switched_from
        state.failure_reason = ""

    def mark_running(self, gpu_key: str) -> None:
        state = self.gpu_states[gpu_key]
        if state.status == "pending":
            state.status = "running"

    def mark_succeeded(self, gpu_key: str) -> None:
        state = self.gpu_states[gpu_key]
        state.status = "succeeded"
        state.failure_reason = ""

    def mark_failed(self, gpu_key: str, reason: str) -> None:
        state = self.gpu_states[gpu_key]
        state.status = "failed"
        state.failure_reason = reason

    def clear_for_retry(self, gpu_key: str, reason: str) -> None:
        state = self.gpu_states[gpu_key]
        state.status = "pending"
        state.job_name = ""
        state.failure_reason = reason
        self.retry_counts[gpu_key] = self.retry_counts.get(gpu_key, 0) + 1

    def block_node_for_gpu(self, gpu_key: str, node_name: str) -> None:
        if node_name:
            self.blocked_nodes_by_gpu.setdefault(gpu_key, set()).add(node_name)

    def blacklist(self, gpu_key: str, now: float, reason: str) -> None:
        self.blacklisted_until[gpu_key] = now + self.blacklist_ttl_seconds
        state = self.gpu_states[gpu_key]
        state.status = "blacklisted"
        state.failure_reason = reason

    def plan_pending_switch(
        self,
        gpu_key: str,
        now: float,
        pod_phase: str,
        unschedulable: bool,
    ) -> SwitchDecision | None:
        state = self.gpu_states[gpu_key]
        if state.status not in {"pending", "running"}:
            return None
        too_old = now - state.submitted_at > self.pending_timeout_seconds
        if not too_old or (pod_phase != "Pending" and not unschedulable):
            return None
        replacement = self.next_replacement_gpu(now)
        reason = "unschedulable" if unschedulable else "pending_timeout"
        self.mark_failed(gpu_key, reason)
        if replacement is None:
            return SwitchDecision(gpu_key, "", reason, state.job_name)
        self.mark_submitted(replacement, "", now=now, switched_from=gpu_key)
        return SwitchDecision(gpu_key, replacement, reason, state.job_name)


def run(cmd: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=input_text, text=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def kubectl(args: list[str], *, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["kubectl", *args], input_text=input_text, check=check)


def namespace_default() -> str:
    proc = kubectl(["config", "view", "--minify", "--output", "jsonpath={..namespace}"], check=False)
    return proc.stdout.strip() or "default"


def clean_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in value.lower()).strip("-")


def write_smoke_model(path: Path) -> None:
    path.write_text(
        dedent(
            '''
            """Small PyTorch model for Nautilus end-to-end label sampling."""

            from __future__ import annotations

            import torch.nn as nn

            MODEL_ID = "nautilus_smoke_mlp"
            INPUT_SHAPE = (16, 32)


            class SmokeModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))

                def forward(self, x):
                    return self.net(x)


            def make_model() -> nn.Module:
                return SmokeModel()
            '''
        ).lstrip(),
        encoding="utf-8",
    )


def prepare_configmap_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2("nrp_calibration_pack/profile/run_profile.py", root / "run_profile.py")
    shutil.copy2("scripts/verify_sampled_labels.py", root / "verify_sampled_labels.py")
    (root / "verify_distinct_gpus.py").write_text(
        dedent(
            '''
            """Verify labels were produced by at least the expected number of distinct GPUs."""

            from __future__ import annotations

            import json
            import sys
            from pathlib import Path


            root = Path(sys.argv[1])
            expected = int(sys.argv[2])
            names = set()
            rows = 0
            for path in root.rglob("results_shard*.jsonl"):
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows += 1
                    if row.get("status") == "ok":
                        hardware = row.get("hardware", {})
                        names.add(hardware.get("gpu_name") or hardware.get("nvidia_smi"))
            names.discard(None)
            print(json.dumps({"rows": rows, "gpu_names": sorted(names), "gpu_count": len(names)}, sort_keys=True))
            if len(names) < expected:
                raise SystemExit(f"expected labels from at least {expected} distinct GPUs")
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    write_smoke_model(root / "smoke_model.py")
    manifest = {
        "model_id": "nautilus_smoke_mlp",
        "model_file": "smoke_model.py",
        "input_shape": [16, 32],
        "precision_config": "fp32_ieee",
        "label_file": "label/label/nautilus_smoke_mlp_fp32_ieee.txt",
    }
    (root / "manifest.jsonl").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


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


def hostname_affinity(node_name: str) -> str:
    return f"""
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
                      - {node_name}"""


def gpu_job_yaml(
    namespace: str,
    run_id: str,
    pvc: str,
    image: str,
    gpu_key: str,
    output_root: str,
    node_name: str = "",
    blocked_nodes: set[str] | None = None,
) -> str:
    preset = GPU_PRESETS[gpu_key]
    name = clean_name(f"perfseer-e2e-{run_id}-{gpu_key}")
    out_dir = f"{output_root}/{gpu_key}"
    work_dir = f"/workspace/perfseer-e2e/{run_id}/work/{gpu_key}"
    placement = hostname_affinity(node_name) if node_name else affinity(preset["products"], blocked_nodes)
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: perfseer-e2e
    perfseer-run-id: {run_id}
    perfseer-gpu-key: {gpu_key}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: perfseer-e2e
        perfseer-run-id: {run_id}
        perfseer-gpu-key: {gpu_key}
    spec:{placement}
      restartPolicy: Never
      containers:
        - name: label-sampler
          image: {image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - >
              set -euo pipefail &&
              rm -rf {work_dir} &&
              mkdir -p {work_dir}/profile {work_dir}/models {work_dir}/manifest {out_dir} &&
              cp /config/run_profile.py {work_dir}/profile/run_profile.py &&
              cp /config/smoke_model.py {work_dir}/models/smoke_model.py &&
              cp /config/manifest.jsonl {work_dir}/manifest/manifest.jsonl &&
              python {work_dir}/profile/run_profile.py
              --manifest {work_dir}/manifest/manifest.jsonl
              --models-dir {work_dir}/models
              --output-dir {out_dir}
              --device cuda
              --warmup-epochs 1
              --profile-epochs 1
              --batches-per-epoch 1
              --sample-interval 0.01
              --optimizer sgd
              --sm-occupancy-source nvml_proxy
              --precision-config fp32_ieee &&
              python /config/verify_sampled_labels.py {out_dir}
          resources:
            requests:
              cpu: "100m"
              memory: "2Gi"
              {preset["resource"]}: "1"
            limits:
              cpu: "1"
              memory: "4Gi"
              {preset["resource"]}: "1"
          volumeMounts:
            - name: work
              mountPath: /workspace
            - name: config
              mountPath: /config
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {pvc}
        - name: config
          configMap:
            name: perfseer-e2e-{run_id}
"""


def verifier_job_yaml(namespace: str, run_id: str, pvc: str, image: str, output_root: str, expected_gpus: int) -> str:
    name = clean_name(f"perfseer-e2e-{run_id}-verify")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: perfseer-e2e
    perfseer-run-id: {run_id}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: perfseer-e2e
        perfseer-run-id: {run_id}
    spec:
      restartPolicy: Never
      containers:
        - name: verifier
          image: {image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - >
              set -euo pipefail &&
              python /config/verify_sampled_labels.py {output_root} &&
              python /config/verify_distinct_gpus.py {output_root} {expected_gpus}
          resources:
            requests:
              cpu: "100m"
              memory: "512Mi"
            limits:
              cpu: "120m"
              memory: "600Mi"
          volumeMounts:
            - name: work
              mountPath: /workspace
            - name: config
              mountPath: /config
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {pvc}
        - name: config
          configMap:
            name: perfseer-e2e-{run_id}
"""


def create_configmap(namespace: str, run_id: str, source_dir: Path) -> None:
    name = f"perfseer-e2e-{run_id}"
    kubectl(["delete", "configmap", name, "-n", namespace], check=False)
    cmd = ["kubectl", "create", "configmap", name, "-n", namespace]
    for path in sorted(source_dir.iterdir()):
        cmd.append(f"--from-file={path.name}={path}")
    print(run(cmd).stdout, end="")


def immediate_diagnostics(namespace: str, run_id: str, record_file: Path) -> None:
    commands = [
        ["get", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"],
        ["get", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"],
        ["get", "events", "-n", namespace, "--sort-by=.lastTimestamp"],
    ]
    with record_file.open("a", encoding="utf-8") as handle:
        for args in commands:
            proc = kubectl(args, check=False)
            output = proc.stdout
            if args[:2] == ["get", "events"]:
                output = "\n".join(output.splitlines()[-80:]) + "\n"
            handle.write(f"\n$ kubectl {' '.join(args)}\n{output}\n")
            print(output)


def append_record(record_file: Path, title: str, body: str) -> None:
    with record_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n{body}\n")


def start_monitor(namespace: str, run_id: str, record_file: Path) -> subprocess.Popen[str]:
    monitor = record_file.with_suffix(".monitor.log")
    script = (
        f"while true; do date -Is; "
        f"kubectl get jobs -n {namespace} -l perfseer-run-id={run_id} -o wide; "
        f"kubectl get pods -n {namespace} -l perfseer-run-id={run_id} -o wide; "
        f"sleep 60; done"
    )
    return subprocess.Popen(["nohup", "bash", "-lc", script], stdout=monitor.open("a"), stderr=subprocess.STDOUT)


def job_name_for(run_id: str, gpu_key: str) -> str:
    return clean_name(f"perfseer-e2e-{run_id}-{gpu_key}")


def submit_gpu_job(
    namespace: str,
    run_id: str,
    pvc: str,
    image: str,
    gpu_key: str,
    output_root: str,
    node_name: str,
    record_file: Path,
    yaml_file: Path,
    state: ControllerState,
    switched_from: str = "",
) -> None:
    yaml_text = gpu_job_yaml(
        namespace,
        run_id,
        pvc,
        image,
        gpu_key,
        output_root,
        node_name,
        state.blocked_nodes_by_gpu.get(gpu_key, set()),
    )
    with yaml_file.open("a", encoding="utf-8") as handle:
        handle.write("---\n")
        handle.write(yaml_text)
    kubectl(["apply", "-f", "-"], input_text=yaml_text)
    state.mark_submitted(gpu_key, job_name_for(run_id, gpu_key), switched_from=switched_from)
    append_record(
        record_file,
        "Submitted Job",
        f"- GPU: `{gpu_key}`\n- Job: `{job_name_for(run_id, gpu_key)}`\n- Switched from: `{switched_from or 'none'}`\n",
    )
    immediate_diagnostics(namespace, run_id, record_file)


def pod_is_unschedulable(pod: dict[str, Any]) -> bool:
    for condition in pod.get("status", {}).get("conditions", []):
        if condition.get("type") == "PodScheduled" and condition.get("status") == "False":
            if condition.get("reason") == "Unschedulable":
                return True
    return False


def classify_failure(logs: str) -> str:
    lowered = logs.lower()
    if "cuda initialization" in lowered or "cuda error" in lowered:
        return "cuda_initialization_failure"
    if "nvidia-smi" in lowered and ("failed" in lowered or "error" in lowered):
        return "nvidia_smi_failure"
    if "xid" in lowered or "hardware" in lowered:
        return "hardware_failure"
    if "importerror" in lowered or "syntaxerror" in lowered or "attributeerror" in lowered:
        return "model_or_code_failure"
    return "job_failure"


def submit_until_active(
    namespace: str,
    run_id: str,
    pvc: str,
    image: str,
    output_root: str,
    node_map: dict[str, str],
    record_file: Path,
    yaml_file: Path,
    state: ControllerState,
) -> None:
    while state.active_count() < state.active_limit:
        next_gpu = state.next_replacement_gpu()
        if next_gpu is None:
            return
        submit_gpu_job(
            namespace,
            run_id,
            pvc,
            image,
            next_gpu,
            output_root,
            node_map.get(next_gpu, ""),
            record_file,
            yaml_file,
            state,
        )


def wait_jobs_with_switching(
    namespace: str,
    run_id: str,
    pvc: str,
    image: str,
    output_root: str,
    node_map: dict[str, str],
    record_file: Path,
    yaml_file: Path,
    timeout_s: int,
    state: ControllerState,
    max_retries_per_gpu: int,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        submit_until_active(namespace, run_id, pvc, image, output_root, node_map, record_file, yaml_file, state)
        jobs_proc = kubectl(["get", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "json"], check=False)
        pods_proc = kubectl(["get", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "json"], check=False)
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
                logs = kubectl(["logs", "-n", namespace, f"job/{gpu_state.job_name}", "--all-containers=true", "--tail=200"], check=False).stdout
                reason = classify_failure(logs)
                if reason in {"cuda_initialization_failure", "nvidia_smi_failure", "hardware_failure"}:
                    node_name = gpu_pods[0].get("spec", {}).get("nodeName", "") if gpu_pods else ""
                    state.block_node_for_gpu(gpu_key, node_name)
                    kubectl(["delete", "job", gpu_state.job_name, "-n", namespace, "--ignore-not-found=true"], check=False)
                    if state.retry_counts.get(gpu_key, 0) >= max_retries_per_gpu:
                        state.blacklist(gpu_key, now, reason)
                        fail_with_diagnostics(namespace, run_id, f"GPU job failed after retries: {gpu_state.job_name} ({reason})")
                    state.clear_for_retry(gpu_key, reason)
                    append_record(
                        record_file,
                        "Retry GPU After Node Failure",
                        f"- GPU: `{gpu_key}`\n- Failed node: `{node_name}`\n- Reason: `{reason}`\n- Retry count: `{state.retry_counts[gpu_key]}`\n\n```text\n{logs[-2000:]}\n```\n",
                    )
                    submit_gpu_job(
                        namespace,
                        run_id,
                        pvc,
                        image,
                        gpu_key,
                        output_root,
                        node_map.get(gpu_key, ""),
                        record_file,
                        yaml_file,
                        state,
                        switched_from=f"{gpu_key}@{node_name}",
                    )
                    continue
                else:
                    state.mark_failed(gpu_key, reason)
                    append_record(record_file, "Failed Job", f"- GPU: `{gpu_key}`\n- Job: `{gpu_state.job_name}`\n- Reason: `{reason}`\n\n```text\n{logs[-2000:]}\n```\n")
                    fail_with_diagnostics(namespace, run_id, f"GPU job failed: {gpu_state.job_name} ({reason})")
            decision = state.plan_pending_switch(gpu_key, now, pod_phase, unschedulable)
            if decision is not None:
                kubectl(["delete", "job", decision.old_job_name, "-n", namespace, "--ignore-not-found=true"], check=False)
                new_gpu_line = f"- New GPU: `{decision.new_gpu}`\n" if decision.new_gpu else "- New GPU: none available\n"
                append_record(
                    record_file,
                    "GPU Switch",
                    f"- Old GPU: `{decision.old_gpu}`\n- Old Job: `{decision.old_job_name}`\n- Reason: `{decision.reason}`\n{new_gpu_line}",
                )
                if not decision.new_gpu:
                    continue
                submit_gpu_job(
                    namespace,
                    run_id,
                    pvc,
                    image,
                    decision.new_gpu,
                    output_root,
                    node_map.get(decision.new_gpu, ""),
                    record_file,
                    yaml_file,
                    state,
                    switched_from=decision.old_gpu,
                )

        summary = {
            "active": state.active_count(),
            "succeeded": state.succeeded_count(),
            "terminal": state.terminal_count(),
            "total": len(state.candidate_gpus),
            "states": {key: item.status for key, item in state.gpu_states.items()},
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if state.succeeded_count() == len(state.candidate_gpus):
            return
        if state.terminal_count() == len(state.candidate_gpus):
            failed = {key: item.failure_reason for key, item in state.gpu_states.items() if item.status != "succeeded"}
            append_record(record_file, "Terminal GPU Failures", f"```json\n{json.dumps(failed, indent=2, sort_keys=True)}\n```\n")
            if state.succeeded_count() == 0:
                raise SystemExit(f"no GPU jobs succeeded: {failed}")
            return
        time.sleep(60)
    fail_with_diagnostics(namespace, run_id, "timeout waiting for GPU jobs with switching")


def wait_jobs(namespace: str, run_id: str, timeout_s: int, expected_gpus: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = kubectl(["get", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "json"], check=False)
        data = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {"items": []}
        jobs = data.get("items", [])
        gpu_jobs = [job for job in jobs if not job["metadata"]["name"].endswith("-verify")]
        succeeded = sum(1 for job in gpu_jobs if int(job.get("status", {}).get("succeeded", 0)) >= 1)
        failed = [job["metadata"]["name"] for job in gpu_jobs if int(job.get("status", {}).get("failed", 0)) > 0]
        print(json.dumps({"gpu_jobs": len(gpu_jobs), "succeeded": succeeded, "failed": failed}, sort_keys=True), flush=True)
        if failed:
            fail_with_diagnostics(namespace, run_id, f"GPU job failed: {failed}")
        if len(gpu_jobs) == expected_gpus and succeeded == expected_gpus:
            return
        time.sleep(60)
    fail_with_diagnostics(namespace, run_id, "timeout waiting for GPU jobs")


def fail_with_diagnostics(namespace: str, run_id: str, message: str) -> None:
    print(kubectl(["get", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"], check=False).stdout)
    print(kubectl(["describe", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}"], check=False).stdout)
    print(kubectl(["logs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "--all-containers=true", "--tail=200"], check=False).stdout)
    raise SystemExit(message)


def verify_parallel(namespace: str, run_id: str, max_start_spread_s: int, expected_gpus: int) -> dict[str, Any]:
    data = json.loads(kubectl(["get", "pods", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "-o", "json"]).stdout)
    gpu_pods = [pod for pod in data["items"] if pod["metadata"]["labels"].get("perfseer-gpu-key")]
    starts = []
    nodes = {}
    for pod in gpu_pods:
        start = pod.get("status", {}).get("startTime")
        if start:
            starts.append(datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(timezone.utc))
        nodes[pod["metadata"]["labels"]["perfseer-gpu-key"]] = pod.get("spec", {}).get("nodeName")
    if len(gpu_pods) != expected_gpus:
        raise SystemExit(f"expected {expected_gpus} GPU pods, got {len(gpu_pods)}")
    if len(starts) >= 2 and (max(starts) - min(starts)).total_seconds() > max_start_spread_s:
        raise SystemExit("GPU pods did not start close enough to verify parallel run")
    return {"gpu_pods": len(gpu_pods), "nodes": nodes, "start_spread_s": (max(starts) - min(starts)).total_seconds() if len(starts) >= 2 else 0}


def wait_verifier(namespace: str, run_id: str, pvc: str, image: str, output_root: str, timeout_s: int, expected_gpus: int) -> str:
    yaml_text = verifier_job_yaml(namespace, run_id, pvc, image, output_root, expected_gpus)
    kubectl(["apply", "-f", "-"], input_text=yaml_text)
    deadline = time.time() + timeout_s
    name = clean_name(f"perfseer-e2e-{run_id}-verify")
    while time.time() < deadline:
        proc = kubectl(["get", "job", name, "-n", namespace, "-o", "json"], check=False)
        if proc.returncode == 0:
            job = json.loads(proc.stdout)
            if int(job.get("status", {}).get("succeeded", 0)) >= 1:
                return kubectl(["logs", "-n", namespace, f"job/{name}", "--all-containers=true"], check=False).stdout
            if int(job.get("status", {}).get("failed", 0)) > 0:
                fail_with_diagnostics(namespace, run_id, "verifier job failed")
        time.sleep(20)
    fail_with_diagnostics(namespace, run_id, "timeout waiting for verifier job")
    return ""


def cleanup(namespace: str, run_id: str) -> None:
    kubectl(["delete", "jobs", "-n", namespace, "-l", f"perfseer-run-id={run_id}", "--ignore-not-found=true"], check=False)
    kubectl(["delete", "configmap", "-n", namespace, f"perfseer-e2e-{run_id}", "--ignore-not-found=true"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nautilus label sampling and verify end to end.")
    parser.add_argument("--namespace", default=namespace_default())
    parser.add_argument("--pvc", default="test-pvc")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--verifier-image", default=VERIFIER_IMAGE)
    parser.add_argument("--gpus", default="a10,l4,t4,v100", help="Comma-separated GPU presets or all-readme.")
    parser.add_argument("--active-gpus", type=int, default=4)
    parser.add_argument("--pending-timeout-seconds", type=int, default=300)
    parser.add_argument("--blacklist-ttl-seconds", type=int, default=21600)
    parser.add_argument("--max-retries-per-gpu", type=int, default=3)
    parser.add_argument("--node-map", default="", help="Comma-separated gpu:nodeName overrides.")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if args.gpus.strip().lower() == "all-readme":
        gpu_keys = list(README_GPU_KEYS)
    else:
        gpu_keys = [item.strip().lower() for item in args.gpus.split(",") if item.strip()]
    if not gpu_keys or len(set(gpu_keys)) != len(gpu_keys):
        raise SystemExit("--gpus must contain distinct GPU presets")
    unknown = [key for key in gpu_keys if key not in GPU_PRESETS]
    if unknown:
        raise SystemExit(f"unknown GPU preset(s): {unknown}")
    node_map = {}
    if args.node_map:
        for item in args.node_map.split(","):
            if not item.strip():
                continue
            key, value = item.split(":", 1)
            node_map[key.strip().lower()] = value.strip()

    run_id = clean_name(args.run_id)
    output_root = f"/workspace/perfseer-e2e/{run_id}/labels"
    record_dir = Path("record")
    record_dir.mkdir(exist_ok=True)
    record_file = record_dir / f"nautilus_e2e_{run_id}.md"
    record_file.write_text(f"# Nautilus E2E {run_id}\n\n- Namespace: {args.namespace}\n\n- PVC: {args.pvc}\n\n- GPUs: {', '.join(gpu_keys)}\n\n", encoding="utf-8")

    monitor: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = Path(tmp)
        prepare_configmap_dir(source_dir)
        create_configmap(args.namespace, run_id, source_dir)
        yaml_file = record_dir / f"nautilus_e2e_{run_id}.yaml"
        yaml_file.write_text("", encoding="utf-8")
        controller_state = ControllerState(
            candidate_gpus=gpu_keys,
            active_limit=min(args.active_gpus, len(gpu_keys)),
            pending_timeout_seconds=args.pending_timeout_seconds,
            blacklist_ttl_seconds=args.blacklist_ttl_seconds,
        )
        monitor = start_monitor(args.namespace, run_id, record_file)
        wait_jobs_with_switching(
            args.namespace,
            run_id,
            args.pvc,
            args.image,
            output_root,
            node_map,
            record_file,
            yaml_file,
            args.timeout_seconds,
            controller_state,
            args.max_retries_per_gpu,
        )
        expected_gpus = controller_state.succeeded_count()
        if expected_gpus <= 0:
            raise SystemExit("no GPU label outputs were produced")
        parallel_report = verify_parallel(args.namespace, run_id, args.timeout_seconds, expected_gpus)
        verifier_logs = wait_verifier(args.namespace, run_id, args.pvc, args.verifier_image, output_root, 900, expected_gpus)
        cluster_summary = kubectl(["get", "jobs", "-n", args.namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"], check=False).stdout
        pod_summary = kubectl(["get", "pods", "-n", args.namespace, "-l", f"perfseer-run-id={run_id}", "-o", "wide"], check=False).stdout
        with record_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Controller States\n\n```json\n{json.dumps({key: value.status for key, value in controller_state.gpu_states.items()}, indent=2, sort_keys=True)}\n```\n")
            handle.write(f"\n## Parallel Report\n\n```json\n{json.dumps(parallel_report, indent=2, sort_keys=True)}\n```\n")
            handle.write(f"\n## Verifier Logs\n\n```text\n{verifier_logs}\n```\n")
            handle.write(f"\n## Jobs\n\n```text\n{cluster_summary}\n```\n")
            handle.write(f"\n## Pods\n\n```text\n{pod_summary}\n```\n")
        print(json.dumps({"run_id": run_id, "record": str(record_file), "parallel": parallel_report, "verifier_logs": verifier_logs}, indent=2, sort_keys=True))
    if monitor is not None:
        monitor.terminate()
    if not args.keep:
        cleanup(args.namespace, run_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
