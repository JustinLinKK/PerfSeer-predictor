#!/usr/bin/env python3
"""Verify the Nautilus label-sampling workflow and produced labels."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import verify_sampled_labels


def fail(message: str) -> None:
    raise SystemExit(message)


def load_yaml_jobs(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return [doc for doc in yaml.safe_load_all(handle) if doc]
    except Exception:
        text = path.read_text(encoding="utf-8")
        jobs = []
        for block in re.split(r"^---\s*$", text, flags=re.MULTILINE):
            if block.strip():
                jobs.append({"raw": block})
        return jobs


def job_command(job: dict[str, Any]) -> str:
    if "raw" in job:
        return str(job["raw"])
    container = job["spec"]["template"]["spec"]["containers"][0]
    args = container.get("args", [])
    return "\n".join(str(item) for item in args)


def job_gpu_identity(job: dict[str, Any]) -> str:
    if "raw" in job:
        raw = str(job["raw"])
        product = re.search(r"nvidia.com/gpu.product.*?values:\s*- ([^\n]+)", raw, re.S)
        resource = re.search(r"(nvidia.com/[a-z0-9-]+): \"1\"", raw)
        return (product.group(1).strip() if product else resource.group(1).strip() if resource else "")
    pod_spec = job["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    limits = container.get("resources", {}).get("limits", {})
    resource = next((key for key in limits if key.startswith("nvidia.com/")), "")
    affinity = pod_spec.get("affinity", {})
    text = json.dumps(affinity, sort_keys=True)
    product = ""
    match = re.search(r"NVIDIA-[A-Za-z0-9-]+|Tesla-[A-Za-z0-9-]+|Quadro-[A-Za-z0-9-]+", text)
    if match:
        product = match.group(0)
    return product or resource


def verify_jobs_yaml(path: Path) -> dict[str, Any]:
    jobs = load_yaml_jobs(path)
    if len(jobs) != 4:
        fail(f"expected 4 jobs, got {len(jobs)}")
    names = []
    gpu_ids = []
    for job in jobs:
        if "raw" in job:
            raw = str(job["raw"])
            name_match = re.search(r"name: ([a-z0-9-]+)", raw)
            names.append(name_match.group(1) if name_match else "")
            parallelism_match = re.search(r"parallelism: (\d+)", raw)
            if not parallelism_match or int(parallelism_match.group(1)) < 1:
                fail("job parallelism missing")
        else:
            if job.get("kind") != "Job":
                fail("YAML document is not a Job")
            names.append(job["metadata"]["name"])
            if int(job["spec"].get("parallelism", 0)) < 1:
                fail(f"{job['metadata']['name']} parallelism invalid")
        command = job_command(job)
        for token in ("--warmup-epochs 1", "--profile-epochs 1", "verify_sampled_labels.py"):
            if token not in command:
                fail(f"missing {token} in job command")
        gpu_ids.append(job_gpu_identity(job))
    if len(set(names)) != 4:
        fail("job names are not distinct")
    if len(set(gpu_ids)) != 4:
        fail(f"GPU requests are not four distinct identities: {gpu_ids}")
    return {"jobs": len(jobs), "job_names": names, "gpu_identities": gpu_ids}


def kubectl_json(args: list[str]) -> dict[str, Any]:
    output = subprocess.check_output(["kubectl", *args, "-o", "json"], text=True)
    return json.loads(output)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def verify_cluster(namespace: str, prefix: str) -> dict[str, Any]:
    jobs = kubectl_json(["get", "jobs", "-n", namespace])
    matched_jobs = [item for item in jobs.get("items", []) if item.get("metadata", {}).get("name", "").startswith(prefix)]
    if len(matched_jobs) != 4:
        fail(f"expected 4 Kubernetes Jobs with prefix {prefix}, got {len(matched_jobs)}")
    for job in matched_jobs:
        spec = job.get("spec", {})
        status = job.get("status", {})
        if int(spec.get("parallelism", 0)) < 1:
            fail(f"{job['metadata']['name']} has invalid parallelism")
        if int(status.get("succeeded", 0)) < int(spec.get("completions", 1)):
            fail(f"{job['metadata']['name']} not completed")
    pods = kubectl_json(["get", "pods", "-n", namespace])
    matched_pods = []
    for pod in pods.get("items", []):
        labels = pod.get("metadata", {}).get("labels", {})
        job_name = labels.get("job-name", "")
        if job_name.startswith(prefix):
            matched_pods.append(pod)
    if not matched_pods:
        fail("no pods found for workflow")
    start_times = [parse_time(pod.get("status", {}).get("startTime")) for pod in matched_pods]
    start_times = [value for value in start_times if value is not None]
    if len(start_times) >= 2 and (max(start_times) - min(start_times)).total_seconds() > 1800:
        fail("pods did not start within 30 minutes; parallel run not verified")
    return {"kubernetes_jobs": len(matched_jobs), "kubernetes_pods": len(matched_pods)}


def verify_results(path: Path) -> dict[str, Any]:
    result_files = sorted(path.rglob("results_shard*.jsonl"))
    if not result_files:
        fail(f"no result files under {path}")
    checked = 0
    ok_rows = 0
    bad_rows = 0
    gpu_names = set()
    for file_path in result_files:
        for _path, _line_number, row in verify_sampled_labels.iter_rows([file_path]):
            checked += 1
            if row.get("status") == "ok":
                ok_rows += 1
                hardware = row.get("hardware", {})
                gpu_name = hardware.get("gpu_name") or hardware.get("nvidia_smi") or hardware.get("device")
                if gpu_name:
                    gpu_names.add(str(gpu_name))
            errors = verify_sampled_labels.verify_row(row)
            if errors:
                bad_rows += 1
                print(f"{file_path}: {row.get('model_id', '<unknown>')}: {', '.join(errors)}", file=sys.stderr)
    if bad_rows:
        fail(f"{bad_rows} label rows failed verification")
    if ok_rows == 0:
        fail("no ok rows found")
    if len(gpu_names) < 4:
        fail(f"expected labels from 4 different GPUs, got {len(gpu_names)}: {sorted(gpu_names)}")
    return {"result_files": len(result_files), "rows": checked, "ok_rows": ok_rows, "gpu_names": sorted(gpu_names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Nautilus label workflow, cluster status, and labels.")
    parser.add_argument("--jobs-yaml", help="Rendered Kubernetes Job YAML.")
    parser.add_argument("--namespace", help="Kubernetes namespace for live verification.")
    parser.add_argument("--job-prefix", default="model-label-sampler")
    parser.add_argument("--results-dir", help="Root output directory containing GPU label outputs.")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.jobs_yaml:
        report["jobs_yaml"] = verify_jobs_yaml(Path(args.jobs_yaml))
    if args.namespace:
        report["cluster"] = verify_cluster(args.namespace, args.job_prefix)
    if args.results_dir:
        report["results"] = verify_results(Path(args.results_dir))
    if not report:
        fail("provide at least one of --jobs-yaml, --namespace, --results-dir")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
