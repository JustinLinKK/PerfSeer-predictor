#!/usr/bin/env python3
"""Run the Kaggle-backed PerfSeer workload workflow on Nautilus."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from run_nrp_source_workflow_local import (
        GPU_PRESETS,
        ROOT,
        clean_name,
        copy_from_workflow_pvc,
        kubectl,
        print_diagnostics,
        run,
        shell_quote,
        stage_local_repo,
        start_monitor,
        validate_kubernetes_name_prefix,
        validate_profile_image_reference,
    )
except ModuleNotFoundError:
    from scripts.run_nrp_source_workflow_local import (
        GPU_PRESETS,
        ROOT,
        clean_name,
        copy_from_workflow_pvc,
        kubectl,
        print_diagnostics,
        run,
        shell_quote,
        stage_local_repo,
        start_monitor,
        validate_kubernetes_name_prefix,
        validate_profile_image_reference,
    )


DEFAULT_DATASET_IDS = (
    "cassava_leaf_disease,"
    "pothole_image_segmentation,"
    "taco_yolo_object_detection,"
    "jigsaw_toxic_comment,"
    "cnn_dailymail_summarization,"
    "animal_audio_classification,"
    "store_sales_time_series,"
    "credit_card_default,"
    "ogbn_products"
)


def parse_gpus(value: str) -> list[str]:
    keys = [item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip()]
    if len(keys) != 1:
        raise ValueError("--gpus must contain exactly one preset for a single four-GPU Indexed Job")
    unknown = [key for key in keys if key not in GPU_PRESETS]
    if unknown:
        raise ValueError(f"unknown GPU preset(s): {', '.join(unknown)}")
    return keys


def workflow_paths(args: argparse.Namespace) -> dict[str, str]:
    root = args.workflow_dir.rstrip("/")
    return {
        "repo": args.repo_dir.rstrip("/"),
        "raw": f"{args.repo_dir.rstrip('/')}/datasets/raw",
        "prepared": f"{args.repo_dir.rstrip('/')}/datasets/prepared",
        "pack": f"{root}/pack",
        "workloads": f"{root}/workload_specs_real",
        "results": f"{root}/results/{args.hardware_id}",
        "package": f"{root}/perfseer_{args.hardware_id}_real_dataset_labels.tar.gz",
    }


def create_kaggle_secret(args: argparse.Namespace) -> None:
    if args.skip_kaggle_secret:
        return
    kaggle_json = Path(args.kaggle_json).expanduser().resolve()
    if not kaggle_json.is_file():
        raise FileNotFoundError(f"Kaggle credential file is missing: {kaggle_json}")
    mode = kaggle_json.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"Kaggle credential file must not be group/world readable: mode={mode:o}")
    kubectl(["delete", "secret", args.kaggle_secret_name, "-n", args.namespace, "--ignore-not-found=true"], check=False)
    kubectl(
        [
            "create",
            "secret",
            "generic",
            args.kaggle_secret_name,
            "-n",
            args.namespace,
            f"--from-file=kaggle.json={kaggle_json}",
        ]
    )
    print(json.dumps({"event": "kaggle_secret_ready", "secret": args.kaggle_secret_name}, sort_keys=True), flush=True)


def prepare_job_yaml(args: argparse.Namespace) -> str:
    paths = workflow_paths(args)
    dataset_ids = ",".join(item.strip() for item in args.dataset_ids.split(",") if item.strip())
    bootstrap = f"{args.bootstrap_command} && " if args.bootstrap_command else ""
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {args.job_prefix}-prepare-real
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: prepare-real
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: prepare-real
    spec:
      restartPolicy: Never
      containers:
      - name: prepare-real
        image: {args.image}
        imagePullPolicy: IfNotPresent
        workingDir: {paths["repo"]}
        env:
        - name: KAGGLE_CONFIG_DIR
          value: /root/.kaggle
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          export PYTHONPATH="{paths["repo"]}/src:{paths["repo"]}:\\${{PYTHONPATH:-}}"
          chmod 600 /root/.kaggle/kaggle.json || true
          {bootstrap}
          python -m pip install --no-cache-dir kaggle ogb torch_geometric networkx scikit-learn tqdm nvidia-ml-py pyyaml pandas
          mkdir -p {shell_quote(paths["raw"])} {shell_quote(paths["prepared"])} {shell_quote(paths["pack"])} {shell_quote(paths["workloads"])}
          python - <<'PY' > {shell_quote(args.workflow_dir.rstrip("/") + "/dataset_plan.tsv")}
          import json
          from pathlib import Path
          wanted = [item.strip() for item in {dataset_ids!r}.split(",") if item.strip()]
          registry = json.loads(Path("dataset_sources/registry.json").read_text())
          rows = registry.get("datasets", [])
          by_id = {{str(row["id"]): row for row in rows}}
          selected = [by_id[item] for item in wanted]
          for row in selected:
              print(f"{{row['id']}}\\t{{row.get('source_type', '')}}\\t{{row.get('task_family', '')}}")
          PY
          while IFS=$'\\t' read -r DATASET_ID SOURCE_TYPE TASK_FAMILY; do
            test -n "$DATASET_ID"
            if [[ "$SOURCE_TYPE" == "pyg_ogb" ]]; then
              python scripts/manage_dataset_sources.py download-ogb "$DATASET_ID" --raw-root {shell_quote(paths["raw"])} --allow-nautilus-only
            else
              python scripts/manage_dataset_sources.py download "$DATASET_ID" --raw-root {shell_quote(paths["raw"])} --allow-nautilus-only
            fi
            python scripts/manage_dataset_sources.py prepare "$DATASET_ID" --raw-root {shell_quote(paths["raw"])} --prepared-root {shell_quote(paths["prepared"])} --force
          done < {shell_quote(args.workflow_dir.rstrip("/") + "/dataset_plan.tsv")}
          python nrp_calibration_pack/generate_model_sources.py \\
            --catalog-mode template \\
            --subset-size {args.subset_size} \\
            --seed {args.seed} \\
            --out-dir {shell_quote(paths["pack"])} \\
            --precision-sweep fp32_ieee \\
            --validation-mode compile \\
            --generation-workers {args.generation_workers} \\
            --force
          python nrp_calibration_pack/profile/make_workload_specs.py \\
            --manifest {shell_quote(paths["pack"] + "/manifest/subset_manifest.jsonl")} \\
            --registry dataset_sources/registry.json \\
            --dataset-profile-root {shell_quote(paths["prepared"])} \\
            --raw-root {shell_quote(paths["raw"])} \\
            --output-dir {shell_quote(paths["workloads"])} \\
            --subset-id {args.workload_subset_id} \\
            --batch-size {args.workload_batch_size} \\
            --precision-sweep {args.workload_precision_sweep} \\
            --optimizer {args.optimizer} \\
            --hardware-id {args.hardware_id} \\
            --force
          python - <<'PY'
          import json
          from pathlib import Path
          workloads = Path({(paths["workloads"] + "/workloads.jsonl")!r})
          rows = [json.loads(line) for line in workloads.read_text().splitlines() if line.strip()]
          if len(rows) != {args.subset_size}:
              raise SystemExit(f"expected {args.subset_size} workload rows, got {{len(rows)}}")
          bad = [row.get("profile_point_id") for row in rows if not row.get("dataset", {{}}).get("real_dataloader_backed")]
          if bad:
              raise SystemExit(f"real_dataloader_backed verifier failed: {{len(bad)}} bad row(s)")
          print(json.dumps({{"event": "real_workload_verifier_ok", "rows": len(rows)}}), flush=True)
          PY
        resources:
          requests:
            cpu: "8"
            memory: "48Gi"
          limits:
            cpu: "8"
            memory: "48Gi"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
        - name: kaggle
          mountPath: /root/.kaggle
          readOnly: true
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: {args.pvc}
      - name: kaggle
        secret:
          secretName: {args.kaggle_secret_name}
          defaultMode: 384
"""


def gpu_affinity(products: list[str]) -> str:
    values = "\n".join(f"                - {product}" for product in products)
    return f"""      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
{values}
"""


def profile_job_yaml(args: argparse.Namespace) -> str:
    paths = workflow_paths(args)
    gpu_key = parse_gpus(args.gpus)[0]
    preset = GPU_PRESETS[gpu_key]
    bootstrap = f"{args.bootstrap_command} && " if args.bootstrap_command else ""
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {args.job_prefix}-profile-real
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: profile-real
spec:
  completionMode: Indexed
  completions: {args.completions}
  parallelism: {args.parallelism}
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: profile-real
    spec:
      restartPolicy: Never
{gpu_affinity(list(preset["products"]))}      containers:
      - name: profile-real
        image: {args.image}
        imagePullPolicy: IfNotPresent
        workingDir: {paths["repo"]}
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        - name: JOB_COMPLETIONS
          value: "{args.completions}"
        command: ["/bin/bash", "-lc"]
        args:
        - |
          set -euo pipefail
          export PYTHONPATH="{paths["repo"]}/src:{paths["repo"]}:\\${{PYTHONPATH:-}}"
          {bootstrap}
          python -m pip install --no-cache-dir ogb torch_geometric networkx scikit-learn tqdm nvidia-ml-py pyyaml pandas
          python nrp_calibration_pack/profile/run_profile.py \\
            --workload-specs {shell_quote(paths["workloads"] + "/workloads.jsonl")} \\
            --models-dir {shell_quote(paths["pack"] + "/models")} \\
            --output-dir {shell_quote(paths["results"])} \\
            --hardware-id {args.hardware_id} \\
            --device cuda \\
            --precision-sweep {args.profile_precision_sweep} \\
            --resume \\
            --num-shards {args.completions} \\
            --shard-index \\${{JOB_COMPLETION_INDEX:-0}} \\
            --warmup {args.warmup} \\
            --infer-repeats {args.infer_repeats} \\
            --train-repeats {args.train_repeats} \\
            --label-time-mode {args.label_time_mode} \\
            --time-label-warmup-epochs {args.time_label_warmup_epochs} \\
            --time-label-measured-epochs {args.time_label_measured_epochs} \\
            --sample-interval {args.sample_interval} \\
            --min-phase-seconds {args.min_phase_seconds} \\
            --min-sampler-samples {args.min_sampler_samples} \\
            --optimizer {args.optimizer} \\
            --sm-occupancy-source {args.sm_occupancy_source} \\
            --resource-profile-mode {args.resource_profile_mode}
        resources:
          requests:
            cpu: "4"
            memory: "24Gi"
            {preset["resource"]}: "1"
          limits:
            cpu: "4"
            memory: "24Gi"
            {preset["resource"]}: "1"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: {args.pvc}
"""


def package_job_yaml(args: argparse.Namespace) -> str:
    paths = workflow_paths(args)
    root = args.workflow_dir.rstrip("/")
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {args.job_prefix}-package-real
  namespace: {args.namespace}
  labels:
    app: {args.job_prefix}
    stage: package-real
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app: {args.job_prefix}
        stage: package-real
    spec:
      restartPolicy: Never
      containers:
      - name: package-real
        image: {args.utility_image}
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh", "-lc"]
        args:
        - |
          set -eu
          cd {shell_quote(root)}
          tar -czf {shell_quote(paths["package"])} \\
            dataset_plan.tsv \\
            pack/manifest \\
            pack/models \\
            workload_specs_real \\
            results/{args.hardware_id}
          tar -tzf {shell_quote(paths["package"])} | tee {shell_quote(root + "/real_dataset_package_contents.txt")}
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "2"
            memory: "8Gi"
        volumeMounts:
        - name: output
          mountPath: /mnt/output
      volumes:
      - name: output
        persistentVolumeClaim:
          claimName: {args.pvc}
"""


def job_json(namespace: str, name: str) -> dict[str, Any] | None:
    proc = kubectl(["get", "job", name, "-n", namespace, "-o", "json"], check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def submit_yaml(yaml_text: str, args: argparse.Namespace, job_name: str) -> None:
    kubectl(["delete", "job", job_name, "-n", args.namespace, "--ignore-not-found=true"], check=False)
    kubectl(["apply", "-f", "-"], input_text=yaml_text)
    print_diagnostics(args.namespace, job_name)


def wait_for_named_job(args: argparse.Namespace, job_name: str) -> None:
    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        data = job_json(args.namespace, job_name)
        if data is None:
            time.sleep(max(args.poll_seconds, 1))
            continue
        status = data.get("status", {})
        spec = data.get("spec", {})
        if int(status.get("succeeded", 0)) >= int(spec.get("completions", 1)):
            print_diagnostics(args.namespace, job_name)
            return
        failed_condition = any(
            condition.get("type") == "Failed" and condition.get("status") == "True"
            for condition in status.get("conditions", [])
        )
        if failed_condition:
            print_diagnostics(args.namespace, job_name)
            raise RuntimeError(f"job failed: {job_name}")
        time.sleep(max(args.poll_seconds, 1))
    print_diagnostics(args.namespace, job_name)
    raise TimeoutError(f"timed out waiting for {job_name}")


def verify_label_package(path: Path, expected_rows: int) -> dict[str, int]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"label package is missing or empty: {path}")
    counts = {"results": 0, "label_v3": 0, "scheduler_resource": 0, "hardware": 0, "models": 0}
    with tarfile.open(path, "r:*") as tar:
        for member in tar.getmembers():
            name = member.name.lstrip("./")
            if not member.isfile():
                continue
            if name.endswith(".py") and name.startswith("pack/models/") and not name.endswith("__init__.py"):
                counts["models"] += 1
            elif "/results_shard" in name and name.endswith(".jsonl"):
                data = tar.extractfile(member)
                if data is not None:
                    counts["results"] += sum(1 for line in data.read().decode("utf-8", errors="ignore").splitlines() if line.strip())
            elif "/label_v3_shard" in name and name.endswith(".jsonl"):
                data = tar.extractfile(member)
                if data is not None:
                    counts["label_v3"] += sum(1 for line in data.read().decode("utf-8", errors="ignore").splitlines() if line.strip())
            elif "/scheduler_resource_shard" in name and name.endswith(".jsonl"):
                data = tar.extractfile(member)
                if data is not None:
                    counts["scheduler_resource"] += sum(1 for line in data.read().decode("utf-8", errors="ignore").splitlines() if line.strip())
            elif "/hardware_shard" in name and name.endswith(".json"):
                counts["hardware"] += 1
    if counts["models"] != expected_rows:
        raise ValueError(f"expected {expected_rows} model source files in package, got {counts['models']}")
    if counts["results"] < expected_rows:
        raise ValueError(f"expected at least {expected_rows} result rows, got {counts['results']}")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kaggle-backed PerfSeer real-dataset labeling on Nautilus.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--allow-mutable-image-tag", action="store_true")
    parser.add_argument("--utility-image", default="alpine:3.20")
    parser.add_argument("--pvc", required=True)
    parser.add_argument("--job-prefix", default=f"perfseer-real-{datetime.now().strftime('%m%d%H%M%S')}")
    parser.add_argument("--workflow-dir", default="/mnt/output/perfseer_real_dataset_workflow")
    parser.add_argument("--repo-dir", default="/workspace/PerfSeer-predictor")
    parser.add_argument("--stage-local-repo", action="store_true")
    parser.add_argument("--local-repo-dir", default=str(ROOT))
    parser.add_argument("--staged-repo-dir", default="")
    parser.add_argument("--staging-pod-name", default="")
    parser.add_argument("--download-pod-name", default="")
    parser.add_argument("--keep-staging-pod", action="store_true")
    parser.add_argument("--keep-download-pod", action="store_true")
    parser.add_argument("--kaggle-json", default="~/.kaggle/kaggle.json")
    parser.add_argument("--kaggle-secret-name", default="")
    parser.add_argument("--skip-kaggle-secret", action="store_true")
    parser.add_argument("--dataset-ids", default=DEFAULT_DATASET_IDS)
    parser.add_argument("--gpus", default="a10")
    parser.add_argument("--hardware-id", default="a10")
    parser.add_argument("--subset-size", type=int, default=10005)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--generation-workers", type=int, default=0)
    parser.add_argument("--workload-subset-id", default="tiny")
    parser.add_argument("--workload-batch-size", type=int, default=8)
    parser.add_argument("--workload-precision-sweep", default="fp32_ieee")
    parser.add_argument("--profile-precision-sweep", default="auto")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--completions", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--infer-repeats", type=int, default=50)
    parser.add_argument("--train-repeats", type=int, default=50)
    parser.add_argument("--label-time-mode", default="measured_epochs", choices=("step_extrapolated", "measured_epochs"))
    parser.add_argument("--time-label-warmup-epochs", type=int, default=1)
    parser.add_argument("--time-label-measured-epochs", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--min-phase-seconds", type=float, default=20)
    parser.add_argument("--min-sampler-samples", type=int, default=100)
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--sm-occupancy-source", default="nvml_proxy", choices=("ncu", "nvml_proxy"))
    parser.add_argument("--resource-profile-mode", default="sustained", choices=("compat", "sustained"))
    parser.add_argument("--bootstrap-command", default="")
    parser.add_argument("--local-output-dir", default="nrp_downloads")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=604800)
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    parser.add_argument("--kubectl-request-timeout", default="30s")
    parser.add_argument("--kubectl-hard-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--monitor-log", default="")
    parser.add_argument("--monitor-interval-seconds", type=int, default=60)
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-local-verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.job_prefix = clean_name(args.job_prefix)
    if not args.kaggle_secret_name:
        args.kaggle_secret_name = clean_name(f"{args.job_prefix}-kaggle")
    os.environ["PERFSEER_KUBECTL_REQUEST_TIMEOUT"] = args.kubectl_request_timeout
    os.environ["PERFSEER_KUBECTL_HARD_TIMEOUT_SECONDS"] = str(args.kubectl_hard_timeout_seconds)
    validate_kubernetes_name_prefix(args.job_prefix, "--job-prefix")
    validate_profile_image_reference(args.image, args.allow_mutable_image_tag)
    parse_gpus(args.gpus)

    if args.stage_local_repo:
        staged_repo_dir = args.staged_repo_dir or f"{args.workflow_dir.rstrip('/')}/repo"
        args.repo_dir = staged_repo_dir

    record_dir = ROOT / "record"
    record_dir.mkdir(exist_ok=True)
    yaml_path = record_dir / f"{args.job_prefix}_real_dataset_workflow.yaml"
    yaml_text = "---\n".join([prepare_job_yaml(args), profile_job_yaml(args), package_job_yaml(args)])
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(json.dumps({"event": "workflow_yaml_written", "path": str(yaml_path)}, sort_keys=True), flush=True)
    if args.dry_run:
        return

    create_kaggle_secret(args)
    monitor_proc, _monitor_log = start_monitor(args)
    try:
        if args.stage_local_repo:
            stage_local_repo(args)
        stages = [
            ("prepare-real", prepare_job_yaml(args)),
            ("profile-real", profile_job_yaml(args)),
            ("package-real", package_job_yaml(args)),
        ]
        for suffix, yaml_doc in stages:
            job_name = f"{args.job_prefix}-{suffix}"
            submit_yaml(yaml_doc, args, job_name)
            wait_for_named_job(args, job_name)
        local_dir = Path(args.local_output_dir).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        package_name = Path(workflow_paths(args)["package"]).name
        local_package = local_dir / package_name
        copy_from_workflow_pvc(args, workflow_paths(args)["package"], local_package)
        report: dict[str, Any] = {"label_package": str(local_package)}
        if not args.skip_local_verify:
            report["verification"] = verify_label_package(local_package, args.subset_size)
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        if monitor_proc is not None:
            monitor_proc.terminate()
        if not args.keep_download_pod:
            pod = args.download_pod_name or f"{args.job_prefix}-download"
            kubectl(["delete", "pod", pod, "-n", args.namespace, "--ignore-not-found=true", "--wait=false"], check=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
