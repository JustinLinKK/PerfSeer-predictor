#!/usr/bin/env python3
"""Render Nautilus GPU label-sampling Jobs for model source folders."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


GPU_PRESETS = {
    "a10": {"resource": "nvidia.com/gpu", "product": "NVIDIA-A10"},
    "a40": {"resource": "nvidia.com/a40", "product": ""},
    "a100": {"resource": "nvidia.com/a100", "product": ""},
    "l4": {"resource": "nvidia.com/gpu", "product": "NVIDIA-L4"},
    "l40": {"resource": "nvidia.com/gpu", "product": "NVIDIA-L40"},
    "l40s": {"resource": "nvidia.com/gpu", "product": "NVIDIA-L40S"},
    "rtx-a4000": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-A4000"},
    "rtx-a5000": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-A5000"},
    "rtx-a6000": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-A6000"},
    "rtx-4000-ada": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-4000-Ada-Generation"},
    "rtx-5000-ada": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-5000-Ada-Generation"},
    "rtx-pro-6000-blackwell": {"resource": "nvidia.com/gpu", "product": "NVIDIA-RTX-PRO-6000-Blackwell"},
    "quadro-rtx-6000": {"resource": "nvidia.com/gpu", "product": "Quadro-RTX-6000"},
    "quadro-rtx-8000": {"resource": "nvidia.com/gpu", "product": "Quadro-RTX-8000"},
    "t4": {"resource": "nvidia.com/gpu", "product": "Tesla-T4"},
    "v100": {"resource": "nvidia.com/gpu", "product": "Tesla-V100"},
}


def clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def node_affinity(product: str) -> str:
    if not product:
        return ""
    return f"""
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
                      - {product}"""


def render_job(args: argparse.Namespace, gpu_key: str) -> str:
    preset = GPU_PRESETS[gpu_key]
    job_name = clean_name(f"{args.name}-{gpu_key}")
    output_dir = f"{args.output_root}/{gpu_key}"
    affinity = node_affinity(preset["product"])
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {args.namespace}
spec:
  completions: {args.completions}
  parallelism: {args.parallelism}
  completionMode: Indexed
  backoffLimit: {args.backoff_limit}
  template:
    spec:{affinity}
      restartPolicy: Never
      containers:
        - name: label-sampler
          image: {args.image}
          imagePullPolicy: IfNotPresent
          command: ["/bin/bash", "-lc"]
          args:
            - >
              python /workspace/nrp_calibration_pack/profile/run_profile.py
              --manifest {args.manifest}
              --models-dir {args.models_dir}
              --output-dir {output_dir}
              --device cuda
              --warmup-epochs {args.warmup_epochs}
              --profile-epochs {args.profile_epochs}
              --batches-per-epoch {args.batches_per_epoch}
              --sample-interval {args.sample_interval}
              --optimizer {args.optimizer}
              --precision-config {args.precision_config}
              &&
              python /workspace/scripts/verify_sampled_labels.py {output_dir}
          resources:
            limits:
              {preset["resource"]}: "1"
            requests:
              cpu: "{args.cpu}"
              memory: "{args.memory}"
              {preset["resource"]}: "1"
          volumeMounts:
            - name: work
              mountPath: /workspace
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {args.pvc}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render four Nautilus GPU label-sampling jobs.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pvc", required=True, help="PVC mounted at /workspace.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--manifest", default="/workspace/manifest/source_manifest.jsonl")
    parser.add_argument("--models-dir", default="/workspace/models")
    parser.add_argument("--output-root", default="/workspace/labels")
    parser.add_argument("--name", default="model-label-sampler")
    parser.add_argument("--gpus", default="a100,a40,l40s,v100")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--completions", type=int, default=64)
    parser.add_argument("--backoff-limit", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--profile-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--precision-config", default="fp32_ieee")
    parser.add_argument("--cpu", default="4")
    parser.add_argument("--memory", default="16Gi")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gpu_keys = [item.strip().lower() for item in args.gpus.split(",") if item.strip()]
    if len(set(gpu_keys)) != 4:
        raise SystemExit("--gpus must contain exactly four distinct GPU presets")
    unknown = [key for key in gpu_keys if key not in GPU_PRESETS]
    if unknown:
        raise SystemExit(f"unknown GPU preset(s): {', '.join(unknown)}")

    docs = [render_job(args, key) for key in gpu_keys]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("---\n".join(docs), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
