#!/usr/bin/env bash
set -euo pipefail
cd /home/downeyflyfan/Research_Projects/AI/Agents/Agents_Scheduler/PerfSeer-predictor

python3 -u scripts/run_nrp_real_dataset_workflow_local.py \
  --namespace ecepxie \
  --image pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel \
  --allow-mutable-image-tag \
  --utility-image alpine:3.20 \
  --pvc perfseer-real-dataset-pvc \
  --job-prefix real-a10-0629060430 \
  --workflow-dir /mnt/output/real-a10-0629060430 \
  --hardware-id a10 \
  --stage-local-repo \
  --subset-size 10005 \
  --completions 64 \
  --parallelism 4 \
  --gpus a10 \
  --warmup 20 \
  --infer-repeats 50 \
  --train-repeats 50 \
  --label-time-mode measured_epochs \
  --time-label-warmup-epochs 1 \
  --time-label-measured-epochs 2 \
  --sample-interval 0.01 \
  --min-phase-seconds 20 \
  --min-sampler-samples 100 \
  --optimizer adam \
  --sm-occupancy-source nvml_proxy \
  --resource-profile-mode sustained \
  --local-output-dir ../labels/real-a10-0629060430 \
  --timeout-seconds 604800 \
  --stage-timeout-seconds 1800 \
  --poll-seconds 60 \
  --kubectl-request-timeout 30s \
  --kubectl-hard-timeout-seconds 180 \
  --monitor-interval-seconds 60
