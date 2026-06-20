# Nautilus Nsight Compute Two-GPU Permission Check

Date: 2026-06-19

Run ID: `ncu4-20260619191658`

Namespace: `ecepxie`

Persistent Volume Claim: `test-pvc`

Container image: `pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel`

Command:

```bash
python3 scripts/run_nautilus_label_sampling_e2e.py --namespace ecepxie --pvc test-pvc --gpus a40,a100 --timeout-seconds 1800 --run-id ncu4-20260619191658
```

## Experiment Setting

- Requested GPU count: 2
- Requested GPU types:
  - NVIDIA A40
  - NVIDIA A100
- Workload: `nautilus_smoke_mlp`
- Warmup: 1 epoch
- Profile: 1 epoch
- Batches per epoch: 1
- Optimizer: Stochastic Gradient Descent
- Precision: `fp32_ieee`
- Streaming Multiprocessor Occupancy source: NVIDIA Nsight Compute command line interface

## Kubernetes Result

```text
perfseer-e2e-ncu4-20260619191658-a100   Failed   0/1
perfseer-e2e-ncu4-20260619191658-a40    Failed   0/1
```

Pods:

```text
perfseer-e2e-ncu4-20260619191658-a100-xsdvq   Error   nautilus-it-gpu08.fullerton.edu
perfseer-e2e-ncu4-20260619191658-a40-6qcsn    Error   bak-hpc1.csub.edu
```

## A40 Result

Exit code: 1

Log:

```text
/usr/local/cuda/bin/ncu
nautilus_smoke_mlp::fp32_ieee: error RuntimeError('ncu failed for infer: ==PROF== Connected to process 78 (/opt/conda/bin/python3.10)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
==PROF== Disconnected from process 78
==WARNING== No kernels were profiled.
')
{"bad_rows": 1, "files": 1, "ok_rows": 0, "rows": 1}
```

## A100 Result

Exit code: 1

Log:

```text
/usr/local/cuda/bin/ncu
nautilus_smoke_mlp::fp32_ieee: error RuntimeError('ncu failed for infer: ==PROF== Connected to process 78 (/opt/conda/bin/python3.10)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
==PROF== Disconnected from process 78
==WARNING== No kernels were profiled.
')
{"bad_rows": 1, "files": 1, "ok_rows": 0, "rows": 1}
```

## Cleanup

Deleted Kubernetes jobs:

```text
perfseer-e2e-ncu4-20260619191658-a100
perfseer-e2e-ncu4-20260619191658-a40
```

Deleted Kubernetes ConfigMap:

```text
perfseer-e2e-ncu4-20260619191658
```

Deleted noisy generated files:

```text
record/nautilus_e2e_ncu4-20260619191658.md
record/nautilus_e2e_ncu4-20260619191658.yaml
record/nautilus_e2e_ncu4-20260619191658.monitor.log
```

