# Nautilus Epoch Workflow Verifier

- Date: 2026-06-11

- Scope: Added verification for one warmup epoch plus second epoch label sampling on four distinct Nautilus GPUs.

- Files changed:

- `nrp_calibration_pack/profile/run_profile.py`

- `scripts/create_nautilus_label_workflow.py`

- `scripts/verify_sampled_labels.py`

- `scripts/verify_nautilus_label_workflow.py`

- Experiment commands:

```bash
python3 -m py_compile scripts/verify_nautilus_label_workflow.py scripts/verify_sampled_labels.py nrp_calibration_pack/profile/run_profile.py scripts/create_nautilus_label_workflow.py
python3 scripts/build_source_manifest.py --models-dir nrp_calibration_pack/models --output /tmp/perfseer_source_manifest_epoch.jsonl --precision-config fp32_ieee
python3 scripts/create_nautilus_label_workflow.py --namespace test-ns --pvc test-pvc --image test/image:latest --output /tmp/perfseer_epoch_jobs.yaml --gpus a100,a40,l40s,v100 --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --optimizer sgd
python3 nrp_calibration_pack/profile/run_profile.py --manifest /tmp/perfseer_one_epoch_manifest.jsonl --models-dir nrp_calibration_pack/models --output-dir /tmp/perfseer_epoch_smoke --device cpu --warmup-epochs 1 --profile-epochs 1 --batches-per-epoch 1 --optimizer sgd
python3 scripts/verify_sampled_labels.py /tmp/perfseer_epoch_smoke
python3 scripts/verify_nautilus_label_workflow.py --jobs-yaml /tmp/perfseer_epoch_jobs.yaml
python3 scripts/verify_nautilus_label_workflow.py --results-dir /tmp/perfseer_fake_four_gpu_results
```

- Results:

- Syntax check: passed.

- Manifest rows: 10000.

- CPU epoch smoke: `calib_0000::fp32_ieee`, status `ok`.

- Label verifier: `bad_rows=0`, `ok_rows=1`.

- Workflow verifier found 4 jobs:

- `model-label-sampler-a100`

- `model-label-sampler-a40`

- `model-label-sampler-l40s`

- `model-label-sampler-v100`

- Workflow verifier found 4 GPU identities:

- `nvidia.com/a100`

- `nvidia.com/a40`

- `NVIDIA-L40S`

- `Tesla-V100`

- Fake four-GPU result verifier: `bad_rows=0`, `ok_rows=4`.

- Nautilus submission: not run in this experiment.
