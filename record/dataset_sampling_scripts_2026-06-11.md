# Dataset Sampling Scripts

- Date: 2026-06-11

- Scope: Extended source-code label sampling scripts for generated PyTorch models.

- Files changed:

- `nrp_calibration_pack/profile/run_profile.py`

- `scripts/build_source_manifest.py`

- Existing workflow renderer verified:

- `scripts/create_nautilus_label_workflow.py`

- Experiment commands:

```bash
python3 -m py_compile nrp_calibration_pack/profile/run_profile.py scripts/build_source_manifest.py scripts/create_nautilus_label_workflow.py
python3 scripts/build_source_manifest.py --models-dir nrp_calibration_pack/models --output /tmp/perfseer_source_manifest.jsonl --precision-config fp32_ieee
python3 scripts/create_nautilus_label_workflow.py --namespace test-ns --pvc test-pvc --image test/image:latest --output /tmp/perfseer_label_jobs.yaml
python3 nrp_calibration_pack/profile/run_profile.py --manifest /tmp/perfseer_one_manifest.jsonl --models-dir nrp_calibration_pack/models --output-dir /tmp/perfseer_profile_smoke --device cpu --warmup 1 --infer-repeats 1 --train-repeats 1 --optimizer adamw
```

- Results:

- Syntax check: passed.

- Manifest rows: 10000.

- Workflow YAML: generated for four GPU jobs.

- CPU smoke model: `calib_0000::fp32_ieee`, status `ok`.

- `label_v2.train` keys:

- `avg_device_memory_usage_mib`

- `avg_host_memory_usage_mib`

- `avg_sm_occupancy_percent`

- `avg_sm_utilization_percent`

- `compile_warmup_time_ms`

- `data_type`

- `dram_activity_percent`

- `mean_iter_ms`

- `metric_notes`

- `optimizer`

- `peak_device_memory_usage_mib`

- `peak_dram_activity_percent`

- `peak_host_memory_usage_mib`

- `peak_sm_occupancy_percent`

- `peak_sm_utilization_percent`

- `time_1_epoch_ms`

- `time_ms_per_sample`

- `warmup_avg_host_memory_usage_mib`

- `warmup_peak_host_memory_usage_mib`

- Notes:

- True SM Occupancy is not available from NVIDIA Management Library. Current field is `null`; NVIDIA Nsight Compute is required for true occupancy.

- DRAM Activity uses NVIDIA Management Library memory utilization proxy, not a hardware throughput counter.

- `time_1_epoch_ms` means one synthetic repeat pass in this script setting.
