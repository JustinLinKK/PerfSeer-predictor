# ABA twelve-label, six-head continuation

- Date: 2026-09-09.
- Remote host: `ssh ABA` (`ecepxiegpu1.ucsd.edu`).
- Latest inspected revisions: MLEvolve `origin/hardware-awared` `93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d`; Predictor `v2` `25d81162fc55959e767b00e4f2aef057c982f1d1`.
- The dirty MLEvolve checkout was not pulled because all seven local modifications overlap incoming commits; a direct fast-forward pull aborted without changing it.
- A clean clone of that latest branch was instead created at `/data1/yufan/MLEvolve-hardware-awared-93371dd`; it is clean at `93371dd64b8e2b888c1bde7cb9d90d7c03ac4e5d`. The current Predictor `v2` tip was also cloned cleanly at `/data1/yufan/PerfSeer-predictor-v2-25d81162` (`25d81162fc55959e767b00e4f2aef057c982f1d1`).
- A file-level comparison establishes that the live trainer is identical to this Predictor `v2` source except for the tested checkpoint-resume and per-epoch `latest.pt` additions that are required for safe continuation.

## Data and architecture

- Training data: `/data1/yufan/perfseer_v32_a100_bs_training_20260908`.
- Combined target rows: 44,820, including 4,800 valid `data/raw_source/a10_bs` labels.
- Split rows: 36,012 train, 4,544 validation, 4,264 test.
- Model: six separate metric heads with widths `(3, 1, 3, 2, 1, 2)`, producing 12 output labels.
- Evaluation: per-label ground-truth relative-error accuracy within 5% and within 10%, emitted to `epoch-*.json` after every epoch. Even-numbered epochs are the required two-epoch reporting points.

## Continuation

- Original run: `training_latest_predictor_six_heads_v12_teacher_batch48_r7`; it exited after epoch 6 and retained its best checkpoint from epoch 3.
- Added contract-checked checkpoint loading and model-weight restoration to `src/perfseer-optimized/train_v12.py`. The focused remote test suite passed: 4 tests.
- The continuation uses a separate output directory, `training_latest_predictor_six_heads_v12_resume_r8`; the original artifacts remain unchanged.
- GPU selection: `CUDA_VISIBLE_DEVICES=1`. Preflight verified GPU 1 was idle; after launch it reached 100% utilization and 52.5 GiB allocated, under the 60 GiB cache threshold.
- The first attempt exited before training because the selected cache contained only one graph feature. Root-cause verification showed the selected rank-0 cache plus sibling rank cache contains all 952 unique train/validation graph inputs. The failed, empty continuation directory was removed before relaunch.
- Active training PID: `1976116`.
- Monitor PID: `1976117`.
- Remote monitor log: `/data1/yufan/perfseer_v32_a100_bs_training_20260908/training_latest_predictor_six_heads_v12_resume_r8/monitor.log`.
- Independent even-epoch accuracy watcher PID: `1989755`; it is verified alive and writes the per-label 5% and 10% ground-truth accuracies to `/data1/yufan/perfseer_v32_a100_bs_training_20260908/training_latest_predictor_six_heads_v12_resume_r8/accuracy_every_2_epochs.log`.
- At 2026-09-09T21:58Z, the training PID had accumulated 34:54 CPU time at 99.0% CPU while it was the sole GPU-1 compute process, holding 49.1 GiB at 100% GPU utilization. Epoch 4 was still running, so no new even-epoch result was yet available.

## Epoch-4 ground-truth accuracy (first continuation checkpoint)

Epoch 4 completed with validation over 4,544 rows. The values below are the fraction of ground-truth labels whose relative error is at most the stated threshold.

| Target | Within 5% | Within 10% |
| --- | ---: | ---: |
| `train_step_wall_ms` | 33.43% | 59.71% |
| `train_step_gpu_ms` | 33.25% | 59.60% |
| `train_epoch_ms` | 31.95% | 59.84% |
| `train_avg_sm_util_percent` | 32.04% | 58.78% |
| `train_avg_vram_mib` | 94.32% | 99.96% |
| `train_peak_vram_mib` | 94.32% | 99.96% |
| `train_peak_torch_allocated_mib` | 83.74% | 91.18% |
| `infer_step_wall_ms` | 51.41% | 88.45% |
| `infer_step_gpu_ms` | 51.67% | 88.64% |
| `infer_avg_sm_util_percent` | 48.88% | 80.13% |
| `infer_avg_vram_mib` | 98.15% | 98.26% |
| `infer_peak_vram_mib` | 98.17% | 98.28% |

The trainer wrote `epoch-0004.json`; the independent watcher copied the same two accuracy dictionaries to `accuracy_every_2_epochs.log`. The PID remained live after the checkpoint, so continuation was not interrupted.

## Epoch-30 completion guard

The trainer was originally launched with `--epochs 800`. To honor the requested 30-epoch total without disrupting the active process, a separate persistent guard (`PID 2018564`) now watches for `epoch-0030.json`. It waits for the corresponding `latest.pt` checkpoint to be present and stable, then sends `SIGINT` to the training process. Its action log is `/data1/yufan/perfseer_v32_a100_bs_training_20260908/training_latest_predictor_six_heads_v12_resume_r8/stop_after_epoch_30.log`; no signal is sent before a completed epoch-30 metric and checkpoint exist.
