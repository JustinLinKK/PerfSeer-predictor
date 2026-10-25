# A10 pretrained student

This package implements D3 pretraining from the same-hardware full-data hybrid
teacher. It contains only the PRETRAIN recipe: 192-wide, four-layer D3,
1,643,470 parameters, scratch initialization, AdamW at 0.001, weight decay
0.0001, 40 epochs, batch 32 and EMA 0.999. The exact published D3 distillation
objective is in pretrain_objective.py. Place student_training/ beside portable/.

For CPU inference, extract the A10 data and S_A10 model archives into one folder.
From the source release root, run (replace EXTRACTED with the extraction path):

```sh
python -B -I tools/predict_student.py --model EXTRACTED/PerfSeer-release-v1/checkpoints/S_A10/student.ts --input EXTRACTED/PerfSeer-release-v1/A10/data/A10/validation-inputs.pt --output predictions.npz
```

This reads inputs only. NPZ arrays are predictions [N,6], sample_ids,
target_names, units, hardware, input_sha256, model_sha256, architecture_sha256
and normalization_sha256. Columns are wall
time (ms), GPU time (ms), epoch time (ms), NVML busy (%), average VRAM (MiB),
and peak VRAM (MiB). IDs preserve input order. Use --batch-size and --cpu-threads
to adjust CPU resource usage. Hardware must match the model. The same command
continues to work when the later complete release replaces student_training/.
The portable/ teacher CLI also supports explicit P_A10 checkpoint inference.

To reproduce D3 training, extract the full A10 training dataset and P_A10 teacher
archive, then generate an absolute-path config:

```sh
python -B -I student_training/make_config.py --assets EXTRACTED --run-output RUNS/S_A10 --config CONFIGS/S_A10.json
python -B -I student_training/cache.py --config CONFIGS/S_A10.json --device cpu
python -B -I student_training/train.py --config CONFIGS/S_A10.json --device cpu
```

The teacher cache includes TRAIN rows only and binds the dataset, teacher,
normalization, architecture and exact split. Training also binds its code.
Validation labels select one checkpoint;
configuration groups cannot cross splits. Training saves atomic optimizer,
scheduler, EMA, RNG, history and selected state; rerunning the command resumes
the last completed epoch. An interrupted epoch is replayed. --smoke uses a
separate cache/output and two epochs of eight TRAIN/eight validation rows.

For CUDA, the GPU owner first runs preflight_cuda.py --config CONFIG --output
RECEIPT. It probes actual forward/backward/update/export on the largest TRAIN
graphs. Only out-of-memory failures permit a smaller batch. Explicitly update
the config if its receipt selects a batch below 32, then pass --preflight RECEIPT
to train.py --device cuda. Preflight and training do not acquire GPU locks.

After validation checkpoint selection, run finalize.py --run RUNS/S_A10
--output MODELS/S_A10. A10 has no independent sealed test; no test score is
claimed. Exported TorchScript is checked against eager outputs, physical
constraints and single-example batching in a fresh interpreter.
