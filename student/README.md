# Student Predictor Package

- Self-contained folder for the distilled SeerNetMulti student predictor (hidden 128, blocks 2, ~708K parameters, <1 ms CPU inference at 4 threads).

## Files and origins

| File | Origin | Role |
|---|---|---|
| `model.py` | copy of `predictor/model.py` | SeerNetMulti architecture |
| `converter.py` | copy of `predictor/converter.py` | torch.fx trace + shape propagation → compute graph |
| `pipeline.py` | copy of `teacher/pipeline.py` | graph featurization (node 53 / edge 3 / global 40 dims) |
| `encoder.py` | copy of `predictor/encoder.py`, imports localized | pure PyTorch `.py` → normalized SeerNet input batch, optional prediction |
| `train_student.py` | copy of `teacher/train_student.py`, imports point at `teacher/` | knowledge-distillation trainer (needs `teacher/train_teacher.py` + a cache) |
| `student_RTX_6000_Blackwell.pt` | moved from `teacher/student_RTX_6000_Blackwell.pt` | student trained on RTX-6000 labels (min10Acc 0.804) |
| `student_A10.pt` | moved from `teacher/student_A10.pt` | student trained on A10 labels (min10Acc 0.701 ± 0.014) |
| `optimize_cpu.py` | new | CPU inference benchmark/export: eager, dynamic-int8, TorchScript, torch.compile, ONNX Runtime |
| `student_a10_cpu_ts.pt` | exported by `optimize_cpu.py` | TorchScript fp32, single-graph, 0.299 ms |
| `student_a10_cpu_ts_int8.pt` | exported by `optimize_cpu.py` | TorchScript int8, single-graph, 0.305 ms, 0.8 MB |
| `student_a10_cpu.onnx` | exported by `optimize_cpu.py` | ONNX fp32, single-graph, 0.282 ms |
| `student_a10_cpu_int8.onnx` | exported by `optimize_cpu.py` | ONNX int8 — fastest CPU backend: 0.251 ms mean, min10Acc 0.7005, 1.0 MB; carries training mean/std in `metadata_props["seer_stats"]` |
| `predict.py` | new | consecutive pipeline: source file → converter → featurizer → normalization → prediction, one `SeerPredictor` class, auto backend (.pt / .onnx) |
| `test_pipeline.py` | new | end-to-end tests: 5 architectures × 4 precisions × 2 backends |

## Usage

- Predict for a pure PyTorch model file (single consecutive pipeline; stats travel inside the artifact):

```bash
.venv/bin/python student/predict.py my_model.py --entry MyNet \
    --input-shapes 8,3,224,224 --precision fp32_ieee \
    --artifact student/student_a10_cpu_int8.onnx   # or student/student_A10.pt
```

- Lower-level encode-only interface:

```bash
.venv/bin/python student/encoder.py my_model.py --entry MyNet \
    --input-shapes 8,3,224,224 --precision fp32_ieee \
    --ckpt student/student_A10.pt
```

- Fast CPU path (ONNX Runtime int8, ~2.15x faster than eager): append `--onnx student/student_a10_cpu_int8.onnx` to the command above.

- Outputs six targets: `train_util` (SM %), `train_mem` (MiB), `train_time`, `infer_util`, `infer_mem`, `infer_time` (time units follow the training labels: ms for A10, s for RTX-6000).

- Retrain / re-distill (uses `teacher/` training code and cache):

```bash
SEER_CACHE=$PWD/teacher/cache_a10 .venv/bin/python student/train_student.py \
    --teacher teacher/teacher_a10_final.pt --hidden 128 --blocks 2 \
    --alpha 0.5 --wts 4,0.2,1,4,0.2,1 --tag student_a10_h128b2
```

- Records: `record/teacher_runs_a10/a10_training.md`, `record/encoder.md`, `record/teacher_runs/student_training.md`.
