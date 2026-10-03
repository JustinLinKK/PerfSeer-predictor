# Predictor v3.2 — student-only package

This directory is the deployable source-and-weight home for the distilled PerfSeer v3.2 student.

## Contents

- `source/`: the exact Python dependency closure used by the student inference path.
- `weights/predictor_v3.2_student.pt`: the final deployable student checkpoint (11.6 MB).
- `weights/validation-summary.json`: selected-epoch validation metadata for the student.
- `.cache/teacher_v3.2_epoch26.pt`: the exact epoch-26 teacher checkpoint used for student distillation (2.24 GB).

The package deliberately excludes every teacher checkpoint, teacher cache, optimizer state, dataset, label file, prediction dump, and training log.

## Output contract

The student has 12 outputs: training-step wall and Graphics Processing Unit time, training-epoch time, training average Streaming Multiprocessor utilization, three training memory measurements, inference-step wall and Graphics Processing Unit time, inference average Streaming Multiprocessor utilization, and two inference memory measurements. It does not predict peak Streaming Multiprocessor utilization.

The student was distilled from the included epoch-26 teacher checkpoint. The student export was selected at epoch 12 and has SHA-256 `988fa6fec1df66349a40c1dd2585ccc5f25ef2a72d88ea38af8f5b97e45850e7`; the teacher has SHA-256 `5792c12088988dfc514d76a0a4d0a7ff4f7155c26dea529486194066cc656a42`. The package excludes teacher output cache, optimizer state, and datasets.

## Import check

```bash
PYTHONPATH=source python -c 'import perfseer_v32.inference'
```
