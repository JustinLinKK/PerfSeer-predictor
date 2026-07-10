# PerfSeer v2 Model Configs

Use exactly these active model configs for v2 per-hardware training:

- `train_hardware_teacher/v2_teacher.yaml`
- `train_deploy_model/v2_student.yaml`

The v2 pair trains on `features.target_source: scheduler_v2_train`, which combines measured per-epoch training time from `label/scheduler_label_v3.jsonl` with sustained resource targets from `label/scheduler_resource_label.jsonl`.

Each hardware class should get one teacher/student pair, for example `v2_teacher_rtx5090` and `v2_student_rtx5090`. Do not use files under `legacy/` for new v2 runs.
