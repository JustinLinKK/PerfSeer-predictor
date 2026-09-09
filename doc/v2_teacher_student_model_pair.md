# v2 Teacher/Student Model Pair

The active PerfSeer v2 workflow keeps exactly one teacher architecture and one
student architecture. Train one pair per hardware ID.

Active configs:

- Teacher: `src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml`
- Student: `src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml`

Target source: `scheduler_v2_train`

Output order:

```text
train_epoch_ms, train_avg_sm_util_percent, train_p95_sm_util_percent,
train_peak_vram_used_mib, train_peak_torch_reserved_mib,
train_peak_memory_controller_util_percent
```

Architecture policy:

- Teacher: large `seernet_multi`, `hidden=1024`, `num_blocks=8`, `head_hidden=1024`.
- Student: compact `seernet_multi`, `hidden=192`, `num_blocks=2`, `head_hidden=192`.

Default per-hardware run IDs:

- `v2_teacher_<hardware-id>`
- `v2_student_<hardware-id>`

Older diagrams and configs live under `doc/legacy/` and
`src/perfseer-optimized/configs/legacy/` for historical reproduction only.
