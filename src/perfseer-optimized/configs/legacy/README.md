# Legacy Configs

These configs are retained only for reproducing older experiments and compatibility checks.

Do not use them for new v2 training. The canonical v2 model pair is:

- `../train_hardware_teacher/v2_teacher.yaml`
- `../train_deploy_model/v2_student.yaml`

Legacy configs may train on the original six-target `label/label/*.txt` path or on pre-v2 scheduler-resource targets that do not include measured epoch time as a direct training target.
