# PerfSeer: A100 student release

**S_A100** is the verified D3 student distilled from A1001 (P_A100). Its 40-epoch
training, selected checkpoint and CPU TorchScript export are complete. The model
and full labeled A100 dataset are distributed through Git LFS.

Start with [A100_QUICKSTART.md](A100_QUICKSTART.md) for selective download,
checksum-verified extraction and CPU inference. The predictor targets A100
workloads; inference itself runs on CPU.

`reports/comparison.json` separates validation and sealed-test results. Selection
used validation only. A selected checkpoint is not a claim that all time, busy
ordering and VRAM quality thresholds pass. A1001 has A10 ancestry, which limits
claims about unseen-hardware transfer.

`portable/` contains the frozen feature runtime and teacher code, `data/` the
workload generator and GPU labeler, and `student_training/` the pretrained D3
recipe. No unselected transfer-student design is included. Follow
`portable/FRESH_DATA.md` for new workloads and labels on the actual target GPU.

`assets/manifest.json` binds every archive/member hash. `reports/lfs/` contains
remote-byte verification. This is a partial release: the A10 teacher/student,
A10G student and remaining teacher/transfer assets will be added in subsequent
commits on this same `paper-release` branch.
