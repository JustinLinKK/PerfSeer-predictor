# Published student training

This package contains the fixed D3 pretrained-student recipe and one selected
transfer-student recipe. Place it beside the packaged `portable/` teacher/runtime
directory. Input paths come exclusively from your JSON config; no historical
trainer or checkpoint is loaded implicitly. `candidate` is PRETRAIN or TS2.

The config requires hardware, teacher {checkpoint, kind, hardware,
identity_override}, dataset, normalization, architecture, validation_key, cache,
output, epochs (40), batch_size (32), ema_decay (.999). Transfer also requires
route A10_A100, immutable subset JSON and source S_A100 initial_student where
the selected design uses a warmstart. Use absolute paths; architecture normally
points at ../portable/architecture.pt. See training_design.json for the fixed
model/initialization/objective/learning rate.

After extracting data and model assets, generate absolute configs with
`python make_config.py --assets EXTRACTED --mode pretrain --hardware A10G
--run-output RUNS/S_A10G --config CONFIGS/S_A10G.json`, or use `--mode transfer
--hardware A10` for the published transfer recipe. EXTRACTED may be the parent
of PerfSeer-release-v1 or that directory itself. Transfer uses the full A10
dataset plus transfer/A10/selection.json, preserving the original row index space.

Run `python cache.py --config CONFIG --device cpu` then
`python train.py --config CONFIG --device cpu`. For CUDA, run the supplied
preflight_cuda.py under the GPU owner, then pass its successful receipt to
train.py using `--preflight RECEIPT`; dataset/code/source/batch changes invalidate
the receipt. OOM fallback requires a recorded, explicit config batch update.
The CUDA verifier probes the published transfer recipe and D3 pretraining only.

Training checkpoints include optimizer, scheduler, EMA, RNG, selected state,
history and completed epoch; an interrupted epoch is replayed from the previous
transaction. `--smoke` checks eight TRAIN/eight selection rows and two epochs,
uses separate output/cache names and cannot enter final evaluation.

Only after checkpoint selection, run `python finalize.py --run RUN --output MODEL`.
This locks the chosen checkpoint before heldout/test evaluation, saves one fixed
prediction set, and reuses it on retries. A10 has no independent sealed test.
Its transfer holdout is historical validation and must retain that disclosure.
TorchScript inference takes the documented 17 tensors and requires only PyTorch.
