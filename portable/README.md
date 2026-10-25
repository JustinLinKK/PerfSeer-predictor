# Portable selected-teacher runtime

This directory contains the P1/SM0 teacher and the selected T7 transfer design,
the pinned feature/runtime code, the frozen architecture metadata, v5 scorer,
and explicit-path training entrypoints. It imports no historical experiment
trainer and reads no implicit source checkpoint. Large checkpoint files are
distributed separately.

| Release model | `--kind` | Parameters | Input hardware identity |
|---|---|---:|---|
| P_A100 | P1 | 253,644,481 | original input token, expected A100 token 190 |
| P_A10G | Adapted | 253,644,481 | mapped to source token 190 |
| P_A10 | Adapted | 253,644,481 | mapped to source token 190 |
| T_A10_A100 | T7 | 255,692,483 | mapped to source token 190 |

P_A100 is the selected P1/SM0 checkpoint formerly named A1001. P_A10G reuses
A10G1. P_A10 is full-data A10 continuation, and T_A10_A100 is A100-to-A10
transfer using the independently generated training design. The A100 source
has earlier A10 ancestry, so this is target re-adaptation, not evidence of
generalization to a previously unseen GPU. Target continuation starts from an
explicit P_A100 checkpoint; it is not training from scratch.

P1 combines the selected timing and Horus busy branches with the VRAM-only
node-9 SeerBlock. T7 adds rank-eight LoRA to the memory trunk's linear layers,
with scale 1, and the original two-value memory offset. It retains the original
state-dictionary names. The factory has no implementations of other transfer
candidates.

## Dependencies and loading

Python, PyTorch, NumPy, SciPy and PyYAML are required; the pinned runtime is
included in `runtime/src` and `runtime/aliases`. CUDA is needed only when the
caller chooses CUDA inference or training. The verification receipts record the
tested environment. Model hub access and the historical checkout are unnecessary.

```bash
python -I portable/infer.py --kind P1 \
  --checkpoint checkpoints/P_A100.pt --architecture portable/architecture.pt \
  --input PerfSeer-release-v1/A100/data/A100/validation-inputs.pt --output predictions.npz

python -I portable/infer.py --kind T7 \
  --checkpoint checkpoints/T_A10_A100.pt --architecture portable/architecture.pt \
  --input a10-input-only.pt --output a10-predictions.npz
```

Paths above are examples; each checkpoint/input path is supplied explicitly.
CPU inference with one thread is the default. `--device cuda`, `--threads` and
`--batch-size` are explicit options. The `.npz` output includes sample IDs,
six predictions, column names/units, 100 MB memory buckets, normalizer identity,
checkpoint/architecture hashes, and epoch protocol.

```python
from portable.teachers import build, load_teacher
base, model, architecture = build("T7", "portable/architecture.pt", "T_A10_A100.pt")
```

`build` strictly loads raw P1 state or wrapped Adapted/T7 state; wrong kinds,
unexpected/missing state keys, conflicting architecture hashes, and incompatible
hardware overrides fail. `load_teacher` accepts a dictionary with `kind` and
`checkpoint` and returns a uniform `.p1` wrapper for distillation taps. Its
optional `identity_override` is None or 190 for P1, and must be 190 for Adapted/T7.
`initialize_target(source_checkpoint, transfer=False|True)` is a separate,
explicit warm-start operation for creating new target-training models.

Importing `portable` or `portable.teachers` bootstraps the pinned runtime before
any `torch.load`, including the custom normalizer dataclasses in `architecture.pt`.
The pinned `load_base`, `PhysicalTiming`, `HorusSM`, feature packing and scorer
remain unchanged. `io_utils` supplies explicit-path checkpoint/logging utilities.

## Input and preprocessing contract

Inference accepts an input-only Torch bundle with `features`, `sample_ids`,
`raw_steps_per_epoch` and `epoch_protocol`; `normalization` may be included.
Label, family and partition tables are rejected by the inference CLI.
The inputs use the frozen A10 source normalizer, semantic hash
`13197c7c5a3aab0017e0e0c72b0047cdb8c2647f81f0af776248441444279748`.
Already-normalized inputs must identify that normalizer. Raw feature inputs are
normalized once through the pinned preprocessing function. A different supplied
normalizer or conflicting feature normalization identity fails.

Inside P1, only the node-9 memory branch transforms source-normalized inputs back
to physical feature values and then applies its original A100 clipping and
normalization. This transform is retained exactly. Adapted/T7 map the hardware
identity to token 190 and retain the inherited A100 physics prior; target-specific
weights represent the target hardware. No automatic rescaling by GPU specs is
introduced.

Output order is wall-step ms, GPU-step ms, wall-epoch ms, NVML busy percent,
average VRAM MiB and peak VRAM MiB. Busy percent is a proxy, not SM occupancy.
`measured_single_step` requires exactly one step per epoch.
`derived_step_times_count` computes wall epoch time as step time times the
explicit positive integer count. This is derived scaling, not a newly measured
multi-step epoch. Memory bucket display rounds upward to decimal 100 MB while
the six-column tensor remains in MiB. Uncertainty is not calibrated.

## Training entrypoints

The two entrypoints use the selected release recipes. They require explicit
source checkpoint, architecture, dataset and output paths. Transfer additionally
requires a sample-ID subset manifest and accepts only the A100-to-A10 route.

```bash
python -I portable/train_pretrain.py --hardware A10 \
  --source checkpoints/P_A100.pt --architecture portable/architecture.pt \
  --dataset PerfSeer-release-v1/A10/data/A10/dataset.pt --output runs/P_A10 --device cuda

python -I portable/train_transfer.py --hardware A10 \
  --source checkpoints/P_A100.pt --architecture portable/architecture.pt \
  --dataset PerfSeer-release-v1/transfer/A10/dataset.pt \
  --subset PerfSeer-release-v1/transfer/A10/selection.json \
  --output runs/T_A10_A100 --device cuda
```

Full continuation uses every TRAIN row and selects on validation: 30 epochs,
effective batch 64, microbatch 8, AdamW with zero weight decay, timing/busy/memory
head/memory trunk learning rates 2e-4/4e-4/1e-5/1e-6, warmup plus cosine decay,
per-group gradient clipping, fixed epoch permutations, log-interval memory loss,
TRAIN-only correct-bucket anchoring and memory L2-SP coefficient .01.

T7 uses the manifest's exact TRAIN IDs, fixed selection IDs and separate
historical-validation holdout IDs. It uses 60 epochs, effective batch 32,
microbatch 8, five calibration epochs, then full timing-branch tuning. The
original memory trunk stays frozen except LoRA; the memory head and two-value
offset train. Learning rates are timing 1e-4, busy 1e-4 during calibration then
2e-4, memory head 2e-5, LoRA 2e-4, offset 2e-3; calibration timing parameters use
3e-4. The symmetric 100 MB gate surrogate, .05 centering and .25 log-interval
term are retained. Both recipes preserve the original timing/busy losses and
pair-order terms, validation-only checkpoint selection, and earlier-epoch ties.

Transfer IDs are mapped to the supplied bundle by sample identity, allowing the
full bundle or its compact transfer cache. Historical split/group isolation is
checked; missing IDs are errors. Full continuation rejects compact transfer
caches outside smoke mode. Feature normalizer identity is checked before training.
No data-driven row replacement, normalizer fitting, or test-based selection occurs.

CUDA training retains bfloat16 autocast; CPU smoke uses FP32. `--micro-batch`
can be 1, 2, 4 or 8 while preserving the effective production batch. A completed
epoch atomically saves model, optimizer, RNG, LR schedule, history, best state and
the next epoch cursor. Restarting the same command resumes it; incompatible
code/configuration/source/data hashes fail. An interrupted incomplete epoch is
replayed. A completed run verifies its checkpoint before returning.

Optional `--test-inputs` and `--test-key` evaluate an identity-aligned sealed test
only after selecting and replaying the best checkpoint. Omitting them performs
no sealed-test evaluation. A10's historical validation holdout is not a new test.
`--smoke --device cpu` is a bounded real-data verification mode with four TRAIN
and four validation rows and shortened epochs; its results are not study scores.

## Verification

P1, Adapted and T7 outputs are bit-identical to their original implementations on
input-only TRAIN probes. All three also passed `python -I` from a temporary
directory outside the checkout, with empty `PYTHONPATH` and CUDA hidden. No
checkpoint was copied for those checks. `verification/*-portable-parity.json`
records parameter counts, input hashes and maximum absolute difference zero.

The training verification records real finite/nonzero gradients and exact
interrupted/resumed model, optimizer, RNG, LR, history and cursor equality on
CPU. Smoke binaries are temporary verification artifacts, not release models.
The source manifest binds the copied runtime, extracted selected classes,
architecture bytes, source files and verification receipts.

For a complete generated-program to physical-label to feature-bundle workflow,
see `FRESH_DATA.md` and `pack_fresh.py`. That guide also records the verified archive
extraction prefixes and the source-only copy allowlist.
