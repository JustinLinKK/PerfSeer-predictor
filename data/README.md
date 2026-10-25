# Reproducible workload generation and GPU labeling

`generate.py` directly constructs workloads from a pinned architecture catalog,
seed, family quotas and executable source templates. It does **not** read a
pretraining tensor bundle, labels, success flags, measured costs, or feature rows
to decide which transfer workloads to generate.

The full design constructs 6,000 models and 48,000 profile points: 32 families,
five modalities, two batch choices, FP32/FP16 AMP and Adam/Muon. The release
transfer design directly emits 8,192 TRAIN profile points. It covers all 5,397
TRAIN model IDs and all 4,771 frozen TRAIN configuration groups, giving every
model one variant before generating additional variants in family-interleaved
rounds. A stable per-config rotation balances batch, precision and optimizer.
Architecture specs are generated from the original sampler; the full 48,000-row
profile set is never materialized or searched as part of transfer generation.

Only the frozen group split map is consulted to keep historical validation/test
configurations out of transfer training. Failed or missing historical labels do
not cause replacement workloads. Generated specifications and available cached
measurements are separate artifacts.

```bash
python data/generate.py --mode pretrain --output my_pretrain
python data/generate.py --mode transfer --train-budget 8192 --output my_transfer
python data/verify.py --generated my_transfer --receipt generated-check.json
python -m unittest discover -s data/tests -v
```

Run these commands from the extracted release root. Python with PyTorch is
required. Generation and cache verification are CPU-only. Exact versions used
for this release are in `receipts/environment.json`; the observed environment
used PyTorch 2.11.0+cu130, Python 3.13.12 and nvidia-ml-py 13.595.45. Install a
PyTorch build compatible with the actual GPU/driver; `nvidia-ml-py` provides
NVML for labeling. The packaged source runtime needs no model hub or datasets.

The published generator outputs are `generated_pretrain/` and
`generated_transfer_final/`. Re-running with the documented options reproduces
their workload identities and source bytes. All 48,000 historical base-source
and standalone variant-source hashes are checked by the release verifier.
Compact archives retain these directories' manifests and generation receipts;
they may omit the thousands of reconstructable `models/` and `variants/` files.
Generate into a fresh directory as shown above before labeling or verifying
source bytes. The code, all 32 templates, frozen split specification and shared
runtime needed to recreate them are included.

## Historical measurements and source lineage

`transfer/<GPU>/` contains `dataset.pt`, `labels_historical.jsonl`,
`missing.jsonl`, `selection.json`, and a hash-bound `join_receipt.json`.
The labels are retained historical measurements. **They are not new A10G,
A10, or A100 measurements from the local RTX 5090.**

| Hardware | Requested TRAIN | Available TRAIN | Missing/ambiguous | Evaluation |
|---|---:|---:|---:|---|
| A10 | 8,192 | 7,494 | 698 | original 512 selection + 1,640 historical-validation holdout |
| A100 | 8,192 | 8,080 | 112 | original 2,296 validation rows |
| A10G | 8,192 | 8,192 | 0 | original 2,352 validation rows |

The 1,640-row A10 holdout comes from historical validation; it is not an untouched
test set. A100/A10G sealed-test artifacts remain in their full dataset packages.
The A100/A10G transfer data packs are data resources; the model transfer experiment
in this release is A100 to A10 only.

Source lineages are explicitly distinct. A10G historical labels identify an
original base program plus `_model_runtime.py`. A10/A100 labels identify
self-contained variant programs that include later SwiGLU/Kimi helper code.
Equal architecture parameters do not prove byte-identical execution across
these two lineages. Both programs are generated, and both hashes are retained.
The historical A10G tensor features were captured through A10's variant-source
static representation, as recorded by the existing conversion. An unqualified
claim that A10G to A10/A100 is a pure hardware-only shift is not justified by
this source evidence.

`capture_templates.py` is a release-building provenance utility. It verified
every one of 48,000 historical standalone source checksums and found exactly
one executable template per family after replacing literal workload parameters.
The packaged 32 templates contain code and parameter placeholders, not selected
dataset rows or labels. `generate.py` constructs source programs from these
templates and independently sampled architecture configurations.

`join_cached.py` checks the original specification, base-source hash, shared
runtime hash, standalone variant-source hash and workload literals before joining
measurements. It then checks hardware, frozen group identity, raw physical
targets and feature membership. Duplicate historical measurements are
quarantined, matching the original native-data policy; no occurrence is silently
preferred. Missing outcomes are recorded explicitly. Cache joining does not
alter the generated design or replace failed rows.

`selection.json` indices refer to the corresponding original full dataset.
The compact `dataset.pt` has those rows in train, selection, holdout order and
adds `release_partition` values 0, 1, 2. Its original `splits` remain unchanged:
both A10 evaluation partitions are historical validation. `manifest` and the
normalizer retain their parent dataset provenance; `release_provenance` and
`selection.json` define the compact pack. Features are `GraphFeaturesV3`, so add
the released `runtime/src` and `runtime/aliases` to `sys.path` before loading:

```python
from pathlib import Path
import sys
import torch
root = Path.cwd()
sys.path[:0] = [str(root / "runtime/src"), str(root / "runtime/aliases")]
data = torch.load(root / "data/transfer/A10/dataset.pt", weights_only=False,
                  map_location="cpu", mmap=True)
train = torch.where(data["release_partition"] == 0)[0]
```

Original historical target units are milliseconds, NVML busy percent, and MiB.
The v5 memory targets round up to decimal 100 MB buckets, **stored in MiB**, with
minimum one bucket. The untouched measurements remain in `raw_targets_v4`.

## Fresh local labeler

The new `label.py` runs the generated standalone variant source on the detected
local NVIDIA GPU. It takes no hardware-name override and checks CUDA and NVML
GPU UUIDs. Each workload runs in a separate process under a nonblocking shared
GPU flock. A busy GPU or held lock prevents starting the next workload.

```bash
python data/label.py detect
python data/label.py run --generated my_transfer --output my_labels \
  --gpu-lock /path/to/shared-gpu.lock --limit 3
python data/label.py verify --output my_labels/nvidia_geforce_rtx_5090_local
```

Choose a lock path shared by every local training/labeling process. Running the
same command resumes from durable per-workload records. It skips completed
successes and failures; it never silently retries an OOM or rewrites its outcome.
A changed GPU, protocol, source manifest or code hash requires a new output
directory. `--id PROFILE_POINT_ID` can be repeated to run specific smoke cases.
`--steps-per-epoch N` defines an explicitly measured synthetic epoch. No real
dataset loader or inferred dataset-size scaling is claimed.

Protocol defaults: one warmup epoch; three repeats of three measured epochs;
one step per epoch, configurable; fixed recorded seed; synthetic inputs;
FP32 IEEE or FP16 AMP with GradScaler; selected Adam or Muon optimizer.
Wall time includes synchronized forward, MSE-to-zero loss, backward and optimizer
step. CUDA events measure elapsed GPU-stream time. Both step and epoch times
are retained in milliseconds, with individual epoch observations and exact
step/epoch arithmetic. Repeats continue the same model/optimizer state.

Resource profiling is a separate sustained training phase of at least one
second and 20 samples, with a 10 ms requested NVML sampling interval. Timestamps,
raw NVML samples, step count, mean/peak device busy percent, mean/peak device
VRAM bytes, Torch peak allocated/reserved bytes and software/hardware versions
are retained. NVML busy is a device busy-time proxy, **not SM occupancy**.
Sampling cadence is requested, not a claim of NVML's underlying update rate.

The CPU parent samples device memory before and after the fresh worker.
Accepted VRAM labels subtract the pre-worker baseline, retaining raw values.
Baseline drift above 64 MiB or post-worker busy above 10% discards the targets
as contamination. This cannot remove arbitrary concurrent activity during the
measurement; exclusive scheduling through the shared lock remains required.
The baseline excludes other processes and pre-existing device allocations, so
this is a different capture protocol from historical cluster measurements.
`historical_protocol_equivalent` is explicitly false.

Every result is fsynced and atomically renamed into `records/`. A crash can lose
only the active workload; successful earlier records survive. Worker logs,
requests and raw outputs remain in `work/`. Errors, OOMs, timeouts and baseline
contamination are recorded without accepted targets. The verifier rejects
hardware/run mismatches, failed records with targets, nonfinite metrics,
invalid units/orderings and incorrect epoch arithmetic. The local verification
uses two steps per epoch to exercise nontrivial epoch arithmetic.

The RTX 5090 smoke outputs and their durable resume/identity evidence live in
`verification_rtx5090/` and `receipts/`. They verify this implementation; they do
not replace historical A10G/A10/A100 labels or constitute a full relabel campaign.
Five workloads passed: text, tabular, audio and graph examples covering FP32 and
FP16 AMP, Adam and Muon. The independent `verify_local.py` recomputed every
accepted metric from individual timing observations and NVML samples. A resumed
run started no new worker and preserved all five result and worker-log hashes.
The CPU suite passed seven tests, and a clean-directory subprocess generated
workloads with empty `PYTHONPATH` using only packaged code, specs and runtime.
