# PerfSeer v3.2 complete A100 training handoff — 2026-09-08

This release includes the native twelve-output T1 teacher and S1 student,
paired training/inference graphs, and all 40,020 dataset records with the active
shorter-time reference labels. Use an A100 to train the predictor; the measurements
being predicted still describe **A10 workloads**.

## Start on your A100 server

Use a Linux shell with Docker, NVIDIA Container Toolkit, and a CUDA-compatible
NVIDIA driver. Extract the ZIP and run these commands inside the extracted folder:

```bash
sha256sum --check --quiet SHA256SUMS
bash train.sh
```

The launcher checks the complete package and builds the CUDA image from the
**code and dataset in this ZIP**. It does not use the old published v3.2 image.
The first build downloads the digest-pinned PyTorch 2.10 / CUDA 12.8 base image
and verifies all paired graphs and label replacements on CPU. Allow several
minutes for this verification. Docker reuses matching build layers on later runs.
The image is not embedded in the ZIP; the first build requires registry access.
No host Python installation or extra dataset download is required for training.

GPU 0 is selected by default. To select another visible GPU:

```bash
PERFSEER_GPU=1 bash train.sh
```

Logs, normalization, feature cache, checkpoints, metrics, and exports persist in
`outputs/v32-twelve-shorter-time/`. Use the same command to resume an interrupted
run. Earlier three-output weights and checkpoints from the original labels are
incompatible; this release uses a fresh output directory and dataset fingerprint.
Do not mix old output files into that directory.

## Model, schedule, and gate

- T1: width 1280, ten shared backbone blocks. S1: width 224, two shared blocks.
- Both use paired training/inference graphs and six heads for twelve outputs.
- Teacher: up to 600 epochs; early stopping after six unimproved epochs once
  epoch 30 is reached. Passing the validation gate does not itself stop training.
- Student: up to 100 epochs, only after the selected teacher passes validation
  and then test. Every output must reach 95% of samples within 5% relative error.
- Effective batch 256, maximum microbatch 4. Complete-batch OOM retry can reduce
  the microbatch. This is a conservative launch setting, not a completed A100
  memory or throughput benchmark of the new design.

The dataset retains the 32,108 / 4,100 / 3,812 train/validation/test assignments
and all 2,352 graph pairs. The active fingerprint is recorded in `MANIFEST.json`.
The shorter-time policy updates 177,425 numeric values in 39,085 records; original
measurements remain in `native_targets` and the preserved original split files.
It applies separately within each split. These are selected reference labels,
not remeasurements. SM conflicts remain, and the strict gate may still fail.
When the teacher gate fails, student training is intentionally blocked.

## Contents and verification

- `src/perfseer_v3.2/`: full model, training, evaluation, export, dataset, and
  container code; the complete prepared dataset and original source provenance.
- `src/perfseer_v3/` and `src/perfseer_v3.1/`: required shared code/resources.
- Both original label/source ZIPs are actual files, with no external symlinks.
- `tests/`: focused v3.2, label-policy, v3.1, and transfer regression tests.
- `record/perfseer-v32/`: independent dataset-update evidence and the preserved native
  model-verification report. Historical report paths identify their original runs.
- `MANIFEST.json` and `SHA256SUMS`: dataset contract, release inventory, and hashes.

The release is verified by fresh extraction, full dataset verification, focused
tests with an isolated PyTorch 2.10 CPU runtime, and launcher checks with a
recording Docker stub. Docker was unavailable
in the packaging WSL session, so a real container build and A100 execution were
not performed. No pretrained weights, training campaign, or accuracy result is
included. See `src/perfseer_v3.2/README.md` for the full model and metric contracts.

## Optional Nautilus deployment

The included `k8s/perfseer-v32-a100-training.yaml` is a template and is not submitted
by the launcher. It requests `nvidia.com/a100` following the
[NRP GPU documentation](https://nrp.ai/documentation/userdocs/running/gpu-pods/).
Use your authorized namespace and replace the image placeholder with the digest
of an image built from this release and pushed to a registry accessible to your
cluster. The old published image does not contain this release.

To build a tagged image without starting local training:

```bash
PERFSEER_BUILD_ONLY=1 PERFSEER_IMAGE=your-registry/perfseer-v32:20260908 bash train.sh
```

The `PERFSEER_IMAGE` variable sets the tag of the image built from the ZIP; it is
not a switch to bypass that build. After publishing it, configure the namespace,
PVC, and immutable image digest in the template according to your cluster setup.
