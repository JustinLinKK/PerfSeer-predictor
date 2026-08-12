# PerfSeer v3 transfer-refactor local verification ledger

Date: 2026-08-05  
Audited Git HEAD: `ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95`

This ledger records local structural, regression, packaging, and prepared-CUDA
verification. It is not production A10G or target-GPU accuracy evidence.

## Focused implementation checks

- Model, training, and hardware checks after the explicit predicted-VRAM /
  capacity OOM input change: **30 passed**.
- Deterministic subset selector after exact optimizer/scheduler-pair coverage:
  **10 passed**.
- Artifact, subset, and model checks after adding all transfer policies to the
  generated artifact schema: **30 passed**.
- Hardware, subset, and artifact checks after complete-profile validation:
  **33 passed**.
- Base-lineage, training, and hardware checks after exact checkpoint and frozen
  normalizer binding: **35 passed**.
- Final transfer subset, artifact/runtime, training, and hardware aggregate:
  **48 passed**, 69 warnings. The warnings were PyTorch/TorchScript deprecation
  warnings and Python `TreeSpec` future warnings.
- Hardware-profile canonicalization/hash regression: **1 passed**, 15 warnings.
- The explicit A10/A100/L4/L40S/H100/RTX 5090/synthetic-future normalization
  matrix is covered in the final exhaustive aggregate. Its focused
  hardware-transfer file passed **14 tests**, 17 warnings.

## Full repository regression

Command:

```text
.venv/bin/python -m pytest -q tests/test_perfseer_v3_*.py
```

Result:

```text
259 passed, 271 warnings, 336 subtests passed in 2909.36s (0:48:29)
```

The warnings were existing PyTorch/TorchScript and PyG deprecations plus two
known compiler side-effect warnings from the representative library corpus.
There were no test failures, errors, skips used to bypass transfer checks, or
unexpected test termination.

## Static and generated-contract checks

The following completed with exit status zero:

```text
git diff --check
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python scripts/build_perfseer_v3_transfer_schemas.py
jsonschema.Draft202012Validator.check_schema(...) for all five v3 schemas
```

The following identity-defining A10 files have no diff from the audited HEAD:

- `src/perfseer_v3/configs/a10g_18k_dataset_pack.yaml`
- `src/perfseer_v3/dataset_pack/contracts.py`
- `src/perfseer_v3/dataset_pack/sampler.py`
- `src/perfseer_v3/dataset_pack/task_registry.py`
- `src/perfseer_v3/dataset_pack/models/factory.py`
- `src/perfseer_v3/dataset_pack/operation_sampler.py`

Current contract hashes:

- Feature schema: `eb2c59805c04deeb32785699be387144b419a03764369454a9ac0b4259fe2e8a`
- Hardware normalization policy: `bfe1e79f8b5b7212379c9c2a22f960adbf98d4024de589b5f9d6bdf8f4afdba7`

## Capacity and local CPU deployment evidence

Command:

```text
.venv/bin/python scripts/benchmark_perfseer_v3_capacity.py \
  --benchmark-students \
  --output reports/perfseer_v3_transfer_capacity_study.json
```

Result:

```text
candidates=7 benchmarked_students=4
report_sha256=2b6def85705ad3a3800806de7600ef35e4fbc1cfca7cad617fdfc38370214ecd
```

The serialized file SHA-256 is
`cddeefdc3ef236135290efb02ea9d5f9003afce64c451182d51bbcfcd1e3c896`.
The content-declared report hash intentionally excludes its own hash field.

## Build and isolated installation

Command:

```text
uv build
```

Result:

```text
Successfully built dist/perfseer-0.1.0.tar.gz
Successfully built dist/perfseer-0.1.0-py3-none-any.whl
```

Distribution hashes:

- sdist: `39b84d21518ad44b545179fdb45f000d55933729cb7598a7f6a01d51367b98ec`
- wheel: `36697f7d1e02e9424a3552eed8f2bd0810c7efb28e73edb60c35a410d2e6b71a`

The final isolated check installed the wheel with `uv pip --target` into
`/tmp/tmp.cPcaSorYFR`. It verified that `perfseer_v3.__file__` resolved inside
that directory, loaded all five JSON schemas, and resolved all eleven transfer
configs as packaged data.

Two harness iterations failed before this final success and were not counted:

1. a cleanup trap was rejected before execution because recursive removal is
   prohibited by the command safety policy;
2. `.venv/bin/python -m pip` failed because this venv intentionally has no
   `pip`, and the subsequent isolation assertion correctly detected the source
   checkout instead of an installation.

The successful verifier used the supported `uv pip` path and `set -e`.

## Prepared Nautilus CUDA verification

`scripts/prepare_perfseer_v3_nautilus_verifier.py` regenerated the tracked
ConfigMap+Job manifest. An independent archive verifier checked every embedded
file byte-for-byte against the checkout.

- Embedded source files: 176
- Embedded source-bundle SHA-256:
  `3494f7d61bf78da111e548a2c45362173b68ed56d566791553fe207a612ccefc`
- Manifest file SHA-256:
  `61254393e51c94136ef5d4d7a702c6bb616c11830903868862632826973a91e9`
- Image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- GPU request/limit: one `nvidia.com/gpu`
- Active deadline: 900 seconds
- Backoff limit: 1
- Monitor shell syntax: valid
- Monitor cadence: 45 seconds for the first five minutes, then 1,200 seconds

No Kubernetes object was submitted. `/home/justin/.kube/config` is absent,
`KUBECONFIG` is unset, `kubectl config current-context` reports that no current
context is set, and `kubectl config get-contexts` lists none.

## Evidence boundary

The repository has no production A10G 18K/54K label corpus, approved measured
operation registry, paired target-GPU labels, trained production base/target
artifacts, or cluster credentials. Therefore no production accuracy,
calibration, label-budget curve, ablation, scheduler-utility, or broad-NVIDIA
claim was verified locally.
