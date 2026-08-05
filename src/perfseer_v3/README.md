# PerfSeer v3

This package is the isolated v3 implementation. It does not replace or mutate
the v2 graph schema, encoder, checkpoints, or scheduler path.

## Corrected v3 contract

- The measured A10G corpus trains one reusable base teacher/student pair. A
  concrete target GPU reuses their hardware-independent workload backbones and
  learns GPU-specific rank-32/rank-16 adapters from a frozen, paired small-label
  subset. Every final artifact still targets exactly one GPU and fails closed
  when the requested workload or hardware-profile hash differs.
- Optimizers and LR schedulers use exact, semantic-family, and stable-hash
  identities. This covers the standard PyTorch optimizer set, Muon (including
  Muon + AdamW parameter groups), LAMB/LARS/Lion, common schedulers, and
  distinguishable custom future names.
- Training inputs include epoch/step progress, current and parameter-group
  learning rates, warmup/decay, weight decay, momentum/betas/epsilon, and
  common optimizer/scheduler controls instead of only batch size and epochs.
- Dtype is retained per operation and tensor edge. FP32-to-BF16 layer changes
  are represented as `mixed` with an explicit conversion rather than reduced
  to one graph-wide precision label.

The base manifest contract is `perfseer_v3_training_manifest_v2`; target
adaptation uses `perfseer_v3_target_training_manifest_v1`. Feature, hardware
profile, transfer-subset, target-manifest, and artifact-metadata schemas live
under `schemas/`. See
`../../docs/perfseer_v3_transfer_learning_runbook.md` for the complete paired
labeling, training, evidence, and rollback workflow.

## Capture contract

`capture_export` uses `torch.export.export(..., strict=True)` first. A strict
failure can fall back to non-strict export only when eager and exported
programs agree on structure, tensor shape, dtype, and values for at least three
legal replay inputs. Every tensor-producing node is converted to `GraphIRV3`;
otherwise capture returns structured failure data.

`capture_training_graph` adds loss, AOT Autograd backward, and optimizer phase
nodes. If the version-pinned AOT path cannot represent a workload, an explicit
analytical summary is used with `backward_capture_quality=estimated`.

## Registered custom operations

A custom operation needs an abstract/fake implementation so `torch.export` can
infer output metadata. For example:

```python
@torch.library.custom_op("my_extension::fused", mutates_args=())
def fused(x: torch.Tensor) -> torch.Tensor:
    return extension_kernel(x)

@fused.register_fake
def fused_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

The graph retains the namespace, operation name, and overload. An operation
without a registry entry receives `family=unknown_or_custom`, `exact_op=UNK`,
a stable nonzero hash identity, generic tensor/cost features, and lower
coverage confidence. An operation that cannot provide export metadata fails
capture; it is never silently removed.

## Profiling and training gates

`ProfileWorkload` accepts only the exact model instance and input signature
used for capture. It checks eager/export equivalence before warm-up and records
raw boundaries, sustained sampler values, CUDA allocator peaks, environment
metadata, and OOM stage. Stochastic correctness checks replay eager and export
from identical CPU/CUDA RNG states. Gradient-only training workloads use
`optimizer_name="none"` instead of inventing optimizer work.

The checked-in operation registry is a bootstrap registry and deliberately has
`training_approved: false`. Production teacher/student training must remain
blocked until a target-hardware profiler-time report identifies the smallest
exact-operation vocabulary covering at least 95% of cumulative operator GPU
time. The matching v3 teacher must be trained before student distillation.

The local diagnostic reports include exact-operation microbenchmarks and a
42-entry P0/representative-real/composite corpus measured with exact
`OpOverload` identities and CUDA event pairs. They can propose a vocabulary,
but do not approve it because they are not a production scheduler-label corpus
or an authenticated Nautilus target-hardware run.

## Main verifiers

```bash
python scripts/build_v3_schema.py
python scripts/build_perfseer_v3_transfer_schemas.py
python scripts/audit_operation_coverage.py --corpus supported
python scripts/build_perfseer_v3_workload_manifest.py
python -m unittest discover -s tests -p 'test_perfseer_v3_*.py' -v
# CUDA:
python scripts/profile_perfseer_v3_microbenchmarks.py
python scripts/profile_perfseer_v3_supported_corpus.py
python scripts/run_perfseer_v3_cuda_verifier.py
```
