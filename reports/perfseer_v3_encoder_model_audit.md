# PerfSeer v3 encoder/model requirement audit

## Outcome

Local implementation requirements passed: 9/9.
Production-evidence requirements blocked by missing data/infrastructure: 3.

| Requirement | Status | Evidence |
|---|---|---|
| versioned canonical graph IR and deterministic hashes | local_pass | graph_ir_v3.py; build_v3_schema.py; schema mismatch tests |
| hierarchical exact/family/hash/phase/dtype/operator encoder | local_pass | model.py HierarchicalNodeEncoder; identity perturbation tests |
| typed edge semantics including external tensors, slots, alias and phase | local_pass | features.py typed external self-loops; edge perturbation tests |
| hardware/execution global encoder and confidence quality inputs | local_pass | HierarchicalGlobalEncoder; hardware embedding tests |
| full training phases, liveness, cost and coarsening representation | local_pass | capture_training.py; liveness_v3.py; cost_v3.py; coarsen_v3.py |
| existing-trunk control, phase-aware pooling and additive optional heads | local_pass | model.py; six-output contract and export tests |
| T0/T1/T2 and S0/S1/S2/S3 capacity sweep | local_pass | /home/justin/PerfSeer-predictor/reports/perfseer_v3_capacity_study.json; 7 exact parameter counts |
| gated CUDA/AMP teacher training and representation distillation runner | local_pass | training_runner.py; run_perfseer_v3_training.py; smoke verification |
| versioned artifact, verified CPU export and scheduler fallback | local_pass | artifact.py; deployment_export.py; runtime.py |
| production measured-GPU-time registry approval | production_blocked | This local diagnostic corpus has no production scheduler labels, limited real-model breadth, and no authenticated Nautilus target-hardware run; it can propose but not approve a production vocabulary. |
| family-held-out teacher/student prediction and calibration gates | production_blocked | no grouped production scheduler-label prediction corpus |
| peak predictor training memory/time and matched-v2 deployment ratios | production_blocked | grouped production scheduler-label corpus; family-held-out validation predictions; predictor-training peak memory and duration; matched v2 graph-bucket latency runner; production calibration and OOM evaluation |

## Feature contract

Schema hash: `7d7966b124db6d473db2391bffa692cad2c74858985ef2cd2f478b8fbe210e78`

Ordered widths: `{'node_continuous': 40, 'node_flags': 21, 'edge_continuous': 14, 'edge_flags': 5, 'global_continuous': 105, 'quality': 8}`

## Remaining production blockers

- Authenticate Nautilus and rerun the final CUDA verifier on target infrastructure.
- Collect grouped production scheduler labels and measured per-operation GPU time.
- Approve a registry revision only from the immutable measured GPU-time report.
- Train and select T/S candidates from held-out accuracy, uncertainty, OOM, latency, size, memory, and duration evidence.

Report SHA-256: `c4474c62677b1b5a4475d2855ebcb4c513aacb01117ed60f68d4c4c19fd9f439`
