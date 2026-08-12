# PerfSeer v3 transfer-refactor completion audit

Date: 2026-08-05  
Source plan: `PerfSeer_v3_A10G_to_NVIDIA_transfer_learning_refactor_plan.md`  
Audited baseline and current HEAD:
`ec36bdc39e6674f6b0dda2b0fed7895ebaf0cd95`

## Status vocabulary

- **Implemented and locally verified**: source/config/schema behavior exists and
  passed focused plus full local regression.
- **Prepared, external execution blocked**: the runnable workflow and verifier
  exist, but credentials or measured inputs are absent.
- **Production evidence blocked**: source support exists, but the plan forbids
  a pass/claim without real grouped GPU measurements.

## Coding goal audit

| Goal | Status | Evidence |
|---|---|---|
| One frozen 18K A10 base teacher/student workflow | Implemented and locally verified; production training blocked | `build_perfseer_v3_base_training_manifest.py`, `training_runner.py`, T0/T1/S0/S1 configs, exact 18K/54K gates |
| New GPU uses at most 1,024 paired labels and parameter-efficient adaptation | Implemented and locally verified | `transfer_subset.py`, rank-32/rank-16 configs, full-backbone rejection |
| Paired target adaptation, calibration, and GPU-specific artifact pair | Implemented and locally verified; production artifacts blocked | paired residual losses, target stages, artifact lineage, calibration fits |
| Scheduler fails closed for wrong GPU/schema/capture/precision/confidence | Implemented and locally verified | runtime/artifact tests and fallback statuses |
| Claims require grouped held-out curves rather than structural tests | Implemented and locally verified as a gate; evidence blocked | strict transfer evaluator rejects incomplete matrices and always denies broad-NVIDIA authorization |

## Architecture and data strategy audit

| Requirement | Status | Evidence |
|---|---|---|
| Hardware-independent workload trunk; no categorical unseen-GPU embedding | Implemented and locally verified | split global encoder and `test_workload_identity_uses_no_categorical_hardware_embedding` |
| Separate static/signature hardware encoder | Implemented and locally verified | 25 static/environment-compatible fields, 40 microbenchmarks, missing mask |
| FiLM identity plus zero-initialized low-rank residual | Implemented and locally verified | model initialization and adapter identity tests |
| Teacher rank 32 and student rank 16 | Implemented and locally verified | transfer configs and artifact rank checks |
| Freeze node/edge/workload blocks and SeerBlocks by default | Implemented and locally verified | named parameter policies and before/after byte checksums |
| Six output-specific paired residual transforms | Implemented and locally verified | log ratios, clipped-logit residuals, round-trip/boundary tests |
| OOM classifier uses peak-live and predicted VRAM/capacity directly | Implemented and locally verified | explicit decoded ratio model path and captured-input regression |
| Exact 18K/54K A10 campaign gate with frozen hashes and retained failures | Implemented and locally verified as a gate; real campaign blocked | production manifest builder and `assert_base_campaign_ready` |
| Grouped 128/256/512/1,024 nested budgets | Implemented and locally verified | exact 96/16/16 through 768/128/128 tests |
| Global strata, exact optimizer/scheduler pairs, then latent diversity | Implemented and locally verified | deterministic selector coverage tests |
| Active scores affect train only; test stays frozen | Implemented and locally verified | nested active-selection regression |
| 10-15% deterministic memory-boundary probes | Implemented and locally verified | fixed 12.5% allocation and power-of-two ladders |
| OOM and repair attempts retained with allocator evidence | Implemented and locally verified | attempt envelopes, manifest schema, materialization tests |
| Workload normalizer fit on A10 train only and reused exactly | Implemented and locally verified | split normalizers, exact lineage, refit rejection |
| Fixed/reference hardware normalizer with separate hash | Implemented and locally verified | hardware policy and distinct-GPU/missingness tests |

## Required source changes audit

All fourteen required source-change groups are implemented and locally
verified:

1. model split, hardware encoder, FiLM/low-rank adapters, named policies, CPU
   TorchScript;
2. feature tensor/normalizer separation and independent clipping;
3. schema version/hash bump and regenerated JSON;
4. canonical hardware profiles, units, missingness, physical/environment checks;
5. transfer config, residuals, freeze audits, and exact base/adapted lineage;
6. paired teacher/student losses and regularization;
7. four training stages, normalization reuse, budget/hash gates, resume, reports;
8. T0/T1, S0/S1, default adapters, and disabled controlled extensions;
9. grouped categorical plus latent/active subset selector;
10. versioned 40-point CUDA hardware profiler;
11. existing five-epoch runner extension for paired target labeling;
12. complete GPU-specific artifact lineage and tamper rejection;
13. exact target artifact selection, residual/calibration runtime, fallbacks;
14. selected-adapter export, eager/reloaded equality, size and CPU latency.

The file-level mapping is in
`reports/perfseer_v3_transfer_file_change_summary.md`.

## PR-sized acceptance-gate audit

| Sequence | Status | Local evidence boundary |
|---|---|---|
| PR 1 contracts/baselines | Implemented and locally verified | baseline HEAD recorded; old artifact/schema mismatch fails closed |
| PR 2 normalization split | Implemented and locally verified | A10/A100/L4/L40S/H100/RTX 5090/future profiles remain distinguishable; workload pairs match |
| PR 3 model conditioning | Implemented and locally verified | identity behavior, trainable fractions, frozen checksums |
| PR 4 paired residuals | Implemented and locally verified | six-output round trips and finite boundary losses |
| PR 5 selector/profiler/labeling | Implemented and locally verified | all four nested budgets, leakage/hash rejection, 40-point profile, retained OOMs |
| PR 6 runners | Implemented and locally verified structurally/synthetically | teacher/student adapter steps, exact artifact contracts, resume rejection; production run blocked |
| PR 7 artifact/runtime/export | Implemented and locally verified | wrong GPU/tamper rejection and eager/reloaded TorchScript equality |
| PR 8 capacity and label-budget experiments | Structural capacity completed; measured experiments blocked | parameter/size/local CPU report exists; production labels/artifacts do not |

## Required test audit

All listed unit-test subjects are covered: profile canonicalization/hashing,
fixed hardware normalization/masks, paired workload invariance, distinct target
features, adapter identity, freeze names/counts/checksums, six residual
round-trips, OOM handling/calibration, deterministic grouping, lineage,
wrong-GPU runtime, and TorchScript equality.

The integration subjects are covered at contract or synthetic execution level:

- a synthetic known A10-to-target metric transformation runs target teacher and
  student adapter steps and verifies frozen bytes;
- target teacher and target student role/subset/profile/base-artifact contracts
  are exact;
- resume lineage is checked and incompatible resumes are rejected;
- wrong-GPU graphs/manifests, target teacher/student mismatch, refitted workload
  normalization, and unapproved operation registries fail closed;
- runtime low-confidence/OOD and scheduler fallbacks are preserved.

Production end-to-end runs are intentionally not simulated into acceptance
evidence because the registry and campaign gates require measured GPU inputs.

The required regression suite, compile check, diff check, build, isolated wheel
import, packaged-data check, and export tests all passed. Exact commands and
results are in `record/perfseer_v3_transfer_local_verification.md`.

## Experiment and evaluation audit

The evaluator requires all 22 named experiment records, with the default
FiLM+low-rank curve at 128/256/512/1,024 and all controls at a frozen 256-label
comparison point. It also requires grouped test hashes, all specified slices,
uncertainty/OOM metrics, deployment ratios, scheduler outcomes, and comparison
gates.

No measured experiment row is available. Therefore every requested empirical
comparison remains **production evidence blocked**:

- no adaptation;
- per-output linear calibration;
- hardware specs/microbenchmark encoder only;
- low-rank only;
- FiLM plus low-rank;
- adapter plus heads;
- adapter plus last SeerBlock;
- absolute labels versus paired residuals;
- random versus categorical selection;
- categorical versus latent-diverse selection;
- fixed 256 versus active 128+128;
- hard-label student versus target-teacher distillation;
- T0 versus T1;
- S0 versus S1;
- static specs versus specs plus microbenchmarks.

Consequently, none of the plan's production accuracy, OOM, uncertainty,
critical-slice, student/teacher, deployment-ratio, pack/admit, missed-OOM,
fallback, batch-regret, or throughput-regret gates has been passed. The
evaluator exists specifically to prevent a partial evidence file from being
reported as success.

## Final deliverables audit

| Deliverable | Status | Location or blocker |
|---|---|---|
| 1. File-by-file change summary | Complete | `reports/perfseer_v3_transfer_file_change_summary.md` |
| 2. Versioned feature/manifest/profile/artifact schemas | Complete | `src/perfseer_v3/schemas/` |
| 3. Base T0/T1 and S0/S1 configs | Complete | `src/perfseer_v3/configs/transfer/` |
| 4. Target teacher/student configs | Complete | same directory |
| 5. Deterministic subset/active tools | Complete | selector and core module |
| 6. Target microbenchmark profiler | Complete | `scripts/profile_perfseer_v3_target_hardware.py` |
| 7. Base/target training commands | Complete | transfer runbook |
| 8. Test/package verification log | Complete | local verification ledger |
| 9. Counts/fractions/local CPU latency/artifact sizes | Complete structurally | capacity study; not production latency acceptance |
| 10. A10 accuracy and target label curves | Production evidence blocked | no real A10/target corpus or trained artifacts |
| 11. Ablation and scheduler-decision reports | Production evidence blocked | no paired target experiment matrix |
| 12. Rollback path | Complete | runbook and implementation status |

## External blockers and next authorized action

Three independent external inputs are missing:

1. the real 18,000 accepted A10G configurations and 54,000 measured epoch
   records;
2. measured operation GPU-time evidence needed to set the registry's
   `training_approved` contract truthfully, followed by paired target-GPU
   labels/artifacts;
3. a Nautilus kubeconfig/context for the prepared CUDA verifier.

The next safe action is to provision Nautilus credentials and/or supply the
frozen production campaign outputs. No schema, measurement, split-isolation,
registry, runtime, or evidence gate was weakened to manufacture completion.

Until the measured multi-generation evidence exists, the only authorized claim
is that PerfSeer v3 provides an A10G-pretrained, parameter-efficient small-label
adaptation workflow for a concrete new NVIDIA GPU. A broad “any NVIDIA GPU”
claim remains prohibited.
