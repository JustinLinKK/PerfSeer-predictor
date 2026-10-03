**PerfSeer v3.2 teammate delivery — verified findings, 2026-09-07**

The uploaded run is now identified. The screenshot is epoch 36 validation, and
the supplied teacher checkpoint is epoch 37. The verified dataset/input ambiguity
from the earlier audit applies to this run. It makes the configured accuracy gate
unattainable for a deterministic predictor using the current inputs. The evidence
does not establish that the graph backbone needs a capacity or head redesign.

**Delivery and provenance**

- Upload: `/home/justin/perfseer_audit_delivery_20260906.tar.zst`, 1,900,273,203 bytes.
- Archive SHA256: `2a8ef7de8304d46b0838c5f2587683db45b72cd2d1fa073602e7a30770ab74ba`.
- [Extracted delivery](../record/perfseer-v32/teammate-audit-20260907/perfseer_audit_delivery_20260906/README.md)
  includes the checkpoint, source, predictions for all splits, split manifests,
  46 completed-epoch reports, and launch/training logs.
- All 66 payload files match the supplied checksums. The checksum list also lists
  itself, and that self-check fails. This packaging defect does not invalidate the
  independently verified payloads; exclude the hash list itself when regenerating it.
- The source archive was safely extracted separately; existing project source and
  the delivered source snapshot were preserved. Five executable files differ from
  this checkout: v3.1 model/training/version and v3.2 training/runner.
- Dataset fingerprint and all split-file hashes match the local dataset. Every
  supplied train/validation truth value equals the corresponding prepared native
  label after the trainer's float32 conversion. Sample IDs, groups, input hashes,
  counts, target order, and validity masks reconcile for all 36,208 rows.
- The twelve-output checkpoint uses the unchanged `perfseer_v32_three_outputs_v1`
  identifier. This is misleading version metadata; the actual ordered targets and
  state dimensions confirm twelve outputs. The local three-output model is a
  different implementation and should not be overwritten with this fork implicitly.
- Checkpoint SHA256: `be8c155c65660790baf6dee92bccca242d3fd467a9017738d0320001d600aefe`.
  Its executable source fingerprint exactly matches the delivered source:
  `3ebe59ccf904990e319b035f9b4e13f2f867dab4644b5b655fe595f28e61098d`.

**What explains the poor timing and SM results**

1. **The inputs cannot distinguish conflicting labels.** Training has 32,108 rows
   but only 700 distinct input hashes; validation has 4,100 rows and 120 distinct
   input hashes. Identical inputs have different measured times and SM utilization.
   The earlier `calib_4679`/`calib_4979` example is present in this run: epoch labels
   are 277.016 and 630.496 ms, with SM labels 15.982% and 7.107%. Their source input
   and node specifications are identical. The loaded checkpoint returns the same
   BF16 prediction for both: approximately 327.536 ms and 13.387% SM.

   Choosing the best possible constant prediction independently for every identical
   input group and every output gives the following optimistic validation bounds.
   This oracle uses validation labels and is an upper-bound diagnostic, not a
   trainable baseline or a generalization result. It assumes one deterministic
   prediction per identical input; additional measured execution features can
   change that contract.

   | Output | Screenshot epoch 36: within 5% | Delivered epoch 37 predictions: within 5% | Oracle maximum within 5% | Oracle maximum within 10% |
   | --- | ---: | ---: | ---: | ---: |
   | Train epoch | 39.49% | 39.68% | 55.37% | 76.27% |
   | Train SM average | 34.78% | 37.90% | 54.37% | 74.71% |
   | Inference SM average | 54.71% | 58.90% | 76.76% | 93.32% |

   The run requires at least 95% within 5% for **every** output. That gate cannot
   pass with these labels and inputs merely by training longer or increasing model
   size. Float32 labels move one SM boundary relative to the earlier float64-label
   audit; the current 10% SM bound is 74.71%, previously 74.73%.

2. **Early loss and checkpoint selection suffered from zero-SM relative errors.**
   The delivered predictions contain six training-zero and seven validation-zero
   train-SM labels, plus two training-zero and six validation-zero inference-SM
   labels. Dividing a nonzero prediction by the evaluation epsilon `1e-6` makes
   those rows dominate mean relative error. Epoch 1's logged maximum batch loss is
   31,107.61. Epochs 1–33 report enormous SM mean errors despite approximately
   percentage-point-scale absolute errors. The original early stop occurred at
   epoch 30 with epoch 16 selected, although epoch 29 had better worst-output
   5%-tolerance accuracy: 37.49% versus 32.22%.

   The delivered source already floors the SM **loss** denominator at 1.0.
   Logged training losses become much smaller before the epoch-34 evaluation
   change; earlier executable revisions were not supplied, so the exact training
   change point is not established. Evaluation's normalized-error scale changes
   sharply at epoch 34, while the reported 5%/10% hit metric retains its original
   epsilon rule. The apparent drop from errors near 15,000 to 0.12 is therefore
   not a comparable accuracy improvement. Do not apply the already-present loss
   repair again or claim that it resolves the conflicting labels.

3. **The near-perfect VRAM scores have a strong simple baseline.** A constant
   training-median prediction already reaches 99.05% within 5% for inference-average
   VRAM on validation, compared with this checkpoint's 99.17%. Memory success alone
   does not show that the representation can predict timing equally well.

4. **A large generalization gap is not the dominant observed symptom.** From the
   supplied epoch-37 predictions, train-epoch 5% accuracy is 43.87% on training and
   39.68% on validation; train-SM is 40.73% and 37.90%. Both training and validation
   performance are low. This is consistent with the verified input ambiguity,
   though optimization may still explain part of the gap below the oracle bound.
   At epoch 37 the learning rate remains 0.00049745, close to the 0.0005 peak under
   the 600-epoch schedule. This is not evidence from a completed cosine schedule.

**Reproduction and remaining run inconsistencies**

All 24 screenshot percentages match `teacher-epoch-0036.json` after rounding. The
delivered per-sample predictions are for epoch 37, so they should not reproduce
epoch 36 exactly. Independent aggregation of all train/validation predictions is
recorded in [analysis.json](../record/perfseer-v32/teammate-audit-20260907/analysis.json).

The supplied checkpoint was loaded using restricted tensor deserialization and
run on 12 validation examples spanning all six modalities, the contradictory pair,
and zero-SM cases. The predictor ran on the local RTX 5090; its output targets still
describe A10 workloads. The maximum relative differences from supplied predictions
were 1.44% in BF16 and 0.81% in FP32. This verifies runnable source/checkpoint
compatibility and approximate prediction agreement, not bitwise reconstruction of
the teammate's GPU/batch execution. The exact prediction-export command is absent.

The included `teacher-gate.json` belongs to epoch 16 and a different checkpoint
hash. It is stale evidence for the epoch-37 teacher. The README's latest-epoch
statement is also older than the epoch-46 and partial-epoch-47 logs. Neither is a
verified current status of the teammate's machine.

The supplied runner additionally constructs teacher and student test `Samples`
without the twelve-target map, although the prepared `targets` dictionary contains
only three targets. This is a later-stage code defect, separate from the screenshot
failure: a passing validation gate would reach missing target keys on test. Both
test constructors need the same `target_map=raw_targets` used by train/validation.
Checkpoint selection still minimizes worst normalized mean error rather than
maximizing worst 5%-tolerance accuracy. Aligning selection with the acceptance
metric requires an explicit new selection identity and fresh or reranked checkpoint
state; old keys must not be compared silently with new keys.

**What is needed to make the accuracy repair**

The delivery is sufficient to identify the actual run, reproduce the screenshot
from its epoch report, and confirm the input ambiguity. The missing evidence is
the original profiler/runtime, timestamped timing and SM telemetry, execution
conditions, and repeated measurements of matched workloads. The delivery explicitly
states raw NVML samples were not retained. These are needed to separate omitted
execution information, unstable profiling, and a reconstruction mismatch.

Preserve the original labels. First reproduce a small matched-workload panel under
controlled execution conditions, including the contradictory pair. If execution
settings explain the differences, include the relevant measurable settings in the
input contract. If the measurements are unstable or incorrect, repair profiling
and recollect/version the affected labels with repeat statistics. Do not add model
IDs as predictive shortcuts, drop hard rows, silently average conflicts, or weaken
the accuracy threshold to force a pass. Recompute the feasibility bounds before
another full training run. Then use matched validation experiments to test any
remaining loss, head, or backbone changes; no such accuracy improvement is proven
by the current audit.

Only the task plan and audit/report artifacts changed. No model, original dataset,
checkpoint, external job, or teammate process was modified. The reproducible
verifiers are [analyze_delivery.py](../record/perfseer-v32/teammate-audit-20260907/analyze_delivery.py)
and [verify_checkpoint_forward.py](../record/perfseer-v32/teammate-audit-20260907/verify_checkpoint_forward.py).
Archive, extraction, payload-checksum, source, and forward-verification records are
colocated under `record/perfseer-v32/teammate-audit-20260907/`. That directory is
ignored by Git, so attach its evidence files explicitly when sharing. No test-label
predictions were evaluated or used to choose changes.
