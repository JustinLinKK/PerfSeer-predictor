# A100 transfer-cause experiment plan

**Goal:** Falsify the two remaining explanations for low held-out training-time
and training Streaming Multiprocessor (SM) accuracy: label noise and inadequate
adaptation of the frozen graph representation.

**Constraints:** Keep the 11,205-row A100 dataset, 12 outputs, six heads,
Ground Truth, seed, optimizer schedule, and train/validation/test group split
fixed. Use only ABA GPU 0 after proving it has no foreign owner. Never touch
GPU 1.

## Experiment A — repeatability floor

1. Choose a pre-registered, stratified panel of 12 test configurations: two
   per modality, balanced across the available batch/precision strata.
2. Execute each configuration three times in fresh worker processes using the
   immutable executable configuration, while retaining the same A100, software,
   warmup, five-second phase minimum, and 10-ms NVML sampling protocol.
3. For each training-time and SM target compute the within-configuration
   coefficient of variation and pairwise relative disagreement.
4. Decision rule: if the median repeat disagreement is materially below 5%,
   label noise cannot explain failure of a 5%-threshold predictor. Otherwise,
   report the measured noise floor as a limit rather than blaming the model.

## Experiment B — frozen-representation ablation

1. Reproduce the completed frozen baseline from its immutable checkpoint and
   final report; it selected epoch 14 and achieved test hits of 12.48% for
   training-step wall time and 13.79% for training-average SM.
2. Train a single ablation that differs only by unfreezing `blocks.9` (the
   final message-passing block) in addition to the existing phase adapters and
   six heads. Keep the parent teacher, split, effective batch 64, microbatch
   16, learning rate 2e-4, maximum 30 epochs, and early stopping rule fixed.
3. Select only by the existing validation `selection_key`; evaluate the selected
   checkpoint once on the untouched test set.
4. Decision rule: a material rise in test training-time/SM hit rates without a
   comparable collapse in inference outputs supports the frozen representation
   as a remediable source. No material rise falsifies this one-block hypothesis;
   it does not justify unfreezing more blocks without a separate experiment.

## Reporting

Report per-label Ground Truth accuracy within 5% and 10%, train-versus-test
generalization gaps, exact checkpoint/data hashes, GPU ownership, and the
repeatability statistics. Do not call a hypothesis confirmed unless its stated
decision rule is satisfied.

## Experiment C — process-level execution control

Experiment A found materially different CUDA-event times for graph-identical,
fresh-process executions.  Source inspection shows that the CUDA start event
is recorded before `make_inputs`: deterministic CPU tensor generation and
host-to-device copies occur inside every timed training call.  When Experiment
B releases GPU 0, rerun two high-variance held-out configurations at least
five times per condition under (a) the original default policy and (b) an
otherwise identical policy that preloads every deterministic GPU input before
the timed phase.  The controller must verify a single semantic captured-graph
hash for all repetitions.  If preloading materially reduces CUDA-event and SM
variation, unobserved host input-pipeline state is a supported cause.  If it
does not, run the already-staged explicit deterministic cuDNN/algorithm mode;
only a reduction there supports process-level kernel dispatch.  If neither
reduces variance, retain unrecorded device-execution state as the explanation
pending direct clock/power telemetry.
