# Clean source-consistent PerfSeer v3.2 training on ABA

## Objective and source archive

- Archive: `.cache/perfseer-v32-full-training-20260908.zip`.
- Archive SHA-256: `1d49dd244daa2bdf4594de31e2e03abe3d13bd228fe7bc42fd1e7e8b7b6bbc6a`.
- Target contract: the archive declares exactly 12 labels.
- Required identity: `provenance.model_source_sha256`, the exact model-source-byte hash.

## Cleaning policy

The archive's prior shorter-time policy leaves source-to-target conflicts, so it cannot ensure a deterministic source-code-to-label mapping. The new policy is `perfseer_v32_source_consistent_observed_vector_v1`:

1. Group every record by exact model source hash, preserving its existing split.
2. Select the lexicographically smallest `sample_id` in each group.
3. Copy that selected record's complete observed 12-label vector to every group member. No per-label synthetic average is created; every resulting target vector was observed.
4. Preserve each original vector in `native_targets` and attach source-consistency provenance.

The cleaner was unit tested (two tests) and reproduced byte-identical artifacts in a separate rebuild. Its verified output is `.cache/perfseer-v32-clean-source-consistent-v1`:

- Rows: 40,020.
- Source identities: 10,005.
- Changed rows: 30,015.
- Remaining source-to-12-label conflicts: 0.
- Clean fingerprint: `3df76bd63dd127bbcea402d996906c86b0e2ca8bf258f0ccbd2e57715b6641ad`.

## Comparison with Justin's latest branch

Justin's current Predictor `v2` is commit `25d81162fc55959e767b00e4f2aef057c982f1d1`. The deployed model definition, `src/perfseer-optimized/model.py`, is byte-identical to that branch (`d57fcc35ac2e4b0411485213e60374f2ab21863762287e430b4204b407e2c427`). Both use the same twelve-output, six-metric-head model contract (`hidden=1024`, eight blocks, six heads). The deployed trainer differs only by focused checkpoint-resume and per-epoch `latest.pt` persistence additions.

## ABA launch

The original unclean continuation was allowed to complete and save epoch 5. It ignored its checkpoint-safe `SIGINT`, so its exact PID was terminated only after that saved checkpoint, freeing GPU 1 for the clean run.

- Staged clean package: `/data1/yufan/perfseer-v32-clean-source-consistent-20260909`.
- Source-consistency verifier output: 40,020 rows, 10,005 sources, zero conflicts.
- Clean dataset view: `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/clean-dataset`.
- Initial clean output: `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/clean_v32_source_consistent_30_epochs`. It was retained after normalization and nine incomplete epoch-1 optimizer steps revealed that its conservative microbatch ceiling of 4 consumed only about 15 GiB of the 80 GiB A100; it has no completed epoch artifact.
- Authoritative training output: `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/clean_v32_source_consistent_30_epochs_mb16`.
- Authoritative training PID: `2055767`.
- The fresh run retains the same effective batch (256) and exact 30-epoch objective, but raises the memory-probe ceiling to 16. The native probe automatically backs off if necessary, so this does not assume a fit or change the model/data contract.
- The even-epoch accuracy and process monitors are attached to the authoritative output. The trainer emits the separate per-label within-5% and within-10% ground-truth accuracies at epochs 2, 4, ..., 30.

## Live evidence

- Epoch 1 completed on the clean 4,100-row validation split in 2,434.91 seconds (126 optimizer steps, effective batch 256, microbatch 16).
- Its artifact is `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/clean_v32_source_consistent_30_epochs_mb16/teacher-epoch-0001.json`; it binds the validation prediction hash `1ae52e27109485e791e6573c4877de5635db3a843d8e4ebf940e0685785e7039` to the twelve-label contract.
- Epoch 2 began normally after epoch-1 validation. The authoritative run remains in progress; the first required monitor report containing separately logged ground-truth within-5% and within-10% accuracies will be produced at epoch 2.

## Even-epoch ground-truth accuracy

Epoch 2 completed on the same 4,100-row clean validation split. The dedicated monitor persisted the result at `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/clean_v32_source_consistent_30_epochs_mb16/accuracy_every_2_epochs.log`.

- Within 5%: training wall/GPU step 51.00%, epoch 46.00%, training SM 40.24%, training average/peak VRAM 99.61%, torch peak allocation 95.71%; inference wall/GPU step 63.68%, inference SM 50.39%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 81.73%, epoch 79.56%, training SM 66.37%, training average/peak VRAM 99.61%, torch peak allocation 95.71%; inference wall step 83.12%, GPU step 83.17%, inference SM 79.93%, inference average/peak VRAM 98.93%.

Epoch 4 completed on the same 4,100-row clean validation split and was independently appended by the dedicated monitor.

- Within 5%: training wall step 63.12%, GPU step 62.27%, epoch 62.73%, training SM 41.05%, training average/peak VRAM 99.61%, torch peak allocation 95.71%; inference wall step 64.10%, GPU step 64.07%, inference SM 51.56%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 83.76%, epoch 86.39%, training SM 65.29%, training average/peak VRAM 99.61%, torch peak allocation 95.71%; inference wall step 87.98%, GPU step 87.95%, inference SM 79.07%, inference average/peak VRAM 99.46%.

Epoch 5 completed on the same 4,100-row clean validation split; epoch 6 then began normally. This is an informational odd-epoch result, while the dedicated accuracy monitor continues to persist the required every-two-epoch reports.

- Within 5%: training wall step 59.05%, GPU step 61.71%, epoch 61.24%, training SM 40.95%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 67.61%, GPU step 67.80%, inference SM 46.46%, inference average VRAM 98.37%, inference peak VRAM 98.93%.
- Within 10%: training wall/GPU step 89.59%, epoch 87.32%, training SM 65.95%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 89.46%, GPU step 89.54%, inference SM 82.07%, inference average/peak VRAM 98.93%.

Epoch 6 completed on the same 4,100-row clean validation split and was persisted by the dedicated every-two-epoch monitor. Epoch 7 began normally.

- Within 5%: training wall step 55.07%, GPU step 56.44%, epoch 60.12%, training SM 42.20%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 55.10%, GPU step 57.27%, inference SM 47.66%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 91.61%, GPU step 93.83%, epoch 98.49%, training SM 67.37%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 80.49%, GPU step 80.51%, inference SM 81.27%, inference average/peak VRAM 98.93%.

## Post-teacher student distillation launcher

`scripts/run_perfseer_v32_student_distillation.py` is staged at `/data1/yufan/perfseer-v32-clean-source-consistent-20260909/run_perfseer_v32_student_distillation.py`. It is deliberately separate from the active teacher process and restores the native `teacher-best.pt` only after verifying all of: the clean-data fingerprint, exact twelve-label order, completed 30-epoch teacher summary, and a valid `teacher-epoch-0030.json` artifact. It then invokes the native `run_stage("student", ..., teacher=teacher)` path, whose student checkpoint embeds the teacher-best SHA-256. Its local completion-guard tests plus the cleaning tests passed (six tests), and the staged remote file passed Python compilation. It must not start before the teacher reaches the required epoch-30 artifact.

To preserve that ordering without touching the teacher, ABA handoff PID `2432409` now waits for authoritative teacher PID `2055767` to exit. It then invokes the guarded launcher with GPU 1, 100 student epochs, effective batch 256, and microbatch 16. The student launcher itself is the final admission gate: a missing/failed/incomplete teacher cannot start distillation. ABA monitor PID `2432822` waits for the student PID file, records process/GPU/log/checkpoint evidence every minute for its first five samples and every 20 minutes after that, and preserves the output at `student-training.monitor.log`.

Epoch 7 completed on the same 4,100-row clean validation split; epoch 8 then began normally. This is an informational odd-epoch result, while the dedicated monitor continues to persist the required every-two-epoch reports.

- Within 5%: training wall step 65.51%, GPU step 62.73%, epoch 57.44%, training SM 40.76%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 58.66%, GPU step 57.07%, inference SM 51.90%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 91.12%, epoch 87.02%, training SM 67.44%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall/GPU step 72.51%, inference SM 83.44%, inference average VRAM 99.20%, inference peak VRAM 99.46%.

Epoch 8 completed on the same 4,100-row clean validation split. Its checkpoint is `teacher-epoch-0008.json` and binds prediction hash `0faac1072652231ea45c7bb0669b457e4447c4ec512268bc9dfee9bbfcd7cbdf` to the exact twelve-label contract.

- Within 5%: training wall step 58.37%, GPU step 63.83%, epoch 59.39%, training SM 41.12%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 64.27%, GPU step 57.68%, inference SM 48.00%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 88.66%, GPU step 88.73%, epoch 88.12%, training SM 67.78%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 86.20%, GPU step 85.49%, inference SM 83.71%, inference average/peak VRAM 98.93%.

Epoch 9 completed on the same 4,100-row clean validation split, with prediction hash `d099a9f38c8da58b6aa132db38388e0ada930dc749c47311eea57c5ac00a9395`. Epoch 10 began normally. This is an informational odd-epoch result; the required dedicated report remains every two epochs.

- Within 5%: training wall step 51.56%, GPU step 55.34%, epoch 59.17%, training SM 39.41%, training average/peak VRAM 99.61%, torch peak allocation 96.98%; inference wall step 61.20%, GPU step 61.46%, inference SM 48.63%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU/epoch step 76.54%, training SM 66.88%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 80.00%, GPU step 80.41%, inference SM 85.88%, inference average/peak VRAM 98.93%.

## A10 batch-size inclusion audit

The cleaned source archive is an A10 corpus (all 40,020 rows carry `a10_calibration` provenance), but it does not contain the literal `labels/a10_bs` source requested for training. The remote source `/data1/yufan/perfseer_v32_a100_bs_training_20260908/data/raw_source/a10_bs` contains 4,800 valid twelve-label records, and its `profile_point_id` set has zero overlap with the current clean dataset's 40,020 `sample_id` values. The already prepared corpus at `/data1/yufan/perfseer_v32_a100_bs_training_20260908/ready_for_train` contains all 44,820 rows (40,020 archive rows plus the 4,800 A10 batch-size rows) and has the same ordered twelve-label contract. It has not yet been substituted into the active clean teacher: doing so would violate the current run's fingerprint and cannot be done without retraining. The active teacher and its post-teacher student handoff therefore remain unchanged while a source-consistent merged-data path is prepared.

The 44,820-row prepared corpus is feasible for the same cleaning policy: it contains 10,005 exact source identities, and direct inspection of its three prepared splits found zero source identities appearing in more than one split. Its row schema is compatible at the data-contract level (`provenance.model_source_sha256`, `sample_id`, `input_path`, exact twelve targets), but differs from the zip-only cleaner's input transport and therefore requires a separate directory-input adapter rather than mutation of either existing dataset.

## Latest required accuracy: epoch 10

Epoch 10 completed on the 4,100-row clean validation ground-truth split. Its artifact is `teacher-epoch-0010.json` with prediction hash `e395f48e7b3457ce596f2e4743ba05a3af78579473e7dddb3052e2be77177d0b`; the dedicated every-two-epoch monitor has persisted the result. Training advanced directly to epoch 11 and remained live on ABA GPU 1.

- Within 5%: training wall step 52.10%, GPU step 54.93%, epoch 57.15%, training SM 41.07%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 72.07%, GPU step 76.78%, inference SM 48.24%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 90.83%, GPU step 90.00%, epoch 90.00%, training SM 65.10%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 91.93%, GPU step 91.95%, inference SM 81.88%, inference average/peak VRAM 98.93%.

Epoch 11 completed on the same 4,100-row clean validation ground-truth split with prediction hash `4414372996b1730d50cbb88c19abdb4eee0758b683b9440f80ebb2b5533a2d22`; epoch 12 began normally. This is an informational odd-epoch result; the dedicated monitor will persist the next required report at epoch 12.

- Within 5%: training wall step 72.15%, GPU step 63.95%, epoch 70.46%, training SM 38.41%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 59.22%, GPU step 59.46%, inference SM 49.12%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 95.54%, GPU step 95.51%, epoch 95.51%, training SM 65.59%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 80.15%, GPU step 80.20%, inference SM 83.17%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 12

Epoch 12 completed on the same 4,100-row clean validation ground-truth split. The ABA teacher PID `2055767` remained live after the report was emitted. Percentages below are the exact proportion of validation predictions whose relative error from Ground Truth is at most the stated threshold.

- Within 5%: training wall step 70.85%, GPU step 66.02%, epoch 70.90%, training SM 40.05%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 70.59%, GPU step 71.66%, inference SM 49.78%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 90.59%, GPU step 91.34%, epoch 90.02%, training SM 67.29%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 90.24%, GPU step 91.02%, inference SM 84.00%, inference average/peak VRAM 98.93%.

## Transfer-learning source package retrieved from Justin-Linux

While the ABA teacher remained active, the verified transfer source package was copied from `/home/justin/PerfSeer-predictor/record/perfseer-v32/transfer-source-data` to `.cache/perfseer-v32-transfer-source-data-20260909` using `scp -r -p`. The local package contains 39 files and is 8.1 GB on disk. It provides nine deterministic source subsets (8,558 samples) spanning audio classification, image classification, text summarization, tabular classification, toxic-comment classification, graph learning, segmentation, time-series forecasting, and object detection.

The integrity gate compared the remote and local `source-data-manifest.json` plus every raw archive SHA-256 value (nine archives) with normalized relative paths; `diff` returned exit code 0. The verified local metadata reports material fingerprint `72c06f7e02e5056e04db45ba6ce4b812218c782c9544c35c6ba8e66ea377c0d1`, source-data fingerprint `47329e44062996e72cab6dc684c172fda894b2c6bb3e11d1342748e03909425f`, and status `verified`. This cache entry remains separate from the active teacher's source-consistent corpus and has not changed its data fingerprint.

## Latest required accuracy: epoch 14

Epoch 14 completed on the unchanged 4,100-row clean validation ground-truth split. Its prediction artifact is `teacher-epoch-0014.json` with SHA-256 `8c6a7d0afc9fd48ae696448ab9546f91479363a77087c4c1fc106767538dbf58`.

- Within 5%: training wall step 62.73%, GPU step 64.39%, epoch 59.51%, training SM 39.44%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 67.34%, GPU step 68.93%, inference SM 48.00%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 91.15%, GPU step 94.93%, epoch 90.61%, training SM 66.54%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 89.10%, GPU step 89.76%, inference SM 84.44%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 16

Epoch 16 completed on the unchanged 4,100-row clean validation ground-truth split. Its prediction artifact is `teacher-epoch-0016.json` with SHA-256 `3f5b5be96abb69597b7a45cee09acbd4b585aa1f2b80cd5877dcee1b99c8eb09`. The teacher then advanced normally to epoch 17 on ABA GPU 1.

- Within 5%: training wall step 63.66%, GPU step 65.22%, epoch 62.10%, training SM 39.83%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 69.37%, GPU step 69.44%, inference SM 49.29%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 92.73%, GPU step 93.80%, epoch 86.34%, training SM 67.02%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall/GPU step 87.71%, inference SM 86.76%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 18

Epoch 18 completed on the unchanged 4,100-row clean validation Ground Truth split. Its prediction artifact is `teacher-epoch-0018.json` with SHA-256 `1dfeeeb5e0203221da4610658825f3175a51eb70fc17162c8a0e57f8e9495afc`. The ABA teacher PID `2055767` remained live after validation.

- Within 5%: training wall step 63.29%, GPU step 61.56%, epoch 63.17%, training SM 41.39%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 73.76%, GPU step 73.93%, inference SM 49.20%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 91.93%, epoch 92.98%, training SM 68.15%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 90.68%, GPU step 90.61%, inference SM 87.54%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 20

Epoch 20 completed on the unchanged 4,100-row clean validation Ground Truth split. Its prediction artifact is `teacher-epoch-0020.json` with SHA-256 `52952d1ad9da0541c175ea6b4a87280e95c73f2e453d9cd9b007b5d4bbeb09db`. The live ABA teacher PID `2055767` remained active after emitting the report. Percentages are exact relative-error hit rates against the twelve Ground Truth labels.

- Within 5%: training wall step 70.29%, GPU step 68.73%, epoch 66.90%, training SM 39.22%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 63.83%, GPU step 63.78%, inference SM 50.61%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 93.46%, GPU step 92.39%, epoch 96.68%, training SM 67.41%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 73.29%, GPU step 73.32%, inference SM 83.88%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 22

Epoch 22 completed on the unchanged 4,100-row clean validation Ground Truth split. Its prediction artifact is `teacher-epoch-0022.json` with SHA-256 `858c5589bc09cd46cda14c4354d8af1609daff082057d3c387b77a3d7151b16c`. The live ABA teacher PID `2055767` advanced normally to epoch 23 after validation. Percentages are exact relative-error hit rates against the twelve Ground Truth labels.

- Within 5%: training wall step 70.29%, GPU step 70.34%, epoch 70.15%, training SM 42.51%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 71.02%, GPU step 71.20%, inference SM 50.39%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 91.12%, epoch 91.12%, training SM 68.24%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall/GPU step 88.49%, inference SM 87.63%, inference average/peak VRAM 98.93%.

## Latest informational accuracy: epoch 23

Epoch 23 completed on the unchanged 4,100-row clean validation Ground Truth split. This is an odd-epoch status requested by the user; the required persisted accuracy cadence remains every two epochs. Its prediction artifact is `teacher-epoch-0023.json` with SHA-256 `d4fbce244679c083d27516a5d1c0f57ae060b1314ff1532af140611ec9e37849`. The live ABA teacher PID `2055767` is continuing toward the next required epoch-24 report.

- Within 5%: training wall step 70.24%, GPU step 70.27%, epoch 70.15%, training SM 43.22%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 71.02%, GPU step 70.98%, inference SM 52.76%, inference average/peak VRAM 98.93%.
- Within 10%: training wall step 94.20%, GPU step 94.54%, epoch 91.12%, training SM 67.44%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall/GPU step 90.93%, inference SM 87.32%, inference average/peak VRAM 98.93%.

## Latest required accuracy: epoch 24

Epoch 24 completed on the unchanged 4,100-row clean validation Ground Truth split. Its prediction artifact is `teacher-epoch-0024.json` with SHA-256 `6e53b5185d78b8033a8c89777359a98e868dc7854dd26a617d88409d16efb0b4`. The live ABA teacher PID `2055767` advanced normally into epoch 25 after validation. Percentages are exact relative-error hit rates over the twelve Ground Truth labels.

- Within 5%: training wall/GPU step 70.24%, epoch 70.22%, training SM 41.37%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall step 70.93%, GPU step 70.95%, inference SM 51.17%, inference average/peak VRAM 98.93%.
- Within 10%: training wall/GPU step 91.76%, epoch 92.39%, training SM 68.80%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall step 92.85%, GPU step 92.88%, inference SM 86.17%, inference average/peak VRAM 98.93%.

## Epoch-24 checkpoint architecture audit

Direct CPU-only inspection of `teacher-latest.pt` after epoch 24 confirms the live teacher contract rather than relying on training configuration alone: `num_outputs=12`, `hidden=1280`, and six independent `prediction_heads`. Their final projection widths are `3 + 1 + 3 + 2 + 1 + 2 = 12`, which proves both the required twelve ordered outputs and the six-head minimum. The checkpoint binds the same ordered target names used by validation and is keyed to the clean-corpus fingerprint `3df76bd63dd127bbcea402d996906c86b0e2ca8bf258f0ccbd2e57715b6641ad`.

## A10 batch-size corpus compatibility audit

The prepared corpus `/data1/yufan/perfseer_v32_a100_bs_training_20260908/ready_for_train` contains 44,820 rows, including 4,800 literal `a10_bs` rows, with the same ordered twelve-label contract and no source identity crossing a train/validation/test split. Its manifest fingerprint is `27db71ea6fa4bb2a36b8b086063e58f78d52fe80a364994bf30265423e8000b9`.

It cannot satisfy the active clean policy if identity is only `model_source_sha256`: a direct audit found all 10,005 source hashes have more than one target vector in this corpus. This reflects the intended batch-size variation; source code is held fixed while profiling configuration changes. A future A10-inclusive deterministic contract must therefore identify an input by source code *and* configuration (at minimum batch size), or discard the batch-size variations. No A10 data have been silently injected into the live teacher, whose original source-only fingerprint remains intact.

## Explicit teacher stop and verified student launch

On explicit user instruction, the live teacher was terminated during epoch 26 after the obsolete automatic handoff was stopped. Its preserved `teacher-latest.pt` snapshot has SHA-256 `99abf01fcc028690b94a0f63f17d01a95c0d7f789f6da505d5a0699c8d0bb09f`, exact twelve-label contract, clean-data fingerprint `3df76bd63dd127bbcea402d996906c86b0e2ca8bf258f0ccbd2e57715b6641ad`, and checkpoint epoch 26. Its last completed Ground Truth validation is epoch 25, artifact prediction hash `c2097cc2db1b78caed25ba049ecadd02e140cabe277609011221d3441c81acea`.

- Epoch-25 within 5%: training wall/GPU/epoch step 70.12%/70.15%/70.05%, training SM 40.41%, training average/peak VRAM 99.61%, torch peak allocation 99.12%; inference wall/GPU step 70.83%/70.76%, inference SM 51.39%, inference average/peak VRAM 98.93%.
- Epoch-25 within 10%: training wall/GPU/epoch step 92.39%, training SM 68.41%, training average/peak VRAM 99.61%, torch peak allocation 100.00%; inference wall/GPU step 92.93%/92.90%, inference SM 86.22%, inference average/peak VRAM 98.93%.

The student override records this distinction rather than fabricating an epoch-26 validation. It accepts an explicitly stopped snapshot only when a twelve-label epoch artifact exists at or before its checkpoint epoch. The remote student PID `3030607` started on ABA GPU 1 and completed steps 1--5 of epoch 1/100 with finite losses `0.184929`, `0.188683`, `0.187743`, `0.185942`, and `0.186335`; GPU use then reached 12,296 MiB. Its command includes `--allow-interrupted-teacher` and uses the verified epoch-26 snapshot with epoch 25 marked as the last validated teacher epoch.

## Cached student-only distillation restart

The initial 100-epoch student path computed the frozen teacher network on every microbatch, which made it needlessly slow despite the student's small 2,888,756-parameter configuration. On explicit user direction, that process was stopped and replaced by a 20-epoch cached-distillation path. The new launcher precomputed the exact epoch-26 teacher outputs once for all 32,108 training rows and wrote the 1.6-GB cache at `cached-student-20-epochs/.cache/teacher-output-cache.pt`.

The first cached launch finished that cache successfully but failed before student epoch 1 because the generic runner hashes `output/teacher-best.pt` to bind its provenance. The tested narrow fix materializes an exact copy of the epoch-26 `teacher-latest.pt` at that expected path and verifies its SHA-256 before training. Local launcher tests passed 7/7 after the fix. The cached result was preserved; the replacement process (`PID 3233825`) began from it, not from a second teacher-cache pass. At the recorded live check it had reached student epoch 1 step 101/126 with finite loss `0.081747`; no student validation artifact exists until that epoch completes.

## Cached student required accuracy: epoch 2

Epoch 2 completed on the unchanged 4,100-row clean validation Ground Truth split; its prediction artifact is `student-epoch-0002.json` with SHA-256 `e7645d7906f77a85ff07594180f1fa97697631507f55a580b01ba6e8e003c4e0`. Exact within-5% / within-10% hit rates are: training wall step 53.02% / 100.00%, training GPU step 51.93% / 100.00%, training epoch 51.12% / 92.12%, training SM 38.88% / 66.20%, training average and peak VRAM 99.61% / 99.61%, Torch peak allocation 95.49% / 95.71%; inference wall step 57.37% / 82.61%, inference GPU step 58.44% / 84.00%, inference SM 50.39% / 81.98%, inference average and peak VRAM 98.93% / 98.93%. The student advanced directly into epoch 3.

## Predictor v3.2 student-only package

The local deployment package is staged at `predictor_v3.2/`. Its `source/` directory contains the exact 67 Python source files imported by the v3.2 student inference closure, including the compatibility modules needed to restore the checkpoint. It passed a clean import and verified the twelve-label contract. The staged package contains no teacher-named file, teacher checkpoint, teacher cache, dataset, optimizer state, prediction dump, training log, or generated bytecode.

The sole student weight is deliberately pending: the live cached-distillation process has not yet emitted `student-export.pt`. On verified completion, the selected exported student checkpoint will be copied into `predictor_v3.2/weights/predictor_v3.2_student.pt`, along with a manifest containing its SHA-256, selected epoch, model configuration, label contract, and validation metrics.

## Predictor V3.3 SM specialization (staged, not launched)

V3.3 is isolated locally in `predictor_v3.3/`; it did not modify the active V3.2 process. V3.2 already assigned each average Streaming Multiprocessor (SM) label a singleton head, but each was a single generic multilayer perceptron and had the lowest observed within-5% accuracy. V3.3 retains all 12 ordered outputs and six output groups, replacing each of the two SM heads with a three-expert, context-gated mixture. It also uses an SM-emphasized loss: the original six-group-balanced relative error is averaged with the two-SM relative error, giving SM misses additional gradient signal without dropping the other ten labels.

The focused behavioral tests passed 2/2: the new head exposes three finite gated experts and the V3.3 SM-only loss penalizes an isolated SM miss more than V3.2. Imports of the V3.3 model, inference, runner, and training modules passed while preserving the 12-label contract. V3.3 has a distinct checkpoint and loss version and requires fresh training; no V3.3 training job has been started.

## Superseding V3.2 student-only package completion

The V3.2 cached student completed and the local `predictor_v3.2/` package now contains the verified deployable student export at `weights/predictor_v3.2_student.pt` (11,622,131 bytes; SHA-256 `988fa6fec1df66349a40c1dd2585ccc5f25ef2a72d88ea38af8f5b97e45850e7`), its 12-label validation metadata, and the 67-file inference dependency closure. The import and CPU checkpoint-load checks passed. The package contains no teacher weights/cache, optimizer state, or dataset.

## V3.3 SM-specialized 20-epoch completion

This supersedes the preceding staged-only note. The V3.3 student trained on ABA to completion with `status: completed`, 20 persisted epoch artifacts (`student-epoch-0001.json` through `student-epoch-0020.json`), `student-best.pt`, `student-export.pt`, and `training-summary.json`. The final deployable export is 12,862,699 bytes with SHA-256 `69f3317f57de9e3ef0dd1257163423155b63cec349fedc24b45e2985d0233d6b`; the selected student checkpoint is epoch 14.

The model retained the required 12 targets and six head groups `((0,1,2),(3,),(4,5,6),(7,8),(9,),(10,11))`. Both singleton SM groups use the V3.3 context-gated head with three experts (`sm_experts: 3`); the V3.3 SM-emphasized distillation loss was used. Ground Truth validation was calculated at every required even epoch (2, 4, 6, 8, 10, 12, 14, 16, 18, and 20) on the unchanged 4,100-row validation split.

Final epoch-20 Ground Truth hit rates (within 5% / within 10%) were: training wall/GPU step 64.20% / 92.49%, training epoch 70.83% / 92.49%, training average SM 42.71% / 69.20%, training average/peak VRAM 99.61% / 99.61%, Torch peak allocation 100.00% / 100.00%; inference wall step 64.15% / 82.90%, inference GPU step 64.17% / 82.93%, inference average SM 52.68% / 83.98%, inference average/peak VRAM 98.93% / 98.93%.

## V3.2 teacher checkpoint added to the local package

On explicit user request, `predictor_v3.2/` now contains both model roles. The teacher is `predictor_v3.2/.cache/teacher_v3.2_epoch26.pt` (2,238,044,505 bytes; SHA-256 `5792c12088988dfc514d76a0a4d0a7ff4f7155c26dea529486194066cc656a42`), which matches the V3.2 student metadata's distillation teacher checksum and epoch 26. A local CPU load verified `role: teacher` and the exact 12-label target contract. The existing student export remains `predictor_v3.2/weights/predictor_v3.2_student.pt` (SHA-256 `988fa6fec1df66349a40c1dd2585ccc5f25ef2a72d88ea38af8f5b97e45850e7`).
