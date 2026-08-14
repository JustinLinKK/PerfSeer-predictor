# PerfSeer V3 Disaster-Tweets Substitution

## Objective

Create `feature/perfseer-v3-nautilus-a10-nonvision-4gpu-disaster-v2`
from the clean non-vision four-A10 branch. Preserve the existing V1 corpus,
manifest, image, and hashes as historical references. Replace the deprecated
Detecting Insults competition with Kaggle Natural Language Processing with
Disaster Tweets while keeping all compute and distribution quotas unchanged.
Build and test locally only; execute no cluster or registry command.

## Contract and lineage

- Add profile `native_a10_nonvision_disaster_v2` and an isolated workspace at
  `/workspace/perfseer-v3-native-a10-nonvision-11200-disaster-v2`.
- Replace `detecting-insults` at its historical ordinal with task
  `disaster-tweets`, Kaggle slug `nlp-getting-started`, modality `nlp`, and a
  binary categorical target. Inputs are ordered keyword, location, and text
  strings; source IDs are unique Kaggle `id` values.
- Freeze the advertised three-file inventory and its SHA-256. Fail closed on
  inventory/archive drift, schema drift, duplicate IDs, empty text, invalid or
  missing target classes, or fewer than 4,096 valid rows.
- Select exactly 4,096 training rows with the existing deterministic SHA-256
  ordering and no replacement.
- Preserve 11,200 candidates, 12 tasks, 22 families, 33,600 measured epochs,
  all modality/precision/execution/regime/checkpoint distributions, and exactly
  2,240 candidates at each effective batch size 32, 64, 128, 256, and 512.
- Preserve the former 1,075-row insults allocation exactly across BERT, MLA,
  BiLSTM-CRF, FastText, and DistilBERT and across every compute dimension.
- Generate a V1-to-V2 crosswalk with 9,050 unchanged rows, 1,075 NLP registry
  rebounds, and 1,075 dataset substitutions. Preserve task-independent compute
  signatures and prohibit silent merging of insults and disaster measurements.
- Bind predecessor IDs/hashes, substitution contract, source locks, and the
  dataset-substitution flag into results and exports.

## Image and workflow

- Add a sibling immutable Disaster V2 image using the existing pinned PyTorch
  2.10/CUDA 12.8 base and dependency lock; keep V1 reproducible.
- Continue using pinned MLE-bench for the other eleven tasks. Use a narrowly
  scoped, hash-verified CSV preparer for Disaster Tweets, copying only the three
  contracted files into the prepared source.
- Lock remote inventory, downloaded archive, and extracted inventory on the
  workspace. Reject V1 state rather than migrating it.
- Keep the unified CLI, four independent production A10 workers, one-worker
  RTX 5090 validation, OOM repair, failure isolation, export, and verification
  behavior.
- Preserve the 32 pilot ordinals; regenerate only its five affected IDs. Render
  V2-specific digest-only pilot, chunk, and export Jobs with the unchanged A10
  resource contract.
- Replace the active Detecting Insults gate with Disaster Tweets. Probe Disaster
  first, TensorFlow Speech Recognition second, and the other ten afterward.
  Require an actual smallest-file download for every competition.

## Acceptance

- Verify exact corpus totals, affected allocations, batch totals, unique IDs,
  historical V1 hashes, and the 9,050/1,075/1,075 crosswalk.
- Test the custom preparer against valid data and every specified schema,
  inventory, archive, count, and class failure.
- Exercise all five affected families and four precision paths with real
  Disaster data when Kaggle access permits.
- Re-run the 11,200-construction audit, 123-route fixture matrix, concurrency and
  injected-failure tests, export reconstruction, dependency/image preflight,
  secret scan, and local YAML verifier.
- Rebuild after executable-source changes and run the 32-label, five-epoch RTX
  5090 workflow only when all twelve real Kaggle download gates pass. If the
  Disaster or Speech agreement remains blocked, record the external blocker and
  do not substitute mirrors or synthetic evidence.
- Run no `kubectl`, Nautilus, image-push, or other cluster command.

## Assumptions

- Availability and operational stability take priority over preserving the old
  insults-domain semantics.
- Existing V1 labels remain a separate corpus; no workspace migration occurs.
- The operator must accept both Disaster Tweets and TensorFlow Speech
  Recognition rules before full real-data validation can complete.
