# Justin Batch-1 ABA Transfer and Distillation Completion

## Dataset and labeling

- Justin-Linux batch-size-one variants: 992 total rows, split into 740 train,
  128 validation, and 124 test rows.
- Mixed A100 corpus: 3,968 rows (2,960 train / 512 validation / 496 test),
  fingerprint `5cfb147f74af382ee1b4ac44379b185910ff40da7a8a9627581d1fbe77477d6f`.
- The ABA label verifier passed all 992 batch-one configurations with the full
  twelve-target physical-label contract.
- The mixed-dataset verifier passed and each split retains the Justin batch-one
  rows above.

## Models

- Transfer teacher: complete, selected epoch 12, twelve outputs and six heads.
- Student: complete, selected epoch 16, twelve outputs, six heads, and
  2,888,744 parameters.  The export and held-out prediction file exist; the
  report hash matches the prediction-file SHA-256.

## Final held-out test result

The selected student test set has 496 rows.  Within-5% / within-10% accuracy:

| Target | 5% | 10% |
| --- | ---: | ---: |
| Train step wall time | 12.70% | 32.66% |
| Train step GPU time | 12.70% | 32.66% |
| Train epoch time | 18.75% | 29.64% |
| Train average SM utilization | 20.36% | 39.11% |
| Train average VRAM | 100.00% | 100.00% |
| Train peak VRAM | 100.00% | 100.00% |
| Train peak PyTorch allocation | 95.16% | 99.19% |
| Inference step wall time | 71.98% | 90.93% |
| Inference step GPU time | 69.35% | 89.92% |
| Inference average SM utilization | 52.62% | 79.44% |
| Inference average VRAM | 100.00% | 100.00% |
| Inference peak VRAM | 100.00% | 100.00% |

The final accuracy gate does not pass because the timing and SM-utilization
outputs remain below the required threshold; this is a completed experimental
result, not a successful accuracy-gate claim.
