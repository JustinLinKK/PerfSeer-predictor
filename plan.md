# Full 10k RTX 5090 Label Generation Plan

- Objective: use the canonical balanced 10,000-model source catalog for local RTX 5090 label generation, not the transformer-only diagnostic pack.

- Step: keep `--low-precision-focus none` for the full dataset.
  - Verifier: generated source command omits `--low-precision-focus te_transformer` and uses `--subset-size 10000`.

- Step: keep `--precision-sweep auto` for environment-aware precision labels.
  - Verifier: profiler help confirms `auto` is accepted and resolves after CUDA/Transformer Engine probes.

- Step: set Adam as the default profiling optimizer across local and Nautilus helper CLIs.
  - Verifier: grep confirms no remaining `default="sgd"` or `Default: sgd` in labeling/profiling helpers.

- Step: explain that a true optimizer sweep is separate from precision sweep.
  - Verifier: materializer label naming is reviewed for optimizer collision risk before recommending multi-optimizer runs.

- Step: provide a corrected copy-paste local 5090 command.
  - Verifier: command uses the balanced pack path, `--precision-sweep auto`, and `--optimizer adam`.
