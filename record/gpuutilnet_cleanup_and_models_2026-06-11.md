# GPUUtilNet Cleanup And Calibration Models Transfer

- Date: 2026-06-11

- Task: identify GPUUtilNet ground truth hardware, remove out-of-memory rows, and copy calibration model files from `Justin-Linux`.

- GPUUtilNet ground truth hardware evidence: the project README says GPUUtilNet data comes from GPUMemNet, and arXiv 2602.17817 states all data collection and experiments use an NVIDIA DGX Station A100 with four NVIDIA A100 GPUs, each with 40 GB High Bandwidth Memory 2.

- Local GPUUtilNet CSV cleanup path: `dataset/gpuutilnet/Analysis/00-Cleaned-NoteBooks/001-visualizations`.

- Local GPUUtilNet CSV cleanup path: `dataset/gpuutilnet/Analysis/00-Cleaned-NoteBooks/002-MLP-based-estimators/data`.

- Removed rows: `CNN.csv` removed 2797 `OOM_CRASH` rows.

- Removed rows: `Transformers.csv` removed 472 `OOM_CRASH` rows.

- Removed rows: `MLP.csv` removed 0 rows.

- Remaining rows per CSV copy: `CNN.csv` 6203, `Transformers.csv` 4539, and `MLP.csv` 3000.

- Remote source path: `Justin-Linux:~/PerfSeer-predictor/nrp_calibration_pack/models`.

- Local destination path: `nrp_calibration_pack/models`.

- Transfer method: `rsync` over `ssh`.

- Transfer status: complete.

- Transfer verification: `nrp_calibration_pack/models` contains 10001 Python files and uses 727 MB.

- Correction: `dataset/gpuutilnet` was absent after cleanup, so GPUUtilNet was cloned again from `git@github.com:itu-rad/GPUUtilNet.git`.

- Correction verification: both CSV data copies were cleaned again after reclone.
