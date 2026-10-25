"""Contracts deliberately incompatible with the legacy six-output artifacts."""

INPUT_SCHEMA_VERSION = "perfseer_v31_model_design_v1"
DATASET_VERSION = "perfseer_v31_a10_dataset_v1"
OUTPUT_CONTRACT_VERSION = "perfseer_v31_three_outputs_v1"
CHECKPOINT_VERSION = "perfseer_v31_checkpoint_v1"
TARGET_NAMES = (
    "train_epoch_ms",
    "train_avg_sm_util_percent",
    "train_peak_vram_mib",
)
HARDWARE_ID = "nvidia_a10_24gb"
