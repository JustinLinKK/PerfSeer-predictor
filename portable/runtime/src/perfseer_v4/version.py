"""Single-training-graph, seven-target PerfSeer v4 contracts."""

from perfseer_v31.version import HARDWARE_ID

INPUT_SCHEMA_VERSION = "perfseer_v4_training_graph_v1"
DATASET_VERSION = "perfseer_v4_training_dataset_v1"
FEATURE_VERSION = "perfseer_v4_training_features_v1"
NUMERIC_FEATURE_VERSION = "perfseer_v32_features_v2"
NORMALIZATION_VERSION = "perfseer_v4_train_moments_v1"
OUTPUT_CONTRACT_VERSION = "perfseer_v4_seven_training_outputs_v1"
CHECKPOINT_VERSION = "perfseer_v4_checkpoint_v1"
LOSS_VERSION = "perfseer_v4_group_balanced_loss_v1"
METRIC_VERSION = "perfseer_v4_relative_hit_v1"
SELECTION_VERSION = "perfseer_v4_worst_hit5_v1"
MODES = ("training",)
TARGET_NAMES = (
    "train_step_wall_ms", "train_step_gpu_ms", "train_epoch_ms",
    "train_avg_sm_util_percent", "train_avg_vram_mib", "train_peak_vram_mib",
    "train_peak_torch_allocated_mib",
)
HEAD_GROUPS = ((0, 1, 2), (3,), (4, 5, 6))
SM_INDICES = (3,)
POSITIVE_INDICES = tuple(i for i in range(len(TARGET_NAMES)) if i not in SM_INDICES)


def validate_targets(values):
    import math

    if len(values) != len(TARGET_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("expected seven finite physical training targets")
    if any(values[i] <= 0 for i in POSITIVE_INDICES) or any(not 0 <= values[i] <= 100 for i in SM_INDICES):
        raise ValueError("target outside physical range")
