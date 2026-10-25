"""Frozen, fail-closed quota and observation protocol for the V100 18K pack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fingerprints import canonical_sha256
from .labeler_profile import PROFILE


_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_QUOTA_CONFIG_PATH = _CONFIG_ROOT / (
    "a10_nonvision_11200_dataset_pack.yaml"
    if PROFILE.is_nonvision_4gpu
    else "v100_18k_dataset_pack.yaml"
)

FROZEN_QUOTA_CELLS = (
    ("vision", "resnet50", "ResNet-50", 400),
    ("vision", "efficientnet_b0_b4", "EfficientNet-B0/B4", 500),
    ("vision", "mobilenet_v3_large", "MobileNetV3-Large", 350),
    ("vision", "inception_v3", "Inception-v3", 350),
    ("vision", "vit_s16", "ViT-S/16", 500),
    ("vision", "swin_t", "Swin-T", 550),
    ("vision", "mlp_mixer_s", "MLP-Mixer-S", 300),
    ("vision", "stn_cnn", "Spatial Transformer Network + CNN", 250),
    ("vision", "mish_resnet", "Mish-ResNet", 250),
    ("vision", "prelu_elu_cnn", "PReLU-Net / ELU-CNN", 400),
    ("vision", "unet_groupnorm", "U-Net with GroupNorm", 550),
    ("vision", "pix2pix", "pix2pix generator + discriminator", 750),
    ("vision", "restormer", "Restormer", 1_000),
    ("nlp", "bert_base", "BERT-base fine-tuning", 550),
    ("nlp", "gpt2_small", "GPT-2-small classification/fine-tuning", 450),
    ("nlp", "t5_small", "T5-small encoder-decoder", 550),
    ("nlp", "llama_small", "Llama-style small decoder", 500),
    ("nlp", "mla_mini_transformer", "Multi-head Latent Attention mini-transformer", 500),
    ("nlp", "kimi_delta_attention", "Kimi Linear/delta-attention block", 500),
    ("nlp", "switch_moe", "Switch-style Mixture-of-Experts classifier", 550),
    ("nlp", "bilstm_crf", "BiLSTM-CRF", 300),
    ("nlp", "gru_rnn_seq2seq", "GRU/RNN seq2seq", 350),
    ("nlp", "fasttext_embeddingbag", "fastText-style EmbeddingBag classifier", 250),
    ("nlp", "distilbert_distillation", "DistilBERT distillation step", 550),
    ("audio", "panns_cnn14", "PANNs CNN14", 550),
    ("audio", "temporal_convolutional_network", "Temporal Convolutional Network", 400),
    ("audio", "m5_waveform_cnn", "M5 raw-waveform CNN", 350),
    ("tabular", "selu_mlp", "SELU self-normalizing MLP", 300),
    ("tabular", "tabtransformer", "TabTransformer", 350),
    ("tabular", "mixture_density_network", "Mixture Density Network", 300),
    ("graph", "cgcnn", "CGCNN", 500),
    ("graph", "gcn", "GCN", 250),
    ("graph", "gat", "GAT", 300),
    ("graph", "graphsage", "GraphSAGE", 250),
    ("generated", "independent_generated", "Independent generated architectures", 3_250),
)
FROZEN_MODALITY_TOTALS = {
    "audio": 1_300,
    "generated": 3_250,
    "graph": 1_300,
    "nlp": 5_050,
    "tabular": 950,
    "vision": 6_150,
}
NONVISION_QUOTA_CELLS = tuple(
    (
        modality,
        family_id,
        display_name,
        2_600 if family_id == "independent_generated" else accepted,
    )
    for modality, family_id, display_name, accepted in FROZEN_QUOTA_CELLS
    if modality != "vision"
)
NONVISION_MODALITY_TOTALS = {
    "audio": 1_300,
    "generated": 2_600,
    "graph": 1_300,
    "nlp": 5_050,
    "tabular": 950,
}


class QuotaConfigError(ValueError):
    """Raised when the frozen quota/protocol config fails closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise QuotaConfigError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise QuotaConfigError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise QuotaConfigError(
            f"{context} keys differ from the frozen schema; missing={missing}, unknown={unknown}"
        )


def _string(value: Any, *, context: str) -> str:
    if type(value) is not str or not value:
        raise QuotaConfigError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, *, context: str) -> int:
    if type(value) is not int:
        raise QuotaConfigError(f"{context} must be an integer")
    return value


def _boolean(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise QuotaConfigError(f"{context} must be a boolean")
    return value


def _integer_tuple(value: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise QuotaConfigError(f"{context} must be a list of integers")
    return tuple(value)


@dataclass(frozen=True)
class QuotaCell:
    modality: str
    family_id: str
    display_name: str
    accepted_configurations: int

    def as_frozen_tuple(self) -> tuple[str, str, str, int]:
        return self.modality, self.family_id, self.display_name, self.accepted_configurations


@dataclass(frozen=True)
class MeasurementProtocol:
    workers: str
    workers_per_qualified_physical_gpu: int
    concurrent_profilers_per_gpu: int
    allow_mps: bool
    allow_multi_job_gpu_packing: bool
    require_fresh_process_per_run: bool
    require_memory_release_before_next_run: bool
    total_epochs_per_run: int
    warmup_epochs: tuple[int, ...]
    measured_epochs: tuple[int, ...]
    maximum_epoch_time_relative_spread: float
    end_to_end_time_label: str
    derive_end_to_end_epoch_time_from_step_time: bool
    failed_attempts_count_toward_quota: bool
    oom_attempts_count_toward_quota: bool
    unstable_attempt_policy: str

    @property
    def measured_epoch_records_per_run(self) -> int:
        return len(self.measured_epochs)

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(self)

    def validate(self) -> None:
        expected = MeasurementProtocol(
            workers="auto",
            workers_per_qualified_physical_gpu=1,
            concurrent_profilers_per_gpu=1,
            allow_mps=False,
            allow_multi_job_gpu_packing=False,
            require_fresh_process_per_run=True,
            require_memory_release_before_next_run=True,
            total_epochs_per_run=5,
            warmup_epochs=(1, 2),
            measured_epochs=(3, 4, 5),
            maximum_epoch_time_relative_spread=0.10,
            end_to_end_time_label="directly_measured_complete_training_epoch",
            derive_end_to_end_epoch_time_from_step_time=False,
            failed_attempts_count_toward_quota=False,
            oom_attempts_count_toward_quota=False,
            unstable_attempt_policy="quarantine_and_replace",
        )
        if self != expected:
            raise QuotaConfigError(
                "measurement protocol differs from the frozen one-run five-epoch contract"
            )


@dataclass(frozen=True)
class QuotaPlan:
    version: str
    target_hardware_id: str
    maximum_cloud_working_set_gib: int
    cloud_free_space_safety_margin_gib: int
    protocol: MeasurementProtocol
    accepted_successful_configurations: int
    accepted_runs_per_configuration: int
    configured_expected_accepted_run_records: int
    configured_expected_measured_epoch_records: int
    cells: tuple[QuotaCell, ...]
    expected_modality_totals: Mapping[str, int]
    generated_lineage_minimum: int
    generated_lineage_maximum_accepted: int
    generated_lineages_held_out_minimum: int

    @property
    def expected_accepted_run_records(self) -> int:
        return self.accepted_successful_configurations * self.accepted_runs_per_configuration

    @property
    def expected_measured_epoch_records(self) -> int:
        return self.expected_accepted_run_records * self.protocol.measured_epoch_records_per_run

    @property
    def family_totals(self) -> dict[str, int]:
        return {cell.family_id: cell.accepted_configurations for cell in self.cells}

    @property
    def modality_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for cell in self.cells:
            totals[cell.modality] = totals.get(cell.modality, 0) + cell.accepted_configurations
        return dict(sorted(totals.items()))

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(self)

    def validate(self) -> None:
        if self.version != PROFILE.quota_version:
            raise QuotaConfigError(f"unsupported quota plan version {self.version!r}")
        if self.target_hardware_id != PROFILE.target_hardware_id:
            raise QuotaConfigError("quota plan target differs from the active profile")
        if self.maximum_cloud_working_set_gib != 600:
            raise QuotaConfigError("maximum cloud working set must be exactly 600 GiB")
        if self.cloud_free_space_safety_margin_gib != 40:
            raise QuotaConfigError("cloud free-space safety margin must be exactly 40 GiB")
        self.protocol.validate()
        expected_candidates = 11_200 if PROFILE.is_nonvision_4gpu else 18_000
        expected_epochs = expected_candidates * 3
        expected_cells = (
            NONVISION_QUOTA_CELLS if PROFILE.is_nonvision_4gpu else FROZEN_QUOTA_CELLS
        )
        expected_modalities = (
            NONVISION_MODALITY_TOTALS
            if PROFILE.is_nonvision_4gpu
            else FROZEN_MODALITY_TOTALS
        )
        if self.accepted_successful_configurations != expected_candidates:
            raise QuotaConfigError(
                f"accepted configuration total must be exactly {expected_candidates:,}"
            )
        if self.accepted_runs_per_configuration != 1:
            raise QuotaConfigError("every accepted configuration must have exactly one run")
        if self.configured_expected_accepted_run_records != self.expected_accepted_run_records:
            raise QuotaConfigError("configured accepted run count differs")
        if self.configured_expected_measured_epoch_records != self.expected_measured_epoch_records:
            raise QuotaConfigError("configured epoch record count differs")
        if self.configured_expected_measured_epoch_records != expected_epochs:
            raise QuotaConfigError("configured measured epoch total differs")
        if tuple(cell.as_frozen_tuple() for cell in self.cells) != expected_cells:
            raise QuotaConfigError("family quota cells differ from the active frozen table")
        if dict(self.expected_modality_totals) != expected_modalities:
            raise QuotaConfigError("expected modality totals differ from the frozen table")
        if self.modality_totals != expected_modalities:
            raise QuotaConfigError("computed modality totals differ from the frozen table")
        minimum_lineages = 40 if PROFILE.is_nonvision_4gpu else 50
        if self.generated_lineage_minimum < minimum_lineages:
            raise QuotaConfigError(
                f"at least {minimum_lineages} independent generated lineages are required"
            )
        if not 1 <= self.generated_lineage_maximum_accepted <= 150:
            raise QuotaConfigError("generated lineage maximum must be positive and at most 150")
        minimum_held_out = 8 if PROFILE.is_nonvision_4gpu else 10
        if self.generated_lineages_held_out_minimum < minimum_held_out:
            raise QuotaConfigError(
                f"at least {minimum_held_out} generated lineages must be held out"
            )


_ROOT_KEYS = {
    "version",
    "target_hardware_id",
    "maximum_cloud_working_set_gib",
    "cloud_free_space_safety_margin_gib",
    "protocol",
    "end_to_end",
    "expected_modality_totals",
    "generated_lineages",
}
_PROTOCOL_KEYS = {
    "workers",
    "workers_per_qualified_physical_gpu",
    "concurrent_profilers_per_gpu",
    "allow_mps",
    "allow_multi_job_gpu_packing",
    "require_fresh_process_per_run",
    "require_memory_release_before_next_run",
    "total_epochs_per_run",
    "warmup_epochs",
    "measured_epochs",
    "maximum_epoch_time_relative_spread",
    "end_to_end_time_label",
    "derive_end_to_end_epoch_time_from_step_time",
    "failed_attempts_count_toward_quota",
    "oom_attempts_count_toward_quota",
    "unstable_attempt_policy",
}


def _parse_protocol(value: Any) -> MeasurementProtocol:
    raw = _mapping(value, context="protocol")
    _exact_keys(raw, _PROTOCOL_KEYS, context="protocol")
    spread = raw["maximum_epoch_time_relative_spread"]
    if isinstance(spread, bool) or not isinstance(spread, (int, float)):
        raise QuotaConfigError("protocol.maximum_epoch_time_relative_spread must be numeric")
    return MeasurementProtocol(
        workers=_string(raw["workers"], context="protocol.workers"),
        workers_per_qualified_physical_gpu=_integer(raw["workers_per_qualified_physical_gpu"], context="protocol.workers_per_qualified_physical_gpu"),
        concurrent_profilers_per_gpu=_integer(raw["concurrent_profilers_per_gpu"], context="protocol.concurrent_profilers_per_gpu"),
        allow_mps=_boolean(raw["allow_mps"], context="protocol.allow_mps"),
        allow_multi_job_gpu_packing=_boolean(raw["allow_multi_job_gpu_packing"], context="protocol.allow_multi_job_gpu_packing"),
        require_fresh_process_per_run=_boolean(raw["require_fresh_process_per_run"], context="protocol.require_fresh_process_per_run"),
        require_memory_release_before_next_run=_boolean(raw["require_memory_release_before_next_run"], context="protocol.require_memory_release_before_next_run"),
        total_epochs_per_run=_integer(raw["total_epochs_per_run"], context="protocol.total_epochs_per_run"),
        warmup_epochs=_integer_tuple(raw["warmup_epochs"], context="protocol.warmup_epochs"),
        measured_epochs=_integer_tuple(raw["measured_epochs"], context="protocol.measured_epochs"),
        maximum_epoch_time_relative_spread=float(spread),
        end_to_end_time_label=_string(raw["end_to_end_time_label"], context="protocol.end_to_end_time_label"),
        derive_end_to_end_epoch_time_from_step_time=_boolean(raw["derive_end_to_end_epoch_time_from_step_time"], context="protocol.derive_end_to_end_epoch_time_from_step_time"),
        failed_attempts_count_toward_quota=_boolean(raw["failed_attempts_count_toward_quota"], context="protocol.failed_attempts_count_toward_quota"),
        oom_attempts_count_toward_quota=_boolean(raw["oom_attempts_count_toward_quota"], context="protocol.oom_attempts_count_toward_quota"),
        unstable_attempt_policy=_string(raw["unstable_attempt_policy"], context="protocol.unstable_attempt_policy"),
    )


def load_quota_plan(path: str | Path = DEFAULT_QUOTA_CONFIG_PATH) -> QuotaPlan:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _mapping(raw, context="quota plan root")
    _exact_keys(root, _ROOT_KEYS, context="quota plan root")
    if PROFILE.name != "v100" and Path(path).resolve() == DEFAULT_QUOTA_CONFIG_PATH.resolve():
        root = {**root, "version": PROFILE.quota_version, "target_hardware_id": PROFILE.target_hardware_id}
    end_to_end = _mapping(root["end_to_end"], context="end_to_end")
    _exact_keys(
        end_to_end,
        {
            "accepted_successful_configurations",
            "accepted_runs_per_configuration",
            "expected_accepted_run_records",
            "expected_measured_epoch_records",
            "family_quotas",
        },
        context="end_to_end",
    )
    raw_cells = end_to_end["family_quotas"]
    if not isinstance(raw_cells, list):
        raise QuotaConfigError("end_to_end.family_quotas must be a list")
    cells: list[QuotaCell] = []
    for index, value in enumerate(raw_cells):
        item = _mapping(value, context=f"quota cell {index}")
        _exact_keys(
            item,
            {"modality", "family_id", "display_name", "accepted_configurations"},
            context=f"quota cell {index}",
        )
        cells.append(
            QuotaCell(
                modality=_string(item["modality"], context=f"quota cell {index}.modality"),
                family_id=_string(item["family_id"], context=f"quota cell {index}.family_id"),
                display_name=_string(item["display_name"], context=f"quota cell {index}.display_name"),
                accepted_configurations=_integer(item["accepted_configurations"], context=f"quota cell {index}.accepted_configurations"),
            )
        )
    expected = _mapping(root["expected_modality_totals"], context="expected_modality_totals")
    expected_modality_keys = (
        set(NONVISION_MODALITY_TOTALS)
        if PROFILE.is_nonvision_4gpu
        else set(FROZEN_MODALITY_TOTALS)
    )
    _exact_keys(expected, expected_modality_keys, context="expected_modality_totals")
    generated = _mapping(root["generated_lineages"], context="generated_lineages")
    _exact_keys(
        generated,
        {"minimum_independent_lineages", "maximum_accepted_per_lineage", "minimum_held_out_lineages"},
        context="generated_lineages",
    )
    plan = QuotaPlan(
        version=_string(root["version"], context="version"),
        target_hardware_id=_string(root["target_hardware_id"], context="target_hardware_id"),
        maximum_cloud_working_set_gib=_integer(root["maximum_cloud_working_set_gib"], context="maximum_cloud_working_set_gib"),
        cloud_free_space_safety_margin_gib=_integer(root["cloud_free_space_safety_margin_gib"], context="cloud_free_space_safety_margin_gib"),
        protocol=_parse_protocol(root["protocol"]),
        accepted_successful_configurations=_integer(end_to_end["accepted_successful_configurations"], context="end_to_end.accepted_successful_configurations"),
        accepted_runs_per_configuration=_integer(end_to_end["accepted_runs_per_configuration"], context="end_to_end.accepted_runs_per_configuration"),
        configured_expected_accepted_run_records=_integer(end_to_end["expected_accepted_run_records"], context="end_to_end.expected_accepted_run_records"),
        configured_expected_measured_epoch_records=_integer(end_to_end["expected_measured_epoch_records"], context="end_to_end.expected_measured_epoch_records"),
        cells=tuple(cells),
        expected_modality_totals={key: _integer(item, context=f"expected_modality_totals.{key}") for key, item in expected.items()},
        generated_lineage_minimum=_integer(generated["minimum_independent_lineages"], context="generated_lineages.minimum_independent_lineages"),
        generated_lineage_maximum_accepted=_integer(generated["maximum_accepted_per_lineage"], context="generated_lineages.maximum_accepted_per_lineage"),
        generated_lineages_held_out_minimum=_integer(generated["minimum_held_out_lineages"], context="generated_lineages.minimum_held_out_lineages"),
    )
    plan.validate()
    return plan


__all__ = [
    "DEFAULT_QUOTA_CONFIG_PATH",
    "FROZEN_MODALITY_TOTALS",
    "FROZEN_QUOTA_CELLS",
    "NONVISION_MODALITY_TOTALS",
    "NONVISION_QUOTA_CELLS",
    "MeasurementProtocol",
    "QuotaCell",
    "QuotaConfigError",
    "QuotaPlan",
    "load_quota_plan",
]
