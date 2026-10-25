"""Versioned Kaggle task and executable-schema registry pinned to MLE-bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import TASK_REGISTRY_VERSION, TaskRegistryEntry
from .fingerprints import canonical_sha256, canonical_value
from .labeler_profile import PROFILE
from .speech_substitution import (
    HISTORICAL_WHALE_TASK_ID,
    SPEECH_KAGGLE_SLUG,
    SPEECH_TASK_ID,
    load_speech_substitution_contract,
)
from .disaster_substitution import (
    DISASTER_KAGGLE_SLUG,
    DISASTER_TASK_ID,
    HISTORICAL_TASK_ID as HISTORICAL_INSULTS_TASK_ID,
    load_disaster_substitution_contract,
)


DEFAULT_TASK_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "registries" / "v100_task_registry.yaml"
)
FROZEN_TASKS = (
    ("histopathologic-cancer", "histopathologic-cancer-detection", "vision"),
    ("dogs-vs-cats", "dogs-vs-cats-redux-kernels-edition", "vision"),
    ("dog-breed", "dog-breed-identification", "vision"),
    ("siim-isic-melanoma", "siim-isic-melanoma-classification", "vision"),
    ("aptos2019", "aptos2019-blindness-detection", "vision"),
    ("aerial-cactus", "aerial-cactus-identification", "vision"),
    ("plant-pathology", "plant-pathology-2020-fgvc7", "vision"),
    ("ranzcr-clip", "ranzcr-clip-catheter-line-classification", "vision"),
    ("leaf-classification", "leaf-classification", "vision"),
    ("denoising-dirty-documents", "denoising-dirty-documents", "vision"),
    ("jigsaw-toxic", "jigsaw-toxic-comment-classification-challenge", "nlp"),
    ("detecting-insults", "detecting-insults-in-social-commentary", "nlp"),
    ("spooky-author", "spooky-author-identification", "nlp"),
    ("random-acts-of-pizza", "random-acts-of-pizza", "nlp"),
    (
        "text-normalization-english",
        "text-normalization-challenge-english-language",
        "nlp",
    ),
    (
        "text-normalization-russian",
        "text-normalization-challenge-russian-language",
        "nlp",
    ),
    ("mlsp-2013-birds", "mlsp-2013-birds", "audio"),
    (
        "icml-2013-whale",
        "the-icml-2013-whale-challenge-right-whale-redux",
        "audio",
    ),
    ("nyc-taxi-fare", "new-york-city-taxi-fare-prediction", "tabular"),
    ("nomad2018", "nomad2018-predict-transparent-conductors", "graph"),
    (
        "tabular-playground-dec-2021",
        "tabular-playground-series-dec-2021",
        "tabular",
    ),
    (
        "tabular-playground-may-2022",
        "tabular-playground-series-may-2022",
        "tabular",
    ),
)
SPEECH_V2_TASKS = tuple(
    (SPEECH_TASK_ID, SPEECH_KAGGLE_SLUG, modality)
    if task_id == HISTORICAL_WHALE_TASK_ID
    else (task_id, slug, modality)
    for task_id, slug, modality in FROZEN_TASKS
)
NONVISION_TASKS = tuple(row for row in SPEECH_V2_TASKS if row[2] != "vision")
DISASTER_V2_TASKS = tuple(
    (DISASTER_TASK_ID, DISASTER_KAGGLE_SLUG, modality)
    if task_id == HISTORICAL_INSULTS_TASK_ID
    else (task_id, slug, modality)
    for task_id, slug, modality in NONVISION_TASKS
)
FROZEN_COMPRESSED_SIZE_HINTS = (
    7_760_000_000,
    850_000_000,
    750_000_000,
    116_160_000_000,
    10_220_000_000,
    25_400_000,
    800_000_000,
    13_130_000_000,
    36_000_000,
    60_000_000,
    60_000_000,
    2_000_000,
    1_900_000,
    3_000_000,
    10_000_000,
    10_000_000,
    585_100_000,
    293_140_000,
    5_700_000_000,
    6_240_000,
    700_000_000,
    570_000_000,
)
SPEECH_V2_COMPRESSED_SIZE_HINTS = tuple(
    3_762_295_706 if index == 17 else value
    for index, value in enumerate(FROZEN_COMPRESSED_SIZE_HINTS)
)
NONVISION_COMPRESSED_SIZE_HINTS = tuple(
    size
    for (_, _, modality), size in zip(
        SPEECH_V2_TASKS, SPEECH_V2_COMPRESSED_SIZE_HINTS, strict=True
    )
    if modality != "vision"
)
DISASTER_V2_COMPRESSED_SIZE_HINTS = tuple(
    1_431_241 if index == 1 else value
    for index, value in enumerate(NONVISION_COMPRESSED_SIZE_HINTS)
)
MLEBENCH_METADATA_REVISION = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
TASK_SCHEMA_VERSION = "perfseer_v3_task_schema_v1"
FROZEN_TASK_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "histopathologic-cancer": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "dogs-vs-cats": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "dog-breed": {"kind": "single_label_classification", "target_width": 120, "target_encoding": "categorical_index"},
    "siim-isic-melanoma": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "aptos2019": {"kind": "single_label_classification", "target_width": 5, "target_encoding": "categorical_index"},
    "aerial-cactus": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "plant-pathology": {"kind": "multilabel_classification", "target_width": 4, "target_encoding": "binary_vector"},
    "ranzcr-clip": {"kind": "multilabel_classification", "target_width": 9, "target_encoding": "binary_vector"},
    "leaf-classification": {"kind": "single_label_classification", "target_width": 99, "target_encoding": "categorical_index"},
    "denoising-dirty-documents": {"kind": "image_restoration", "target_width": 3, "target_encoding": "paired_image"},
    "jigsaw-toxic": {"kind": "multilabel_classification", "target_width": 6, "target_encoding": "binary_vector"},
    "detecting-insults": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "spooky-author": {"kind": "single_label_classification", "target_width": 3, "target_encoding": "categorical_index"},
    "random-acts-of-pizza": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "text-normalization-english": {"kind": "teacher_forced_seq2seq", "target_width": 32, "target_encoding": "stable_token_bucket_32_v1"},
    "text-normalization-russian": {"kind": "teacher_forced_seq2seq", "target_width": 32, "target_encoding": "stable_token_bucket_32_v1"},
    "mlsp-2013-birds": {"kind": "multilabel_classification", "target_width": 19, "target_encoding": "binary_vector"},
    "icml-2013-whale": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
    "nyc-taxi-fare": {"kind": "regression", "target_width": 1, "target_encoding": "float_vector"},
    "nomad2018": {"kind": "graph_regression", "target_width": 2, "target_encoding": "float_vector"},
    "tabular-playground-dec-2021": {"kind": "single_label_classification", "target_width": 7, "target_encoding": "categorical_index"},
    "tabular-playground-may-2022": {"kind": "single_label_classification", "target_width": 2, "target_encoding": "categorical_index"},
}
SPEECH_V2_TASK_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    **{
        task_id: schema
        for task_id, schema in FROZEN_TASK_SCHEMAS.items()
        if task_id != HISTORICAL_WHALE_TASK_ID
    },
    SPEECH_TASK_ID: {
        "kind": "single_label_classification",
        "target_width": 2,
        "target_encoding": "categorical_index",
    },
}
NONVISION_TASK_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    task_id: SPEECH_V2_TASK_SCHEMAS[task_id]
    for task_id, _, _ in NONVISION_TASKS
}
DISASTER_V2_TASK_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    **{
        task_id: schema
        for task_id, schema in NONVISION_TASK_SCHEMAS.items()
        if task_id != HISTORICAL_INSULTS_TASK_ID
    },
    DISASTER_TASK_ID: {
        "kind": "single_label_classification",
        "target_width": 2,
        "target_encoding": "categorical_index",
    },
}
MLEBENCH_SIZE_HINT_SOURCE_URL = (
    "https://github.com/openai/mle-bench/blob/"
    f"{MLEBENCH_METADATA_REVISION}/README.md#lite-evaluation"
)
_ROOT_KEYS = {
    "version",
    "target_hardware_id",
    "training_approved",
    "metadata_source_url",
    "metadata_revision",
    "size_hint_source_url",
    "archive_checksum_state",
    "expected_train_examples_per_task",
    "entries",
}
_SOURCE_ENTRY_KEYS = {
    "task_id",
    "kaggle_slug",
    "modality",
    "compressed_size_hint_bytes",
}
_SERIALIZED_ENTRY_KEYS = set(TaskRegistryEntry.__dataclass_fields__)
_SERIALIZED_REGISTRY_KEYS = {
    "version",
    "target_hardware_id",
    "training_approved",
    "metadata_source_url",
    "metadata_revision",
    "size_hint_source_url",
    "archive_checksum_state",
    "entries",
    "registry_sha256",
}


class TaskRegistryError(ValueError):
    """Raised when task metadata is missing, guessed, stale, or malformed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise TaskRegistryError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TaskRegistryError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise TaskRegistryError(
            f"{context} schema differs; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value:
        raise TaskRegistryError(f"{context} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(type(item) is not str or not item for item in value):
        raise TaskRegistryError(f"{context} must be a string sequence")
    return tuple(value)


def task_entry_from_dict(value: Mapping[str, Any]) -> TaskRegistryEntry:
    raw = _mapping(value, context="serialized task entry")
    _exact_keys(raw, _SERIALIZED_ENTRY_KEYS, context="serialized task entry")
    mappings = {
        name: _mapping(raw[name], context=f"task entry {name}")
        for name in (
            "source_checksums",
            "split_recipe",
            "target_schema",
            "validation_rules",
            "prepared_view_recipe",
            "cache_policy",
            "eviction_policy",
        )
    }
    entry = TaskRegistryEntry(
        registry_version=_text(raw["registry_version"], context="registry_version"),
        task_id=_text(raw["task_id"], context="task_id"),
        kaggle_kind=_text(raw["kaggle_kind"], context="kaggle_kind"),
        kaggle_slug=_text(raw["kaggle_slug"], context="kaggle_slug"),
        modality=_text(raw["modality"], context="modality"),
        license_or_rules_url=_text(raw["license_or_rules_url"], context="license_or_rules_url"),
        manual_acceptance_required=raw["manual_acceptance_required"],
        expected_archive_files=_string_tuple(
            raw["expected_archive_files"], context="expected_archive_files"
        ),
        required_file_patterns=_string_tuple(
            raw["required_file_patterns"], context="required_file_patterns"
        ),
        optional_file_patterns=_string_tuple(
            raw["optional_file_patterns"], context="optional_file_patterns"
        ),
        compressed_size_hint_bytes=raw["compressed_size_hint_bytes"],
        maximum_extracted_bytes=raw["maximum_extracted_bytes"],
        expected_train_examples=raw["expected_train_examples"],
        dataset_revision=_text(raw["dataset_revision"], context="dataset_revision"),
        **mappings,
    )
    entry.validate()
    return entry


@dataclass(frozen=True)
class TaskRegistry:
    version: str
    target_hardware_id: str
    training_approved: bool
    metadata_source_url: str
    metadata_revision: str
    size_hint_source_url: str
    archive_checksum_state: str
    entries: tuple[TaskRegistryEntry, ...]

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_value(asdict(self))
        payload["registry_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRegistry":
        raw = _mapping(value, context="serialized task registry")
        _exact_keys(raw, _SERIALIZED_REGISTRY_KEYS, context="serialized task registry")
        entries = raw["entries"]
        if not isinstance(entries, (list, tuple)):
            raise TaskRegistryError("serialized task entries must be a sequence")
        registry = cls(
            version=_text(raw["version"], context="version"),
            target_hardware_id=_text(raw["target_hardware_id"], context="target_hardware_id"),
            training_approved=raw["training_approved"],
            metadata_source_url=_text(raw["metadata_source_url"], context="metadata_source_url"),
            metadata_revision=_text(raw["metadata_revision"], context="metadata_revision"),
            size_hint_source_url=_text(raw["size_hint_source_url"], context="size_hint_source_url"),
            archive_checksum_state=_text(
                raw["archive_checksum_state"], context="archive_checksum_state"
            ),
            entries=tuple(task_entry_from_dict(entry) for entry in entries),
        )
        registry.validate()
        if raw["registry_sha256"] != registry.sha256:
            raise TaskRegistryError("serialized task registry hash mismatch")
        return registry

    def validate(self) -> None:
        if self.version != TASK_REGISTRY_VERSION:
            raise TaskRegistryError("task registry version mismatch")
        if self.target_hardware_id != PROFILE.target_hardware_id:
            raise TaskRegistryError("task registry target differs from the active profile")
        if type(self.training_approved) is not bool or self.training_approved:
            raise TaskRegistryError("local task registry must remain explicitly unapproved")
        if self.metadata_source_url != "https://github.com/openai/mle-bench":
            raise TaskRegistryError("task identifiers must be sourced from the pinned MLE-bench repository")
        if self.metadata_revision != MLEBENCH_METADATA_REVISION:
            raise TaskRegistryError("MLE-bench metadata revision differs from the audited commit")
        if self.size_hint_source_url != MLEBENCH_SIZE_HINT_SOURCE_URL:
            raise TaskRegistryError("size-hint source URL must be the pinned MLE-bench Lite table")
        if self.archive_checksum_state != "materialization_required":
            raise TaskRegistryError("archive checksums must be resolved during materialization")
        speech_v2 = PROFILE.uses_speech_v2
        expected_tasks = (
            DISASTER_V2_TASKS
            if PROFILE.uses_disaster_v2
            else NONVISION_TASKS
            if PROFILE.is_nonvision_4gpu
            else SPEECH_V2_TASKS
            if speech_v2
            else FROZEN_TASKS
        )
        expected_sizes = (
            DISASTER_V2_COMPRESSED_SIZE_HINTS
            if PROFILE.uses_disaster_v2
            else NONVISION_COMPRESSED_SIZE_HINTS
            if PROFILE.is_nonvision_4gpu
            else SPEECH_V2_COMPRESSED_SIZE_HINTS
            if speech_v2
            else FROZEN_COMPRESSED_SIZE_HINTS
        )
        expected_schemas = (
            DISASTER_V2_TASK_SCHEMAS
            if PROFILE.uses_disaster_v2
            else NONVISION_TASK_SCHEMAS
            if PROFILE.is_nonvision_4gpu
            else SPEECH_V2_TASK_SCHEMAS
            if speech_v2
            else FROZEN_TASK_SCHEMAS
        )
        substitution = load_speech_substitution_contract() if speech_v2 else None
        disaster = (
            load_disaster_substitution_contract()
            if PROFILE.uses_disaster_v2
            else None
        )
        identities = tuple((entry.task_id, entry.kaggle_slug, entry.modality) for entry in self.entries)
        if identities != expected_tasks:
            raise TaskRegistryError("task IDs/slugs/modalities differ from the frozen MLE-bench mapping")
        if tuple(entry.compressed_size_hint_bytes for entry in self.entries) != expected_sizes:
            raise TaskRegistryError("task size hints differ from the pinned MLE-bench Lite table")
        if len({entry.task_id for entry in self.entries}) != len(self.entries):
            raise TaskRegistryError("task IDs must be unique")
        if len({entry.kaggle_slug for entry in self.entries}) != len(self.entries):
            raise TaskRegistryError("Kaggle slugs must be unique")
        for entry in self.entries:
            entry.validate()
            if entry.registry_version != self.version or entry.kaggle_kind != "competition":
                raise TaskRegistryError("task entry registry/kind mismatch")
            if entry.license_or_rules_url != f"https://www.kaggle.com/competitions/{entry.kaggle_slug}/rules":
                raise TaskRegistryError("task rules URL does not match its canonical competition slug")
            if entry.manual_acceptance_required is not True:
                raise TaskRegistryError("Kaggle competition rules require explicit acceptance preflight")
            if entry.expected_archive_files != (f"{entry.kaggle_slug}.zip",):
                raise TaskRegistryError("expected Kaggle archive name drifted")
            expected_revision = (
                f"kaggle:{entry.kaggle_slug}:unmaterialized@mlebench:{self.metadata_revision}"
                f"@task_schema:{TASK_SCHEMA_VERSION}"
            )
            if speech_v2 and entry.task_id == SPEECH_TASK_ID:
                assert substitution is not None
                expected_revision += (
                    "@prepared_view:binary_yes_no_v2"
                    f"@substitution:{substitution.sha256}"
                )
            if PROFILE.uses_disaster_v2 and entry.task_id == DISASTER_TASK_ID:
                assert disaster is not None
                expected_revision += (
                    "@prepared_view:disaster_tweets_csv_v1"
                    f"@substitution:{disaster.sha256}"
                )
            if entry.dataset_revision != expected_revision:
                raise TaskRegistryError("task dataset revision is not pinned to its source metadata")
            if entry.source_checksums:
                raise TaskRegistryError("local source registry must not fabricate archive checksums")
            if entry.validation_rules.get("archive_checksum_state") != self.archive_checksum_state:
                raise TaskRegistryError("task entry must fail closed until archive materialization")
            expected_schema = {
                "schema_version": TASK_SCHEMA_VERSION,
                **expected_schemas[entry.task_id],
            }
            if entry.target_schema != expected_schema:
                raise TaskRegistryError("task executable target schema drifted")
            if entry.maximum_extracted_bytes != entry.compressed_size_hint_bytes * 4:
                raise TaskRegistryError("extracted-size admission ceiling must use the frozen 4x bound")
            if entry.expected_train_examples != 4_096:
                raise TaskRegistryError("prepared epoch cardinality must be the frozen 4,096 examples")
            expected_view_recipe: Mapping[str, Any] = {
                "kind": "pinned_mlebench_preparer",
                "atomic_publish": True,
                "expected_train_examples": entry.expected_train_examples,
                "require_exact_prepared_count": True,
            }
            if speech_v2 and entry.task_id == SPEECH_TASK_ID:
                assert substitution is not None
                expected_view_recipe = {
                    "kind": "perfseer_binary_speech_view_v2",
                    "atomic_publish": True,
                    "expected_train_examples": entry.expected_train_examples,
                    "require_exact_prepared_count": True,
                    "classes": {"no": 0, "yes": 1},
                    "valid_wavs_per_class": 2_048,
                    "sample_rate_hz": 16_000,
                    "selection": "sha256_relative_path_then_relative_path_v1",
                    "ignore_mlebench_test_split": True,
                    "substitution_contract_sha256": substitution.sha256,
                }
            if PROFILE.uses_disaster_v2 and entry.task_id == DISASTER_TASK_ID:
                assert disaster is not None
                expected_view_recipe = {
                    "kind": "perfseer_disaster_tweets_csv_v1",
                    "atomic_publish": True,
                    "expected_train_examples": entry.expected_train_examples,
                    "require_exact_prepared_count": True,
                    "columns": ["id", "keyword", "location", "text", "target"],
                    "input_columns": ["keyword", "location", "text"],
                    "target_values": [0, 1],
                    "source_row_count": 7_613,
                    "selection": "sha256_order_then_deterministic_cycle_v1",
                    "substitution_contract_sha256": disaster.sha256,
                }
            if entry.prepared_view_recipe != expected_view_recipe:
                raise TaskRegistryError("prepared-view exact-count contract drifted")
            if entry.cache_policy != {"kind": "single_current_task"} or entry.eviction_policy != {
                "delete_after_task_complete": True
            }:
                raise TaskRegistryError("task cache/eviction policy must remain sequential and simple")


def load_task_registry(path: str | Path = DEFAULT_TASK_REGISTRY_PATH) -> TaskRegistry:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _mapping(raw, context="task registry root")
    _exact_keys(root, _ROOT_KEYS, context="task registry root")
    default_source = Path(path).resolve() == DEFAULT_TASK_REGISTRY_PATH.resolve()
    if PROFILE.name != "v100" and default_source:
        root = {**root, "version": PROFILE.task_registry_version, "target_hardware_id": PROFILE.target_hardware_id}
    if PROFILE.uses_speech_v2 and default_source:
        substitution = load_speech_substitution_contract()
        source_entries = list(root["entries"])
        ordinal = int(substitution.payload["new_task"]["ordinal"])
        expected_old = {
            key: substitution.payload["old_task"][key]
            for key in _SOURCE_ENTRY_KEYS
        }
        if canonical_value(source_entries[ordinal]) != canonical_value(expected_old):
            raise TaskRegistryError("V1 task at the substitution ordinal drifted")
        source_entries[ordinal] = {
            key: substitution.payload["new_task"][key]
            for key in _SOURCE_ENTRY_KEYS
        }
        root = {**root, "entries": source_entries}
    if PROFILE.is_nonvision_4gpu and default_source:
        root = {
            **root,
            "entries": [
                entry for entry in root["entries"] if entry.get("modality") != "vision"
            ],
        }
    if PROFILE.uses_disaster_v2 and default_source:
        disaster = load_disaster_substitution_contract()
        source_entries = list(root["entries"])
        ordinal = int(disaster.payload["new_task"]["nonvision_ordinal"])
        expected_old = {
            "task_id": HISTORICAL_INSULTS_TASK_ID,
            "kaggle_slug": "detecting-insults-in-social-commentary",
            "modality": "nlp",
            "compressed_size_hint_bytes": 2_000_000,
        }
        if canonical_value(source_entries[ordinal]) != canonical_value(expected_old):
            raise TaskRegistryError("non-vision V1 insults task drifted")
        source_entries[ordinal] = {
            key: disaster.payload["new_task"][key]
            for key in _SOURCE_ENTRY_KEYS
        }
        root = {**root, "entries": source_entries}
    if root["version"] != TASK_REGISTRY_VERSION:
        raise TaskRegistryError("task registry source version mismatch")
    entries = root["entries"]
    if not isinstance(entries, list):
        raise TaskRegistryError("task registry entries must be a list")
    metadata_revision = _text(root["metadata_revision"], context="metadata_revision")
    expected_train_examples = root["expected_train_examples_per_task"]
    if type(expected_train_examples) is not int or expected_train_examples != 4_096:
        raise TaskRegistryError("expected_train_examples_per_task must be exactly 4,096")
    expanded = []
    substitution = (
        load_speech_substitution_contract()
        if PROFILE.uses_speech_v2
        else None
    )
    for index, value in enumerate(entries):
        source = _mapping(value, context=f"task source entry {index}")
        _exact_keys(source, _SOURCE_ENTRY_KEYS, context=f"task source entry {index}")
        task_id = _text(source["task_id"], context="task_id")
        slug = _text(source["kaggle_slug"], context="kaggle_slug")
        modality = _text(source["modality"], context="modality")
        size = source["compressed_size_hint_bytes"]
        if type(size) is not int or size < 1:
            raise TaskRegistryError("compressed size hint must be a positive integer")
        dataset_revision = (
            f"kaggle:{slug}:unmaterialized@mlebench:{metadata_revision}"
            f"@task_schema:{TASK_SCHEMA_VERSION}"
        )
        split_recipe: Mapping[str, Any] = {
            "kind": "pinned_mlebench_preparer",
            "metadata_revision": metadata_revision,
            "seed": 0,
        }
        prepared_view_recipe: Mapping[str, Any] = {
            "kind": "pinned_mlebench_preparer",
            "atomic_publish": True,
            "expected_train_examples": expected_train_examples,
            "require_exact_prepared_count": True,
        }
        if PROFILE.uses_speech_v2 and task_id == SPEECH_TASK_ID:
            assert substitution is not None
            dataset_revision += (
                "@prepared_view:binary_yes_no_v2"
                f"@substitution:{substitution.sha256}"
            )
            split_recipe = {
                "kind": "pinned_mlebench_public_training_extraction_then_perfseer_binary_view",
                "metadata_revision": metadata_revision,
                "seed": 0,
                "ignore_mlebench_test_split": True,
                "substitution_contract_sha256": substitution.sha256,
            }
            prepared_view_recipe = {
                "kind": "perfseer_binary_speech_view_v2",
                "atomic_publish": True,
                "expected_train_examples": expected_train_examples,
                "require_exact_prepared_count": True,
                "classes": {"no": 0, "yes": 1},
                "valid_wavs_per_class": 2_048,
                "sample_rate_hz": 16_000,
                "selection": "sha256_relative_path_then_relative_path_v1",
                "ignore_mlebench_test_split": True,
                "substitution_contract_sha256": substitution.sha256,
            }
        if PROFILE.uses_disaster_v2 and task_id == DISASTER_TASK_ID:
            disaster = load_disaster_substitution_contract()
            dataset_revision += (
                "@prepared_view:disaster_tweets_csv_v1"
                f"@substitution:{disaster.sha256}"
            )
            split_recipe = {
                "kind": "perfseer_disaster_tweets_csv_v1",
                "seed": 0,
                "substitution_contract_sha256": disaster.sha256,
            }
            prepared_view_recipe = {
                "kind": "perfseer_disaster_tweets_csv_v1",
                "atomic_publish": True,
                "expected_train_examples": expected_train_examples,
                "require_exact_prepared_count": True,
                "columns": ["id", "keyword", "location", "text", "target"],
                "input_columns": ["keyword", "location", "text"],
                "target_values": [0, 1],
                "source_row_count": 7_613,
                "selection": "sha256_order_then_deterministic_cycle_v1",
                "substitution_contract_sha256": disaster.sha256,
            }
        schemas = (
            DISASTER_V2_TASK_SCHEMAS
            if PROFILE.uses_disaster_v2
            else NONVISION_TASK_SCHEMAS
            if PROFILE.is_nonvision_4gpu
            else SPEECH_V2_TASK_SCHEMAS
            if PROFILE.uses_speech_v2
            else FROZEN_TASK_SCHEMAS
        )
        expanded.append(
            TaskRegistryEntry(
                registry_version=TASK_REGISTRY_VERSION,
                task_id=task_id,
                kaggle_kind="competition",
                kaggle_slug=slug,
                modality=modality,
                license_or_rules_url=f"https://www.kaggle.com/competitions/{slug}/rules",
                manual_acceptance_required=True,
                expected_archive_files=(f"{slug}.zip",),
                required_file_patterns=("**/*",),
                optional_file_patterns=("**/test*", "**/sample_submission*"),
                compressed_size_hint_bytes=size,
                maximum_extracted_bytes=size * 4,
                expected_train_examples=expected_train_examples,
                dataset_revision=dataset_revision,
                source_checksums={},
                split_recipe=split_recipe,
                target_schema={
                    "schema_version": TASK_SCHEMA_VERSION,
                    **schemas[task_id],
                },
                validation_rules={
                    "archive_checksum_state": root["archive_checksum_state"],
                    "minimum_files": 1,
                    "reject_symlinks_escaping_root": True,
                },
                prepared_view_recipe=prepared_view_recipe,
                cache_policy={"kind": "single_current_task"},
                eviction_policy={
                    "delete_after_task_complete": True,
                },
            )
        )
    registry = TaskRegistry(
        version=_text(root["version"], context="version"),
        target_hardware_id=_text(root["target_hardware_id"], context="target_hardware_id"),
        training_approved=root["training_approved"],
        metadata_source_url=_text(root["metadata_source_url"], context="metadata_source_url"),
        metadata_revision=metadata_revision,
        size_hint_source_url=_text(root["size_hint_source_url"], context="size_hint_source_url"),
        archive_checksum_state=_text(
            root["archive_checksum_state"], context="archive_checksum_state"
        ),
        entries=tuple(expanded),
    )
    registry.validate()
    return registry


__all__ = [
    "DEFAULT_TASK_REGISTRY_PATH",
    "FROZEN_COMPRESSED_SIZE_HINTS",
    "FROZEN_TASK_SCHEMAS",
    "FROZEN_TASKS",
    "MLEBENCH_METADATA_REVISION",
    "MLEBENCH_SIZE_HINT_SOURCE_URL",
    "TASK_SCHEMA_VERSION",
    "SPEECH_V2_COMPRESSED_SIZE_HINTS",
    "SPEECH_V2_TASK_SCHEMAS",
    "SPEECH_V2_TASKS",
    "NONVISION_COMPRESSED_SIZE_HINTS",
    "NONVISION_TASK_SCHEMAS",
    "NONVISION_TASKS",
    "DISASTER_V2_COMPRESSED_SIZE_HINTS",
    "DISASTER_V2_TASK_SCHEMAS",
    "DISASTER_V2_TASKS",
    "TaskRegistry",
    "TaskRegistryError",
    "load_task_registry",
    "task_entry_from_dict",
]
