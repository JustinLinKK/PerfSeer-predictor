"""Strict declarative contract for the non-vision Disaster Tweets substitution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fingerprints import canonical_sha256, canonical_value


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "registries"
    / "native_a10_nonvision_disaster_v2_substitution.yaml"
)
CONTRACT_VERSION = (
    "perfseer_v3_native_a10_nonvision_disaster_substitution_contract_v2"
)
HISTORICAL_TASK_ID = "detecting-insults"
DISASTER_TASK_ID = "disaster-tweets"
DISASTER_KAGGLE_SLUG = "nlp-getting-started"
EXPECTED_INVENTORY = (
    ("sample_submission.csv", 22_746),
    ("test.csv", 420_783),
    ("train.csv", 987_712),
)
EXPECTED_INVENTORY_SHA256 = (
    "c1d9ad5c3af3130823c5687d033a8ce4b3cd9961c859b8e17d6f39cc2302e847"
)


class DisasterSubstitutionError(ValueError):
    """Raised when the V2 substitution declaration drifts."""


@dataclass(frozen=True)
class DisasterSubstitutionContract:
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload)

    def validate(self) -> None:
        root = self.payload
        if set(root) != {
            "version",
            "old_task",
            "new_task",
            "affected_families",
            "affected_precision_counts",
            "affected_effective_batch_counts",
            "lineage_expectations",
        } or root["version"] != CONTRACT_VERSION:
            raise DisasterSubstitutionError("substitution root contract differs")
        if root["old_task"] != {
            "historical_ordinal": 11,
            "nonvision_ordinal": 1,
            "task_id": HISTORICAL_TASK_ID,
            "kaggle_slug": "detecting-insults-in-social-commentary",
            "modality": "nlp",
            "compressed_size_hint_bytes": 2_000_000,
        }:
            raise DisasterSubstitutionError("historical insults declaration differs")
        expected_new = {
            "historical_ordinal": 11,
            "nonvision_ordinal": 1,
            "task_id": DISASTER_TASK_ID,
            "kaggle_slug": DISASTER_KAGGLE_SLUG,
            "modality": "nlp",
            "compressed_size_hint_bytes": 1_431_241,
            "source_inventory": [
                {"name": name, "size_bytes": size}
                for name, size in EXPECTED_INVENTORY
            ],
            "source_inventory_sha256": EXPECTED_INVENTORY_SHA256,
            "source_row_count": 7_613,
            "columns": ["id", "keyword", "location", "text", "target"],
            "input_columns": ["keyword", "location", "text"],
            "target_values": [0, 1],
            "prepared_example_count": 4_096,
            "selection": "sha256_order_then_deterministic_cycle_v1",
        }
        if canonical_value(root["new_task"]) != canonical_value(expected_new):
            raise DisasterSubstitutionError("Disaster Tweets declaration differs")
        if root["affected_families"] != {
            "bert_base": 275,
            "bilstm_crf": 150,
            "distilbert_distillation": 275,
            "fasttext_embeddingbag": 125,
            "mla_mini_transformer": 250,
        }:
            raise DisasterSubstitutionError("affected family allocation differs")
        if root["affected_precision_counts"] != {
            "fp32_tf32": 214,
            "bf16": 321,
            "fp16_grad_scaler": 324,
            "mixed_structured": 216,
        }:
            raise DisasterSubstitutionError("affected precision allocation differs")
        if root["affected_effective_batch_counts"] != {
            "32": 215,
            "64": 215,
            "128": 215,
            "256": 215,
            "512": 215,
        }:
            raise DisasterSubstitutionError("affected batch allocation differs")
        if root["lineage_expectations"] != {
            "unchanged": 9_050,
            "nlp_registry_rebound": 1_075,
            "dataset_substitution": 1_075,
        }:
            raise DisasterSubstitutionError("lineage allocation differs")


def load_disaster_substitution_contract(
    path: str | Path = CONTRACT_PATH,
) -> DisasterSubstitutionContract:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise DisasterSubstitutionError("substitution contract is unreadable") from error
    if not isinstance(payload, Mapping):
        raise DisasterSubstitutionError("substitution contract must be a mapping")
    contract = DisasterSubstitutionContract(canonical_value(payload))
    contract.validate()
    return contract


__all__ = [
    "CONTRACT_PATH",
    "CONTRACT_VERSION",
    "DISASTER_KAGGLE_SLUG",
    "DISASTER_TASK_ID",
    "EXPECTED_INVENTORY",
    "EXPECTED_INVENTORY_SHA256",
    "HISTORICAL_TASK_ID",
    "DisasterSubstitutionContract",
    "DisasterSubstitutionError",
    "load_disaster_substitution_contract",
]
