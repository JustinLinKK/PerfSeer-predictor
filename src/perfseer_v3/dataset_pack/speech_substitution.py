"""Strict declarative contract for the native-A10 speech substitution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fingerprints import canonical_sha256, canonical_value


SUBSTITUTION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "registries"
    / "native_a10_speech_v2_substitution.yaml"
)
SUBSTITUTION_CONTRACT_VERSION = (
    "perfseer_v3_native_a10_speech_substitution_contract_v2"
)
SPEECH_TASK_ID = "tensorflow-speech-yes-no"
SPEECH_KAGGLE_SLUG = "tensorflow-speech-recognition-challenge"
HISTORICAL_WHALE_TASK_ID = "icml-2013-whale"


class SpeechSubstitutionError(ValueError):
    """Raised when the declarative V2 substitution contract drifts."""


@dataclass(frozen=True)
class SpeechSubstitutionContract:
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
            "lineage_expectations",
        }:
            raise SpeechSubstitutionError("substitution root schema differs")
        if root["version"] != SUBSTITUTION_CONTRACT_VERSION:
            raise SpeechSubstitutionError("substitution version differs")
        old = root["old_task"]
        new = root["new_task"]
        if not isinstance(old, Mapping) or old != {
            "ordinal": 17,
            "task_id": HISTORICAL_WHALE_TASK_ID,
            "kaggle_slug": "the-icml-2013-whale-challenge-right-whale-redux",
            "modality": "audio",
            "compressed_size_hint_bytes": 293_140_000,
        }:
            raise SpeechSubstitutionError("historical task declaration differs")
        if not isinstance(new, Mapping) or set(new) != {
            "ordinal",
            "task_id",
            "kaggle_slug",
            "modality",
            "compressed_size_hint_bytes",
            "source_inventory",
            "public_training_root",
            "classes",
            "required_valid_wavs_per_class",
            "required_sample_rate_hz",
            "selection",
            "use_mlebench_public_training_extraction_only",
            "ignore_mlebench_test_split",
        }:
            raise SpeechSubstitutionError("replacement task schema differs")
        expected_new = {
            "ordinal": 17,
            "task_id": SPEECH_TASK_ID,
            "kaggle_slug": SPEECH_KAGGLE_SLUG,
            "modality": "audio",
            "compressed_size_hint_bytes": 3_762_295_706,
            "source_inventory": [
                {"name": "link_to_gcp_credits_form.txt", "size_bytes": 50},
                {"name": "sample_submission.7z", "size_bytes": 512_684},
                {"name": "test.7z", "size_bytes": 2_640_679_130},
                {"name": "train.7z", "size_bytes": 1_121_103_842},
            ],
            "public_training_root": "train/audio",
            "classes": {"no": 0, "yes": 1},
            "required_valid_wavs_per_class": 2_048,
            "required_sample_rate_hz": 16_000,
            "selection": "sha256_relative_path_then_relative_path_v1",
            "use_mlebench_public_training_extraction_only": True,
            "ignore_mlebench_test_split": True,
        }
        if canonical_value(new) != canonical_value(expected_new):
            raise SpeechSubstitutionError("replacement task declaration differs")
        if root["affected_families"] != {
            "panns_cnn14": 275,
            "temporal_convolutional_network": 200,
            "m5_waveform_cnn": 175,
        }:
            raise SpeechSubstitutionError("affected family allocation differs")
        if root["affected_precision_counts"] != {
            "fp32_tf32": 130,
            "bf16": 195,
            "fp16_grad_scaler": 195,
            "mixed_structured": 130,
        }:
            raise SpeechSubstitutionError("affected precision allocation differs")
        if root["lineage_expectations"] != {
            "unchanged": 16_700,
            "audio_registry_rebound": 650,
            "dataset_substitution": 650,
        }:
            raise SpeechSubstitutionError("lineage allocation differs")


def load_speech_substitution_contract(
    path: str | Path = SUBSTITUTION_CONTRACT_PATH,
) -> SpeechSubstitutionContract:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SpeechSubstitutionError("substitution contract is unreadable") from error
    if not isinstance(payload, Mapping):
        raise SpeechSubstitutionError("substitution contract must be a mapping")
    contract = SpeechSubstitutionContract(canonical_value(payload))
    contract.validate()
    return contract


__all__ = [
    "HISTORICAL_WHALE_TASK_ID",
    "SPEECH_KAGGLE_SLUG",
    "SPEECH_TASK_ID",
    "SUBSTITUTION_CONTRACT_PATH",
    "SUBSTITUTION_CONTRACT_VERSION",
    "SpeechSubstitutionContract",
    "SpeechSubstitutionError",
    "load_speech_substitution_contract",
]
