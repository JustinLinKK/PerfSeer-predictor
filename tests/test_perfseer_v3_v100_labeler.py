from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from perfseer_v3.dataset_pack.campaign import (
    CAMPAIGN_CANDIDATE_COUNT,
    CAMPAIGN_COUNTS,
    CAMPAIGN_MEASURED_EPOCH_COUNT,
    build_campaign_contract,
    freeze_campaign_contract,
    pilot_candidates,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.local_smoke import select_smoke_candidate
from perfseer_v3.dataset_pack.modality_sharding import ModalityShardError, _image_digest
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.dataset_pack.supervisor import (
    SupervisorError,
    discover_v100_probes,
    validate_v100_gpu_identity,
)


def test_full_design_and_campaign_are_exact_v100_projections() -> None:
    manifest = build_target_manifest()
    campaign = build_campaign_contract()
    assert manifest.target_hardware_id == "nvidia_tesla_v100_sxm2_32gb_nrp"
    assert len(manifest.candidates) == len({row.candidate_id for row in manifest.candidates}) == 18_000
    assert campaign.candidate_count == CAMPAIGN_CANDIDATE_COUNT == 3_550
    assert campaign.measured_epoch_count == CAMPAIGN_MEASURED_EPOCH_COUNT == 10_650
    assert campaign.modality_counts == CAMPAIGN_COUNTS
    selected = [row for row in manifest.candidates if row.candidate_id in set(campaign.candidate_ids)]
    assert Counter(row.quota_modality for row in selected) == CAMPAIGN_COUNTS
    assert not any(row.quota_modality == "generated" for row in selected)


def test_precision_translation_has_no_tf32_or_bfloat16() -> None:
    manifest = build_target_manifest()
    modes = Counter(row.precision_policy["policy_id"] for row in manifest.candidates)
    assert modes == {
        "fp32_ieee": 5_479,
        "fp16_grad_scaler": 9_117,
        "mixed_structured": 3_404,
    }
    assert not any(
        token in str(row.precision_policy).lower()
        for row in manifest.candidates
        for token in ("tf32", "bf16", "bfloat16")
    )


@pytest.mark.parametrize(
    ("name", "capability", "memory"),
    [
        ("NVIDIA A10", (8, 6), 24 * 1024**3),
        ("NVIDIA GeForce RTX 5090", (12, 0), 32 * 1024**3),
        ("Tesla V100-SXM2-32GB", (7, 0), 16 * 1024**3),
        ("Tesla V100-SXM2-32GB", (7, 5), 32 * 1024**3),
    ],
)
def test_v100_identity_rejects_hardware_mismatches(name, capability, memory) -> None:
    with pytest.raises(SupervisorError):
        validate_v100_gpu_identity(name, capability, memory)


def test_v100_identity_accepts_exact_sxm2_32gb() -> None:
    validate_v100_gpu_identity("Tesla V100-SXM2-32GB", (7, 0), 32 * 1024**3)


def test_discovery_requires_four_unique_non_mixed_probes(monkeypatch) -> None:
    import perfseer_v3.dataset_pack.supervisor as module

    monkeypatch.setitem(__import__("sys").modules, "pynvml", SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 4,
    ))

    @dataclass
    class Probe:
        physical_index: int

        def __post_init__(self):
            self.gpu_uuid = f"GPU-{self.physical_index}"
            self.hardware_provenance = {
                "name": "Tesla V100-SXM2-32GB",
                "compute_capability": [7, 0],
                "total_memory_bytes": 32 * 1024**3,
            }

    monkeypatch.setattr(module, "ParentNvmlProbe", Probe)
    assert len(discover_v100_probes()) == 4


def test_campaign_contract_freeze_is_atomic_and_resume_safe(tmp_path: Path) -> None:
    first = freeze_campaign_contract(tmp_path)
    path = tmp_path / "state/campaign_contract.json"
    assert json.loads(path.read_text())["contract_sha256"] == first.contract_sha256
    assert freeze_campaign_contract(tmp_path) == first
    assert not tuple(path.parent.glob("*.tmp"))


def test_atomic_json_replaces_complete_payload(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2, "sha256": canonical_sha256({"value": 2})})
    assert json.loads(path.read_text())["value"] == 2
    assert not tuple(tmp_path.glob("*.tmp"))


def test_local_smoke_candidates_are_non_bf16_rules() -> None:
    panns = select_smoke_candidate("panns_cnn14")
    cgcnn = select_smoke_candidate("cgcnn")
    assert (panns.task_id, panns.precision_policy["policy_id"]) == (
        "mlsp-2013-birds",
        "fp32_ieee",
    )
    assert (cgcnn.task_id, cgcnn.precision_policy["policy_id"]) == (
        "nomad2018",
        "fp16_grad_scaler",
    )


def test_pilot_is_four_unique_named_families() -> None:
    rows = pilot_candidates()
    assert len(rows) == len({row.candidate_id for row in rows}) == 4
    assert tuple(row.family_id for row in rows) == (
        "panns_cnn14",
        "m5_waveform_cnn",
        "tabtransformer",
        "cgcnn",
    )


def test_full_registry_digest_is_valid_but_mutable_tag_is_not() -> None:
    _image_digest("registry.example/perfseer@sha256:" + "1" * 64)
    with pytest.raises(ModalityShardError):
        _image_digest("registry.example/perfseer:latest@sha256:" + "1" * 64)
