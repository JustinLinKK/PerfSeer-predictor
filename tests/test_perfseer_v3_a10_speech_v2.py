from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if os.environ.get("PERFSEER_LABELER_PROFILE") != "native_a10_speech_v2":
    pytest.skip(
        "speech V2 tests require PERFSEER_LABELER_PROFILE=native_a10_speech_v2",
        allow_module_level=True,
    )


def _profile_script(profile: str, source: str, *, timeout: int = 300) -> str:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = profile
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def test_v1_hashes_remain_exact() -> None:
    output = _profile_script(
        "native_a10",
        "from perfseer_v3.dataset_pack.sampler import build_target_manifest; "
        "from perfseer_v3.dataset_pack.a10_crosswalk import build_crosswalk; "
        "m=build_target_manifest(); x=build_crosswalk(m); "
        "print(m.sha256,x.summary['crosswalk_sha256'])",
    )
    assert output.split() == [
        "361f31ed6fe6b75c508c7e5405aad6c893cc69f5a3cfe33d71fcc823c88aaba5",
        "d66c4f098665f9bab003825abde591c6d142dd68d09178e93475bac2e3db9a69",
    ]


def test_exact_v2_manifest_crosswalk_and_distribution() -> None:
    from perfseer_v3.dataset_pack.a10_speech_crosswalk import build_crosswalk
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    manifest = build_target_manifest()
    crosswalk = build_crosswalk(manifest)
    speech = [
        row for row in manifest.candidates if row.task_id == "tensorflow-speech-yes-no"
    ]
    assert manifest.sha256 == "c8fb3fe0d51073c63645f539a53247204bde45cf199671263db7cab5e79f5e39"
    assert len(manifest.candidates) == 18_000
    assert manifest.to_summary()["precision_counts"] == {
        "bf16": 5_232,
        "fp16_grad_scaler": 4_171,
        "fp32_tf32": 5_193,
        "mixed_structured": 3_404,
    }
    assert len(speech) == 650
    assert Counter(row.family_id for row in speech) == {
        "panns_cnn14": 275,
        "temporal_convolutional_network": 200,
        "m5_waveform_cnn": 175,
    }
    assert Counter(row.precision_policy["policy_id"] for row in speech) == {
        "fp32_tf32": 130,
        "bf16": 195,
        "fp16_grad_scaler": 195,
        "mixed_structured": 130,
    }
    assert crosswalk.summary["classification_counts"] == {
        "unchanged": 16_700,
        "audio_registry_rebound": 650,
        "dataset_substitution": 650,
    }
    assert sum(
        row["native_v1_candidate_id"] == row["native_v2_candidate_id"]
        for row in crosswalk.rows
    ) == 16_700
    assert all(
        len(row["task_independent_compute_signature"]) == 64
        for row in crosswalk.rows
    )
    assert all(
        row["old_semantic_signature"] != row["new_semantic_signature"]
        for row in crosswalk.rows
        if row["row_classification"] == "dataset_substitution"
    )
    assert len({row.task_id for row in manifest.candidates}) == 22
    assert len({row.family_id for row in manifest.candidates}) == 35
    assert len(manifest.candidates) * 3 == 54_000


def test_balanced_speech_view_is_exact_and_deterministic(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.local_task_fixture import _populate
    from perfseer_v3.dataset_pack.prepared_view import (
        build_shared_prepared_view,
        load_and_verify_prepared_view,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = load_task_registry().entries[17]
    public = tmp_path / "public"
    _populate(entry.task_id, public)
    # These invalid files are outside the independently selected public-train
    # yes/no roots and therefore cannot leak MLE-bench test semantics.
    (public / "test/audio").mkdir(parents=True)
    (public / "test/audio/invalid.wav").write_bytes(b"not a wave")
    (public / "train/audio/up").mkdir(parents=True)
    (public / "train/audio/up/invalid.wav").write_bytes(b"not a wave")
    first = build_shared_prepared_view(
        entry, public, tmp_path / "prepared-a", archive_sha256="a" * 64
    )
    second = build_shared_prepared_view(
        entry, public, tmp_path / "prepared-b", archive_sha256="a" * 64
    )
    assert first == second
    assert first.source_example_count == first.prepared_example_count == 4_096
    assert first.sampling_with_replacement is False
    load_and_verify_prepared_view(
        entry, public, tmp_path / "prepared-a", archive_sha256="a" * 64
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "prepared-a/samples.jsonl").read_text().splitlines()
    ]
    assert Counter(row["target"] for row in rows) == {0: 2_048, 1: 2_048}
    for class_name, offset in (("no", 0), ("yes", 2_048)):
        relatives = [
            path.relative_to(public).as_posix()
            for path in (public / "train/audio" / class_name).glob("*.wav")
        ]
        expected = sorted(
            relatives,
            key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
        )[:2_048]
        assert [row["source_sample_id"] for row in rows[offset : offset + 2_048]] == expected


def test_speech_view_rejects_corruption_and_insufficient_class(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.local_task_fixture import _populate, _wave_bytes
    from perfseer_v3.dataset_pack.prepared_view import PreparedViewError, build_shared_prepared_view
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = load_task_registry().entries[17]
    public = tmp_path / "public"
    _populate(entry.task_id, public)
    victim = public / "train/audio/no/speaker_0000.wav"
    victim.write_bytes(b"corrupt")
    with pytest.raises(PreparedViewError, match="corrupt"):
        build_shared_prepared_view(
            entry, public, tmp_path / "prepared-corrupt", archive_sha256="b" * 64
        )
    victim.write_bytes(_wave_bytes(16_000))
    victim.unlink()
    with pytest.raises(PreparedViewError, match="at least 2048"):
        build_shared_prepared_view(
            entry, public, tmp_path / "prepared-short", archive_sha256="b" * 64
        )


def test_speech_source_lock_rejects_archive_and_inventory_drift(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.materialization import (
        MATERIALIZATION_STATE_VERSION,
        TaskMaterializationError,
        TaskMaterializationState,
        TaskMaterializer,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = load_task_registry().entries[17]
    materializer = TaskMaterializer(
        workspace=tmp_path / "workspace",
        repository_root=ROOT,
        kaggle=SimpleNamespace(),
        preparer=SimpleNamespace(),
    )
    state = TaskMaterializationState(
        version=MATERIALIZATION_STATE_VERSION,
        task_id=entry.task_id,
        kaggle_slug=entry.kaggle_slug,
        stage="inspected",
        credential_source="api_token_environment",
        remote_inventory_sha256="a" * 64,
        archive_sha256="b" * 64,
        archive_inventory_sha256="c" * 64,
    )
    inventory = SimpleNamespace(archive_sha256="b" * 64)
    first = materializer._freeze_speech_source_lock(entry, state, inventory)
    assert first is not None
    assert materializer._freeze_speech_source_lock(entry, state, inventory) == first
    drifted = TaskMaterializationState(
        **{**state.__dict__, "remote_inventory_sha256": "d" * 64}
    )
    with pytest.raises(TaskMaterializationError, match="drifted"):
        materializer._freeze_speech_source_lock(entry, drifted, inventory)


def test_v2_workspace_rejects_v1_state(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.a10_campaign import (
        A10CampaignError,
        _assert_workspace_generation,
    )

    state = tmp_path / "state"
    state.mkdir()
    (state / "a10_campaign_contract.json").write_text("{}")
    with pytest.raises(A10CampaignError, match="V1 workspace"):
        _assert_workspace_generation(tmp_path)


def test_every_v2_record_binds_three_way_lineage_and_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
    from perfseer_v3.dataset_pack import supervisor

    candidate_id = "1" * 64
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lineage = {
        "native_v2_candidate_id": candidate_id,
        "original_a10g_candidate_id": "2" * 64,
        "native_v1_candidate_id": "3" * 64,
        "row_classification": "dataset_substitution",
        "old_semantic_signature": "4" * 64,
        "new_semantic_signature": "5" * 64,
        "task_independent_compute_signature": "6" * 64,
    }
    (state_dir / "a10_speech_v2_crosswalk.jsonl").write_text(
        json.dumps(lineage) + "\n"
    )
    task_cache = tmp_path / "task_cache"
    task_cache.mkdir()
    materialization = {
        "task_id": "tensorflow-speech-yes-no",
        "stage": "view_ready",
        "archive_sha256": "7" * 64,
        "remote_inventory_sha256": "8" * 64,
        "dataset_fingerprint": "9" * 64,
    }
    materialization["state_sha256"] = canonical_sha256(materialization)
    (task_cache / "materialization_state.json").write_text(
        json.dumps(materialization)
    )
    locks = state_dir / "source_locks"
    locks.mkdir()
    (locks / "tensorflow-speech-yes-no.json").write_text(
        json.dumps(
            {
                "archive_sha256": "7" * 64,
                "remote_inventory_sha256": "8" * 64,
                "lock_sha256": "a" * 64,
            }
        )
    )

    @dataclass(frozen=True)
    class FakeRecord:
        fingerprints: object
        production_eligible: bool = False
        reference_provenance: object = None
        build_identity: object = None

        def validate_against_configuration(self, candidate: object, task: object) -> None:
            return None

    record = FakeRecord(SimpleNamespace(dataset_sha256="9" * 64))
    candidate = SimpleNamespace(
        candidate_id=candidate_id,
        mutation_specification={},
    )
    task = SimpleNamespace(task_id="tensorflow-speech-yes-no")
    monkeypatch.setattr(
        supervisor,
        "environment_provenance",
        lambda: {
            "source_revision": "b" * 40,
            "source_tree_sha256": "c" * 64,
            "dependency_lock_sha256": "d" * 64,
            "image_identity": "image@sha256:" + "e" * 64,
        },
    )
    monkeypatch.setattr(
        supervisor, "semantic_distribution_signature", lambda _: "f" * 64
    )
    bound = supervisor._bind_native_a10_provenance(
        record, candidate, task, tmp_path
    )
    provenance = bound.reference_provenance
    assert provenance["original_a10g_candidate_id"] == "2" * 64
    assert provenance["native_v1_candidate_id"] == "3" * 64
    assert provenance["dataset_substitution"] is True
    assert provenance["source_archive_sha256"] == "7" * 64
    assert provenance["remote_inventory_sha256"] == "8" * 64
    assert provenance["speech_source_lock_sha256"] == "a" * 64


def test_v2_offline_job_and_pvc_contract(tmp_path: Path) -> None:
    path = ROOT / "scripts/render_a10_speech_v2_nautilus_job.py"
    spec = importlib.util.spec_from_file_location("render_a10_speech_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = (
        "gitlab-registry.nrp-nautilus.io/group/perfseer-a10-speech-v2-labeler"
        "@sha256:" + "b" * 64
    )
    revision = "a" * 40
    pilot = tmp_path / "pilot.yaml"
    chunk = tmp_path / "chunk.yaml"
    assert module.main(
        [
            "--output",
            str(pilot),
            "--namespace",
            "test-ns",
            "--image",
            image,
            "--source-revision",
            revision,
            "--mode",
            "pilot",
        ]
    ) == 0
    assert module.main(
        [
            "--output",
            str(chunk),
            "--namespace",
            "test-ns",
            "--image",
            image,
            "--source-revision",
            revision,
            "--mode",
            "chunk",
            "--chunk-index",
            "69",
        ]
    ) == 0
    module.verify_job(module._load(pilot), mode="pilot")
    module.verify_job(module._load(chunk), mode="chunk", chunk_index=69)
    pvc = yaml.safe_load(
        (ROOT / "k8s/a10-speech-v2-labeler-pvc.yaml").read_text()
    )
    assert pvc["spec"]["accessModes"] == ["ReadWriteMany"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "700Gi"


def test_v2_image_contract_is_baked_and_secret_free() -> None:
    dockerfile = (ROOT / "containers/a10-speech-v2-labeler/Dockerfile").read_text()
    assert "PERFSEER_A10_IMAGE_PROFILE=native_a10_speech_v2" in dockerfile
    assert "PERFSEER_LABELER_PROFILE=native_a10_speech_v2" in dockerfile
    assert "containers/a10-labeler/requirements.lock" in dockerfile
    assert "sm_86" not in dockerfile  # architecture support is checked at runtime
    assert "kaggle.json" in dockerfile and "test ! -e" in dockerfile
    assert "apt-get" not in dockerfile.split("FROM --platform=linux/amd64 ${BASE_IMAGE} AS runtime", 1)[1].split("ENTRYPOINT", 1)[1]


def test_speech_gate_is_first_and_requires_the_50_byte_download() -> None:
    script = (ROOT / "scripts/run_perfseer_v3_a10_labeling.py").read_text()
    assert 'speech.get("remote_name") != "link_to_gcp_credits_form.txt"' in script
    assert 'speech.get("downloaded_bytes") != 50' in script
