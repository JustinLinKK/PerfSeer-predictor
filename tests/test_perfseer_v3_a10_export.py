from __future__ import annotations

from dataclasses import dataclass, replace
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "native_a10_nonvision_disaster_v2"
if os.environ.get("PERFSEER_LABELER_PROFILE") != PROFILE:
    pytest.skip(
        "A10 export tests require PERFSEER_LABELER_PROFILE=" + PROFILE,
        allow_module_level=True,
    )
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from perfseer_v3.dataset_pack import a10_export
from perfseer_v3.dataset_pack.contracts import (
    AttemptStatus,
    EPOCH_MEASUREMENT_VERSION,
    EpochMeasurement,
    FailureStage,
    FingerprintBundle,
    GpuCleanupEvidence,
    LABEL_RUN_RECORD_VERSION,
    LabelRunRecord,
    TelemetrySample,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256, file_sha256
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.storage import atomic_write_json, atomic_write_jsonl


@dataclass(frozen=True)
class ExportFixture:
    archive: Path
    result: dict[str, Any]
    candidates: tuple[Any, ...]


def _measurement(candidate: Any, epoch: int, epoch_ms: float) -> EpochMeasurement:
    expected = candidate.batch_plan.expected_train_examples
    batches = math.ceil(expected / candidate.microbatch_size)
    optimizer_steps = math.ceil(batches / candidate.gradient_accumulation_steps)
    backend = str(candidate.execution["backend_id"])
    return EpochMeasurement(
        version=EPOCH_MEASUREMENT_VERSION,
        epoch=epoch,
        epoch_completed=True,
        epoch_ms=epoch_ms,
        examples_seen=expected,
        batches_seen=batches,
        microsteps=batches,
        optimizer_steps=optimizer_steps,
        loss_finite=True,
        gradients_finite=True,
        telemetry_complete=True,
        telemetry_samples=(
            TelemetrySample(
                timestamp_offset_s=float(epoch),
                duration_s=1.0,
                sm_util_percent=50.0,
                memory_controller_util_percent=25.0,
                device_used_vram_mib=1_024.0,
            ),
        ),
        peak_torch_reserved_mib=512.0,
        requested_backend_id=backend,
        observed_backend_id=backend,
        foreign_process_detected=False,
    )


def _accepted_record(candidate: Any) -> LabelRunRecord:
    record = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id="export-fixture-" + candidate.candidate_id[:16],
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.ACCEPTED,
        failure_stage=FailureStage.NONE,
        fingerprints=FingerprintBundle(
            source_sha256=candidate.source_sha256,
            graph_sha256="2" * 64,
            environment_sha256="3" * 64,
            hardware_sha256="4" * 64,
            support_contract_sha256="5" * 64,
            dataset_sha256="6" * 64,
        ),
        gpu_uuid="GPU-export-fixture",
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode=str(candidate.execution["mode"]),
        compile_completed_before_epoch_1=True,
        expected_train_examples=candidate.batch_plan.expected_train_examples,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=(1, 2, 3, 4, 5),
        finite_loss_epochs=(1, 2, 3, 4, 5),
        finite_gradient_epochs=(1, 2, 3, 4, 5),
        epoch_measurements=(
            _measurement(candidate, 3, 100.0),
            _measurement(candidate, 4, 101.0),
            _measurement(candidate, 5, 99.0),
        ),
        cleanup=GpuCleanupEvidence(
            pre_sample_device_used_vram_mib=100.0,
            post_cleanup_device_used_vram_mib=100.0,
            release_tolerance_mib=64.0,
            cleanup_seconds=1.0,
            stable_dwell_seconds=1.0,
            child_process_tree_exited=True,
            owned_gpu_processes_remaining=0,
            passed=True,
        ),
        production_eligible=True,
        build_identity={
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "dependency_lock_sha256": "c" * 64,
            "image_identity": "fixture@sha256:" + "d" * 64,
        },
    )
    record = replace(record, targets=record.aggregate_targets())
    record.validate()
    return record


def _fixture_candidates() -> tuple[Any, ...]:
    manifest = build_target_manifest()
    generated = tuple(
        row for row in manifest.candidates if row.family_id == "independent_generated"
    )
    other = next(row for row in manifest.candidates if row.family_id == "selu_mlp")
    assert len(generated) >= 2
    return generated[0], generated[1], other


def _configure_export(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_count: int,
) -> tuple[Any, ...]:
    candidates = _fixture_candidates()
    records = {row.candidate_id: _accepted_record(row) for row in candidates}
    fake_manifest = SimpleNamespace(candidates=candidates, sha256="e" * 64)
    monkeypatch.setattr(a10_export, "TOTAL_CANDIDATES", len(candidates))
    monkeypatch.setattr(a10_export, "build_target_manifest", lambda: fake_manifest)
    monkeypatch.setattr(
        a10_export,
        "build_campaign_contract",
        lambda: {"contract_sha256": "f" * 64},
    )

    def load(_: Path, root_candidate: Any) -> tuple[Any, LabelRunRecord] | None:
        if root_candidate not in candidates[:resolved_count]:
            return None
        return SimpleNamespace(current_candidate=root_candidate), records[root_candidate.candidate_id]

    monkeypatch.setattr(a10_export, "_load_record_for_root", load)
    return candidates


def _export_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ExportFixture:
    candidates = _configure_export(monkeypatch, resolved_count=3)
    result = dict(
        a10_export.export_release(
            tmp_path / "workspace",
            ROOT,
            tmp_path / "releases",
            complete=True,
        )
    )
    return ExportFixture(Path(result["archive"]), result, candidates)


def _extract(archive: Path, destination: Path) -> Path:
    destination.mkdir()
    a10_export._safe_extract(archive, destination)
    return destination / "release"


def _repack(
    release: Path,
    destination: Path,
    *,
    refresh_checksums: bool,
) -> Path:
    if refresh_checksums:
        a10_export._write_sha256sums(release)
    a10_export._deterministic_tar(release, destination)
    return destination


def test_real_export_round_trip_pairs_sources_configurations_and_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _export_fixture(monkeypatch, tmp_path)
    verification = a10_export.verify_release(fixture.archive)
    assert verification["status"] == "passed"
    assert verification["label_count"] == 3
    assert verification["reconstructed_configuration_count"] == 3
    assert fixture.result["archive_sha256"] == file_sha256(fixture.archive)
    sidecar = fixture.archive.with_suffix(fixture.archive.suffix + ".json")
    assert json.loads(sidecar.read_text()) == fixture.result

    release = _extract(fixture.archive, tmp_path / "inspect")
    labels = (release / "labels.jsonl").read_text().splitlines()
    indexes = [
        json.loads(line) for line in (release / "candidate-index.jsonl").read_text().splitlines()
    ]
    assert len(labels) == len(indexes) == 3
    assert [row["label_line"] for row in indexes] == [1, 2, 3]
    assert indexes[0]["model_source_bundle_path"] == indexes[1]["model_source_bundle_path"]
    assert indexes[0]["model_source_bundle_path"] != indexes[2]["model_source_bundle_path"]
    assert len({row["configuration_path"] for row in indexes}) == 3


def test_complete_export_rejects_an_unresolved_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_export(monkeypatch, resolved_count=2)
    with pytest.raises(a10_export.A10ExportError, match="all 11,200 accepted labels"):
        a10_export.export_release(
            tmp_path / "workspace",
            ROOT,
            tmp_path / "releases",
            complete=True,
        )


def _corrupt_label_order(release: Path) -> None:
    path = release / "candidate-index.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["label_line"] = 2
    atomic_write_jsonl(path, rows)


def _corrupt_configuration_join(release: Path) -> None:
    index_path = release / "candidate-index.jsonl"
    indexes = [json.loads(line) for line in index_path.read_text().splitlines()]
    configuration_path = release / indexes[0]["configuration_path"]
    configuration = json.loads(configuration_path.read_text())
    configuration["family_id"] = "wrong-family"
    atomic_write_json(configuration_path, configuration)
    indexes[0]["configuration_sha256"] = canonical_sha256(configuration)
    atomic_write_jsonl(index_path, indexes)


def _corrupt_source_file(release: Path) -> None:
    index = json.loads((release / "candidate-index.jsonl").read_text().splitlines()[0])
    bundle = release / index["model_source_bundle_path"]
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    family_path = next(path for path in manifest["files"] if path.endswith(".py"))
    (bundle / family_path).write_text("CORRUPTED = True\n", encoding="utf-8")


def _corrupt_bundle_manifest(release: Path) -> None:
    index = json.loads((release / "candidate-index.jsonl").read_text().splitlines()[0])
    path = release / index["model_source_bundle_path"] / "bundle-manifest.json"
    manifest = json.loads(path.read_text())
    first = next(iter(manifest["files"]))
    manifest["files"][first] = "0" * 64
    atomic_write_json(path, manifest)


def _add_forbidden_weight(release: Path) -> None:
    (release / "model.ckpt").write_bytes(b"forbidden")


def _add_unlisted_file(release: Path) -> None:
    (release / "unlisted.txt").write_text("not checksummed\n", encoding="utf-8")


def _add_checksummed_extra_file(release: Path) -> None:
    (release / "extra.txt").write_text("checksummed but not contracted\n", encoding="utf-8")


def _add_unreferenced_configuration(release: Path) -> None:
    atomic_write_json(release / "configurations" / f"{'0' * 64}.json", {"extra": True})


def _add_unreferenced_source_bundle(release: Path) -> None:
    path = release / "source-bundles" / "model" / ("0" * 64)
    atomic_write_json(path / "bundle-manifest.json", {"extra": True})


@pytest.mark.parametrize(
    ("mutate", "refresh_checksums", "message"),
    [
        (_corrupt_label_order, True, "schema or label order"),
        (_corrupt_configuration_join, True, "join differs"),
        (_corrupt_source_file, True, "source bundle checksum differs"),
        (_corrupt_bundle_manifest, True, "source bundle digest differs"),
        (_add_forbidden_weight, True, "weights, checkpoints, or bytecode"),
        (_add_unlisted_file, False, "coverage differs"),
        (_add_checksummed_extra_file, True, "outside the artifact contract"),
        (_add_unreferenced_configuration, True, "configuration file coverage differs"),
        (_add_unreferenced_source_bundle, True, "unreferenced source bundle"),
    ],
)
def test_verifier_rejects_corrupt_or_uncovered_release_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Callable[[Path], None],
    refresh_checksums: bool,
    message: str,
) -> None:
    fixture = _export_fixture(monkeypatch, tmp_path)
    release = _extract(fixture.archive, tmp_path / "corrupt")
    mutate(release)
    archive = _repack(
        release,
        tmp_path / "corrupt.tar.zst",
        refresh_checksums=refresh_checksums,
    )
    with pytest.raises(a10_export.A10ExportError, match=message):
        a10_export.verify_release(archive, reconstruct=False)


def test_verifier_rejects_unsafe_tar_member(tmp_path: Path) -> None:
    raw = tmp_path / "unsafe.tar"
    with tarfile.open(raw, "w") as bundle:
        payload = b"escape"
        member = tarfile.TarInfo("../escape")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    archive = tmp_path / "unsafe.tar.zst"
    archive.write_bytes(a10_export._zstd_compress(raw.read_bytes()))
    with pytest.raises(a10_export.A10ExportError, match="safe relative POSIX path"):
        a10_export.verify_release(archive, reconstruct=False)
