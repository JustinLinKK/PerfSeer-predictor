from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, replace
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "native_a10_nonvision_disaster_v2"
if os.environ.get("PERFSEER_LABELER_PROFILE") != PROFILE:
    pytest.skip(
        "Disaster V2 tests require PERFSEER_LABELER_PROFILE=" + PROFILE,
        allow_module_level=True,
    )
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _render_train(
    *,
    fieldnames: list[str] | None = None,
    row_count: int = 7_613,
    duplicate_ids: bool = False,
    invalid_target: bool = False,
    empty_text: bool = False,
    one_class: bool = False,
) -> bytes:
    fields = fieldnames or ["id", "keyword", "location", "text", "target"]

    def payload(padding: int) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
        writer.writeheader()
        padding_each, padding_remainder = divmod(padding, row_count)
        for index in range(row_count):
            row_padding = padding_each + (1 if index < padding_remainder else 0)
            row = {
                "id": "1" if duplicate_ids and index == 1 else str(index + 1),
                "keyword": "fire" if index % 3 == 0 else "",
                "location": "coast" if index % 5 == 0 else "",
                "text": (
                    " " * (len("tweet 2") + row_padding)
                    if empty_text and index == 1
                    else f"tweet {index + 1}" + ("x" * row_padding)
                ),
                "target": "2" if invalid_target and index == 1 else "0" if one_class else str(index % 2),
                "extra": "",
            }
            writer.writerow({name: row.get(name, "") for name in fields})
        return stream.getvalue().encode("utf-8")

    initial = payload(0)
    padding = 987_712 - len(initial)
    if padding < 0:
        raise AssertionError("fixture CSV exceeds the contracted size")
    result = payload(padding)
    assert len(result) == 987_712
    return result


def _raw_source(path: Path, **train_options: object) -> Path:
    path.mkdir()
    (path / "sample_submission.csv").write_bytes(b"s" * 22_746)
    (path / "test.csv").write_bytes(b"t" * 420_783)
    (path / "train.csv").write_bytes(_render_train(**train_options))
    return path


def _replacement_audit_quarantine(target: Any) -> Any:
    from perfseer_v3.dataset_pack.contracts import FailureStage
    from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
    from perfseer_v3.dataset_pack.repair import QUARANTINE_VERSION, QuarantineRecord

    draft = QuarantineRecord(
        version=QUARANTINE_VERSION,
        quarantine_id="0" * 64,
        root_candidate_id=target.candidate_id,
        terminal_candidate_id=target.candidate_id,
        terminal_failure_record_sha256="1" * 64,
        oom_repair_attempt_ids=(),
        failure_stage=FailureStage.FORWARD.value,
        reason_code="replacement_contract_audit",
        family_id=target.family_id,
        task_id=target.task_id,
        regime=target.regime,
        coverage_cell_ids=target.coverage_cell_ids,
        batch_one_exhausted=False,
    )
    return replace(
        draft,
        quarantine_id=canonical_sha256(draft.unhashed_payload()),
    )


def test_manifest_crosswalk_and_historical_v1_identities() -> None:
    from perfseer_v3.dataset_pack.a10_disaster_crosswalk import build_crosswalk
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    manifest = build_target_manifest()
    affected = tuple(row for row in manifest.candidates if row.task_id == "disaster-tweets")
    assert manifest.sha256 == "8de16ba359c99743107f25b41fad6999598d4e83a7ddeca9fad8da35e23d699f"
    assert len(manifest.candidates) == 11_200
    assert len({row.task_id for row in manifest.candidates}) == 12
    assert len({row.family_id for row in manifest.candidates}) == 22
    assert len(affected) == 1_075
    assert Counter(row.family_id for row in affected) == {
        "bert_base": 275,
        "mla_mini_transformer": 250,
        "bilstm_crf": 150,
        "fasttext_embeddingbag": 125,
        "distilbert_distillation": 275,
    }
    assert Counter(row.precision_policy["policy_id"] for row in affected) == {
        "fp32_tf32": 214,
        "bf16": 321,
        "fp16_grad_scaler": 324,
        "mixed_structured": 216,
    }
    assert Counter(
        row.microbatch_size * row.gradient_accumulation_steps for row in affected
    ) == {32: 215, 64: 215, 128: 215, 256: 215, 512: 215}
    crosswalk = build_crosswalk(manifest)
    assert Counter(row["row_classification"] for row in crosswalk.rows) == {
        "unchanged": 9_050,
        "nlp_registry_rebound": 1_075,
        "dataset_substitution": 1_075,
    }
    assert sum(row["v1_candidate_id"] == row["v2_candidate_id"] for row in crosswalk.rows) == 9_050
    assert all(row["old_compute_signature"] == row["new_compute_signature"] for row in crosswalk.rows)

    environment = {**os.environ, "PERFSEER_LABELER_PROFILE": "native_a10_nonvision_4gpu_v1", "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", "from perfseer_v3.dataset_pack.sampler import build_target_manifest; from perfseer_v3.dataset_pack.task_registry import load_task_registry; from perfseer_v3.dataset_pack.model_registry import load_model_registry; print(build_target_manifest().sha256,load_task_registry().sha256,load_model_registry().sha256)"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.stdout.split() == [
        "05e4d98b94c55d787a3d325b9c35f8ab3ff5dfbbb211be22844ff49bd881515b",
        "dddc68689d6fa3e59426ea7f9cccd37a5787a633112492e1903cfad7bc60e298",
        "9953e4d249688f244ddbb9cf9475b988476144ffed2e4e2870df36056b18f513",
    ]


def test_fasttext_replacements_rebind_sparse_optimizer_contract() -> None:
    from perfseer_v3.dataset_pack.compatibility import DEPLOYMENT_OPTIMIZERS
    from perfseer_v3.dataset_pack.repair import make_quota_replacement
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    targets = tuple(
        row
        for row in build_target_manifest().candidates
        if row.family_id == "fasttext_embeddingbag"
    )
    replacements = []
    for target in targets:
        quarantine = _replacement_audit_quarantine(target)
        for replacement_index in range(3):
            replacement = make_quota_replacement(
                target,
                quarantine,
                replacement_index=replacement_index,
            )
            assert replacement.candidate.optimizer == target.optimizer
            assert replacement.candidate.architecture_parameters["sparse_gradients"] is (
                target.optimizer["name"] == "sparse_adam"
            )
            replacements.append(replacement)

    assert len(targets) == 250
    assert len(replacements) == 750
    assert len({row.candidate.candidate_id for row in replacements}) == 750
    assert {row.candidate.optimizer["name"] for row in replacements} == set(
        DEPLOYMENT_OPTIMIZERS
    )


def test_quota_replacement_skips_an_incompatible_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfseer_v3.dataset_pack import repair
    from perfseer_v3.dataset_pack.compatibility import (
        COMPATIBILITY_VERSION,
        CompatibilityDecision,
    )
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    target = next(
        row
        for row in build_target_manifest().candidates
        if row.family_id != "fasttext_embeddingbag"
    )
    quarantine = _replacement_audit_quarantine(target)
    evaluate_compatibility = repair.evaluate_compatibility
    requests = []

    def reject_first_proposal(request: Any) -> CompatibilityDecision:
        requests.append(request)
        if len(requests) == 1:
            return CompatibilityDecision(
                version=COMPATIBILITY_VERSION,
                request_sha256=request.sha256,
                compatible=False,
                reason_codes=("optimizer_parameter_contract_invalid",),
            )
        return evaluate_compatibility(request)

    monkeypatch.setattr(repair, "evaluate_compatibility", reject_first_proposal)
    replacement = repair.make_quota_replacement(
        target,
        quarantine,
        replacement_index=0,
    )

    assert len(requests) == 2
    assert replacement.candidate.ordinal == 18_001 + target.ordinal * 1_000 + 3
    assert replacement.candidate.mutation_specification["quota_replacement"] == {
        "quarantine_id": quarantine.quarantine_id,
        "replacement_index": 0,
        "target_candidate_id": target.candidate_id,
        "proposal_index": 1,
        "proposal_ordinal": replacement.candidate.ordinal,
    }


def test_fixture_smoke_executes_the_declared_compiled_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from perfseer_v3.dataset_pack.adapters import adapter_for_task
    from perfseer_v3.dataset_pack.local_smoke import _execute_one_batch_update
    from perfseer_v3.dataset_pack.local_task_fixture import (
        build_local_real_format_batch,
    )
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    candidate = next(
        row
        for row in build_target_manifest().candidates
        if row.family_id == "independent_generated"
        and row.source_modality == "tabular"
        and row.execution["mode"] == "compiled"
        and row.precision_policy["policy_id"] == "fp32_tf32"
    )
    adapter = adapter_for_task(candidate.task_id)
    _, raw = build_local_real_format_batch(adapter)
    compile_calls: list[dict[str, object]] = []
    compiler_resets: list[bool] = []

    def fake_compile(function: object, **kwargs: object) -> object:
        compile_calls.append(dict(kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        torch.compiler,
        "reset",
        lambda: compiler_resets.append(True),
    )
    result = _execute_one_batch_update(
        candidate,
        torch.device("cpu"),
        adapter=adapter,
        raw=raw,
    )

    assert compile_calls == [
        {"backend": "inductor", "fullgraph": False, "dynamic": False}
    ]
    assert compiler_resets == [True]
    assert result["requested_execution_mode"] == "compiled"
    assert result["observed_backend_id"] == "inductor_cuda"


def test_fixture_matrix_covers_generated_runtime_interactions() -> None:
    from perfseer_v3.dataset_pack.local_smoke import (
        select_nonvision_fixture_matrix,
    )
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    manifest_generated = tuple(
        row
        for row in build_target_manifest().candidates
        if row.family_id == "independent_generated"
    )
    selected = select_nonvision_fixture_matrix()
    selected_generated = tuple(
        row for row in selected if row.family_id == "independent_generated"
    )

    assert len(selected) == 177
    assert len(selected_generated) == 94
    assert Counter(row.execution["mode"] for row in selected_generated) == {
        "eager": 54,
        "compiled": 40,
    }
    for projection in (
        lambda row: (row.source_lineage, row.execution["mode"]),
        lambda row: (row.source_lineage, row.precision_policy["policy_id"]),
        lambda row: (row.optimizer["name"], row.source_modality),
    ):
        assert {projection(row) for row in selected_generated} == {
            projection(row) for row in manifest_generated
        }


def test_fixture_matrix_uses_an_isolated_writable_compiler_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfseer_v3.dataset_pack import local_smoke
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    candidate = next(
        row
        for row in build_target_manifest().candidates
        if row.family_id == "independent_generated"
        and row.execution["mode"] == "compiled"
    )
    previous = {
        "TORCHINDUCTOR_CACHE_DIR": "/read-only/inductor",
        "TRITON_CACHE_DIR": "/read-only/triton",
        "PYTORCH_KERNEL_CACHE_PATH": "/read-only/torch-kernels",
    }
    for name, value in previous.items():
        monkeypatch.setenv(name, value)
    observed: dict[str, str] = {}

    def execute(_candidate: object, _device: object) -> dict[str, object]:
        for name in previous:
            value = os.environ[name]
            observed[name] = value
            assert Path(value).is_dir()
            assert Path(value).is_relative_to(tmp_path)
        return {
            "configuration_id": candidate.candidate_id,
            "family_id": candidate.family_id,
            "task_id": candidate.task_id,
            "one_batch_update": "passed",
        }

    monkeypatch.setattr(
        local_smoke,
        "Rtx5090TelemetryBackend",
        lambda: SimpleNamespace(hardware_provenance={"scope": "fixture"}),
    )
    monkeypatch.setattr(
        local_smoke,
        "select_nonvision_fixture_matrix",
        lambda: (candidate,),
    )
    monkeypatch.setattr(local_smoke, "_run_fixture_one_batch_update", execute)

    local_smoke.run_nonvision_fixture_matrix_smoke(tmp_path)

    assert set(observed) == set(previous)
    assert all(os.environ[name] == value for name, value in previous.items())
    assert all(not Path(value).exists() for value in observed.values())


def test_local_validation_preserves_unstable_generated_measurements() -> None:
    from dataclasses import replace
    import math

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
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    candidate = next(
        row
        for row in build_target_manifest().candidates
        if row.family_id == "independent_generated"
        and row.execution["mode"] == "compiled"
    )
    batches = math.ceil(4_096 / candidate.microbatch_size)
    optimizer_steps = math.ceil(batches / candidate.gradient_accumulation_steps)

    def measurement(epoch: int, epoch_ms: float) -> EpochMeasurement:
        return EpochMeasurement(
            version=EPOCH_MEASUREMENT_VERSION,
            epoch=epoch,
            epoch_completed=True,
            epoch_ms=epoch_ms,
            examples_seen=4_096,
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
            requested_backend_id=candidate.execution["backend_id"],
            observed_backend_id=candidate.execution["backend_id"],
            foreign_process_detected=False,
        )

    record = LabelRunRecord(
        version=LABEL_RUN_RECORD_VERSION,
        run_id="local-generated-instability-fixture",
        configuration_id=candidate.candidate_id,
        status=AttemptStatus.ACCEPTED,
        failure_stage=FailureStage.NONE,
        fingerprints=FingerprintBundle(
            source_sha256="1" * 64,
            graph_sha256="2" * 64,
            environment_sha256="3" * 64,
            hardware_sha256="4" * 64,
            support_contract_sha256="5" * 64,
            dataset_sha256="6" * 64,
        ),
        gpu_uuid="GPU-local-fixture",
        target_hardware_id=candidate.target_hardware_id,
        capture_workload_sha256=candidate.candidate_id,
        profile_workload_sha256=candidate.candidate_id,
        coverage_cell_ids=candidate.coverage_cell_ids,
        task_id=candidate.task_id,
        execution_mode="compiled",
        compile_completed_before_epoch_1=True,
        expected_train_examples=4_096,
        microbatch_size=candidate.microbatch_size,
        gradient_accumulation_steps=candidate.gradient_accumulation_steps,
        total_epochs=5,
        warmup_epochs=(1, 2),
        measured_epochs=(3, 4, 5),
        completed_epochs=(1, 2, 3, 4, 5),
        finite_loss_epochs=(1, 2, 3, 4, 5),
        finite_gradient_epochs=(1, 2, 3, 4, 5),
        epoch_measurements=(
            measurement(3, 899.0),
            measurement(4, 1_045.0),
            measurement(5, 876.0),
        ),
        cleanup=GpuCleanupEvidence(
            pre_sample_device_used_vram_mib=100.0,
            post_cleanup_device_used_vram_mib=100.0,
            release_tolerance_mib=4_096.0,
            cleanup_seconds=1.0,
            stable_dwell_seconds=1.0,
            child_process_tree_exited=True,
            owned_gpu_processes_remaining=0,
            passed=True,
        ),
        production_eligible=False,
    )
    record = replace(record, targets=record.aggregate_targets())
    assert record.epoch_time_relative_spread is not None
    assert record.epoch_time_relative_spread > 0.10
    record.validate()
    with pytest.raises(ValueError, match=r"epoch stability gate \(10%\)"):
        replace(record, production_eligible=True).validate()
    far_too_unstable = replace(
        record,
        epoch_measurements=(
            measurement(3, 100.0),
            measurement(4, 1_000.0),
            measurement(5, 100.0),
        ),
    )
    far_too_unstable = replace(
        far_too_unstable,
        targets=far_too_unstable.aggregate_targets(),
    )
    with pytest.raises(ValueError, match=r"epoch stability gate \(60%\)"):
        far_too_unstable.validate()


@pytest.mark.parametrize(
    "options",
    [
        {"fieldnames": ["id", "keyword", "text", "target"]},
        {"fieldnames": ["id", "keyword", "location", "text", "target", "extra"]},
        {"duplicate_ids": True},
        {"invalid_target": True},
        {"empty_text": True},
        {"row_count": 4_095},
        {"one_class": True},
    ],
)
def test_custom_preparer_rejects_bad_csv_contract(
    tmp_path: Path, options: dict[str, object]
) -> None:
    from perfseer_v3.dataset_pack.substitution_preparer import (
        SubstitutionPreparationError,
        _prepare_disaster_tweets,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = next(row for row in load_task_registry().entries if row.task_id == "disaster-tweets")
    raw = _raw_source(tmp_path / "raw", **options)
    with pytest.raises(SubstitutionPreparationError):
        _prepare_disaster_tweets(entry, raw, tmp_path / "public", tmp_path / "private")


def test_custom_preparer_and_prepared_view_are_exact_and_deterministic(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.prepared_view import (
        build_shared_prepared_view,
        load_and_verify_prepared_view,
    )
    from perfseer_v3.dataset_pack.substitution_preparer import _prepare_disaster_tweets
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = next(row for row in load_task_registry().entries if row.task_id == "disaster-tweets")
    raw = _raw_source(tmp_path / "raw")
    public, private = tmp_path / "public", tmp_path / "private"
    _prepare_disaster_tweets(entry, raw, public, private)
    first = build_shared_prepared_view(entry, public, tmp_path / "prepared-a", archive_sha256="a" * 64)
    second = build_shared_prepared_view(entry, public, tmp_path / "prepared-b", archive_sha256="a" * 64)
    assert first == second
    assert first.source_example_count == 7_613
    assert first.prepared_example_count == 4_096
    assert first.sampling_with_replacement is False
    load_and_verify_prepared_view(entry, public, tmp_path / "prepared-a", archive_sha256="a" * 64)
    rows = tuple(json.loads(line) for line in (tmp_path / "prepared-a/samples.jsonl").read_text().splitlines())
    assert len(rows) == len({row["source_sample_id"] for row in rows}) == 4_096
    assert set(row["target"] for row in rows) == {0, 1}


def test_large_nyc_view_uses_bounded_deterministic_streaming(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
    from perfseer_v3.dataset_pack.prepared_view import (
        _stream_tabular_smallest,
        build_shared_prepared_view,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = next(row for row in load_task_registry().entries if row.task_id == "nyc-taxi-fare")
    public = tmp_path / "public"
    public.mkdir()
    identities = [f"taxi-{index:05d}" for index in range(5_500)]
    with (public / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("key", "fare_amount", "pickup_longitude"),
        )
        writer.writeheader()
        for index, identity in enumerate(identities):
            writer.writerow(
                {
                    "key": identity,
                    "fare_amount": f"{5 + index % 40}.25",
                    "pickup_longitude": f"{-74 + index / 100_000:.6f}",
                }
            )
    selected, source_count = _stream_tabular_smallest(
        entry,
        public,
        table="labels.csv",
        id_column="key",
        target_columns=("fare_amount",),
    )
    expected = sorted(
        identities,
        key=lambda identity: (
            canonical_sha256(
                {"task_id": entry.task_id, "source_sample_id": identity}
            ),
            identity,
        ),
    )[:4_096]
    assert source_count == 5_500
    assert [row.source_sample_id for row in selected] == expected
    assert len({row.source_sample_id for row in selected}) == 4_096
    manifest = build_shared_prepared_view(
        entry,
        public,
        tmp_path / "prepared",
        archive_sha256="f" * 64,
    )
    assert manifest.source_example_count == 5_500
    assert manifest.prepared_example_count == 4_096
    assert manifest.sampling_with_replacement is False


def test_lingering_child_group_is_killed_and_rechecked() -> None:
    from perfseer_v3.dataset_pack.supervisor import (
        cleanup_release_tolerance_mib,
        retained_epoch_spread_limit,
        terminate_lingering_process_group,
    )

    assert cleanup_release_tolerance_mib(production_eligible=True) == 64.0
    assert cleanup_release_tolerance_mib(production_eligible=False) == 4_096.0
    assert retained_epoch_spread_limit(production_eligible=True) == 0.10
    assert retained_epoch_spread_limit(production_eligible=False) == 0.60

    probes = iter((None, None, ProcessLookupError()))
    signals: list[int] = []

    def killpg(_pid: int, sent_signal: int) -> None:
        signals.append(sent_signal)
        if sent_signal == 0:
            outcome = next(probes)
            if isinstance(outcome, Exception):
                raise outcome

    with mock.patch(
        "perfseer_v3.dataset_pack.supervisor.os.killpg", side_effect=killpg
    ):
        assert terminate_lingering_process_group(
            12345, timeout_seconds=1.0, poll_seconds=0.0
        )
    assert signals.count(0) == 3

    with mock.patch("perfseer_v3.dataset_pack.supervisor.os.killpg") as kill:
        assert not terminate_lingering_process_group(12345, timeout_seconds=0.0)
    assert kill.call_count == 2


def test_worker_diagnostic_is_bounded_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfseer_v3.dataset_pack.label_worker import _diagnostic_message
    from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
    from perfseer_v3.dataset_pack.supervisor import (
        build_worker_failure_diagnostic,
        worker_failure_summary,
    )

    monkeypatch.setenv("KAGGLE_KEY", "private-value")
    message = _diagnostic_message(
        RuntimeError("token=public-leak private-value " + "x" * 3_000)
    )
    assert "private-value" not in message
    assert "public-leak" not in message
    assert "<redacted>" in message
    assert len(message) == 2_000
    diagnostic = build_worker_failure_diagnostic(
        candidate_id="a" * 64,
        run_id="b" * 64,
        attempt_index=3,
        return_code=21,
        child_log="attempts/logs/failure.log",
        worker_diagnostic={
            "failure_stage": "forward",
            "reason_code": "builtins.RuntimeError",
            "reason_message": message,
        },
    )
    unhashed = dict(diagnostic)
    declared = unhashed.pop("diagnostic_sha256")
    assert declared == canonical_sha256(unhashed)
    assert diagnostic["reason_message"] == message
    code, summary = worker_failure_summary(
        {
            "reason_code": "builtins.ValueError",
            "reason_message": "NVML token=public-leak private-value",
        }
    )
    assert code == "builtins.ValueError"
    assert "private-value" not in summary
    assert "public-leak" not in summary
    assert summary == "NVML token=<redacted> <redacted>"


def test_four_gpu_child_assignments_bind_cuda_nvml_uuid_and_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfseer_v3.dataset_pack.supervisor import child_gpu_environment
    from perfseer_v3.dataset_pack.v100_runner import NvmlTelemetryBackend
    import perfseer_v3.dataset_pack.v100_runner as runner

    uuids = tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(4))
    handles: list[int] = []

    def handle_by_index(index: int) -> int:
        handles.append(index)
        return index

    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=handle_by_index,
        nvmlDeviceGetName=lambda _handle: "NVIDIA A10",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            total=24 * 1024**3,
            used=(handle + 1) * 1024**2,
        ),
        nvmlDeviceGetUUID=lambda handle: uuids[handle],
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(
            gpu=10 + handle,
            memory=20 + handle,
        ),
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda _handle: 0,
        nvmlDeviceGetComputeRunningProcesses=lambda handle: (
            SimpleNamespace(pid=10_000 + handle),
        ),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(major=8, minor=6),
    )

    fingerprints = []
    for index, gpu_uuid in enumerate(uuids):
        probe = SimpleNamespace(physical_index=index, gpu_uuid=gpu_uuid)
        environment = child_gpu_environment(probe)
        assert environment == {
            "CUDA_VISIBLE_DEVICES": str(index),
            "PERFSEER_ASSIGNED_GPU_UUID": gpu_uuid,
        }
        backend = NvmlTelemetryBackend(index, expected_gpu_uuid=gpu_uuid)
        reading = backend.read()
        assert backend.gpu_uuid == gpu_uuid
        assert reading.compute_process_ids == (10_000 + index,)
        assert reading.device_used_vram_mib == float(index + 1)
        fingerprints.append(backend.hardware_fingerprint)

    assert handles == [0, 1, 2, 3]
    assert len(set(fingerprints)) == 4


def test_child_gpu_assignment_rejects_missing_malformed_and_mismatched_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfseer_v3.dataset_pack.label_worker import (
        LabelWorkerError,
        _physical_assignment,
    )
    from perfseer_v3.dataset_pack.supervisor import SupervisorError, child_gpu_environment
    from perfseer_v3.dataset_pack.v100_runner import NvmlTelemetryBackend, V100RunError
    import perfseer_v3.dataset_pack.v100_runner as runner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("PERFSEER_ASSIGNED_GPU_UUID", raising=False)
    with pytest.raises(LabelWorkerError, match="numeric CUDA_VISIBLE_DEVICES"):
        _physical_assignment()
    for value in ("0,1", "GPU-fixture", "-1", " 1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(LabelWorkerError, match="numeric CUDA_VISIBLE_DEVICES"):
            _physical_assignment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(LabelWorkerError, match="physical GPU UUID"):
        _physical_assignment()
    monkeypatch.setenv("PERFSEER_ASSIGNED_GPU_UUID", "GPU-fixture-3")
    assert _physical_assignment() == (3, "GPU-fixture-3")

    with pytest.raises(SupervisorError, match="index"):
        child_gpu_environment(SimpleNamespace(physical_index=-1, gpu_uuid="GPU-fixture"))
    with pytest.raises(SupervisorError, match="UUID"):
        child_gpu_environment(SimpleNamespace(physical_index=0, gpu_uuid="bad uuid"))

    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetName=lambda _handle: "NVIDIA A10",
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(total=24 * 1024**3),
        nvmlDeviceGetUUID=lambda _handle: "GPU-observed",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(V100RunError, match="UUID differs"):
        NvmlTelemetryBackend(2, expected_gpu_uuid="GPU-assigned")
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(V100RunError, match="exactly one CUDA-visible"):
        NvmlTelemetryBackend(2, expected_gpu_uuid="GPU-observed")


def test_four_gpu_batch_rejects_duplicate_probe_assignments() -> None:
    from perfseer_v3.dataset_pack.a10_campaign import (
        A10CampaignError,
        run_isolated_worker_batch,
    )

    roots = tuple(SimpleNamespace(candidate_id=str(index)) for index in range(2))
    probes = (
        SimpleNamespace(gpu_uuid="GPU-duplicate"),
        SimpleNamespace(gpu_uuid="GPU-duplicate"),
    )
    with pytest.raises(A10CampaignError, match="more than once"):
        run_isolated_worker_batch(roots, probes, lambda _root, _probe: None)


def test_nomad_geometry_decoder_accepts_conventional_and_ase_forms() -> None:
    from perfseer_v3.dataset_pack.real_data import (
        RealPreparedDataError,
        _geometry_nodes,
    )

    conventional = _geometry_nodes(b"2\nfixture\nAl 0 1 2\nO 3 4 5\n")
    ase = _geometry_nodes(
        b"# ASE\nlattice_vector 1 0 0\natom 0 1 2 Al\natom_frac 0.5 0.5 0.5 O\n"
    )
    assert conventional.shape == ase.shape == (2, 6)
    assert conventional[:, 1:4].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert ase[:, 1:4].tolist() == [[0.0, 1.0, 2.0], [0.5, 0.5, 0.5]]
    with pytest.raises(RealPreparedDataError, match="atom row"):
        _geometry_nodes(b"# ASE\natom 0 1 Al\n")
    with pytest.raises(RealPreparedDataError, match="invalid atom count"):
        _geometry_nodes(b"# ASE\nlattice_vector 1 0 0\n")


def test_inventory_and_archive_source_lock_drift_fail_closed(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.disaster_substitution import EXPECTED_INVENTORY_SHA256
    from perfseer_v3.dataset_pack.materialization import (
        MATERIALIZATION_STATE_VERSION,
        TaskMaterializationError,
        TaskMaterializationState,
        TaskMaterializer,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = next(row for row in load_task_registry().entries if row.task_id == "disaster-tweets")
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
        remote_inventory_sha256=EXPECTED_INVENTORY_SHA256,
        archive_sha256="b" * 64,
        archive_inventory_sha256="c" * 64,
    )
    first = materializer._freeze_disaster_source_lock(
        entry, state, SimpleNamespace(archive_sha256="b" * 64)
    )
    assert first is not None
    with pytest.raises(TaskMaterializationError):
        materializer._freeze_disaster_source_lock(
            entry, state, SimpleNamespace(archive_sha256="d" * 64)
        )
    drifted = TaskMaterializationState(
        **{**state.__dict__, "remote_inventory_sha256": "e" * 64}
    )
    with pytest.raises(TaskMaterializationError):
        materializer._freeze_disaster_source_lock(
            entry, drifted, SimpleNamespace(archive_sha256="b" * 64)
        )


def test_live_inventory_repairs_stale_extraction_hint_without_identity_change() -> None:
    from perfseer_v3.dataset_pack.kaggle import (
        KaggleCompetitionProbe,
        KaggleRemoteFile,
    )
    from perfseer_v3.dataset_pack.materialization import (
        live_archive_extraction_ceiling,
    )
    from perfseer_v3.dataset_pack.task_registry import load_task_registry

    entry = next(
        row for row in load_task_registry().entries if row.task_id == "random-acts-of-pizza"
    )
    assert entry.maximum_extracted_bytes == 12_000_000
    probe = KaggleCompetitionProbe(
        entry.kaggle_slug,
        (KaggleRemoteFile("random-acts-of-pizza.zip", 4_973_912, "frozen"),),
    )
    ceiling = live_archive_extraction_ceiling(entry, probe)
    assert ceiling == 4_973_912 * 16
    assert ceiling > 17_967_936


def test_disaster_record_binds_predecessor_and_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from perfseer_v3.dataset_pack import supervisor
    from perfseer_v3.dataset_pack.fingerprints import canonical_sha256

    candidate_id = "1" * 64
    predecessor_id = "2" * 64
    state = tmp_path / "state"
    state.mkdir()
    lineage = {
        "v1_candidate_id": predecessor_id,
        "v2_candidate_id": candidate_id,
        "row_classification": "dataset_substitution",
        "old_task_id": "detecting-insults",
        "new_task_id": "disaster-tweets",
        "old_semantic_signature": "3" * 64,
        "new_semantic_signature": "4" * 64,
        "new_compute_signature": "5" * 64,
    }
    (state / "a10_disaster_crosswalk.jsonl").write_text(json.dumps(lineage) + "\n")
    task_cache = tmp_path / "task_cache"
    task_cache.mkdir()
    materialization = {
        "task_id": "disaster-tweets",
        "stage": "view_ready",
        "archive_sha256": "6" * 64,
        "remote_inventory_sha256": "7" * 64,
        "dataset_fingerprint": "8" * 64,
    }
    materialization["state_sha256"] = canonical_sha256(materialization)
    (task_cache / "materialization_state.json").write_text(json.dumps(materialization))
    locks = state / "source_locks"
    locks.mkdir()
    (locks / "disaster-tweets.json").write_text(
        json.dumps(
            {
                "archive_sha256": "6" * 64,
                "remote_inventory_sha256": "7" * 64,
                "lock_sha256": "9" * 64,
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

    monkeypatch.setattr(
        supervisor,
        "environment_provenance",
        lambda: {
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "dependency_lock_sha256": "c" * 64,
            "image_identity": "image@sha256:" + "d" * 64,
        },
    )
    monkeypatch.setattr(
        supervisor, "semantic_distribution_signature", lambda _: "e" * 64
    )
    bound = supervisor._bind_native_a10_provenance(
        FakeRecord(SimpleNamespace(dataset_sha256="8" * 64)),
        SimpleNamespace(candidate_id=candidate_id, mutation_specification={}),
        SimpleNamespace(task_id="disaster-tweets"),
        tmp_path,
    )
    provenance = bound.reference_provenance
    assert provenance["predecessor_candidate_id"] == predecessor_id
    assert provenance["v2_root_candidate_id"] == candidate_id
    assert provenance["dataset_substitution"] is True
    assert provenance["source_archive_sha256"] == "6" * 64
    assert provenance["remote_inventory_sha256"] == "7" * 64
    assert provenance["disaster_source_lock_sha256"] == "9" * 64
    assert provenance["speech_source_lock_sha256"] is None


def test_pilot_lineage_workspace_and_offline_job_contract(tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.a10_campaign import (
        A10CampaignError,
        _assert_workspace_generation,
        pilot_candidates,
    )

    pilot = pilot_candidates()
    assert len(pilot) == 32
    assert sum(row.task_id == "disaster-tweets" for row in pilot) == 1
    old = yaml.safe_load((ROOT / "src/perfseer_v3/registries/native_a10_nonvision_pilot_v1.yaml").read_text())
    new = yaml.safe_load((ROOT / "src/perfseer_v3/registries/native_a10_nonvision_disaster_pilot_v2.yaml").read_text())
    assert sum(left == right for left, right in zip(old["candidate_ids"], new["candidate_ids"], strict=True)) == 27
    state = tmp_path / "old" / "state"
    state.mkdir(parents=True)
    (state / "a10_nonvision_campaign_contract.json").write_text("{}")
    with pytest.raises(A10CampaignError, match="older A10 workspace"):
        _assert_workspace_generation(tmp_path / "old")

    path = ROOT / "scripts/render_a10_disaster_v2_nautilus_job.py"
    spec = importlib.util.spec_from_file_location("render_disaster_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(module)
    image = "gitlab-registry.nrp-nautilus.io/group/project@sha256:" + "b" * 64
    output = tmp_path / "pilot.yaml"
    assert module.main([
        "--output", str(output), "--namespace", "test-ns", "--image", image,
        "--source-revision", "a" * 40, "--pvc", "disaster-v2", "--secret",
        "kaggle-disaster", "--mode", "pilot",
    ]) == 0
    value = yaml.safe_load(output.read_text())
    pod = value["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert (
        value["metadata"]["name"]
        == "perfseer-v3-a10-nonvision-disaster-v2-nvml-v1-pilot"
    )
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
    assert container["args"][container["args"].index("--workspace") + 1].endswith(
        "disaster-v2-nvml-v1"
    )
    assert pod["initContainers"][0]["name"] == "stage-kaggle-credential"
    assert pod["initContainers"][0]["image"] == image
    volumes = {row["name"]: row for row in pod["volumes"]}
    assert volumes["kaggle-credential-source"]["secret"]["secretName"] == "kaggle-disaster"
    assert volumes["kaggle-credential-private"]["emptyDir"]["medium"] == "Memory"


def test_image_gate_order_and_active_runbook() -> None:
    dockerfile = (ROOT / "containers/a10-nonvision-disaster-v2-labeler/Dockerfile").read_text()
    cli = (ROOT / "scripts/run_perfseer_v3_a10_labeling.py").read_text()
    runbook = (ROOT / "NAUTILUS_SUBMISSION.md").read_text()
    assert "native_a10_nonvision_disaster_v2" in dockerfile
    assert "COPY kaggle.json" not in dockerfile
    assert '("disaster-tweets", "tensorflow-speech-yes-no")' in cli
    assert 'disaster.get("downloaded_bytes") != 22_746' in cli
    assert "ICML 2013 Whale Challenge" in runbook
    assert "Detecting Insults in Social Commentary" in runbook
    assert "tensorflow-speech-recognition-challenge/rules" in runbook
    assert "nlp-getting-started/rules" in runbook
    assert "detecting-insults-in-social-commentary/rules" not in runbook
    assert "the-icml-2013-whale-challenge-right-whale-redux/rules" not in runbook
    assert "scripts/render_a10_disaster_v2_nautilus_job.py" in runbook
    assert "docker push" in runbook and "kubectl apply" in runbook and "nohup" in runbook
