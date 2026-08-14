from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "native_a10_nonvision_disaster_v2"
if os.environ.get("PERFSEER_LABELER_PROFILE") != PROFILE:
    pytest.skip(
        "continuous tests require PERFSEER_LABELER_PROFILE=" + PROFILE,
        allow_module_level=True,
    )
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from perfseer_v3.dataset_pack.a10_continuous import (
    A10ContinuousError,
    CONTINUOUS_CHUNK_COUNT,
    CONTINUOUS_PILOT_LABELS,
    CONTINUOUS_TOTAL_LABELS,
    ContinuousDependencies,
    continuous_progress_path,
    continuous_receipt_path,
    run_continuous_campaign,
)
from perfseer_v3.dataset_pack.fingerprints import file_sha256


REVISION = "a" * 40
DIGEST = "registry.example/project@sha256:" + "b" * 64


class FakePhases:
    def __init__(
        self,
        workspace: Path,
        *,
        fail_phase: str | None = None,
        preexisting: set[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.fail_phase = fail_phase
        self.completed = set(preexisting or ())
        self.calls: list[str] = []

    def preflight(self) -> None:
        self.calls.append("preflight")

    def run_phase(self, **arguments: Any) -> dict[str, Any]:
        phase = (
            "pilot"
            if arguments.get("pilot") is True
            else f"chunk-{arguments['chunk_index']:02d}"
        )
        self.calls.append(
            ("resume:" if phase in self.completed else "run:") + phase
        )
        if phase == self.fail_phase:
            raise RuntimeError(f"injected {phase} failure")
        self.completed.add(phase)
        ordinal = 1 if phase == "pilot" else int(phase.split("-")[1]) + 2
        return {"receipt_sha256": f"{ordinal:064x}"}

    def verify_campaign(self, _: Path, *, complete: bool) -> dict[str, Any]:
        self.calls.append("verify:complete" if complete else "verify:partial")
        accepted = CONTINUOUS_PILOT_LABELS if "pilot" in self.completed else 0
        accepted += sum(
            160 if index == CONTINUOUS_CHUNK_COUNT - 1 else 256
            for index in range(CONTINUOUS_CHUNK_COUNT)
            if f"chunk-{index:02d}" in self.completed
        )
        if complete and accepted != CONTINUOUS_TOTAL_LABELS:
            raise RuntimeError("incomplete fake campaign")
        return {
            "accepted_labels": accepted,
            "campaign_contract_sha256": "c" * 64,
        }

    def export_release(
        self, _: Path, __: Path, output_directory: Path, *, complete: bool
    ) -> dict[str, Any]:
        self.calls.append("export")
        output_directory.mkdir(parents=True, exist_ok=True)
        archive = output_directory / "fake-complete.tar.zst"
        archive.write_bytes(b"verified fake release")
        return {
            "archive": str(archive),
            "archive_sha256": file_sha256(archive),
            "release_sha256": "d" * 64,
            "label_count": CONTINUOUS_TOTAL_LABELS,
            "complete": complete,
        }

    def verify_release(self, archive: Path) -> dict[str, Any]:
        self.calls.append("verify-release")
        return {
            "archive_sha256": file_sha256(archive),
            "label_count": CONTINUOUS_TOTAL_LABELS,
        }

    def dependencies(self) -> ContinuousDependencies:
        return ContinuousDependencies(
            run_phase=self.run_phase,
            verify_campaign=self.verify_campaign,
            export_release=self.export_release,
            verify_release=self.verify_release,
        )


def _run(workspace: Path, fake: FakePhases) -> dict[str, Any]:
    return dict(
        run_continuous_campaign(
            workspace=workspace,
            repository_root=ROOT,
            mlebench_checkout=ROOT,
            repository_revision=REVISION,
            image_digest=DIGEST,
            output_directory=workspace / "releases",
            preflight=fake.preflight,
            dependencies=fake.dependencies(),
            event_sink=None,
        )
    )


def test_exact_order_and_verified_terminal_export(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    fake = FakePhases(workspace)
    receipt = _run(workspace, fake)

    expected = ["preflight", "run:pilot", "verify:partial"]
    for index in range(CONTINUOUS_CHUNK_COUNT):
        expected.extend([f"run:chunk-{index:02d}", "verify:partial"])
    expected.extend(["verify:complete", "export", "verify-release"])
    assert fake.calls == expected
    assert receipt["accepted_labels"] == CONTINUOUS_TOTAL_LABELS
    assert receipt["completed_chunks"] == CONTINUOUS_CHUNK_COUNT
    assert Path(receipt["archive"]).is_file()
    assert json.loads(continuous_receipt_path(workspace).read_text()) == receipt
    progress = json.loads(continuous_progress_path(workspace).read_text())
    assert progress["event"] == "campaign_complete"
    assert progress["details"]["archive_sha256"] == receipt["archive_sha256"]


@pytest.mark.parametrize("failed_phase", ["pilot", "chunk-03"])
def test_failed_phase_stops_later_work(tmp_path: Path, failed_phase: str) -> None:
    workspace = tmp_path / failed_phase
    fake = FakePhases(workspace, fail_phase=failed_phase)
    with pytest.raises(RuntimeError, match="injected"):
        _run(workspace, fake)

    progress = json.loads(continuous_progress_path(workspace).read_text())
    assert progress["event"] == "campaign_failed"
    assert progress["phase"] == failed_phase
    assert "export" not in fake.calls
    if failed_phase == "pilot":
        assert not any(call.startswith("run:chunk") for call in fake.calls)
    else:
        assert "run:chunk-04" not in fake.calls


def test_receipt_resume_and_terminal_fast_exit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    preexisting = {"pilot", *(f"chunk-{index:02d}" for index in range(6))}
    first = FakePhases(workspace, preexisting=preexisting)
    terminal = _run(workspace, first)
    assert first.calls[1:3] == ["resume:pilot", "verify:partial"]
    assert "resume:chunk-05" in first.calls
    assert "run:chunk-06" in first.calls

    second = FakePhases(
        workspace,
        preexisting={"pilot", *(f"chunk-{index:02d}" for index in range(44))},
        fail_phase="pilot",
    )
    returned = _run(workspace, second)
    assert returned == terminal
    assert second.calls == ["verify:complete", "verify-release"]


def test_corrupt_terminal_and_invalid_phase_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    receipt_path = continuous_receipt_path(workspace)
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text('{"receipt_sha256":"bad"}\n', encoding="utf-8")
    with pytest.raises(A10ContinuousError, match="receipt hash differs"):
        _run(workspace, FakePhases(workspace))

    receipt_path.unlink()
    fake = FakePhases(workspace)
    original = fake.run_phase

    def invalid_receipt(**arguments: Any) -> dict[str, Any]:
        result = original(**arguments)
        result["receipt_sha256"] = "z" * 64
        return result

    dependencies = fake.dependencies()
    dependencies = ContinuousDependencies(
        run_phase=invalid_receipt,
        verify_campaign=dependencies.verify_campaign,
        export_release=dependencies.export_release,
        verify_release=dependencies.verify_release,
    )
    with pytest.raises(A10ContinuousError, match="no valid receipt hash"):
        run_continuous_campaign(
            workspace=workspace,
            repository_root=ROOT,
            mlebench_checkout=ROOT,
            repository_revision=REVISION,
            image_digest=DIGEST,
            output_directory=workspace / "releases",
            dependencies=dependencies,
            event_sink=None,
        )


def test_release_must_remain_inside_workspace(tmp_path: Path) -> None:
    with pytest.raises(A10ContinuousError, match="under the workspace"):
        run_continuous_campaign(
            workspace=tmp_path / "workspace",
            repository_root=ROOT,
            mlebench_checkout=ROOT,
            repository_revision=REVISION,
            image_digest=DIGEST,
            output_directory=tmp_path / "elsewhere",
            dependencies=FakePhases(tmp_path).dependencies(),
            event_sink=None,
        )


def test_failed_archive_verification_never_writes_terminal_receipt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    fake = FakePhases(workspace)
    dependencies = fake.dependencies()

    def reject_archive(_: Path) -> dict[str, Any]:
        fake.calls.append("verify-release")
        raise RuntimeError("injected archive verification failure")

    dependencies = ContinuousDependencies(
        run_phase=dependencies.run_phase,
        verify_campaign=dependencies.verify_campaign,
        export_release=dependencies.export_release,
        verify_release=reject_archive,
    )
    with pytest.raises(RuntimeError, match="archive verification failure"):
        run_continuous_campaign(
            workspace=workspace,
            repository_root=ROOT,
            mlebench_checkout=ROOT,
            repository_revision=REVISION,
            image_digest=DIGEST,
            output_directory=workspace / "releases",
            preflight=fake.preflight,
            dependencies=dependencies,
            event_sink=None,
        )

    assert fake.calls[-2:] == ["export", "verify-release"]
    assert not continuous_receipt_path(workspace).exists()
    progress = json.loads(continuous_progress_path(workspace).read_text())
    assert progress["event"] == "campaign_failed"
    assert progress["phase"] == "export"


def test_continuous_job_contract_and_monitor_terminal_following(
    tmp_path: Path,
) -> None:
    path = ROOT / "scripts/render_a10_disaster_v2_nautilus_job.py"
    spec = importlib.util.spec_from_file_location("render_continuous", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "continuous.yaml"
    arguments = SimpleNamespace(
        template=None,
        output=output,
        namespace="ecepxie",
        image="gitlab-registry.nrp-nautilus.io/group/project@sha256:" + "e" * 64,
        source_revision="f" * 40,
        pvc="perfseer-panns-jingbin-260808-a0af09",
        secret="perfseer-kaggle-disaster-v2",
        mode="continuous",
        chunk_index=None,
    )
    rendered = module.render(arguments)
    module.verify_job(rendered, mode="continuous")
    value = yaml.safe_load(yaml.safe_dump(rendered))
    spec_value = value["spec"]
    pod = spec_value["template"]["spec"]
    container = pod["containers"][0]
    assert value["metadata"]["namespace"] == "ecepxie"
    assert spec_value["backoffLimit"] == 1
    assert spec_value["activeDeadlineSeconds"] == 604_800
    assert pod["terminationGracePeriodSeconds"] == 120
    assert container["args"][0] == "run-continuous"
    assert container["resources"]["limits"] == container["resources"]["requests"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    volumes = {row["name"]: row for row in pod["volumes"]}
    assert volumes["workspace"]["persistentVolumeClaim"]["claimName"] == arguments.pvc
    assert volumes["kaggle-credential"]["secret"]["secretName"] == arguments.secret

    monitor = (ROOT / "scripts/monitor_a10_nonvision_job.sh").read_text()
    assert "monitor_pods=" in monitor
    assert "Complete|Failed" in monitor
    assert "break" in monitor
