from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from perfseer_v3.dataset_pack.finalization import (
    FinalizationError,
    _resolve_workspace_slots,
)
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256, file_sha256
from perfseer_v3.dataset_pack.materialization import (
    TASK_COMPLETION_VERSION,
    TASK_LOOP_STATE_VERSION,
    TaskCompletionReceipt,
    TaskLoopState,
    freeze_initial_target_manifest,
)
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.sharding import (
    EXPECTED_SHARD_ROOT_COUNTS,
    EXPECTED_SHARD_TASK_COUNTS,
    PRODUCTION_SOURCE_LOCK_VERSION,
    SHARD_IDS,
    DurableArtifact,
    ProductionSourceLock,
    ShardContract,
    ShardError,
    ShardTaskLoopState,
    _FINAL_ARTIFACT_PATHS,
    _assert_merged_workspace_allowlist,
    _copy_artifact,
    _direct_json_files,
    _merge_copy_plan,
    build_shard_contract,
    finalize_shard_workspace,
    freeze_shard_contract,
    initial_shard_task_loop,
    merge_shard_workspaces,
    save_shard_task_loop,
    validate_shard_partition,
)
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.dataset_pack.task_registry import load_task_registry
from perfseer_v3.dataset_pack.workflow import (
    WorkflowError,
    _pilot_candidate_order,
    _record_indexes,
    run_task_workflow,
)
from tests.test_perfseer_v3_dataset_finalization import _accepted, _failed_oom


def _receipt(task_id: str, manifest_sha: str) -> TaskCompletionReceipt:
    resolutions = (("1" * 64, "2" * 64),)
    hashes = ("3" * 64,)
    records_manifest = canonical_sha256(
        {"resolutions": resolutions, "accepted_record_sha256s": hashes}
    )
    draft = TaskCompletionReceipt(
        version=TASK_COMPLETION_VERSION,
        task_id=task_id,
        target_manifest_sha256=manifest_sha,
        dataset_fingerprint="4" * 64,
        resolutions=resolutions,
        accepted_record_sha256s=hashes,
        records_manifest_sha256=records_manifest,
        receipt_sha256="",
    )
    return replace(draft, receipt_sha256=canonical_sha256(draft.unhashed_payload()))


def _source_lock() -> ProductionSourceLock:
    files = (("src/fixture.py", 1, "a" * 64),)
    draft = ProductionSourceLock(
        version=PRODUCTION_SOURCE_LOCK_VERSION,
        git_commit="b" * 40,
        file_manifest=files,
        file_manifest_sha256=canonical_sha256(files),
        source_lock_sha256="",
    )
    return replace(
        draft,
        source_lock_sha256=canonical_sha256(draft.unhashed_payload()),
    )


class DatasetShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_target_manifest()
        cls.tasks = load_task_registry()
        cls.contracts = {
            shard_id: build_shard_contract(cls.manifest, shard_id)
            for shard_id in SHARD_IDS
        }

    def test_contracts_are_exact_task_and_root_partition(self) -> None:
        validate_shard_partition(self.manifest, tuple(self.contracts.values()))
        for shard_id, contract in self.contracts.items():
            self.assertEqual(contract.task_count, EXPECTED_SHARD_TASK_COUNTS[shard_id])
            self.assertEqual(contract.root_candidate_count, EXPECTED_SHARD_ROOT_COUNTS[shard_id])
        root_by_id = {row.candidate_id: row for row in self.manifest.candidates}
        self.assertTrue(
            all(
                root_by_id[root_id].source_modality == "nlp"
                for root_id in self.contracts["nlp"].root_candidate_ids
            )
        )
        self.assertTrue(
            all(
                root_by_id[root_id].source_modality == "vision"
                for root_id in self.contracts["vision"].root_candidate_ids
            )
        )
        self.assertEqual(
            {
                root_by_id[root_id].source_modality
                for root_id in self.contracts["rest"].root_candidate_ids
            },
            {"audio", "graph", "tabular"},
        )

    def test_contract_round_trip_and_tamper_rejection(self) -> None:
        contract = self.contracts["nlp"]
        payload = contract.to_dict(self.manifest, self.tasks)
        self.assertEqual(
            ShardContract.from_dict(payload, self.manifest, self.tasks), contract
        )
        payload = dict(payload)
        payload["shard_id"] = "vision"
        with self.assertRaisesRegex(ShardError, "identity|projection"):
            ShardContract.from_dict(payload, self.manifest, self.tasks)

    def test_wrong_shard_resume_is_rejected_from_empty_loop(self) -> None:
        nlp = self.contracts["nlp"]
        vision = self.contracts["vision"]
        state = initial_shard_task_loop(nlp)
        payload = state.to_dict(nlp)
        with self.assertRaisesRegex(ShardError, "contract"):
            ShardTaskLoopState.from_dict(payload, vision)
        self.assertEqual(state.select_next(nlp.task_ids), nlp.task_ids[0])
        state = state.begin(nlp.task_ids[0], nlp.task_ids)
        state = ShardTaskLoopState.from_dict(state.to_dict(nlp), nlp)
        self.assertEqual(state.active_task_id, nlp.task_ids[0])
        receipt = _receipt(nlp.task_ids[0], self.manifest.sha256)
        state = state.complete(nlp.task_ids[0], receipt, nlp.task_ids)
        state.validate(nlp)
        self.assertEqual(state.completed_task_ids, (nlp.task_ids[0],))

    def test_shard_contract_refuses_group_switch_and_legacy_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = freeze_shard_contract(root, self.manifest, "nlp")
            self.assertEqual(
                freeze_shard_contract(root, self.manifest, "nlp"), first
            )
            with self.assertRaisesRegex(ShardError, "requested shard"):
                freeze_shard_contract(root, self.manifest, "vision")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "state" / "task_loop.json", {"legacy": True})
            with self.assertRaisesRegex(ShardError, "unsharded workspace"):
                freeze_shard_contract(root, self.manifest, "nlp")

    def test_subset_slot_reconstruction_remains_root_local(self) -> None:
        root_candidate = self.manifest.candidates[0]
        task = next(row for row in self.tasks.entries if row.task_id == root_candidate.task_id)
        record = _accepted(root_candidate, task)
        with tempfile.TemporaryDirectory() as directory:
            resolved = _resolve_workspace_slots(
                Path(directory),
                self.manifest,
                {root_candidate.candidate_id: root_candidate.candidate_id},
                {root_candidate.candidate_id: record},
                (),
                root_candidates=(root_candidate,),
            )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].root_candidate, root_candidate)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FinalizationError, "selected frozen"):
                _resolve_workspace_slots(
                    Path(directory),
                    self.manifest,
                    {},
                    {},
                    (),
                    root_candidates=(root_candidate,),
                )

    def test_completed_shard_workflow_calls_only_shard_finalizer(self) -> None:
        contract = self.contracts["nlp"]
        loop = replace(
            initial_shard_task_loop(contract),
            completed_task_ids=contract.task_ids,
            completed_receipt_sha256s=tuple("a" * 64 for _ in contract.task_ids),
        )
        loop.validate(contract)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (
                patch(
                    "perfseer_v3.dataset_pack.workflow.freeze_initial_target_manifest",
                    return_value=(self.manifest, workspace / "manifests/initial_targets.jsonl"),
                ),
                patch(
                    "perfseer_v3.dataset_pack.sharding.freeze_shard_contract",
                    return_value=contract,
                ),
                patch("perfseer_v3.dataset_pack.sharding.freeze_production_source_lock"),
                patch(
                    "perfseer_v3.dataset_pack.sharding.load_shard_task_loop",
                    return_value=loop,
                ),
                patch("perfseer_v3.dataset_pack.workflow.TaskMaterializer"),
                patch("perfseer_v3.dataset_pack.workflow.KaggleCliClient"),
                patch("perfseer_v3.dataset_pack.workflow.PinnedMleBenchPreparer"),
                patch("perfseer_v3.dataset_pack.workflow._verify_completed_prefix"),
                patch("perfseer_v3.dataset_pack.workflow.discover_a10g_probes", return_value=()),
                patch("perfseer_v3.dataset_pack.workflow.lock_campaign_environment"),
                patch("perfseer_v3.dataset_pack.sharding.finalize_shard_workspace") as finalize,
                patch("perfseer_v3.dataset_pack.finalization.finalize_workspace") as full_finalize,
            ):
                run_task_workflow(
                    workspace=workspace,
                    repository_root=ROOT,
                    mlebench_checkout=ROOT,
                    task_group="nlp",
                )
            finalize.assert_called_once_with(workspace.resolve(), ROOT)
            full_finalize.assert_not_called()

    def test_unsharded_workflow_rejects_shard_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            atomic_write_json(workspace / "state/shard_contract.json", {"shard": "nlp"})
            with patch(
                "perfseer_v3.dataset_pack.workflow.freeze_initial_target_manifest",
                return_value=(self.manifest, workspace / "manifests/initial_targets.jsonl"),
            ):
                with self.assertRaisesRegex(WorkflowError, "cannot resume"):
                    run_task_workflow(
                        workspace=workspace,
                        repository_root=ROOT,
                        mlebench_checkout=ROOT,
                    )

    def test_pilot_limit_requires_a_positive_labeling_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaisesRegex(WorkflowError, "positive integer"):
                run_task_workflow(
                    workspace=workspace,
                    repository_root=ROOT,
                    mlebench_checkout=ROOT,
                    max_new_accepted=0,
                )
            with self.assertRaisesRegex(WorkflowError, "materialize-only"):
                run_task_workflow(
                    workspace=workspace,
                    repository_root=ROOT,
                    mlebench_checkout=ROOT,
                    materialize_only=True,
                    max_new_accepted=1,
                )

    def test_pilot_limit_returns_only_after_complete_worker_batch(self) -> None:
        first_task_id = self.tasks.entries[0].task_id
        candidates = tuple(
            row for row in self.manifest.candidates if row.task_id == first_task_id
        )[:2]
        self.assertEqual(len(candidates), 2)
        task = self.tasks.entries[0]
        loop = TaskLoopState(
            version=TASK_LOOP_STATE_VERSION,
            target_manifest_sha256=self.manifest.sha256,
            task_registry_sha256=self.tasks.sha256,
            completed_task_ids=(),
            completed_receipt_sha256s=(),
            active_task_id=None,
        )
        loop.validate(tuple(row.task_id for row in self.tasks.entries))
        fake_manifest = SimpleNamespace(
            candidates=candidates,
            sha256=self.manifest.sha256,
        )
        materialized = SimpleNamespace(
            view_manifest=object(),
            public=Path("/fixture/public"),
            prepared_view=Path("/fixture/prepared"),
            inventory=SimpleNamespace(archive_sha256="a" * 64),
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)

            def accept(candidate, *_args, **_kwargs):
                record = _accepted(candidate, task)
                atomic_write_json(
                    workspace
                    / "attempts/accepted"
                    / f"{record.configuration_id}.json",
                    asdict(record),
                )
                return record

            supervisor = SimpleNamespace(run=accept)
            with (
                patch(
                    "perfseer_v3.dataset_pack.workflow.freeze_initial_target_manifest",
                    return_value=(
                        fake_manifest,
                        workspace / "manifests/initial_targets.jsonl",
                    ),
                ),
                patch(
                    "perfseer_v3.dataset_pack.workflow.load_task_loop_state",
                    return_value=loop,
                ),
                patch(
                    "perfseer_v3.dataset_pack.workflow.TaskMaterializer"
                ) as materializer_type,
                patch("perfseer_v3.dataset_pack.workflow.KaggleCliClient"),
                patch("perfseer_v3.dataset_pack.workflow.PinnedMleBenchPreparer"),
                patch("perfseer_v3.dataset_pack.workflow._verify_completed_prefix"),
                patch("perfseer_v3.dataset_pack.workflow.lock_campaign_environment"),
                patch(
                    "perfseer_v3.dataset_pack.workflow.AttemptSupervisor",
                    return_value=supervisor,
                ),
            ):
                materializer_type.return_value.materialize.return_value = materialized
                run_task_workflow(
                    workspace=workspace,
                    repository_root=ROOT,
                    mlebench_checkout=ROOT,
                    probes=(object(), object()),
                    max_new_accepted=1,
                )
            accepted, _ = _record_indexes(workspace)
            self.assertEqual(set(accepted), {row.candidate_id for row in candidates})
            materializer_type.return_value.cleanup_completed_task.assert_not_called()
            self.assertFalse(
                (workspace / "state/completed_materializations" / f"{first_task_id}.json").exists()
            )

    def test_pilot_prefix_covers_available_execution_factors(self) -> None:
        root_by_id = {row.candidate_id: row for row in self.manifest.candidates}
        for shard_id in SHARD_IDS:
            contract = self.contracts[shard_id]
            task_id = contract.task_ids[0]
            candidates = tuple(
                root_by_id[root_id]
                for root_id in contract.root_candidate_ids
                if root_by_id[root_id].task_id == task_id
            )
            prefix = _pilot_candidate_order(candidates, 32)[:32]
            self.assertEqual(len(prefix), 32)
            factors = (
                lambda row: row.execution["mode"],
                lambda row: row.batch_plan.effective_tier,
                lambda row: row.precision_policy["policy_id"],
                lambda row: bool(row.activation_checkpointing.get("enabled", False)),
                lambda row: row.optimizer["name"],
                lambda row: row.scheduler["name"],
                lambda row: row.regime,
                lambda row: row.training_step_id,
            )
            for factor in factors:
                available = {factor(row) for row in candidates}
                if len(available) <= len(prefix):
                    self.assertEqual({factor(row) for row in prefix}, available)
            family_count = len({row.family_id for row in candidates})
            self.assertEqual(
                len({row.family_id for row in prefix}),
                min(32, family_count),
            )

    def test_merge_copy_collisions_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("one", "two")]
            relative = "provenance/environment/identity.json"
            for root in roots:
                path = root / relative
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
            artifact = DurableArtifact(relative, 3, file_sha256(roots[0] / relative))
            shards = tuple(
                SimpleNamespace(
                    workspace=root,
                    completion=SimpleNamespace(durable_artifacts=(artifact,)),
                )
                for root in roots
            )
            self.assertEqual(len(_merge_copy_plan(shards)), 1)
            accepted_relative = "attempts/accepted/same.json"
            for root in roots:
                path = root / accepted_relative
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
            accepted = DurableArtifact(
                accepted_relative, 3, file_sha256(roots[0] / accepted_relative)
            )
            colliding = tuple(
                SimpleNamespace(
                    workspace=root,
                    completion=SimpleNamespace(durable_artifacts=(accepted,)),
                )
                for root in roots
            )
            with self.assertRaisesRegex(ShardError, "non-deduplicable"):
                _merge_copy_plan(colliding)
            (roots[1] / relative).write_text('{"changed":true}\n', encoding="utf-8")
            conflicting = (
                shards[0],
                SimpleNamespace(
                    workspace=roots[1],
                    completion=SimpleNamespace(
                        durable_artifacts=(
                            DurableArtifact(
                                relative,
                                (roots[1] / relative).stat().st_size,
                                file_sha256(roots[1] / relative),
                            ),
                        )
                    ),
                ),
            )
            with self.assertRaisesRegex(ShardError, "conflicting"):
                _merge_copy_plan(conflicting)

    def test_merge_allowlist_permits_only_resumable_final_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = ("state/merge_journal.json",)
            atomic_write_json(root / expected[0], {"journal": True})
            atomic_write_json(root / "state/merge_receipt.json", {"receipt": True})
            partial = next(iter(_FINAL_ARTIFACT_PATHS))
            (root / partial).parent.mkdir(parents=True, exist_ok=True)
            (root / partial).write_text("partial", encoding="utf-8")
            _assert_merged_workspace_allowlist(root, expected, finalized=False)
            (root / "unexpected.txt").write_text("no", encoding="utf-8")
            with self.assertRaisesRegex(ShardError, "allowlist"):
                _assert_merged_workspace_allowlist(root, expected, finalized=False)

    def test_merge_copy_is_atomic_idempotent_and_conflict_detecting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            destination = root / "merged" / "artifact.json"
            source.write_text('{"stable":true}\n', encoding="utf-8")
            artifact = DurableArtifact(
                "merged/artifact.json", source.stat().st_size, file_sha256(source)
            )
            _copy_artifact(source, destination, artifact)
            first = destination.read_bytes()
            _copy_artifact(source, destination, artifact)
            self.assertEqual(destination.read_bytes(), first)
            destination.write_text('{"different":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ShardError, "destination conflicts"):
                _copy_artifact(source, destination, artifact)

    def test_durable_semantic_directories_reject_nested_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            nested = workspace / "attempts/accepted/unreferenced/nested.json"
            atomic_write_json(nested, {"unreferenced": True})
            with self.assertRaisesRegex(ShardError, "direct JSON"):
                _direct_json_files(workspace, "attempts/accepted")

    def test_terminal_record_resume_removes_accepted_and_oom_transients(self) -> None:
        accepted_candidate, failed_candidate = self.manifest.candidates[:2]
        task_by_id = {row.task_id: row for row in self.tasks.entries}
        accepted = _accepted(accepted_candidate, task_by_id[accepted_candidate.task_id])
        failed = _failed_oom(failed_candidate, task_by_id[failed_candidate.task_id])
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            atomic_write_json(
                workspace / "attempts/accepted" / f"{accepted.configuration_id}.json",
                asdict(accepted),
            )
            atomic_write_json(
                workspace / "attempts/failed" / f"{failed.run_id}.json",
                asdict(failed),
            )
            for configuration_id in (
                accepted.configuration_id,
                failed.configuration_id,
            ):
                atomic_write_json(
                    workspace / "attempts/staging" / f"{configuration_id}.json",
                    {"stale": True},
                )
                atomic_write_json(
                    workspace / "state/dispatch" / f"{configuration_id}.json",
                    {"stale": True},
                )
            accepted_index, failure_index = _record_indexes(workspace)
            self.assertEqual(set(accepted_index), {accepted.configuration_id})
            self.assertEqual(set(failure_index), {failed.configuration_id})
            self.assertFalse(any((workspace / "attempts/staging").glob("*.json")))
            self.assertFalse(any((workspace / "state/dispatch").glob("*.json")))

    def _build_completed_shards(
        self,
        parent: Path,
        source_lock: ProductionSourceLock,
    ) -> dict[str, Path]:
        environment = {"environment": "fixture"}
        environment_sha = canonical_sha256(environment)
        hardware = {
            "target_hardware_id": "nvidia_a10g_24gb_aws_g5",
            "name": "NVIDIA A10G",
            "compute_capability": [8, 6],
            "total_memory_bytes": 24 * 1024**3,
        }
        hardware_sha = canonical_sha256(hardware)
        task_by_id = {row.task_id: row for row in self.tasks.entries}
        root_by_id = {row.candidate_id: row for row in self.manifest.candidates}
        workspaces: dict[str, Path] = {}
        for shard_id in SHARD_IDS:
            workspace = parent / shard_id
            manifest, _ = freeze_initial_target_manifest(workspace)
            self.assertEqual(manifest.sha256, self.manifest.sha256)
            contract = freeze_shard_contract(workspace, manifest, shard_id)
            atomic_write_json(
                workspace / "state/production_source.json",
                source_lock.to_dict(),
            )
            atomic_write_json(
                workspace / "state/campaign_environment.json",
                {
                    "version": "perfseer_v3_a10g_campaign_environment_v1",
                    "environment": environment,
                    "environment_sha256": environment_sha,
                },
            )
            atomic_write_json(
                workspace / "provenance/environment" / f"{environment_sha}.json",
                environment,
            )
            atomic_write_json(
                workspace / "provenance/hardware" / f"{hardware_sha}.json",
                hardware,
            )
            receipt_hashes = []
            for task_id in contract.task_ids:
                task = task_by_id[task_id]
                candidates = tuple(
                    root_by_id[root_id]
                    for root_id in contract.root_candidate_ids
                    if root_by_id[root_id].task_id == task_id
                )
                resolutions = []
                record_hashes = []
                for candidate in candidates:
                    record = _accepted(candidate, task)
                    record = replace(
                        record,
                        fingerprints=replace(
                            record.fingerprints,
                            hardware_sha256=hardware_sha,
                        ),
                    )
                    record.validate_against_configuration(candidate, task)
                    record_sha = canonical_sha256(asdict(record))
                    atomic_write_json(
                        workspace
                        / "attempts/accepted"
                        / f"{record.configuration_id}.json",
                        asdict(record),
                    )
                    resolutions.append(
                        (candidate.candidate_id, record.configuration_id)
                    )
                    record_hashes.append(record_sha)
                receipt_payload = {
                    "resolutions": tuple(resolutions),
                    "accepted_record_sha256s": tuple(record_hashes),
                }
                draft = TaskCompletionReceipt(
                    version=TASK_COMPLETION_VERSION,
                    task_id=task_id,
                    target_manifest_sha256=manifest.sha256,
                    dataset_fingerprint=canonical_sha256({"task_id": task_id}),
                    resolutions=tuple(resolutions),
                    accepted_record_sha256s=tuple(record_hashes),
                    records_manifest_sha256=canonical_sha256(receipt_payload),
                    receipt_sha256="",
                )
                receipt = replace(
                    draft,
                    receipt_sha256=canonical_sha256(draft.unhashed_payload()),
                )
                receipt.validate()
                atomic_write_json(
                    workspace
                    / "state/completed_materializations"
                    / f"{task_id}.json",
                    receipt.to_dict(),
                )
                receipt_hashes.append(receipt.receipt_sha256)
            loop = replace(
                initial_shard_task_loop(contract),
                completed_task_ids=contract.task_ids,
                completed_receipt_sha256s=tuple(receipt_hashes),
            )
            save_shard_task_loop(
                workspace / "state/shard_task_loop.json",
                loop,
                contract,
            )
            finalize_shard_workspace(workspace, ROOT)
            workspaces[shard_id] = workspace
        return workspaces

    def test_real_three_shard_finalize_interrupted_merge_and_verify(self) -> None:
        source_lock = _source_lock()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with patch(
                "perfseer_v3.dataset_pack.sharding.build_production_source_lock",
                return_value=source_lock,
            ):
                workspaces = self._build_completed_shards(parent, source_lock)
                completion_path = workspaces["nlp"] / "state/shard_completion.json"
                completion_bytes = completion_path.read_bytes()
                completion_path.write_text("{}\n", encoding="utf-8")
                with self.assertRaises(ShardError):
                    merge_shard_workspaces(
                        workspaces,
                        parent / "merged",
                        ROOT,
                    )
                completion_path.write_bytes(completion_bytes)

                environment_path = (
                    workspaces["vision"] / "state/campaign_environment.json"
                )
                environment_bytes = environment_path.read_bytes()
                atomic_write_json(
                    environment_path,
                    {
                        "version": "perfseer_v3_a10g_campaign_environment_v1",
                        "environment": {"environment": "drifted"},
                        "environment_sha256": canonical_sha256(
                            {"environment": "drifted"}
                        ),
                    },
                )
                with self.assertRaises(ShardError):
                    merge_shard_workspaces(
                        workspaces,
                        parent / "merged",
                        ROOT,
                    )
                environment_path.write_bytes(environment_bytes)

                output = parent / "merged"
                with patch(
                    "perfseer_v3.dataset_pack.finalization.finalize_workspace",
                    side_effect=RuntimeError("simulated merge interruption"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        merge_shard_workspaces(workspaces, output, ROOT)
                self.assertFalse(output.exists())
                staging = tuple(parent.glob(".merged.merge-*"))
                self.assertEqual(len(staging), 1)

                receipt = merge_shard_workspaces(workspaces, output, ROOT)
                self.assertTrue(output.is_dir())
                self.assertFalse(staging[0].exists())
                self.assertEqual(
                    receipt,
                    merge_shard_workspaces(
                        workspaces,
                        output,
                        ROOT,
                        verify_only=True,
                    ),
                )
                audit = json.loads(
                    (output / "final/audit_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(audit["accepted_run_count"], 18_000)
                self.assertEqual(audit["measured_epoch_record_count"], 54_000)

    def test_cli_exposes_explicit_shard_and_merge_interfaces(self) -> None:
        runner = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_a10g_18k_pack.py"), "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )
        merger = subprocess.run(
            [sys.executable, str(ROOT / "scripts/merge_a10g_18k_shards.py"), "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertIn("--task-group {nlp,vision,rest}", runner.stdout)
        self.assertIn("--max-new-accepted MAX_NEW_ACCEPTED", runner.stdout)
        self.assertEqual(merger.returncode, 0, merger.stderr)
        for option in (
            "--nlp-workspace",
            "--vision-workspace",
            "--rest-workspace",
            "--output-workspace",
            "--verify-only",
        ):
            self.assertIn(option, merger.stdout)


if __name__ == "__main__":
    unittest.main()
