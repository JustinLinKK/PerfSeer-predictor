"""Exact audio/tabular/graph production campaign for four V100 workers."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fingerprints import canonical_sha256, canonical_value
from .modality_sharding import (
    LOGICAL_TARGET_HARDWARE_ID,
    MEASURED_EPOCHS_PER_LABEL,
    V100_FAMILY_ID,
    ModalityCompletion,
    ModalityShardError,
    build_modality_contract,
    candidates_for_modality,
    verify_modality_workspace,
)
from .storage import atomic_write_json
from .supervisor import (
    AttemptSupervisor,
    GpuProbe,
    discover_v100_probes,
    lock_campaign_environment,
)


CAMPAIGN_MODALITIES = ("audio", "tabular", "graph")
CAMPAIGN_COUNTS = {"audio": 1_300, "tabular": 950, "graph": 1_300}
CAMPAIGN_CANDIDATE_COUNT = 3_550
CAMPAIGN_MEASURED_EPOCH_COUNT = 10_650
CAMPAIGN_CONTRACT_VERSION = "perfseer_v3_v100_atg_campaign_contract_v1"
CAMPAIGN_COMPLETION_VERSION = "perfseer_v3_v100_atg_campaign_completion_v1"
PILOT_COMPLETION_VERSION = "perfseer_v3_v100_four_label_pilot_completion_v1"
PILOT_RULES = (
    ("panns_cnn14", "mlsp-2013-birds", "fp32_ieee"),
    ("m5_waveform_cnn", "mlsp-2013-birds", "fp16_grad_scaler"),
    ("tabtransformer", "tabular-playground-dec-2021", "fp16_grad_scaler"),
    ("cgcnn", "nomad2018", "fp16_grad_scaler"),
)


class CampaignError(RuntimeError):
    """Raised when the production campaign is not an exact frozen projection."""


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign artifact {path}") from error
    if not isinstance(value, Mapping):
        raise CampaignError("campaign artifact must be a JSON object")
    return value


@dataclass(frozen=True)
class CampaignContract:
    version: str
    hardware_family_id: str
    logical_target_hardware_id: str
    target_manifest_sha256: str
    task_registry_sha256: str
    modalities: tuple[str, ...]
    modality_contract_sha256s: Mapping[str, str]
    task_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_ids_sha256: str
    candidate_count: int
    measured_epoch_count: int
    modality_counts: Mapping[str, int]
    family_counts: Mapping[str, int]
    contract_sha256: str

    def unhashed_payload(self) -> Mapping[str, Any]:
        value = canonical_value(asdict(self))
        value.pop("contract_sha256")
        return value

    def validate(self) -> None:
        expected = _build_campaign_contract_unchecked()
        if self.unhashed_payload() != expected.unhashed_payload():
            raise CampaignError("campaign contract differs from the frozen projection")
        if self.contract_sha256 != canonical_sha256(self.unhashed_payload()):
            raise CampaignError("campaign contract hash differs")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate()
        return canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignContract":
        if set(value) != set(cls.__dataclass_fields__):
            raise CampaignError("campaign contract schema differs")
        result = cls(
            **{
                **dict(value),
                "modalities": tuple(value["modalities"]),
                "task_ids": tuple(value["task_ids"]),
                "candidate_ids": tuple(value["candidate_ids"]),
            }
        )
        result.validate()
        return result


def _build_campaign_contract_unchecked() -> CampaignContract:
    contracts = tuple(build_modality_contract(name) for name in CAMPAIGN_MODALITIES)
    rows = tuple(
        candidate
        for modality in CAMPAIGN_MODALITIES
        for candidate in candidates_for_modality(modality)
    )
    candidate_ids = tuple(row.candidate_id for row in rows)
    if len(candidate_ids) != CAMPAIGN_CANDIDATE_COUNT:
        raise CampaignError("campaign does not contain exactly 3,550 candidates")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CampaignError("campaign candidate IDs overlap")
    if any(row.quota_modality not in CAMPAIGN_MODALITIES for row in rows):
        raise CampaignError("generated or unrelated candidates entered the campaign")
    if any(
        token in str(row.precision_policy).lower()
        for row in rows
        for token in ("tf32", "bf16", "bfloat16")
    ):
        raise CampaignError("V100 campaign contains TF32 or BF16")
    task_ids = tuple(dict.fromkeys(task for contract in contracts for task in contract.task_ids))
    draft = CampaignContract(
        version=CAMPAIGN_CONTRACT_VERSION,
        hardware_family_id=V100_FAMILY_ID,
        logical_target_hardware_id=LOGICAL_TARGET_HARDWARE_ID,
        target_manifest_sha256=contracts[0].target_manifest_sha256,
        task_registry_sha256=contracts[0].task_registry_sha256,
        modalities=CAMPAIGN_MODALITIES,
        modality_contract_sha256s={
            contract.modality: contract.contract_sha256 for contract in contracts
        },
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        candidate_ids_sha256=canonical_sha256(candidate_ids),
        candidate_count=len(candidate_ids),
        measured_epoch_count=len(candidate_ids) * MEASURED_EPOCHS_PER_LABEL,
        modality_counts=dict(CAMPAIGN_COUNTS),
        family_counts=dict(sorted(Counter(row.family_id for row in rows).items())),
        contract_sha256="",
    )
    if draft.measured_epoch_count != CAMPAIGN_MEASURED_EPOCH_COUNT:
        raise CampaignError("campaign measured-epoch count differs")
    return replace(
        draft, contract_sha256=canonical_sha256(draft.unhashed_payload())
    )


def build_campaign_contract() -> CampaignContract:
    result = _build_campaign_contract_unchecked()
    result.validate()
    return result


def freeze_campaign_contract(workspace: str | Path) -> CampaignContract:
    root = Path(workspace).resolve()
    path = root / "state" / "campaign_contract.json"
    contract = build_campaign_contract()
    if path.is_file():
        if CampaignContract.from_dict(_load_json(path)) != contract:
            raise CampaignError("workspace is pinned to another campaign contract")
    elif path.exists() or path.is_symlink():
        raise CampaignError("campaign contract path is not a regular file")
    else:
        atomic_write_json(path, contract.to_dict())
    return contract


def analysis_summary() -> Mapping[str, Any]:
    contract = build_campaign_contract()
    return {
        "target_hardware_id": contract.logical_target_hardware_id,
        "target_manifest_sha256": contract.target_manifest_sha256,
        "contract_sha256": contract.contract_sha256,
        "candidate_count": contract.candidate_count,
        "measured_epoch_count": contract.measured_epoch_count,
        "modality_counts": contract.modality_counts,
        "family_counts": contract.family_counts,
        "task_ids": contract.task_ids,
        "generated_excluded": True,
        "worker_count": 4,
        "pilot": [
            {
                "family_id": family,
                "task_id": task,
                "precision_policy": precision,
                "configuration_id": candidate.candidate_id,
            }
            for (family, task, precision), candidate in zip(
                PILOT_RULES, pilot_candidates(), strict=True
            )
        ],
    }


def pilot_candidates() -> tuple[Any, ...]:
    rows = tuple(
        candidate
        for modality in CAMPAIGN_MODALITIES
        for candidate in candidates_for_modality(modality)
    )
    selected = []
    for family, task, precision in PILOT_RULES:
        matches = tuple(
            row
            for row in rows
            if row.family_id == family
            and row.task_id == task
            and row.precision_policy["policy_id"] == precision
            and row.execution["mode"] == "eager"
            and row.optimizer["name"] == "adamw"
        )
        if not matches:
            raise CampaignError(f"pilot rule for {family} has no frozen candidate")
        selected.append(
            sorted(
                matches,
                key=lambda row: (
                    -row.microbatch_size,
                    row.gradient_accumulation_steps,
                    row.candidate_id,
                ),
            )[0]
        )
    if len({row.candidate_id for row in selected}) != 4:
        raise CampaignError("pilot candidates are not four unique configurations")
    return tuple(selected)


def run_pilot(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    repository_revision: str,
    image_digest: str,
    kaggle_executable: str = "kaggle",
    probes: Sequence[GpuProbe] | None = None,
) -> Mapping[str, Any]:
    """Prepare three tasks, then dispatch the four named labels concurrently."""

    from .kaggle import KaggleCliClient
    from .materialization import (
        TaskMaterializer,
        freeze_initial_target_manifest,
        seal_worker_inputs,
    )
    from .mlebench_bridge import PinnedMleBenchPreparer
    from .modality_sharding import freeze_modality_contract, freeze_run_identity
    from .task_registry import load_task_registry
    from .workflow import (
        _advance_failed_slot,
        _latest_failure,
        _load_slot,
        _record_indexes,
        _save_slot,
    )

    root = Path(workspace).resolve()
    freeze_campaign_contract(root)
    workers = tuple(probes or discover_v100_probes())
    if len(workers) != 4 or len({worker.gpu_uuid for worker in workers}) != 4:
        raise CampaignError("pilot requires exactly four unique V100 GPU workers")
    candidates = pilot_candidates()
    tasks = {row.task_id: row for row in load_task_registry().entries}
    prepared: dict[tuple[str, str], Any] = {}
    modality_roots: dict[str, Path] = {}
    for modality in CAMPAIGN_MODALITIES:
        modality_root = root / modality
        modality_roots[modality] = modality_root
        contract = freeze_modality_contract(modality_root, modality)
        freeze_run_identity(
            modality_root,
            repository_revision=repository_revision,
            image_digest=image_digest,
        )
        manifest, _ = freeze_initial_target_manifest(modality_root)
        if manifest.sha256 != contract.target_manifest_sha256:
            raise CampaignError("pilot modality contract and manifest differ")
        lock_campaign_environment(modality_root)
        materializer = TaskMaterializer(
            workspace=modality_root,
            repository_root=Path(repository_root),
            kaggle=KaggleCliClient(executable=kaggle_executable),
            preparer=PinnedMleBenchPreparer(Path(mlebench_checkout)),
        )
        for task_id in dict.fromkeys(
            row.task_id for row in candidates if row.quota_modality == modality
        ):
            materialized = materializer.materialize(tasks[task_id])
            seal_worker_inputs(materialized)
            prepared[(modality, task_id)] = materialized

    jobs = []
    slot_by_root: dict[str, Any] = {}
    for candidate in candidates:
        modality_root = modality_roots[candidate.quota_modality]
        slot_path = modality_root / "state/slots" / f"{candidate.candidate_id}.json"
        slot = _load_slot(slot_path, candidate)
        accepted, failures = _record_indexes(modality_root)
        if slot.current_candidate.candidate_id in accepted:
            slot_by_root[candidate.candidate_id] = slot
            continue
        failure = _latest_failure(failures.get(slot.current_candidate.candidate_id, ()))
        if failure is not None:
            slot = _advance_failed_slot(modality_root, slot, failure, candidate)
            slot = _save_slot(slot_path, slot)
        else:
            slot = _save_slot(slot_path, slot)
        slot_by_root[candidate.candidate_id] = slot
        materialized = prepared[(candidate.quota_modality, candidate.task_id)]
        jobs.append((candidate, slot, modality_root, materialized, failures))
    if len(jobs) > 4:
        raise CampaignError("pilot scheduled more than four unresolved labels")
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
        futures = []
        for probe, (_, slot, modality_root, materialized, failures) in zip(
            workers[: len(jobs)], jobs, strict=True
        ):
            current = slot.current_candidate
            futures.append(
                executor.submit(
                    AttemptSupervisor(modality_root).run,
                    current,
                    tasks[current.task_id],
                    materialized.view_manifest,
                    public_directory=materialized.public,
                    prepared_directory=materialized.prepared_view,
                    archive_sha256=materialized.inventory.archive_sha256,
                    probe=probe,
                    attempt_index=len(failures.get(current.candidate_id, ())),
                )
            )
        for future in futures:
            future.result()
    resolutions = []
    for root_candidate in candidates:
        modality_root = modality_roots[root_candidate.quota_modality]
        accepted, _ = _record_indexes(modality_root)
        current = slot_by_root[root_candidate.candidate_id].current_candidate
        record = accepted.get(current.candidate_id)
        if record is None:
            raise CampaignError("pilot did not publish all four accepted records")
        resolutions.append(
            {
                "family_id": root_candidate.family_id,
                "root_configuration_id": root_candidate.candidate_id,
                "resolved_configuration_id": current.candidate_id,
                "record_sha256": canonical_sha256(asdict(record)),
            }
        )
    receipt: dict[str, Any] = {
        "version": PILOT_COMPLETION_VERSION,
        "campaign_contract_sha256": build_campaign_contract().contract_sha256,
        "repository_revision": repository_revision,
        "image_digest": image_digest,
        "worker_count": 4,
        "labels": resolutions,
    }
    receipt["completion_sha256"] = canonical_sha256(receipt)
    atomic_write_json(root / "state/pilot_completion.json", receipt)
    return canonical_value(receipt)


def run_campaign(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    repository_revision: str,
    image_digest: str,
    kaggle_executable: str = "kaggle",
    probes: Sequence[GpuProbe] | None = None,
    pilot: bool = False,
) -> None:
    """Run or resume all three modality workspaces with the same four GPUs."""

    root = Path(workspace).resolve()
    freeze_campaign_contract(root)
    workers = tuple(probes or discover_v100_probes())
    if len(workers) != 4 or len({worker.gpu_uuid for worker in workers}) != 4:
        raise CampaignError("campaign requires exactly four unique V100 GPU workers")
    if pilot:
        run_pilot(
            workspace=root,
            repository_root=repository_root,
            mlebench_checkout=mlebench_checkout,
            repository_revision=repository_revision,
            image_digest=image_digest,
            kaggle_executable=kaggle_executable,
            probes=workers,
        )
        return
    from .modality_workflow import run_modality_workflow

    for modality in CAMPAIGN_MODALITIES:
        run_modality_workflow(
            workspace=root / modality,
            repository_root=repository_root,
            mlebench_checkout=mlebench_checkout,
            modality=modality,
            repository_revision=repository_revision,
            image_digest=image_digest,
            kaggle_executable=kaggle_executable,
            probes=workers,
        )
    verify_campaign(
        root,
        repository_revision=repository_revision,
        image_digest=image_digest,
    )


def verify_campaign(
    workspace: str | Path,
    *,
    repository_revision: str,
    image_digest: str,
    publish: bool = True,
) -> Mapping[str, Any]:
    root = Path(workspace).resolve()
    contract = freeze_campaign_contract(root)
    completions: dict[str, ModalityCompletion] = {}
    for modality in CAMPAIGN_MODALITIES:
        try:
            completions[modality] = verify_modality_workspace(
                root / modality,
                modality,
                repository_revision=repository_revision,
                image_digest=image_digest,
            )
        except ModalityShardError as error:
            raise CampaignError(f"{modality} workspace is incomplete or invalid") from error
    resolved = tuple(
        candidate
        for modality in CAMPAIGN_MODALITIES
        for candidate in completions[modality].resolved_candidate_ids
    )
    if len(resolved) != len(set(resolved)) or len(resolved) != contract.candidate_count:
        raise CampaignError("campaign completion is not an exact disjoint union")
    receipt: dict[str, Any] = {
        "version": CAMPAIGN_COMPLETION_VERSION,
        "contract_sha256": contract.contract_sha256,
        "repository_revision": repository_revision,
        "image_digest": image_digest,
        "candidate_count": len(resolved),
        "measured_epoch_count": len(resolved) * MEASURED_EPOCHS_PER_LABEL,
        "modality_completion_sha256s": {
            key: value.completion_sha256 for key, value in completions.items()
        },
    }
    receipt["completion_sha256"] = canonical_sha256(receipt)
    if publish:
        atomic_write_json(root / "state" / "campaign_completion.json", receipt)
    return canonical_value(receipt)


__all__ = [
    "CAMPAIGN_CANDIDATE_COUNT",
    "CAMPAIGN_COUNTS",
    "CAMPAIGN_MEASURED_EPOCH_COUNT",
    "CAMPAIGN_MODALITIES",
    "CampaignContract",
    "CampaignError",
    "analysis_summary",
    "build_campaign_contract",
    "freeze_campaign_contract",
    "run_campaign",
    "run_pilot",
    "pilot_candidates",
    "verify_campaign",
]
