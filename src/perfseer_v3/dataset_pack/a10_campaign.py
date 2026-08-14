"""Resumable native Nautilus A10 campaign orchestration."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from .a10_crosswalk import REFERENCE_MANIFEST_SHA256
from .fingerprints import canonical_sha256, canonical_value
from .labeler_profile import PROFILE
from .storage import atomic_write_json


_SPEECH_V2 = PROFILE.uses_speech_v2
_NONVISION = PROFILE.is_nonvision_4gpu
A10_CAMPAIGN_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_11200_campaign_v1"
    if _NONVISION
    else "perfseer_v3_nrp_a10_speech_18k_campaign_v2"
    if _SPEECH_V2
    else "perfseer_v3_nrp_a10_18k_campaign_v1"
)
A10_RUN_IDENTITY_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_run_identity_v1"
    if _NONVISION
    else "perfseer_v3_nrp_a10_speech_run_identity_v2"
    if _SPEECH_V2
    else "perfseer_v3_nrp_a10_run_identity_v1"
)
A10_PILOT_RECEIPT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_32_pilot_receipt_v1"
    if _NONVISION
    else "perfseer_v3_nrp_a10_speech_96_pilot_receipt_v2"
    if _SPEECH_V2
    else "perfseer_v3_nrp_a10_96_pilot_receipt_v1"
)
A10_CHUNK_RECEIPT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_chunk_receipt_v1"
    if _NONVISION
    else "perfseer_v3_nrp_a10_speech_chunk_receipt_v2"
    if _SPEECH_V2
    else "perfseer_v3_nrp_a10_chunk_receipt_v1"
)
A10_TASK_RECEIPT_VERSION = (
    "perfseer_v3_nrp_a10_nonvision_partial_task_receipt_v1"
    if _NONVISION
    else "perfseer_v3_nrp_a10_speech_partial_task_receipt_v2"
    if _SPEECH_V2
    else "perfseer_v3_nrp_a10_partial_task_receipt_v1"
)
LOCAL_VALIDATION_RECEIPT_VERSION = "perfseer_v3_nrp_a10_nonvision_local_validation_receipt_v1"
PILOT_SIZE = 32 if _NONVISION else 96
CHUNK_SIZE = 256
PRODUCTION_CHUNK_COUNT = 44 if _NONVISION else 70
TOTAL_CANDIDATES = 11_200 if _NONVISION else 18_000
MEASURED_EPOCHS_PER_LABEL = 3
MAX_REPLACEMENTS_PER_SLOT = 3 if _NONVISION else 100


class A10CampaignError(RuntimeError):
    """Raised when campaign ordering, resume state, or verification fails closed."""


def run_isolated_worker_batch(
    roots: Sequence[Any],
    probes: Sequence[Any],
    callback: Callable[[Any, Any], Any],
    *,
    isolated_exceptions: tuple[type[BaseException], ...] = (),
) -> tuple[tuple[Any, ...], tuple[tuple[Any, BaseException], ...]]:
    """Run one candidate per unique GPU and observe every sibling future."""

    if not roots or len(roots) > len(probes):
        raise A10CampaignError("worker batch size is outside the available probe count")
    selected_probes = tuple(probes[: len(roots)])
    if len({probe.gpu_uuid for probe in selected_probes}) != len(selected_probes):
        raise A10CampaignError("worker batch assigns one GPU more than once")
    successes: list[Any] = []
    isolated: list[tuple[Any, BaseException]] = []
    global_failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(roots)) as pool:
        future_roots = {
            pool.submit(callback, root, probe): root
            for root, probe in zip(roots, selected_probes, strict=True)
        }
        for future in as_completed(future_roots):
            root = future_roots[future]
            try:
                successes.append(future.result())
            except isolated_exceptions as error:
                isolated.append((root, error))
            except BaseException as error:
                global_failures.append(error)
    if global_failures:
        raise global_failures[0]
    return tuple(successes), tuple(isolated)


def _crosswalk_functions() -> tuple[Callable[..., Any], Callable[..., Any]]:
    if _NONVISION:
        from .a10_nonvision_crosswalk import build_crosswalk, freeze_crosswalk
    elif _SPEECH_V2:
        from .a10_speech_crosswalk import build_crosswalk, freeze_crosswalk
    else:
        from .a10_crosswalk import build_crosswalk, freeze_crosswalk
    return build_crosswalk, freeze_crosswalk


def _campaign_contract_path(root: Path) -> Path:
    name = (
        "a10_nonvision_campaign_contract.json"
        if _NONVISION
        else "a10_speech_v2_campaign_contract.json"
        if _SPEECH_V2
        else "a10_campaign_contract.json"
    )
    return root / "state" / name


def _pilot_receipt_path(root: Path) -> Path:
    name = (
        "nonvision_pilot_receipt.json"
        if _NONVISION
        else "speech_v2_pilot_receipt.json"
        if _SPEECH_V2
        else "pilot_receipt.json"
    )
    return root / "state" / name


def _chunk_receipt_path(root: Path, index: int) -> Path:
    directory = (
        "nonvision_chunk_receipts"
        if _NONVISION
        else "speech_v2_chunk_receipts"
        if _SPEECH_V2
        else "chunk_receipts"
    )
    return root / "state" / directory / f"chunk-{index:02d}.json"


def _partial_task_receipt_path(root: Path, task_id: str) -> Path:
    directory = (
        "nonvision_partial_task_receipts"
        if _NONVISION
        else "speech_v2_partial_task_receipts"
        if _SPEECH_V2
        else "partial_task_receipts"
    )
    return root / "state" / directory / f"{task_id}.json"


def _incomplete_receipt_path(root: Path, name: str) -> Path:
    return root / "state" / "incomplete_receipts" / f"{name}.json"


def _failure_ledger_path(root: Path) -> Path:
    return root / "state" / "failure_ledger.json"


def _assert_workspace_generation(root: Path) -> None:
    if not _SPEECH_V2:
        return
    forbidden = [
        root / "state" / "a10_campaign_contract.json",
        root / "state" / "a10_crosswalk_summary.json",
        root / "state" / "a10_crosswalk.jsonl",
        root / "state" / "campaign.lock",
        root / "state" / "pilot_receipt.json",
        root / "state" / "chunk_receipts",
        root / "state" / "partial_task_receipts",
    ]
    if _NONVISION:
        forbidden.extend(
            (
                root / "state" / "a10_speech_v2_campaign_contract.json",
                root / "state" / "a10_speech_v2_crosswalk_summary.json",
                root / "state" / "a10_speech_v2_crosswalk.jsonl",
                root / "state" / "campaign-speech-v2.lock",
                root / "state" / "speech_v2_pilot_receipt.json",
                root / "state" / "speech_v2_chunk_receipts",
                root / "state" / "speech_v2_partial_task_receipts",
            )
        )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise A10CampaignError("active profile refuses an older A10 workspace")


def _coverage_tokens(candidate: Any) -> set[tuple[str, str]]:
    return {
        ("task", candidate.task_id),
        ("family", candidate.family_id),
        ("precision", str(candidate.precision_policy["policy_id"])),
        ("execution", str(candidate.execution["mode"])),
        ("regime", candidate.regime),
        ("checkpoint", str(bool(candidate.activation_checkpointing["enabled"]))),
        ("memory_tier", candidate.batch_plan.effective_tier),
        (
            "requested_effective_batch",
            str(candidate.microbatch_size * candidate.gradient_accumulation_steps),
        ),
    }


def _nonvision_pilot_candidates(source: Sequence[Any], manifest_sha256: str) -> tuple[Any, ...]:
    path = (
        Path(__file__).resolve().parents[1]
        / "registries"
        / "native_a10_nonvision_pilot_v1.yaml"
    )
    try:
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise A10CampaignError("non-vision pilot contract is unreadable") from error
    if not isinstance(contract, Mapping):
        raise A10CampaignError("non-vision pilot contract must be an object")
    expected_keys = {
        "version",
        "manifest_sha256",
        "candidate_count",
        "category_counts",
        "minimum_factor_counts",
        "candidate_ids",
    }
    if set(contract) != expected_keys or (
        contract["version"] != "perfseer_v3_nrp_a10_nonvision_32_pilot_v1"
        or contract["manifest_sha256"] != manifest_sha256
        or contract["candidate_count"] != 32
    ):
        raise A10CampaignError("non-vision pilot identity differs")
    ids = tuple(contract["candidate_ids"])
    by_id = {row.candidate_id: row for row in source}
    if len(ids) != 32 or len(set(ids)) != 32 or any(value not in by_id for value in ids):
        raise A10CampaignError("non-vision pilot candidate IDs differ")
    selected = tuple(by_id[value] for value in ids)
    categories = Counter(row.quota_modality for row in selected)
    if dict(sorted(categories.items())) != dict(contract["category_counts"]):
        raise A10CampaignError("non-vision pilot category totals differ")
    if len({row.family_id for row in selected}) != 22:
        raise A10CampaignError("non-vision pilot does not cover all 22 families")
    if len({row.task_id for row in selected}) != 12:
        raise A10CampaignError("non-vision pilot does not cover all 12 tasks")
    factors = {
        "requested_effective_batch": Counter(
            row.microbatch_size * row.gradient_accumulation_steps for row in selected
        ),
        "precision": Counter(row.precision_policy["policy_id"] for row in selected),
        "execution": Counter(row.execution["mode"] for row in selected),
        "regime": Counter(row.regime for row in selected),
        "checkpoint": Counter(
            bool(row.activation_checkpointing["enabled"]) for row in selected
        ),
    }
    for factor, minimum in contract["minimum_factor_counts"].items():
        if not factors[factor] or min(factors[factor].values()) < int(minimum):
            raise A10CampaignError(f"non-vision pilot {factor} coverage differs")
    if set(factors["requested_effective_batch"]) != {32, 64, 128, 256, 512}:
        raise A10CampaignError("non-vision pilot batch-size coverage differs")
    return selected


def pilot_candidates(manifest: Any | None = None) -> tuple[Any, ...]:
    from .sampler import build_target_manifest

    frozen = manifest or build_target_manifest()
    source = frozen.candidates
    if _NONVISION:
        return _nonvision_pilot_candidates(source, frozen.sha256)
    universe = set().union(*(_coverage_tokens(row) for row in source))
    remaining = set(universe)
    selected: list[Any] = []
    selected_ids: set[str] = set()
    while remaining:
        chosen = max(
            (row for row in source if row.candidate_id not in selected_ids),
            key=lambda row: (len(_coverage_tokens(row) & remaining), -row.ordinal),
        )
        if not (_coverage_tokens(chosen) & remaining):
            raise A10CampaignError("pilot factor coverage cannot make progress")
        selected.append(chosen)
        selected_ids.add(chosen.candidate_id)
        remaining -= _coverage_tokens(chosen)
    selected.extend(
        row
        for row in source
        if row.candidate_id not in selected_ids
        and len(selected) < PILOT_SIZE
    )
    if len(selected) != PILOT_SIZE:
        raise A10CampaignError(f"pilot does not contain exactly {PILOT_SIZE} candidates")
    selected_tokens = set().union(*(_coverage_tokens(row) for row in selected))
    if selected_tokens != universe or len({row.candidate_id for row in selected}) != PILOT_SIZE:
        raise A10CampaignError("pilot is not a unique complete factor cover")
    return tuple(selected)


def production_candidates(manifest: Any | None = None) -> tuple[Any, ...]:
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    frozen = manifest or build_target_manifest()
    pilot_ids = {row.candidate_id for row in pilot_candidates(frozen)}
    task_order = {row.task_id: index for index, row in enumerate(load_task_registry().entries)}
    result = tuple(
        sorted(
            (row for row in frozen.candidates if row.candidate_id not in pilot_ids),
            key=lambda row: (task_order[row.task_id], row.ordinal),
        )
    )
    expected = TOTAL_CANDIDATES - PILOT_SIZE
    if len(result) != expected or len({row.candidate_id for row in result}) != len(result):
        raise A10CampaignError(
            f"production projection must contain {expected:,} unique candidates"
        )
    return result


def chunk_candidates(chunk_index: int, manifest: Any | None = None) -> tuple[Any, ...]:
    if type(chunk_index) is not int or not 0 <= chunk_index < PRODUCTION_CHUNK_COUNT:
        raise A10CampaignError(
            f"chunk index must be in [0, {PRODUCTION_CHUNK_COUNT - 1}]"
        )
    rows = production_candidates(manifest)
    start = chunk_index * CHUNK_SIZE
    result = rows[start : start + CHUNK_SIZE]
    expected = (
        CHUNK_SIZE
        if chunk_index < PRODUCTION_CHUNK_COUNT - 1
        else (TOTAL_CANDIDATES - PILOT_SIZE) % CHUNK_SIZE or CHUNK_SIZE
    )
    if len(result) != expected:
        raise A10CampaignError("chunk candidate count differs from the frozen sequence")
    return result


def build_campaign_contract(
    manifest: Any | None = None,
    crosswalk: Any | None = None,
) -> Mapping[str, Any]:
    from .sampler import build_target_manifest
    from .task_registry import load_task_registry

    if not PROFILE.is_native_a10:
        raise A10CampaignError("A10 campaign requires a native A10 process profile")
    manifest = manifest or build_target_manifest()
    if crosswalk is None:
        build_crosswalk, _ = _crosswalk_functions()
        crosswalk = build_crosswalk(manifest)
    pilot = pilot_candidates(manifest)
    production = production_candidates(manifest)
    family_counts = Counter(row.family_id for row in manifest.candidates)
    modality_counts = Counter(row.quota_modality for row in manifest.candidates)
    precision_counts = Counter(row.precision_policy["policy_id"] for row in manifest.candidates)
    coverage = set().union(*(_coverage_tokens(row) for row in pilot))
    payload: dict[str, Any] = {
        "version": A10_CAMPAIGN_VERSION,
        "target_hardware_id": PROFILE.target_hardware_id,
        "hardware_family_id": PROFILE.hardware_family_id,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "native_manifest_sha256": manifest.sha256,
        "task_registry_sha256": load_task_registry().sha256,
        "crosswalk_sha256": crosswalk.summary["crosswalk_sha256"],
        "candidate_count": len(manifest.candidates),
        "measured_epoch_count": len(manifest.candidates) * MEASURED_EPOCHS_PER_LABEL,
        "task_count": len({row.task_id for row in manifest.candidates}),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "precision_counts": dict(sorted(precision_counts.items())),
        "pilot_candidate_ids": tuple(row.candidate_id for row in pilot),
        "pilot_coverage": tuple(sorted(coverage)),
        "production_candidate_ids_sha256": canonical_sha256(
            tuple(row.candidate_id for row in production)
        ),
        "production_chunk_count": PRODUCTION_CHUNK_COUNT,
        "production_chunk_sizes": (
            (*((CHUNK_SIZE,) * 43), 160)
            if _NONVISION
            else (*((CHUNK_SIZE,) * 69), 240)
        ),
        "worker_count": 4 if _NONVISION else 1,
    }
    if _NONVISION:
        from .a10_nonvision_crosswalk import PREDECESSOR_MANIFEST_SHA256
        from .speech_substitution import load_speech_substitution_contract

        payload.update(
            {
                "predecessor_manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
                "substitution_contract_sha256": load_speech_substitution_contract().sha256,
                "lineage_disposition_counts": crosswalk.summary[
                    "disposition_counts"
                ],
                "lineage_exclusion_counts": crosswalk.summary["exclusion_counts"],
                "workspace_generation": "native_a10_nonvision_4gpu_v1",
            }
        )
    elif _SPEECH_V2:
        from .a10_speech_crosswalk import NATIVE_V1_MANIFEST_SHA256
        from .speech_substitution import load_speech_substitution_contract

        payload.update(
            {
                "native_v1_manifest_sha256": NATIVE_V1_MANIFEST_SHA256,
                "substitution_contract_sha256": load_speech_substitution_contract().sha256,
                "lineage_classification_counts": crosswalk.summary[
                    "classification_counts"
                ],
                "workspace_generation": "native_a10_speech_v2",
            }
        )
    expected_tasks = 12 if _NONVISION else 22
    expected_families = 22 if _NONVISION else 35
    expected_chunks = (
        (*((256,) * 43), 160)
        if _NONVISION
        else (*((256,) * 69), 240)
    )
    if (
        payload["candidate_count"] != TOTAL_CANDIDATES
        or payload["measured_epoch_count"]
        != TOTAL_CANDIDATES * MEASURED_EPOCHS_PER_LABEL
        or payload["task_count"] != expected_tasks
        or payload["family_count"] != expected_families
        or payload["production_chunk_sizes"] != expected_chunks
    ):
        raise A10CampaignError("campaign counts differ from the approved contract")
    payload["contract_sha256"] = canonical_sha256(payload)
    return canonical_value(payload)


def analysis_summary() -> Mapping[str, Any]:
    contract = build_campaign_contract()
    return {
        **contract,
        "pilot_size": PILOT_SIZE,
        "remaining_after_pilot": TOTAL_CANDIDATES - PILOT_SIZE,
        "production_eligible_hardware_only": True,
    }


@contextmanager
def exclusive_workspace_lock(workspace: str | Path) -> Iterator[Path]:
    root = Path(workspace).resolve()
    lock_path = root / "state" / (
        "campaign-nonvision-4gpu.lock"
        if _NONVISION
        else "campaign-speech-v2.lock"
        if _SPEECH_V2
        else "campaign.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise A10CampaignError(
                "another writer holds the CephFS campaign workspace lock"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _freeze_json(path: Path, expected: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    if path.is_file():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise A10CampaignError(f"{context} is unreadable") from error
        if actual != expected:
            raise A10CampaignError(f"{context} differs from the frozen workspace")
    elif path.exists() or path.is_symlink():
        raise A10CampaignError(f"{context} path is not a regular file")
    else:
        atomic_write_json(path, expected)
    return expected


def freeze_campaign_state(
    workspace: str | Path,
    *,
    repository_revision: str,
    image_digest: str,
) -> Mapping[str, Any]:
    from .materialization import freeze_initial_target_manifest
    from .supervisor import lock_campaign_environment

    root = Path(workspace).resolve()
    _assert_workspace_generation(root)
    manifest, _ = freeze_initial_target_manifest(root)
    _, freeze_crosswalk = _crosswalk_functions()
    crosswalk = freeze_crosswalk(root, manifest)
    contract = build_campaign_contract(manifest, crosswalk)
    if manifest.sha256 != contract["native_manifest_sha256"]:
        raise A10CampaignError("workspace manifest differs from campaign contract")
    _freeze_json(_campaign_contract_path(root), contract, context="campaign contract")
    identity: dict[str, Any] = {
        "version": A10_RUN_IDENTITY_VERSION,
        "repository_revision": repository_revision,
        "image_digest": image_digest,
        "campaign_contract_sha256": contract["contract_sha256"],
        "crosswalk_sha256": crosswalk.summary["crosswalk_sha256"],
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    _freeze_json(root / "state" / "run_identity.json", identity, context="run identity")
    lock_campaign_environment(root)
    return contract


def _load_record_for_root(root: Path, candidate: Any) -> tuple[Any, Any] | None:
    from .workflow import _load_slot, _record_indexes

    slot = _load_slot(root / "state" / "slots" / f"{candidate.candidate_id}.json", candidate)
    accepted, _ = _record_indexes(root)
    record = accepted.get(slot.current_candidate.candidate_id)
    return None if record is None else (slot, record)


def _verify_native_record(
    record: Any,
    resolved: Any,
    root_candidate: Any,
    *,
    production_eligible: bool,
) -> None:
    record.validate()
    reference = record.reference_provenance
    common_invalid = (
        record.status.value != "accepted"
        or record.production_eligible is not production_eligible
        or record.target_hardware_id != PROFILE.target_hardware_id
        or not isinstance(reference, Mapping)
        or not record.build_identity
    )
    if common_invalid:
        raise A10CampaignError("accepted record lacks native/reference/build provenance")
    assert isinstance(reference, Mapping)
    if _NONVISION:
        from .a10_nonvision_crosswalk import PREDECESSOR_MANIFEST_SHA256
        from .a10_speech_crosswalk import NATIVE_V1_MANIFEST_SHA256
        from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract

        substituted = reference.get("dataset_substitution") is True
        provenance_invalid = (
            reference.get("nonvision_root_candidate_id") != root_candidate.candidate_id
            or reference.get("predecessor_manifest_sha256")
            != PREDECESSOR_MANIFEST_SHA256
            or reference.get("original_a10g_manifest_sha256")
            != REFERENCE_MANIFEST_SHA256
            or reference.get("native_v1_manifest_sha256")
            != NATIVE_V1_MANIFEST_SHA256
            or reference.get("substitution_contract_sha256")
            != load_speech_substitution_contract().sha256
            or reference.get("lineage_disposition") != "retained"
            or type(reference.get("historical_ordinal")) is not int
            or type(reference.get("nonvision_ordinal")) is not int
            or type(reference.get("dataset_substitution")) is not bool
            or substituted != (root_candidate.task_id == SPEECH_TASK_ID)
            or any(
                not isinstance(reference.get(key), str)
                or len(str(reference.get(key))) != 64
                for key in (
                    "original_a10g_candidate_id",
                    "native_v1_candidate_id",
                    "predecessor_candidate_id",
                    "predecessor_manifest_sha256",
                    "source_archive_sha256",
                    "remote_inventory_sha256",
                    "old_nonbatch_semantic_signature",
                    "new_nonbatch_semantic_signature",
                )
            )
        )
    elif _SPEECH_V2:
        from .a10_speech_crosswalk import NATIVE_V1_MANIFEST_SHA256
        from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract

        substituted = reference.get("dataset_substitution") is True
        provenance_invalid = (
            reference.get("native_v2_root_candidate_id") != root_candidate.candidate_id
            or reference.get("original_a10g_manifest_sha256")
            != REFERENCE_MANIFEST_SHA256
            or reference.get("native_v1_manifest_sha256")
            != NATIVE_V1_MANIFEST_SHA256
            or reference.get("substitution_contract_sha256")
            != load_speech_substitution_contract().sha256
            or type(reference.get("dataset_substitution")) is not bool
            or substituted != (root_candidate.task_id == SPEECH_TASK_ID)
            or (
                substituted
                and (
                    not isinstance(reference.get("speech_source_lock_sha256"), str)
                    or len(str(reference.get("speech_source_lock_sha256"))) != 64
                )
            )
            or (
                not substituted
                and reference.get("speech_source_lock_sha256") is not None
            )
            or any(
                not isinstance(reference.get(key), str)
                or len(str(reference.get(key))) != 64
                for key in (
                    "original_a10g_candidate_id",
                    "native_v1_candidate_id",
                    "native_v1_manifest_sha256",
                    "substitution_contract_sha256",
                    "source_archive_sha256",
                    "remote_inventory_sha256",
                )
            )
        )
    else:
        provenance_invalid = (
            reference.get("native_root_candidate_id") != root_candidate.candidate_id
            or reference.get("reference_manifest_sha256") != REFERENCE_MANIFEST_SHA256
        )
    if provenance_invalid:
        raise A10CampaignError("accepted record lacks native/reference/build provenance")
    if record.configuration_id != resolved.candidate_id:
        raise A10CampaignError("accepted record differs from its resolved candidate")


def _receipt(
    *,
    root: Path,
    version: str,
    name: str,
    candidates: Sequence[Any],
    contract_sha256: str,
    production_eligible: bool = True,
) -> Mapping[str, Any]:
    labels = []
    for candidate in candidates:
        resolved = _load_record_for_root(root, candidate)
        if resolved is None:
            raise A10CampaignError(f"{name} candidate {candidate.candidate_id} is unresolved")
        slot, record = resolved
        _verify_native_record(
            record,
            slot.current_candidate,
            candidate,
            production_eligible=production_eligible,
        )
        labels.append(
            {
                "root_candidate_id": candidate.candidate_id,
                "resolved_candidate_id": slot.current_candidate.candidate_id,
                "record_sha256": canonical_sha256(asdict(record)),
            }
        )
    if _NONVISION:
        hardware_payloads = []
        for path in sorted((root / "provenance" / "hardware").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if canonical_sha256(payload) != path.stem:
                raise A10CampaignError("hardware provenance hash differs")
            hardware_payloads.append(payload)
        expected_name = "NVIDIA A10" if production_eligible else "NVIDIA GeForce RTX 5090"
        expected_capability = [8, 6] if production_eligible else [12, 0]
        expected_mode = "production-a10" if production_eligible else "local-rtx5090"
        expected_count = 4 if production_eligible else 1
        # Limit qualification to GPUs actually referenced by this receipt.
        record_uuids = set()
        for candidate in candidates:
            resolved = _load_record_for_root(root, candidate)
            assert resolved is not None
            record_uuids.add(resolved[1].gpu_uuid)
        by_uuid = {row.get("uuid"): row for row in hardware_payloads}
        identities = set(record_uuids)
        if (
            len(identities) != expected_count
            or any(uuid not in by_uuid for uuid in identities)
            or any(by_uuid[uuid].get("name") != expected_name for uuid in identities)
            or any(
                by_uuid[uuid].get("compute_capability") != expected_capability
                for uuid in identities
            )
            or any(by_uuid[uuid].get("hardware_mode") != expected_mode for uuid in identities)
        ):
            raise A10CampaignError("receipt GPU identity/worker count differs")
    value: dict[str, Any] = {
        "version": version,
        "name": name,
        "campaign_contract_sha256": contract_sha256,
        "new_accepted_target": len(candidates),
        "labels": labels,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return canonical_value(value)


def _execute_candidates(
    *,
    workspace: Path,
    repository_root: Path,
    mlebench_checkout: Path,
    candidates: Sequence[Any],
    kaggle_executable: str,
    probes: Sequence[Any],
    production_eligible: bool,
) -> tuple[Mapping[str, Any], ...]:
    from .kaggle import KaggleCliClient
    from .materialization import TaskMaterializer, seal_worker_inputs
    from .mlebench_bridge import PinnedMleBenchPreparer
    from .supervisor import AttemptSupervisor
    from .task_registry import load_task_registry
    from .workflow import (
        _advance_failed_slot,
        _latest_failure,
        _load_slot,
        _record_indexes,
        _save_slot,
        SlotExhaustedError,
    )

    expected_workers = 4 if (_NONVISION and production_eligible) else 1
    if len(probes) != expected_workers:
        raise A10CampaignError("worker probe count differs from the execution mode")
    if len({probe.gpu_uuid for probe in probes}) != len(probes):
        raise A10CampaignError("worker probe UUID assignment is duplicated")
    tasks = {row.task_id: row for row in load_task_registry().entries}
    task_order = tuple(dict.fromkeys(row.task_id for row in candidates))
    exhausted: list[Mapping[str, Any]] = []
    for task_id in task_order:
        task_candidates = tuple(row for row in candidates if row.task_id == task_id)
        unresolved = tuple(row for row in task_candidates if _load_record_for_root(workspace, row) is None)
        if not unresolved:
            continue
        materializer = TaskMaterializer(
            workspace=workspace,
            repository_root=repository_root,
            kaggle=KaggleCliClient(executable=kaggle_executable),
            preparer=PinnedMleBenchPreparer(mlebench_checkout),
        )
        materialized = materializer.materialize(tasks[task_id])
        seal_worker_inputs(materialized)
        try:
            def process_root(root_candidate: Any, worker_probe: Any) -> None:
                while True:
                    slot_path = workspace / "state" / "slots" / f"{root_candidate.candidate_id}.json"
                    slot = _load_slot(slot_path, root_candidate)
                    accepted, failures = _record_indexes(workspace)
                    if slot.current_candidate.candidate_id in accepted:
                        break
                    failure = _latest_failure(failures.get(slot.current_candidate.candidate_id, ()))
                    if failure is not None:
                        slot = _advance_failed_slot(workspace, slot, failure, root_candidate)
                    slot = _save_slot(slot_path, slot)
                    current = slot.current_candidate
                    AttemptSupervisor(
                        workspace,
                        production_eligible=production_eligible,
                    ).run(
                        current,
                        tasks[task_id],
                        materialized.view_manifest,
                        public_directory=materialized.public,
                        prepared_directory=materialized.prepared_view,
                        archive_sha256=materialized.inventory.archive_sha256,
                        probe=worker_probe,
                        attempt_index=len(failures.get(current.candidate_id, ())),
                    )

            for offset in range(0, len(unresolved), len(probes)):
                batch = unresolved[offset : offset + len(probes)]
                _, isolated = run_isolated_worker_batch(
                    batch,
                    probes,
                    process_root,
                    isolated_exceptions=(SlotExhaustedError,),
                )
                exhausted.extend(
                    {
                        "root_candidate_id": root_candidate.candidate_id,
                        "task_id": root_candidate.task_id,
                        "reason": str(error),
                    }
                    for root_candidate, error in isolated
                )
        finally:
            durable = []
            for candidate in task_candidates:
                resolved = _load_record_for_root(workspace, candidate)
                if resolved is not None:
                    _, record = resolved
                    durable.append([candidate.candidate_id, canonical_sha256(asdict(record))])
            partial: dict[str, Any] = {
                "version": A10_TASK_RECEIPT_VERSION,
                "task_id": task_id,
                "dataset_fingerprint": materialized.view_manifest.dataset_fingerprint,
                "durable_records": durable,
                "unresolved_root_candidate_ids": [
                    candidate.candidate_id
                    for candidate in task_candidates
                    if _load_record_for_root(workspace, candidate) is None
                ],
            }
            partial["receipt_sha256"] = canonical_sha256(partial)
            atomic_write_json(
                _partial_task_receipt_path(workspace, task_id),
                partial,
            )
            materializer._delete_task_cache()
            if materializer.task_cache.exists() or materializer.task_cache.is_symlink():
                raise A10CampaignError("task cache cleanup did not complete")
    return tuple(exhausted)


def run_campaign(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    repository_revision: str,
    image_digest: str,
    kaggle_executable: str = "kaggle",
    pilot: bool = False,
    local_validation: bool = False,
    chunk_index: int | None = None,
    max_new_accepted: int = CHUNK_SIZE,
    probe: Any | None = None,
    probes: Sequence[Any] | None = None,
    executor: Callable[[Sequence[Any]], None] | None = None,
) -> Mapping[str, Any]:
    from .sampler import build_target_manifest
    from .supervisor import (
        discover_a10_probe,
        discover_a10_probes,
        discover_local_rtx5090_probe,
    )

    if sum((pilot, local_validation, chunk_index is not None)) != 1:
        raise A10CampaignError(
            "select exactly one of local_validation, pilot, or chunk_index"
        )
    if chunk_index is not None and max_new_accepted != CHUNK_SIZE:
        raise A10CampaignError("production Jobs must cap new accepted labels at exactly 256")
    if probe is not None and probes is not None:
        raise A10CampaignError("probe and probes are mutually exclusive")
    root = Path(workspace).resolve()
    _assert_workspace_generation(root)
    with exclusive_workspace_lock(root):
        contract = freeze_campaign_state(
            root,
            repository_revision=repository_revision,
            image_digest=image_digest,
        )
        manifest = build_target_manifest()
        if local_validation:
            candidates = pilot_candidates(manifest)
            name = "local-validation"
            version = LOCAL_VALIDATION_RECEIPT_VERSION
            receipt_path = root / "state" / "local_validation_receipt.json"
        elif pilot:
            candidates = pilot_candidates(manifest)
            name = "pilot"
            version = A10_PILOT_RECEIPT_VERSION
            receipt_path = _pilot_receipt_path(root)
        else:
            assert chunk_index is not None
            pilot_path = _pilot_receipt_path(root)
            if not pilot_path.is_file():
                raise A10CampaignError("production cannot start before the pilot receipt")
            if chunk_index > 0 and not _chunk_receipt_path(
                root, chunk_index - 1
            ).is_file():
                raise A10CampaignError("production chunks must run in strict index order")
            candidates = chunk_candidates(chunk_index, manifest)
            name = f"chunk-{chunk_index:02d}"
            version = A10_CHUNK_RECEIPT_VERSION
            receipt_path = _chunk_receipt_path(root, chunk_index)
        if receipt_path.is_file():
            expected = _receipt(
                root=root,
                version=version,
                name=name,
                candidates=candidates,
                contract_sha256=str(contract["contract_sha256"]),
                production_eligible=not local_validation,
            )
            return _freeze_json(receipt_path, expected, context=f"{name} receipt")
        if executor is not None:
            executor(candidates)
        else:
            if probes is not None:
                worker_probes = tuple(probes)
            elif probe is not None:
                worker_probes = (probe,)
            elif local_validation:
                worker_probes = (discover_local_rtx5090_probe(),)
            elif _NONVISION:
                worker_probes = discover_a10_probes()
            else:
                worker_probes = (discover_a10_probe(),)
            unresolved = _execute_candidates(
                workspace=root,
                repository_root=Path(repository_root).resolve(),
                mlebench_checkout=Path(mlebench_checkout).resolve(),
                candidates=candidates,
                kaggle_executable=kaggle_executable,
                probes=worker_probes,
                production_eligible=not local_validation,
            )
            if unresolved:
                ledger: dict[str, Any] = {
                    "version": "perfseer_v3_nrp_a10_nonvision_failure_ledger_v1",
                    "campaign_name": name,
                    "failures": unresolved,
                }
                ledger["ledger_sha256"] = canonical_sha256(ledger)
                atomic_write_json(_failure_ledger_path(root), ledger)
                incomplete: dict[str, Any] = {
                    "version": "perfseer_v3_nrp_a10_nonvision_incomplete_receipt_v1",
                    "campaign_name": name,
                    "campaign_contract_sha256": contract["contract_sha256"],
                    "unresolved_root_candidate_ids": [
                        row["root_candidate_id"] for row in unresolved
                    ],
                    "failure_ledger_sha256": ledger["ledger_sha256"],
                }
                incomplete["receipt_sha256"] = canonical_sha256(incomplete)
                atomic_write_json(_incomplete_receipt_path(root, name), incomplete)
                raise A10CampaignError(
                    f"{name} drained all schedulable work with {len(unresolved)} "
                    "unresolved quota slots"
                )
        receipt = _receipt(
            root=root,
            version=version,
            name=name,
            candidates=candidates,
            contract_sha256=str(contract["contract_sha256"]),
            production_eligible=not local_validation,
        )
        atomic_write_json(receipt_path, receipt)
        return receipt


def verify_campaign(
    workspace: str | Path,
    *,
    complete: bool,
) -> Mapping[str, Any]:
    from .sampler import build_target_manifest

    root = Path(workspace).resolve()
    _assert_workspace_generation(root)
    contract = build_campaign_contract()
    _freeze_json(_campaign_contract_path(root), contract, context="campaign contract")
    pilot = pilot_candidates(build_target_manifest())
    pilot_receipt = _receipt(
        root=root,
        version=A10_PILOT_RECEIPT_VERSION,
        name="pilot",
        candidates=pilot,
        contract_sha256=str(contract["contract_sha256"]),
    )
    _freeze_json(_pilot_receipt_path(root), pilot_receipt, context="pilot receipt")
    completed_chunks = 0
    accepted = len(pilot)
    for index in range(PRODUCTION_CHUNK_COUNT):
        path = _chunk_receipt_path(root, index)
        if not path.exists():
            break
        candidates = chunk_candidates(index)
        expected = _receipt(
            root=root,
            version=A10_CHUNK_RECEIPT_VERSION,
            name=f"chunk-{index:02d}",
            candidates=candidates,
            contract_sha256=str(contract["contract_sha256"]),
        )
        _freeze_json(path, expected, context=f"chunk {index} receipt")
        completed_chunks += 1
        accepted += len(candidates)
    if complete and (completed_chunks != PRODUCTION_CHUNK_COUNT or accepted != TOTAL_CANDIDATES):
        raise A10CampaignError(
            f"complete verification requires pilot plus all {PRODUCTION_CHUNK_COUNT} chunks"
        )
    result = {
        "status": "complete" if accepted == TOTAL_CANDIDATES else "partial",
        "accepted_labels": accepted,
        "measured_epoch_records": accepted * MEASURED_EPOCHS_PER_LABEL,
        "completed_chunks": completed_chunks,
        "remaining_labels": TOTAL_CANDIDATES - accepted,
        "campaign_contract_sha256": contract["contract_sha256"],
    }
    return canonical_value(result)


def verify_local_validation(workspace: str | Path) -> Mapping[str, Any]:
    if not _NONVISION:
        raise A10CampaignError("local validation receipt requires the non-vision profile")
    from .sampler import build_target_manifest

    root = Path(workspace).resolve()
    contract = build_campaign_contract()
    candidates = pilot_candidates(build_target_manifest())
    receipt = _receipt(
        root=root,
        version=LOCAL_VALIDATION_RECEIPT_VERSION,
        name="local-validation",
        candidates=candidates,
        contract_sha256=str(contract["contract_sha256"]),
        production_eligible=False,
    )
    _freeze_json(
        root / "state" / "local_validation_receipt.json",
        receipt,
        context="local validation receipt",
    )
    hardware_payloads = []
    for path in sorted((root / "provenance" / "hardware").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(payload) != path.stem:
            raise A10CampaignError("local hardware provenance hash differs")
        hardware_payloads.append(payload)
    identities = {row.get("uuid") for row in hardware_payloads}
    if (
        len(identities) != 1
        or any(row.get("name") != "NVIDIA GeForce RTX 5090" for row in hardware_payloads)
        or any(row.get("compute_capability") != [12, 0] for row in hardware_payloads)
        or any(row.get("hardware_mode") != "local-rtx5090" for row in hardware_payloads)
    ):
        raise A10CampaignError("local validation hardware is not one qualified RTX 5090")
    return canonical_value(
        {
            "status": "passed",
            "accepted_labels": len(candidates),
            "measured_epoch_records": len(candidates) * MEASURED_EPOCHS_PER_LABEL,
            "production_eligible": False,
            "gpu_uuid": next(iter(identities)),
            "receipt_sha256": receipt["receipt_sha256"],
        }
    )


__all__ = [
    "A10CampaignError",
    "CHUNK_SIZE",
    "PILOT_SIZE",
    "PRODUCTION_CHUNK_COUNT",
    "analysis_summary",
    "build_campaign_contract",
    "chunk_candidates",
    "exclusive_workspace_lock",
    "freeze_campaign_state",
    "pilot_candidates",
    "production_candidates",
    "run_isolated_worker_batch",
    "run_campaign",
    "verify_campaign",
    "verify_local_validation",
]
