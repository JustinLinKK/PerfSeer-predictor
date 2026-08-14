"""Sequential real-data RTX 5090 smoke labels that can never enter V100 data."""

from __future__ import annotations

from dataclasses import asdict
import gc
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .fingerprints import canonical_sha256, canonical_value, file_sha256
from .labeler_profile import PROFILE
from .quota import FROZEN_QUOTA_CELLS, NONVISION_QUOTA_CELLS
from .kaggle import KaggleCliClient
from .materialization import TaskMaterializer
from .mlebench_bridge import PinnedMleBenchPreparer
from .adapters import adapter_for_task
from .local_runtime import (
    _build_model,
    _build_optimizer,
    _build_scheduler,
    _forward_loss,
    _gradients_finite,
    _move,
    _precision_context,
    _update,
    bind_candidate_batch_shape,
    configure_precision_backends,
    five_epoch_optimizer_steps,
)
from .local_task_fixture import build_local_real_format_batch
from .sampler import TargetCandidate, build_target_manifest
from .storage import atomic_write_json
from .task_registry import TaskRegistryEntry, load_task_registry
from .v100_runner import (
    FiveEpochRunResult,
    TelemetryReading,
    run_five_epoch_training,
)


LOCAL_SMOKE_VERSION = (
    "perfseer_v3_nrp_a10_speech_rtx5090_real_label_smoke_v2"
    if PROFILE.uses_speech_v2
    else "perfseer_v3_nrp_a10_rtx5090_real_label_smoke_v1"
    if PROFILE.is_native_a10
    else "perfseer_v3_rtx5090_real_label_smoke_v1"
)
LOCAL_SMOKE_MODELS = (
    tuple(row[1] for row in NONVISION_QUOTA_CELLS)
    if PROFILE.is_nonvision_4gpu
    else ("panns_cnn14",)
    if PROFILE.uses_speech_v2
    else ("panns_cnn14", "cgcnn")
)
_V100_CAMPAIGN_FAMILIES = (
    "panns_cnn14",
    "temporal_convolutional_network",
    "m5_waveform_cnn",
    "selu_mlp",
    "tabtransformer",
    "mixture_density_network",
    "gcn",
    "gat",
    "graphsage",
    "cgcnn",
)
CAMPAIGN_FAMILIES = (
    tuple(
        row[1]
        for row in (
            NONVISION_QUOTA_CELLS
            if PROFILE.is_nonvision_4gpu
            else FROZEN_QUOTA_CELLS
        )
    )
    if PROFILE.is_native_a10
    else _V100_CAMPAIGN_FAMILIES
)
_MODEL_RULES = {
    "panns_cnn14": (
        "tensorflow-speech-yes-no"
        if PROFILE.uses_speech_v2
        else "mlsp-2013-birds",
        "fp32_tf32" if PROFILE.is_native_a10 else "fp32_ieee",
    ),
    "cgcnn": (
        "nomad2018",
        "bf16" if PROFILE.is_native_a10 else "fp16_grad_scaler",
    ),
}


class LocalSmokeError(RuntimeError):
    """Raised when a non-production local smoke result is not self-consistent."""


class Rtx5090TelemetryBackend:
    """Telemetry adapter with an explicit non-production RTX 5090 identity."""

    def __init__(self, physical_index: int = 0) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            uuid = pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            properties = torch.cuda.get_device_properties(0)
            if (
                str(name) != "NVIDIA GeForce RTX 5090"
                or (properties.major, properties.minor) != (12, 0)
                or not 30 * 1024**3 <= int(memory.total) <= 34 * 1024**3
            ):
                raise LocalSmokeError("local smoke requires the 32 GiB RTX 5090")
            self._gpu_uuid = str(uuid)
            self.hardware_provenance = {
                "observed_hardware_id": "nvidia_geforce_rtx_5090_32gb_local",
                "name": str(name),
                "uuid": self._gpu_uuid,
                "total_memory_bytes": int(memory.total),
                "compute_capability": [properties.major, properties.minor],
                "production_eligible": False,
            }
            self._hardware_fingerprint = canonical_sha256(self.hardware_provenance)
        except LocalSmokeError:
            raise
        except Exception as error:
            raise LocalSmokeError("cannot qualify the local RTX 5090 with NVML") from error

    @property
    def gpu_uuid(self) -> str:
        return self._gpu_uuid

    @property
    def hardware_fingerprint(self) -> str:
        return self._hardware_fingerprint

    def read(self) -> TelemetryReading:
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        return TelemetryReading(
            sm_util_percent=float(utilization.gpu),
            memory_controller_util_percent=float(utilization.memory),
            device_used_vram_mib=float(memory.used / 1024**2),
            throttle_reason_bits=0,
            compute_process_ids=tuple(sorted(int(row.pid) for row in processes)),
        )


def select_smoke_candidate(model: str) -> TargetCandidate:
    try:
        task_id, precision = _MODEL_RULES[model]
    except KeyError as error:
        raise LocalSmokeError(f"unsupported local smoke model {model!r}") from error
    matches = tuple(
        row
        for row in build_target_manifest().candidates
        if row.family_id == model
        and row.task_id == task_id
        and row.precision_policy["policy_id"] == precision
        and row.execution["mode"] == "eager"
        and row.optimizer["name"] == "adamw"
    )
    if not matches:
        raise LocalSmokeError("frozen manifest has no safe smoke candidate")
    # Prefer a large microbatch and one-step accumulation so five real epochs
    # finish promptly, then use the immutable candidate ID as the tie-breaker.
    return sorted(
        matches,
        key=lambda row: (
            -row.microbatch_size,
            row.gradient_accumulation_steps,
            row.candidate_id,
        ),
    )[0]


def select_family_matrix_candidate(family: str) -> TargetCandidate:
    if family not in CAMPAIGN_FAMILIES:
        raise LocalSmokeError(f"family {family!r} is outside the campaign")
    precision_order = (
        ("fp32_tf32", "bf16", "fp16_grad_scaler", "mixed_structured")
        if PROFILE.is_native_a10
        else ("fp32_ieee",)
    )
    desired = precision_order[CAMPAIGN_FAMILIES.index(family) % len(precision_order)]
    matches = tuple(
        row
        for row in build_target_manifest().candidates
        if row.family_id == family
        and (
            PROFILE.is_native_a10
            or row.quota_modality in {"audio", "tabular", "graph"}
        )
        and row.precision_policy["policy_id"] == desired
        and row.execution["mode"] == "eager"
    )
    if not matches and PROFILE.is_native_a10:
        matches = tuple(
            row
            for row in build_target_manifest().candidates
            if row.family_id == family and row.execution["mode"] == "eager"
        )
    if not matches:
        raise LocalSmokeError(f"no deterministic one-batch candidate for {family}")
    return sorted(
        matches,
        key=lambda row: (
            row.microbatch_size,
            row.gradient_accumulation_steps,
            row.candidate_id,
        ),
    )[0]


def _execute_one_batch_update(
    candidate: TargetCandidate,
    device: torch.device,
    *,
    adapter: Any,
    raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    configure_precision_backends(candidate)
    batch = bind_candidate_batch_shape(candidate, adapter, _move(raw, device))
    model = _build_model(candidate, adapter, device)
    optimizer = _build_optimizer(candidate, model)
    scheduler = _build_scheduler(
        candidate,
        optimizer,
        total_optimizer_steps=five_epoch_optimizer_steps(
            candidate, expected_train_examples=4_096
        ),
    )
    with _precision_context(candidate, device):
        _, loss = _forward_loss(
            model,
            adapter,
            batch,
            bool(candidate.activation_checkpointing["enabled"]),
        )
    if not bool(torch.isfinite(loss)):
        raise LocalSmokeError(f"{candidate.family_id} fixture forward loss is non-finite")
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1_000)
        if candidate.precision_policy["gradient_scaler"]
        else None
    )
    _, updated_loss = _update(
        candidate=candidate,
        model=model,
        batch=batch,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        forward_loss=lambda: _forward_loss(
            model,
            adapter,
            batch,
            bool(candidate.activation_checkpointing["enabled"]),
        ),
        scaler=scaler,
    )
    if not bool(torch.isfinite(updated_loss)) or not _gradients_finite(model):
        raise LocalSmokeError(f"{candidate.family_id} fixture update is non-finite")
    return {
        "family_id": candidate.family_id,
        "task_id": candidate.task_id,
        "configuration_id": candidate.candidate_id,
        "precision_policy": candidate.precision_policy["policy_id"],
        "one_batch_update": "passed",
    }


def _run_fixture_one_batch_update(
    candidate: TargetCandidate, device: torch.device
) -> Mapping[str, Any]:
    adapter = adapter_for_task(candidate.task_id)
    fixture_sha256, raw = build_local_real_format_batch(adapter)
    result = {
        **_execute_one_batch_update(
            candidate,
            device,
            adapter=adapter,
            raw=raw,
        ),
        "fixture_sha256": fixture_sha256,
    }
    if PROFILE.uses_speech_v2:
        result["input_kind"] = "local_real_format_fixture_only"
    return result


def run_family_matrix_smoke(workspace: str | Path) -> Mapping[str, Any]:
    """Run one real-format fixture optimizer update for each campaign family."""

    backend = Rtx5090TelemetryBackend()
    device = torch.device("cuda")
    rows = [
        _run_fixture_one_batch_update(select_family_matrix_candidate(family), device)
        for family in CAMPAIGN_FAMILIES
    ]
    result: dict[str, Any] = {
        "version": (
            "perfseer_v3_nrp_a10_speech_rtx5090_family_matrix_smoke_v2"
            if PROFILE.uses_speech_v2
            else "perfseer_v3_nrp_a10_rtx5090_family_matrix_smoke_v1"
            if PROFILE.is_native_a10
            else "perfseer_v3_rtx5090_family_matrix_smoke_v1"
        ),
        "production_eligible": False,
        "hardware_provenance": backend.hardware_provenance,
        "family_count": len(rows),
        "families": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_write_json(Path(workspace).resolve() / "family-matrix.json", result)
    return canonical_value(result)


def _fixture_coverage_tokens(candidate: TargetCandidate) -> set[tuple[str, str]]:
    tokens = {
        ("family_precision", f"{candidate.family_id}:{candidate.precision_policy['policy_id']}"),
        ("execution", str(candidate.execution["mode"])),
        (
            "requested_effective_batch",
            str(candidate.microbatch_size * candidate.gradient_accumulation_steps),
        ),
    }
    if candidate.family_id == "independent_generated":
        tokens.add(("generated_lineage", candidate.source_lineage))
        tokens.add(
            (
                "generated_structure",
                str(candidate.architecture_parameters["architecture_specification"]),
            )
        )
    return tokens


def select_nonvision_fixture_matrix() -> tuple[TargetCandidate, ...]:
    """Greedily cover every executable family/precision and generated structure."""

    if not PROFILE.is_nonvision_4gpu:
        raise LocalSmokeError("non-vision fixture matrix requires its baked profile")
    manifest = build_target_manifest()
    required = set().union(
        *(_fixture_coverage_tokens(row) for row in manifest.candidates)
    )
    uncovered = set(required)
    remaining = list(manifest.candidates)
    selected: list[TargetCandidate] = []
    while uncovered:
        scored = [
            (
                len(_fixture_coverage_tokens(row) & uncovered),
                -row.microbatch_size,
                row.candidate_id,
                row,
            )
            for row in remaining
        ]
        score, _, _, chosen = max(scored, key=lambda value: value[:3])
        if score == 0:
            raise LocalSmokeError("fixture matrix cannot cover its required routes")
        selected.append(chosen)
        uncovered -= _fixture_coverage_tokens(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def run_nonvision_fixture_matrix_smoke(workspace: str | Path) -> Mapping[str, Any]:
    backend = Rtx5090TelemetryBackend()
    device = torch.device("cuda")
    candidates = select_nonvision_fixture_matrix()
    rows = [
        _run_fixture_one_batch_update(candidate, device) for candidate in candidates
    ]
    covered = set().union(*(_fixture_coverage_tokens(row) for row in candidates))
    result: dict[str, Any] = {
        "version": "perfseer_v3_nrp_a10_nonvision_rtx5090_fixture_matrix_v1",
        "production_eligible": False,
        "hardware_provenance": backend.hardware_provenance,
        "candidate_count": len(candidates),
        "family_count": len({row.family_id for row in candidates}),
        "coverage_tokens": sorted([list(row) for row in covered]),
        "updates": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_write_json(Path(workspace).resolve() / "nonvision-fixture-matrix.json", result)
    return canonical_value(result)


def run_manifest_construction_audit(workspace: str | Path) -> Mapping[str, Any]:
    """Instantiate every active configuration on the meta device without weights."""

    if not PROFILE.is_nonvision_4gpu:
        raise LocalSmokeError("construction audit requires the non-vision profile")
    from .models.base import FamilyModel

    task_kinds = {
        row.task_id: str(row.target_schema["kind"])
        for row in load_task_registry().entries
    }
    rows = []
    for candidate in build_target_manifest().candidates:
        module = importlib.import_module(candidate.factory_id)
        with torch.device("meta"):
            model = module.build_model(
                output_width=candidate.target_width,
                task_kind=task_kinds[candidate.task_id],
                seed=int(candidate.seed_policy["seed"]),
                architecture_parameters=candidate.architecture_parameters,
            )
        if not isinstance(model, FamilyModel) or model.family_id != candidate.family_id:
            raise LocalSmokeError("construction audit factory returned another family")
        parameters = tuple(model.named_parameters())
        parameter_count = sum(row.numel() for _, row in parameters)
        if parameter_count < 1 or any(row.device.type != "meta" for _, row in parameters):
            raise LocalSmokeError("construction audit allocated weights or made an empty model")
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family_id": candidate.family_id,
                "factory_entrypoint": f"{candidate.factory_id}:build_model",
                "parameter_count": parameter_count,
                "state_schema_sha256": canonical_sha256(
                    tuple((name, list(value.shape), str(value.dtype)) for name, value in parameters)
                ),
            }
        )
        del model, parameters
        if len(rows) % 100 == 0:
            gc.collect()
    if (
        len(rows) != 11_200
        or len({row["candidate_id"] for row in rows}) != 11_200
        or len({row["family_id"] for row in rows}) != 22
    ):
        raise LocalSmokeError("construction audit totals differ from the active corpus")
    result: dict[str, Any] = {
        "version": "perfseer_v3_nrp_a10_nonvision_manifest_construction_audit_v1",
        "candidate_count": len(rows),
        "family_count": len({row["family_id"] for row in rows}),
        "production_eligible": False,
        "rows": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_write_json(
        Path(workspace).resolve() / "manifest-construction-audit.json", result
    )
    return canonical_value(
        {
            key: value for key, value in result.items() if key != "rows"
        }
    )


def run_speech_precision_matrix_smoke(
    workspace: str | Path,
    *,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    kaggle_executable: str = "kaggle",
) -> Mapping[str, Any]:
    """Exercise all rebound audio/precision paths on one verified real speech view."""

    if not PROFILE.uses_speech_v2:
        raise LocalSmokeError("speech precision matrix requires the speech V2 profile")
    backend = Rtx5090TelemetryBackend()
    device = torch.device("cuda")
    manifest = build_target_manifest()
    entry = _entry("tensorflow-speech-yes-no")
    materialized = TaskMaterializer(
        workspace=Path(workspace).resolve() / "materialized" / entry.task_id,
        repository_root=Path(repository_root).resolve(),
        kaggle=KaggleCliClient(executable=kaggle_executable),
        preparer=PinnedMleBenchPreparer(Path(mlebench_checkout)),
    ).materialize(entry)
    from .a10_speech_crosswalk import build_crosswalk
    from .real_data import VerifiedPreparedDataset
    from .speech_substitution import load_speech_substitution_contract

    lineage = {
        str(row["native_v2_candidate_id"]): row for row in build_crosswalk(manifest).rows
    }
    dataset = VerifiedPreparedDataset(
        entry,
        materialized.public,
        materialized.prepared_view,
        materialized.inventory.archive_sha256,
    )
    rows = []
    families = (
        "panns_cnn14",
        "temporal_convolutional_network",
        "m5_waveform_cnn",
    )
    precisions = ("fp32_tf32", "bf16", "fp16_grad_scaler", "mixed_structured")
    for family in families:
        for precision in precisions:
            matches = tuple(
                row
                for row in manifest.candidates
                if row.task_id == "tensorflow-speech-yes-no"
                and row.family_id == family
                and row.precision_policy["policy_id"] == precision
                and row.execution["mode"] == "eager"
            )
            if not matches:
                raise LocalSmokeError(
                    f"speech V2 has no eager {family}/{precision} candidate"
                )
            candidate = sorted(
                matches,
                key=lambda row: (
                    row.microbatch_size,
                    row.gradient_accumulation_steps,
                    row.candidate_id,
                ),
            )[0]
            raw = dataset.build_batch(tuple(range(candidate.microbatch_size)))
            update = _execute_one_batch_update(
                candidate,
                device,
                adapter=adapter_for_task(candidate.task_id),
                raw=raw,
            )
            row_lineage = lineage[candidate.candidate_id]
            rows.append(
                {
                    **update,
                    "input_kind": "verified_real_speech_prepared_view",
                    "dataset_fingerprint": dataset.dataset_fingerprint,
                    "original_a10g_candidate_id": row_lineage[
                        "original_a10g_candidate_id"
                    ],
                    "native_v1_candidate_id": row_lineage["native_v1_candidate_id"],
                    "dataset_substitution": True,
                }
            )
    result: dict[str, Any] = {
        "version": "perfseer_v3_nrp_a10_speech_rtx5090_precision_matrix_v2",
        "production_eligible": False,
        "hardware_provenance": backend.hardware_provenance,
        "family_count": len(families),
        "precision_count": len(precisions),
        "update_count": len(rows),
        "dataset_fingerprint": materialized.view_manifest.dataset_fingerprint,
        "source_archive_sha256": materialized.inventory.archive_sha256,
        "remote_inventory_sha256": materialized.state.remote_inventory_sha256,
        "speech_source_lock_sha256": (
            materialized.source_lock.lock_sha256
            if materialized.source_lock is not None
            else None
        ),
        "substitution_contract_sha256": load_speech_substitution_contract().sha256,
        "updates": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_write_json(
        Path(workspace).resolve() / "speech-precision-matrix.json", result
    )
    return canonical_value(result)


def _entry(task_id: str) -> TaskRegistryEntry:
    matches = tuple(row for row in load_task_registry().entries if row.task_id == task_id)
    if len(matches) != 1:
        raise LocalSmokeError("smoke task is not unique in the task registry")
    return matches[0]


def run_local_smoke(
    *,
    workspace: str | Path,
    repository_root: str | Path,
    mlebench_checkout: str | Path,
    dependency_lock: str | Path,
    source_revision: str,
    source_tree_sha256: str,
    image_identity: str,
    models: Sequence[str] = LOCAL_SMOKE_MODELS,
    kaggle_executable: str = "kaggle",
) -> tuple[Mapping[str, Any], ...]:
    """Run selected labels sequentially and publish non-production records."""

    if tuple(models) != tuple(dict.fromkeys(models)) or not models:
        raise LocalSmokeError("smoke model selection must be non-empty and unique")
    root = Path(workspace).resolve()
    repository = Path(repository_root).resolve()
    lock_path = Path(dependency_lock).resolve()
    if len(source_revision) != 40 or len(source_tree_sha256) != 64:
        raise LocalSmokeError("source revision/tree identities are invalid")
    if not lock_path.is_file():
        raise LocalSmokeError("dependency lock is missing")
    backend = Rtx5090TelemetryBackend()
    preparer = PinnedMleBenchPreparer(Path(mlebench_checkout))
    records: list[Mapping[str, Any]] = []
    speech_lineage: Mapping[str, Mapping[str, Any]] = {}
    if PROFILE.uses_speech_v2:
        from .a10_speech_crosswalk import build_crosswalk

        speech_lineage = {
            str(row["native_v2_candidate_id"]): row
            for row in build_crosswalk().rows
        }
    for model in models:
        candidate = select_smoke_candidate(model)
        entry = _entry(candidate.task_id)
        task_root = root / "materialized" / candidate.task_id
        materialized = TaskMaterializer(
            workspace=task_root,
            repository_root=repository,
            kaggle=KaggleCliClient(executable=kaggle_executable),
            preparer=preparer,
        ).materialize(entry)
        result = run_five_epoch_training(
            candidate,
            entry,
            public_directory=materialized.public,
            prepared_directory=materialized.prepared_view,
            archive_sha256=materialized.inventory.archive_sha256,
            telemetry_backend=backend,
            device="cuda",
            allow_non_v100_test_device=True,
        )
        record: dict[str, Any] = {
            "version": LOCAL_SMOKE_VERSION,
            "production_eligible": False,
            "exclusion_reason": "observed hardware is RTX 5090, not the production target",
            "target_hardware_id": candidate.target_hardware_id,
            "hardware_provenance": backend.hardware_provenance,
            "hardware_sha256": backend.hardware_fingerprint,
            "source_revision": source_revision,
            "source_tree_sha256": source_tree_sha256,
            "dependency_lock_sha256": file_sha256(lock_path),
            "image_identity": image_identity,
            "configuration_id": candidate.candidate_id,
            "candidate": canonical_value(asdict(candidate)),
            "family_id": candidate.family_id,
            "task_id": candidate.task_id,
            "precision_policy": candidate.precision_policy,
            "dataset_fingerprint": materialized.view_manifest.dataset_fingerprint,
            "archive_sha256": materialized.inventory.archive_sha256,
            "total_epochs": 5,
            "warmup_epochs": [1, 2],
            "measured_epochs": [3, 4, 5],
            "run": result.to_dict(),
        }
        if PROFILE.uses_speech_v2:
            from .a10_crosswalk import REFERENCE_MANIFEST_SHA256
            from .a10_speech_crosswalk import NATIVE_V1_MANIFEST_SHA256
            from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract

            lineage = speech_lineage.get(candidate.candidate_id)
            if lineage is None:
                raise LocalSmokeError("speech V2 smoke candidate has no lineage row")
            record["reference_provenance"] = {
                "original_a10g_candidate_id": lineage["original_a10g_candidate_id"],
                "native_v1_candidate_id": lineage["native_v1_candidate_id"],
                "native_v2_root_candidate_id": candidate.candidate_id,
                "original_a10g_manifest_sha256": REFERENCE_MANIFEST_SHA256,
                "native_v1_manifest_sha256": NATIVE_V1_MANIFEST_SHA256,
                "substitution_contract_sha256": load_speech_substitution_contract().sha256,
                "dataset_substitution": candidate.task_id == SPEECH_TASK_ID,
                "source_archive_sha256": materialized.inventory.archive_sha256,
                "remote_inventory_sha256": materialized.state.remote_inventory_sha256,
                "speech_source_lock_sha256": (
                    materialized.source_lock.lock_sha256
                    if materialized.source_lock is not None
                    else None
                ),
            }
        record["record_sha256"] = canonical_sha256(record)
        atomic_write_json(root / "records" / f"{model}.json", record)
        records.append(canonical_value(record))
    verify_local_smoke(root, expected_models=models)
    return tuple(records)


def verify_local_smoke(
    workspace: str | Path,
    *,
    expected_models: Sequence[str] = LOCAL_SMOKE_MODELS,
) -> tuple[Mapping[str, Any], ...]:
    root = Path(workspace).resolve()
    records: list[Mapping[str, Any]] = []
    fingerprints: set[str] = set()
    for model in expected_models:
        path = root / "records" / f"{model}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LocalSmokeError(f"missing or invalid smoke record for {model}") from error
        unhashed = dict(value)
        declared = unhashed.pop("record_sha256", None)
        run = FiveEpochRunResult.from_dict(value.get("run", {}))
        candidate = select_smoke_candidate(model)
        if (
            declared != canonical_sha256(unhashed)
            or value.get("version") != LOCAL_SMOKE_VERSION
            or value.get("production_eligible") is not False
            or value.get("family_id") != model
            or value.get("task_id") != candidate.task_id
            or value.get("configuration_id") != candidate.candidate_id
            or value.get("precision_policy") != candidate.precision_policy
            or value.get("measured_epochs") != [3, 4, 5]
            or value.get("total_epochs") != 5
            or value.get("hardware_provenance", {}).get("name")
            != "NVIDIA GeForce RTX 5090"
            or value.get("hardware_provenance", {}).get("compute_capability") != [12, 0]
            or run.completed_epochs != (1, 2, 3, 4, 5)
            or tuple(row.epoch for row in run.epoch_measurements) != (3, 4, 5)
        ):
            raise LocalSmokeError(f"smoke record for {model} violates the contract")
        if PROFILE.uses_speech_v2:
            provenance = value.get("reference_provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("dataset_substitution") is not True
                or any(
                    not isinstance(provenance.get(key), str)
                    or len(str(provenance.get(key))) != 64
                    for key in (
                        "original_a10g_candidate_id",
                        "native_v1_candidate_id",
                        "original_a10g_manifest_sha256",
                        "native_v1_manifest_sha256",
                        "substitution_contract_sha256",
                        "source_archive_sha256",
                        "remote_inventory_sha256",
                        "speech_source_lock_sha256",
                    )
                )
            ):
                raise LocalSmokeError("speech V2 smoke lineage/source provenance differs")
        dataset_fingerprint = value.get("dataset_fingerprint")
        if not isinstance(dataset_fingerprint, str) or len(dataset_fingerprint) != 64:
            raise LocalSmokeError("smoke record has no real dataset fingerprint")
        fingerprints.add(dataset_fingerprint)
        records.append(value)
    if len(fingerprints) != len(records):
        raise LocalSmokeError("different smoke tasks unexpectedly share a dataset fingerprint")
    return tuple(records)


__all__ = [
    "LOCAL_SMOKE_MODELS",
    "CAMPAIGN_FAMILIES",
    "LOCAL_SMOKE_VERSION",
    "LocalSmokeError",
    "Rtx5090TelemetryBackend",
    "run_local_smoke",
    "run_family_matrix_smoke",
    "run_nonvision_fixture_matrix_smoke",
    "run_manifest_construction_audit",
    "run_speech_precision_matrix_smoke",
    "select_smoke_candidate",
    "select_nonvision_fixture_matrix",
    "verify_local_smoke",
]
