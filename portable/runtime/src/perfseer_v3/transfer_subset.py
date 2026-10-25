"""Deterministic grouped, stratified, latent-diverse transfer subset selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline import canonical_json
from .hardware import require_specific_hardware_id
from .hardware_transfer import (
    BaseTransferLineageV3,
    TRANSFER_LABEL_BUDGETS,
    TRANSFER_SPLIT_COUNTS,
)
from .version import TRANSFER_MANIFEST_VERSION


_ID_FIELDS = ("configuration_id", "sample_id", "config_id")
_GROUP_FIELDS = ("source_group", "source_lineage", "lineage_id")
_SIGNATURE_FIELDS = ("graph_signature", "graph_sha256")
_STRATUM_ALIASES = {
    "modality": ("modality", "quota_modality"),
    "model_family": ("model_family", "family_id"),
    "precision_policy": ("precision_policy", "precision", "precision_id"),
    "execution_mode": ("execution_mode",),
    "resource_regime": ("resource_regime",),
    "optimizer": ("optimizer", "optimizer_id"),
    "scheduler": ("scheduler", "scheduler_id"),
    "optimizer_scheduler": ("optimizer_scheduler",),
    "operation_cost_regime": ("operation_cost_regime",),
    "high_cost_operation_family": ("high_cost_operation_family",),
    "unknown_custom_regime": ("unknown_custom_regime",),
    "activation_checkpointing": ("activation_checkpointing",),
    "batch_extremity": ("batch_extremity",),
}
_STRATUM_FIELDS = tuple(_STRATUM_ALIASES)


def _first(row: Mapping[str, Any], fields: Sequence[str], *, context: str) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"{context} is missing all supported fields {tuple(fields)}")


def _row_id(row: Mapping[str, Any]) -> str:
    return _first(row, _ID_FIELDS, context="transfer candidate ID")


def _source_group(row: Mapping[str, Any]) -> str:
    return _first(row, _GROUP_FIELDS, context=f"candidate {_row_id(row)!r} source group")


def _graph_signature(row: Mapping[str, Any]) -> str:
    return _first(
        row,
        _SIGNATURE_FIELDS,
        context=f"candidate {_row_id(row)!r} graph signature",
    )


def _split(row: Mapping[str, Any]) -> str:
    value = str(row.get("split", ""))
    if value == "val":
        value = "validation"
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"candidate {_row_id(row)!r} has invalid grouped split {value!r}")
    return value


def _stratum_value(row: Mapping[str, Any], field: str) -> str | None:
    for alias in _STRATUM_ALIASES[field]:
        value = row.get(alias)
        if value not in (None, ""):
            if isinstance(value, Mapping):
                for nested_name in ("policy_id", "name", "enabled"):
                    if value.get(nested_name) not in (None, ""):
                        return str(value[nested_name])
            return str(value)
    if field == "optimizer_scheduler":
        optimizer = _stratum_value(row, "optimizer")
        scheduler = _stratum_value(row, "scheduler")
        if optimizer is not None and scheduler is not None:
            return f"{optimizer}|{scheduler}"
    return None


def _normalized_embedding(value: Sequence[Any], *, row_id: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"embedding for {row_id!r} must be a finite nonempty vector")
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _validate_group_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, str] = {}
    signatures: dict[str, str] = {}
    for row in rows:
        split = _split(row)
        for value, seen, label in (
            (_source_group(row), groups, "source group"),
            (_graph_signature(row), signatures, "graph signature"),
        ):
            prior = seen.setdefault(value, split)
            if prior != split:
                raise ValueError(f"{label} {value!r} leaks across {prior!r} and {split!r}")


def _validate_prior_manifest(
    prior_manifest: Mapping[str, Any],
    *,
    request: "TransferSubsetRequestV3",
    row_by_id: Mapping[str, Mapping[str, Any]],
    base_lineage: BaseTransferLineageV3,
) -> dict[str, list[str]]:
    payload = dict(prior_manifest)
    observed_sha256 = str(payload.pop("subset_sha256", ""))
    expected_sha256 = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("prior transfer subset content hash mismatch")
    if payload.get("manifest_version") != TRANSFER_MANIFEST_VERSION:
        raise ValueError("prior transfer subset manifest version mismatch")
    if payload.get("base_hardware_id") != request.base_hardware_id:
        raise ValueError("prior subset uses a different base GPU")
    if payload.get("target_hardware_id") != request.target_hardware_id:
        raise ValueError("prior subset targets a different GPU")
    try:
        prior_lineage = BaseTransferLineageV3.from_dict(payload.get("base_lineage", {}))
    except ValueError as exc:
        raise ValueError(f"prior subset base lineage is invalid: {exc}") from exc
    if prior_lineage != base_lineage:
        raise ValueError("prior subset uses different frozen A10 base lineage")
    prior_budget = int(payload.get("label_budget", 0))
    if prior_budget not in TRANSFER_LABEL_BUDGETS:
        raise ValueError("prior subset label budget is outside the frozen gates")
    budget_index = TRANSFER_LABEL_BUDGETS.index(request.label_budget)
    if budget_index == 0:
        raise ValueError("the 128-label pilot cannot have a prior subset")
    expected_prior_budget = TRANSFER_LABEL_BUDGETS[budget_index - 1]
    if prior_budget != expected_prior_budget:
        raise ValueError(
            f"nested transfer expansion requires the preceding {expected_prior_budget}-label subset"
        )
    expected_counts = dict(
        zip(("train", "validation", "test"), TRANSFER_SPLIT_COUNTS[prior_budget])
    )
    if payload.get("split_counts") != expected_counts:
        raise ValueError("prior subset split counts do not match its frozen label budget")
    if payload.get("test_selection_uses_model_errors") is not False:
        raise ValueError("prior subset does not attest frozen test selection")
    selection = payload.get("selection")
    if not isinstance(selection, list) or len(selection) != prior_budget:
        raise ValueError("prior subset selection count does not match its label budget")
    selected: dict[str, list[str]] = {
        split: [] for split in ("train", "validation", "test")
    }
    seen: set[str] = set()
    for item in selection:
        if not isinstance(item, Mapping):
            raise ValueError("prior subset selection rows must be mappings")
        configuration_id = str(item.get("configuration_id", ""))
        split = str(item.get("split", ""))
        if configuration_id in seen:
            raise ValueError("prior subset repeats a configuration ID")
        if configuration_id not in row_by_id:
            raise ValueError("prior subset contains an unknown configuration ID")
        row = row_by_id[configuration_id]
        if split not in selected or split != _split(row):
            raise ValueError("prior subset changed a configuration's grouped split")
        if item.get("source_group") != _source_group(row):
            raise ValueError("prior subset source-group lineage mismatch")
        if item.get("graph_signature") != _graph_signature(row):
            raise ValueError("prior subset graph-signature lineage mismatch")
        if str(item.get("graph_path", "")) != str(row.get("graph_path", "")):
            raise ValueError("prior subset graph-path lineage mismatch")
        seen.add(configuration_id)
        selected[split].append(configuration_id)
    if {split: len(values) for split, values in selected.items()} != expected_counts:
        raise ValueError("prior subset selection rows disagree with frozen split counts")
    probe_rows = payload.get("memory_boundary_probes")
    expected_probe_count = max(1, round(0.125 * prior_budget))
    if not isinstance(probe_rows, list) or len(probe_rows) != expected_probe_count:
        raise ValueError("prior subset memory-probe count differs from its budget")
    probe_ids: set[str] = set()
    selected_split_by_id = {
        configuration_id: split
        for split, identifiers in selected.items()
        for configuration_id in identifiers
    }
    for probe in probe_rows:
        if not isinstance(probe, Mapping):
            raise ValueError("prior subset memory-probe row is invalid")
        configuration_id = str(probe.get("configuration_id", ""))
        if configuration_id in probe_ids or configuration_id not in selected_split_by_id:
            raise ValueError("prior subset memory-probe identity is invalid")
        if probe.get("grouped_split") != selected_split_by_id[configuration_id]:
            raise ValueError("prior subset memory-probe split is invalid")
        if (
            probe.get("paired_batch_first") is not True
            or probe.get("retain_oom_and_repair") is not True
            or probe.get("batch_ladder")
            != "deterministic_power_of_two_until_first_oom_or_ceiling"
        ):
            raise ValueError("prior subset memory-probe policy is invalid")
        probe_ids.add(configuration_id)
    return selected


@dataclass(frozen=True)
class TransferSubsetRequestV3:
    target_hardware_id: str
    label_budget: int
    base_hardware_id: str = "nvidia_a10g_24gb_aws_g5"
    active_learning: bool = False

    def validate(self) -> None:
        base = require_specific_hardware_id(
            self.base_hardware_id, context="transfer subset base_hardware_id"
        )
        if base != "nvidia_a10g_24gb_aws_g5":
            raise ValueError("transfer subset base must be the frozen A10G corpus")
        target = require_specific_hardware_id(
            self.target_hardware_id, context="transfer subset target_hardware_id"
        )
        if target == self.base_hardware_id:
            raise ValueError("transfer subset target must differ from the A10 base")
        if self.label_budget not in TRANSFER_LABEL_BUDGETS:
            raise ValueError(f"label budget must be one of {TRANSFER_LABEL_BUDGETS}")


def _latent_distance(
    candidate_id: str,
    selected_ids: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> float:
    if not selected_ids:
        return 1.0
    candidate = embeddings[candidate_id]
    return min(float(np.linalg.norm(candidate - embeddings[item])) for item in selected_ids)


def _select_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    embeddings: Mapping[str, np.ndarray],
    already_selected: Sequence[str],
    active_scores: Mapping[str, float],
    active_learning: bool,
    coverage_requirements: set[tuple[str, str]],
    coverage_weights: Mapping[tuple[str, str], float],
) -> tuple[list[str], dict[str, str], set[tuple[str, str]]]:
    by_id = {_row_id(row): row for row in rows}
    selected = [item for item in already_selected if item in by_id]
    if len(selected) > count:
        raise ValueError("prior subset contains more rows than the requested nested budget")
    reasons = {item: "retained_from_smaller_budget" for item in selected}
    remaining = set(by_id) - set(selected)

    # Cache each candidate's nearest selected embedding. Incremental updates
    # reduce nested 1,024-label selection from cubic-like repeated scans to
    # O(pool * selected * embedding_dim).
    minimum_distances = {candidate_id: 1.0 for candidate_id in remaining}
    if selected:
        for candidate_id in remaining:
            minimum_distances[candidate_id] = _latent_distance(
                candidate_id,
                selected,
                embeddings,
            )

    def record_selection(chosen: str) -> None:
        selected.append(chosen)
        remaining.remove(chosen)
        minimum_distances.pop(chosen, None)
        chosen_embedding = embeddings[chosen]
        for candidate_id in remaining:
            distance = float(np.linalg.norm(embeddings[candidate_id] - chosen_embedding))
            minimum_distances[candidate_id] = min(
                minimum_distances[candidate_id],
                distance,
            )

    # Cover the remaining globally required strata that are available in this
    # frozen split. Coverage is assessed across the complete subset, not
    # independently inside a 16-row validation/test quota.
    uncovered = {
        requirement
        for requirement in coverage_requirements
        if any(
            _stratum_value(row, requirement[0]) == requirement[1]
            for row in rows
        )
    }
    covered = {
        requirement
        for requirement in coverage_requirements
        if any(
            _stratum_value(by_id[item], requirement[0]) == requirement[1]
            for item in selected
        )
    }
    uncovered -= covered
    while uncovered and remaining and len(selected) < count:
        candidates: list[tuple[float, float, str]] = []
        for candidate_id in sorted(remaining):
            row = by_id[candidate_id]
            coverage = sum(
                coverage_weights.get((field, value), 1.0)
                for field, value in uncovered
                if _stratum_value(row, field) == value
            )
            candidates.append(
                (
                    coverage,
                    minimum_distances[candidate_id],
                    candidate_id,
                )
            )
        coverage, _, chosen = max(candidates, key=lambda item: (item[0], item[1], -len(item[2]), item[2]))
        if coverage <= 0:
            break
        record_selection(chosen)
        row = by_id[chosen]
        newly_covered = {
            pair
            for pair in uncovered
            if _stratum_value(row, pair[0]) == pair[1]
        }
        reasons[chosen] = "categorical_coverage:" + ",".join(
            f"{field}={value}" for field, value in sorted(newly_covered)
        )
        uncovered -= newly_covered
        covered |= newly_covered

    while remaining and len(selected) < count:
        scored: list[tuple[float, str, str]] = []
        selected_strata = Counter(
            (field, value)
            for item in selected
            for field in _STRATUM_FIELDS
            if (value := _stratum_value(by_id[item], field)) is not None
        )
        for candidate_id in sorted(remaining):
            row = by_id[candidate_id]
            distance = minimum_distances[candidate_id]
            if active_learning:
                uncertainty = float(active_scores.get(candidate_id, 0.0))
                stratum_terms = [
                    1.0 / (1.0 + selected_strata[(field, value)])
                    for field in _STRATUM_FIELDS
                    if (value := _stratum_value(row, field)) is not None
                ]
                strata_bonus = sum(stratum_terms) / max(1, len(stratum_terms))
                memory_bonus = float(
                    str(_stratum_value(row, "resource_regime") or "").lower()
                    in {"heavy", "capacity_bound", "memory_bound"}
                )
                scheduler_bonus = float(row.get("scheduler_decision_sensitivity", 0.0) or 0.0)
                score = uncertainty + distance + strata_bonus + memory_bonus + scheduler_bonus
                reason = (
                    "active_learning:uncertainty+latent_distance+stratum+"
                    "memory_boundary+scheduler_sensitivity"
                )
            else:
                score = distance
                reason = "latent_farthest_point"
            scored.append((score, candidate_id, reason))
        _, chosen, reason = max(scored, key=lambda item: (item[0], item[1]))
        record_selection(chosen)
        reasons[chosen] = reason
    if len(selected) != count:
        raise ValueError(f"split has only {len(selected)} selectable rows; expected {count}")
    return selected, reasons, covered


def select_transfer_subset(
    rows: Sequence[Mapping[str, Any]],
    embeddings_raw: Mapping[str, Sequence[Any]],
    request: TransferSubsetRequestV3,
    *,
    base_lineage: Mapping[str, Any] | BaseTransferLineageV3,
    prior_manifest: Mapping[str, Any] | None = None,
    active_scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    request.validate()
    if isinstance(base_lineage, BaseTransferLineageV3):
        validated_base_lineage = base_lineage
        validated_base_lineage.validate()
    else:
        validated_base_lineage = BaseTransferLineageV3.from_dict(base_lineage)
    if len(rows) < request.label_budget:
        raise ValueError("accepted A10 corpus is smaller than the requested transfer budget")
    _validate_group_isolation(rows)
    ids = [_row_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("accepted A10 configuration IDs must be unique")
    if any(row.get("graph_path") in (None, "") for row in rows):
        raise ValueError(
            "every transfer candidate requires a frozen A10 graph path"
        )
    embeddings = {
        row_id: _normalized_embedding(embeddings_raw[row_id], row_id=row_id)
        for row_id in ids
        if row_id in embeddings_raw
    }
    missing_embeddings = sorted(set(ids) - set(embeddings))
    if missing_embeddings:
        raise ValueError(
            f"base-teacher embeddings are missing {len(missing_embeddings)} candidates"
        )
    dimensions = {vector.size for vector in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError("base-teacher embeddings have inconsistent dimensions")

    prior_selection: dict[str, list[str]] = {
        split: [] for split in ("train", "validation", "test")
    }
    row_by_id = {_row_id(row): row for row in rows}
    if prior_manifest is not None:
        prior_selection = _validate_prior_manifest(
            prior_manifest,
            request=request,
            row_by_id=row_by_id,
            base_lineage=validated_base_lineage,
        )
    elif request.label_budget != TRANSFER_LABEL_BUDGETS[0]:
        raise ValueError("nested transfer budgets require the preceding frozen subset")
    if request.active_learning and prior_manifest is None:
        raise ValueError("active learning requires the frozen pilot/prior subset")
    if active_scores is not None and not request.active_learning:
        raise ValueError("active scores require active_learning=True")
    if request.active_learning:
        prior_train = set(prior_selection["train"])
        train_pool = {
            _row_id(row)
            for row in rows
            if _split(row) == "train" and _row_id(row) not in prior_train
        }
        missing_scores = train_pool - set(active_scores or {})
        if missing_scores:
            raise ValueError(
                f"active-learning scores are missing {len(missing_scores)} unmeasured train rows"
            )
        invalid_scores = [
            key
            for key, value in (active_scores or {}).items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid_scores:
            raise ValueError("active-learning scores must be finite numeric values")

    coverage_requirements = {
        (field, value)
        for field in _STRATUM_FIELDS
        for row in rows
        if (value := _stratum_value(row, field)) is not None
    }
    requirement_candidate_counts = Counter(
        (field, value)
        for row in rows
        for field in _STRATUM_FIELDS
        if (value := _stratum_value(row, field)) is not None
    )
    requirement_split_counts = {
        requirement: len(
            {
                _split(row)
                for row in rows
                if _stratum_value(row, requirement[0]) == requirement[1]
            }
        )
        for requirement in coverage_requirements
    }
    coverage_weights = {
        requirement: (
            10.0 / requirement_split_counts[requirement]
            + 1.0 / requirement_candidate_counts[requirement]
        )
        for requirement in coverage_requirements
    }
    for split, identifiers in prior_selection.items():
        for identifier in identifiers:
            row = row_by_id[identifier]
            coverage_requirements -= {
                (field, value)
                for field in _STRATUM_FIELDS
                if (value := _stratum_value(row, field)) is not None
            }

    counts = dict(zip(("train", "validation", "test"), TRANSFER_SPLIT_COUNTS[request.label_budget]))
    selected_by_split: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if _split(row) == split]
        # Active learning never observes validation/test errors; validation and
        # test are extended only by the precommitted deterministic selector.
        use_active = bool(request.active_learning and split == "train")
        selected, split_reasons, covered = _select_split(
            split_rows,
            count=counts[split],
            embeddings=embeddings,
            already_selected=prior_selection[split],
            active_scores=active_scores or {},
            active_learning=use_active,
            coverage_requirements=coverage_requirements,
            coverage_weights=coverage_weights,
        )
        selected_by_split[split] = selected
        reasons.update(split_reasons)
        coverage_requirements -= covered
    if coverage_requirements:
        preview = ", ".join(
            f"{field}={value}"
            for field, value in sorted(coverage_requirements)[:10]
        )
        raise ValueError(
            "transfer label budget cannot cover all required categorical strata: "
            f"{preview}"
        )

    selection = [
        {
            "configuration_id": item,
            "split": split,
            "source_group": _source_group(row_by_id[item]),
            "graph_signature": _graph_signature(row_by_id[item]),
            "selection_reason": reasons[item],
            **(
                {"graph_path": str(row_by_id[item]["graph_path"])}
                if row_by_id[item].get("graph_path") not in (None, "")
                else {}
            ),
        }
        for split in ("train", "validation", "test")
        for item in selected_by_split[split]
    ]
    probe_count = max(1, round(0.125 * request.label_budget))
    probe_split_counts = {
        "train": round(0.75 * probe_count),
        "validation": round(0.125 * probe_count),
    }
    probe_split_counts["test"] = probe_count - sum(probe_split_counts.values())
    memory_probes = []
    for split in ("train", "validation", "test"):
        split_selection = [item for item in selection if item["split"] == split]
        heavy = [
            item
            for item in split_selection
            if str(
                _stratum_value(
                    row_by_id[item["configuration_id"]], "resource_regime"
                )
                or ""
            ).lower()
            in {"heavy", "capacity_bound", "memory_bound"}
        ]
        ordered_candidates = [
            *heavy,
            *[item for item in split_selection if item not in heavy],
        ]
        for item in ordered_candidates[: probe_split_counts[split]]:
            memory_probes.append(
                {
                    "configuration_id": item["configuration_id"],
                    "grouped_split": split,
                    "paired_batch_first": True,
                    "batch_ladder": (
                        "deterministic_power_of_two_until_first_oom_or_ceiling"
                    ),
                    "retain_oom_and_repair": True,
                }
            )
    payload: dict[str, Any] = {
        "manifest_version": TRANSFER_MANIFEST_VERSION,
        "base_lineage": validated_base_lineage.to_dict(),
        "base_hardware_id": request.base_hardware_id,
        "target_hardware_id": request.target_hardware_id,
        "label_budget": request.label_budget,
        "split_counts": counts,
        "selection_method": (
            "grouped_categorical_latent_diversity_active"
            if request.active_learning
            else "grouped_categorical_latent_diversity"
        ),
        "test_selection_uses_model_errors": False,
        "embedding_dimension": next(iter(dimensions)),
        "selection": selection,
        "memory_boundary_probes": memory_probes,
    }
    payload["subset_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def read_embeddings(path: str | Path) -> dict[str, Sequence[Any]]:
    source = Path(path)
    if source.suffix == ".npz":
        payload = np.load(source, allow_pickle=False)
        return {str(name): payload[name].tolist() for name in payload.files}
    if source.suffix == ".jsonl":
        return {
            _row_id(row): row["embedding"]
            for row in read_jsonl(source)
        }
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("embedding JSON must map configuration IDs to vectors")
    return {str(key): value for key, value in payload.items()}


__all__ = [
    "TransferSubsetRequestV3",
    "read_embeddings",
    "read_jsonl",
    "select_transfer_subset",
]
