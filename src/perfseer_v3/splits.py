"""Leakage-safe source-family grouped dataset splitting."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .baseline import canonical_json
from .workloads import WorkloadDescriptor


EVALUATION_SLICE_NAMES = (
    "in_distribution_validation",
    "architecture_source_family_held_out",
    "operation_combination_held_out",
    "generated_code_robustness",
    "dynamic_shape_extrapolation",
    "precision_optimizer_held_out",
    "custom_oov_suite",
    "v2_compatible_matched_test",
)


def _allocate_group_counts(
    group_count: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    names = tuple(fractions)
    raw = {name: group_count * fractions[name] for name in names}
    counts = {name: math.floor(raw[name]) for name in names}
    for name in sorted(
        names,
        key=lambda candidate: (-(raw[candidate] - counts[candidate]), candidate),
    )[: group_count - sum(counts.values())]:
        counts[name] += 1

    # When a stratum has enough source groups, every requested split receives
    # at least one whole group. This avoids a valid-but-useless manifest where,
    # for example, every microbenchmark operation hashes into the training set.
    if group_count >= len(names):
        for empty_name in (name for name in names if counts[name] == 0):
            donors = sorted(
                (name for name in names if counts[name] > 1),
                key=lambda candidate: (-counts[candidate], -fractions[candidate], candidate),
            )
            if not donors:
                raise AssertionError("unable to allocate one source group per split")
            counts[donors[0]] -= 1
            counts[empty_name] += 1
    if sum(counts.values()) != group_count:
        raise AssertionError("group allocation did not preserve the group count")
    return counts


def grouped_split(
    descriptors: Iterable[WorkloadDescriptor],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 42,
) -> dict[str, tuple[WorkloadDescriptor, ...]]:
    fractions = dict(fractions or {"train": 0.8, "validation": 0.1, "test": 0.1})
    if not fractions or any(value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive")
    total = sum(fractions.values())
    normalized = {name: value / total for name, value in fractions.items()}
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name, fraction in normalized.items():
        running += fraction
        cumulative.append((name, running))

    groups: dict[str, list[WorkloadDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        descriptor.validate()
        groups[descriptor.source_group].append(descriptor)

    result: dict[str, list[WorkloadDescriptor]] = {name: [] for name in normalized}
    strata: dict[str, list[str]] = defaultdict(list)
    for source_group, rows in groups.items():
        data_layers = {row.data_layer for row in rows}
        if len(data_layers) != 1:
            raise ValueError(
                f"source group {source_group!r} spans multiple data layers"
            )
        strata[next(iter(data_layers))].append(source_group)

    for data_layer in sorted(strata):
        source_groups = sorted(
            strata[data_layer],
            key=lambda source_group: (
                hashlib.sha256(f"{seed}:{source_group}".encode("utf-8")).hexdigest(),
                source_group,
            ),
        )
        counts = _allocate_group_counts(len(source_groups), normalized)
        offset = 0
        for split_name in normalized:
            for source_group in source_groups[offset : offset + counts[split_name]]:
                result[split_name].extend(groups[source_group])
            offset += counts[split_name]
    return {
        name: tuple(sorted(rows, key=lambda descriptor: descriptor.workload_id))
        for name, rows in result.items()
    }


def validate_group_isolation(splits: Mapping[str, Iterable[WorkloadDescriptor]]) -> None:
    ownership: dict[str, str] = {}
    workload_ids: set[str] = set()
    for split_name, rows in splits.items():
        for descriptor in rows:
            if descriptor.workload_id in workload_ids:
                raise ValueError(f"duplicate workload ID {descriptor.workload_id!r} across splits")
            workload_ids.add(descriptor.workload_id)
            previous = ownership.setdefault(descriptor.source_group, split_name)
            if previous != split_name:
                raise ValueError(
                    f"source group {descriptor.source_group!r} leaks across {previous!r} and {split_name!r}"
                )


def split_manifest_payload(
    splits: Mapping[str, Iterable[WorkloadDescriptor]],
) -> dict[str, object]:
    materialized = {
        str(name): tuple(rows)
        for name, rows in splits.items()
    }
    validate_group_isolation(materialized)
    split_rows = {
        name: {
            "workload_ids": sorted(row.workload_id for row in rows),
            "source_groups": sorted({row.source_group for row in rows}),
        }
        for name, rows in sorted(materialized.items())
    }
    payload: dict[str, object] = {
        "split_manifest_version": "perfseer_v3_grouped_split_v1",
        "splits": split_rows,
    }
    payload["sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def evaluation_slice_manifest_payload(
    splits: Mapping[str, Iterable[WorkloadDescriptor]],
) -> dict[str, Any]:
    """Build the eight required, leakage-safe evaluation slice declarations.

    Empty required suites stay explicit and make ``complete`` false. Production
    data collection can therefore fill them without the bootstrap corpus
    silently claiming evaluation readiness.
    """

    materialized = {
        str(name): tuple(rows)
        for name, rows in splits.items()
    }
    validate_group_isolation(materialized)
    missing_partitions = {"train", "validation", "test"} - set(materialized)
    if missing_partitions:
        raise ValueError(
            "evaluation slices require train/validation/test partitions: "
            + ", ".join(sorted(missing_partitions))
        )
    train_groups = {row.source_group for row in materialized["train"]}
    validation = materialized["validation"]
    test = materialized["test"]
    held_out = (*validation, *test)

    selected: dict[str, tuple[WorkloadDescriptor, ...]] = {
        "in_distribution_validation": tuple(validation),
        "architecture_source_family_held_out": tuple(test),
        "operation_combination_held_out": tuple(
            row for row in test if row.data_layer == "composite"
        ),
        "generated_code_robustness": tuple(
            row for row in held_out if row.data_layer == "generated"
        ),
        "dynamic_shape_extrapolation": tuple(
            row for row in test if row.shape_regime == "boundary"
        ),
        "precision_optimizer_held_out": tuple(
            row
            for row in test
            if row.dtype in {"float16", "bfloat16"} or row.optimizer is not None
        ),
        "custom_oov_suite": tuple(
            row
            for row in held_out
            if bool(row.config.get("custom_oov"))
            or any(
                not operation.startswith(("aten::", "prims::"))
                for operation in row.declared_operations
            )
        ),
        "v2_compatible_matched_test": tuple(
            row for row in test if bool(row.config.get("v2_compatible"))
        ),
    }
    if tuple(selected) != EVALUATION_SLICE_NAMES:
        raise AssertionError("required evaluation slice names drifted")

    slice_rows: dict[str, dict[str, Any]] = {}
    for name, rows in selected.items():
        ordered = tuple(sorted(rows, key=lambda row: row.workload_id))
        leaked = sorted(
            {row.source_group for row in ordered} & train_groups
        )
        if leaked:
            raise ValueError(
                f"evaluation slice {name!r} leaks training source groups: "
                + ", ".join(leaked)
            )
        slice_rows[name] = {
            "available": bool(ordered),
            "workload_count": len(ordered),
            "workload_ids": [row.workload_id for row in ordered],
            "source_groups": sorted({row.source_group for row in ordered}),
        }

    missing = [
        name for name in EVALUATION_SLICE_NAMES
        if not slice_rows[name]["available"]
    ]
    payload: dict[str, Any] = {
        "evaluation_slice_manifest_version": "perfseer_v3_evaluation_slices_v1",
        "split_manifest_sha256": split_manifest_payload(materialized)["sha256"],
        "required_slices": list(EVALUATION_SLICE_NAMES),
        "complete": not missing,
        "missing_required_slices": missing,
        "slices": slice_rows,
    }
    payload["sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


__all__ = [
    "EVALUATION_SLICE_NAMES",
    "evaluation_slice_manifest_payload",
    "grouped_split",
    "split_manifest_payload",
    "validate_group_isolation",
]
