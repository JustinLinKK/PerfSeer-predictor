"""Versioned ATen/custom operation registry with fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .baseline import canonical_json
from .version import OP_REGISTRY_VERSION


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("op_registry_v3.yaml")
SUPPORTED_COST_FORMULAS = frozenset(
    {
        "attention",
        "convolution",
        "copy",
        "dropout",
        "einsum",
        "elementwise",
        "embedding",
        "indexing",
        "linear",
        "loss",
        "matmul",
        "normalization",
        "optimizer",
        "pooling",
        "recurrent",
        "reduction",
        "resample",
        "softmax",
        "sort_select",
        "unknown",
        "view",
        "view_or_copy",
    }
)
SEMANTIC_PRESERVE_FAMILIES = frozenset(
    {
        "attention",
        "convolution",
        "custom_fused",
        "dense_matrix",
        "embedding_sequence",
        "normalization",
        "optimizer",
        "quantized_low_precision",
        "sparse_graph",
    }
)


@dataclass(frozen=True)
class OperationRule:
    raw: str
    canonical_id: str
    family: str
    exact_id: int
    cost_formula: str
    flags: tuple[str, ...]
    decomposition: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedOperation:
    raw_target: str
    canonical_id: str
    family: str
    family_id: int
    exact_id: int
    hash_bucket: int
    cost_formula: str
    flags: tuple[str, ...]
    decomposition: str
    is_custom: bool
    is_known: bool


class RegistryValidationError(ValueError):
    """Raised when a registry could produce unstable or ambiguous identities."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that fails instead of silently replacing duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RegistryValidationError(f"duplicate YAML mapping key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class OperationRegistry:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = json.loads(json.dumps(payload))
        self._validate()
        self.version = str(self.payload["version"])
        self.hash_buckets = int(self.payload["hash_buckets"])
        self.training_approved = bool(self.payload["selection"]["training_approved"])
        self.selection = dict(self.payload["selection"])
        self.families = tuple(str(item["name"]) for item in self.payload["families"])
        self.family_to_id = {name: index for index, name in enumerate(self.families)}
        self.rules = tuple(self._parse_rule(raw) for raw in self.payload["operations"])
        self._targets: dict[str, OperationRule] = {}
        self._alias_flags: dict[str, tuple[str, ...]] = {}
        for raw_payload, rule in zip(self.payload["operations"], self.rules):
            self._targets[rule.raw] = rule
            for raw_alias in raw_payload.get("aliases", ()):
                alias = self._alias_raw(raw_alias)
                self._targets[alias] = rule
                self._alias_flags[alias] = self._alias_extra_flags(raw_alias)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTRY_PATH) -> "OperationRegistry":
        payload = yaml.load(
            Path(path).read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
        if not isinstance(payload, Mapping):
            raise RegistryValidationError("registry root must be a mapping")
        return cls(payload)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.payload).encode("utf-8")).hexdigest()

    def _parse_rule(self, raw: Mapping[str, Any]) -> OperationRule:
        exact_value = raw.get("exact_id", "UNK")
        exact_id = 0 if exact_value == "UNK" else int(exact_value)
        return OperationRule(
            raw=str(raw["raw"]),
            canonical_id=str(raw["canonical_id"]),
            family=str(raw["family"]),
            exact_id=exact_id,
            cost_formula=str(raw.get("cost_formula", "unknown")),
            flags=tuple(str(value) for value in raw.get("flags", ())),
            decomposition=str(raw.get("decomposition", "preserve")),
            aliases=tuple(self._alias_raw(value) for value in raw.get("aliases", ())),
        )

    @staticmethod
    def _alias_raw(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("raw", ""))
        return str(value)

    @staticmethod
    def _alias_extra_flags(value: Any) -> tuple[str, ...]:
        if not isinstance(value, Mapping):
            return ()
        return tuple(str(flag) for flag in value.get("flags", ()))

    def _validate(self) -> None:
        required = {"version", "hash_buckets", "selection", "families", "operations"}
        missing = required - set(self.payload)
        if missing:
            raise RegistryValidationError(f"registry is missing keys: {sorted(missing)}")
        if self.payload["version"] != OP_REGISTRY_VERSION:
            raise RegistryValidationError(
                f"registry version {self.payload['version']!r} does not match {OP_REGISTRY_VERSION!r}"
            )
        if int(self.payload["hash_buckets"]) < 2:
            raise RegistryValidationError("hash_buckets must be >= 2 so bucket zero remains reserved")
        families = list(self.payload["families"])
        family_names = [str(item.get("name")) for item in families]
        family_ids = [int(item.get("id", -1)) for item in families]
        if family_ids != list(range(len(families))):
            raise RegistryValidationError("family IDs must be consecutive and preserve file order")
        if not family_names or family_names[0] != "unknown_or_custom":
            raise RegistryValidationError("family ID zero must be unknown_or_custom")
        if len(set(family_names)) != len(family_names):
            raise RegistryValidationError("family names must be unique")

        targets: set[str] = set()
        canonical_ids: set[str] = set()
        exact_ids: list[int] = []
        for index, raw in enumerate(self.payload["operations"]):
            if not isinstance(raw, Mapping):
                raise RegistryValidationError(f"operation {index} must be a mapping")
            target = str(raw.get("raw", ""))
            canonical = str(raw.get("canonical_id", ""))
            family = str(raw.get("family", ""))
            if not target or not canonical:
                raise RegistryValidationError(f"operation {index} has an empty raw/canonical identity")
            if target in targets:
                raise RegistryValidationError(f"duplicate operation target {target!r}")
            targets.add(target)
            if canonical in canonical_ids:
                raise RegistryValidationError(f"duplicate canonical operation ID {canonical!r}")
            canonical_ids.add(canonical)
            if family not in family_names:
                raise RegistryValidationError(f"operation {target!r} uses unknown family {family!r}")
            cost_formula = str(raw.get("cost_formula", "unknown"))
            if cost_formula not in SUPPORTED_COST_FORMULAS:
                raise RegistryValidationError(
                    f"operation {target!r} uses unsupported cost formula "
                    f"{cost_formula!r}"
                )
            decomposition = str(raw.get("decomposition", "preserve"))
            if decomposition not in {"preserve", "decompose"}:
                raise RegistryValidationError(
                    f"operation {target!r} uses invalid decomposition policy "
                    f"{decomposition!r}"
                )
            if (
                family in SEMANTIC_PRESERVE_FAMILIES
                and decomposition != "preserve"
            ):
                raise RegistryValidationError(
                    f"semantic operation {target!r} in family {family!r} "
                    "must be preserved"
                )
            for raw_alias in raw.get("aliases", ()):
                alias = self._alias_raw(raw_alias)
                if not alias:
                    raise RegistryValidationError(
                        f"operation {target!r} has an empty alias identity"
                    )
                if alias in targets:
                    raise RegistryValidationError(f"conflicting operation alias {alias!r}")
                targets.add(alias)
            exact_value = raw.get("exact_id", "UNK")
            if exact_value != "UNK":
                exact_id = int(exact_value)
                if exact_id <= 0:
                    raise RegistryValidationError("exact ID zero is reserved for UNK")
                exact_ids.append(exact_id)
        if sorted(exact_ids) != list(range(1, len(exact_ids) + 1)):
            raise RegistryValidationError("non-UNK exact IDs must be unique and consecutive")

        selection = self.payload["selection"]
        if not isinstance(selection, Mapping) or "training_approved" not in selection:
            raise RegistryValidationError("selection.training_approved is required")
        if bool(selection["training_approved"]) and not selection.get("gpu_time_report_sha256"):
            raise RegistryValidationError("training-approved registry requires a measured GPU-time report hash")

    def stable_hash_bucket(self, raw_target: str) -> int:
        digest = hashlib.sha256(raw_target.encode("utf-8")).digest()
        return 1 + int.from_bytes(digest[:8], "big") % (self.hash_buckets - 1)

    def resolve(self, raw_target: str) -> ResolvedOperation:
        raw = str(raw_target)
        rule = self._targets.get(raw)
        if rule is None:
            namespace = raw.split("::", 1)[0] if "::" in raw else ""
            is_custom = bool(namespace and namespace not in {"aten", "prims", "prim"})
            return ResolvedOperation(
                raw_target=raw,
                canonical_id="UNK",
                family="unknown_or_custom",
                family_id=0,
                exact_id=0,
                hash_bucket=self.stable_hash_bucket(raw),
                cost_formula="unknown",
                flags=("custom",) if is_custom else (),
                decomposition="preserve",
                is_custom=is_custom,
                is_known=False,
            )
        flags = tuple(dict.fromkeys((*rule.flags, *self._alias_flags.get(raw, ()))))
        return ResolvedOperation(
            raw_target=raw,
            canonical_id=rule.canonical_id,
            family=rule.family,
            family_id=self.family_to_id[rule.family],
            exact_id=rule.exact_id,
            hash_bucket=self.stable_hash_bucket(rule.canonical_id),
            cost_formula=rule.cost_formula,
            flags=flags,
            decomposition=rule.decomposition,
            is_custom=rule.family == "custom_fused",
            is_known=True,
        )

    def runtime_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "hash_buckets": self.hash_buckets,
            "training_approved": self.training_approved,
            "selection": self.selection,
            "families": list(self.families),
            "operations": [
                {
                    "raw": rule.raw,
                    "canonical_id": rule.canonical_id,
                    "family_id": self.family_to_id[rule.family],
                    "family": rule.family,
                    "exact_id": rule.exact_id,
                    "cost_formula": rule.cost_formula,
                    "flags": list(rule.flags),
                    "decomposition": rule.decomposition,
                    "aliases": [
                        (
                            {
                                "raw": alias,
                                "flags": list(self._alias_flags[alias]),
                            }
                            if self._alias_flags.get(alias)
                            else alias
                        )
                        for alias in rule.aliases
                    ],
                }
                for rule in self.rules
            ],
        }


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "OperationRegistry",
    "OperationRule",
    "RegistryValidationError",
    "ResolvedOperation",
    "SEMANTIC_PRESERVE_FAMILIES",
    "SUPPORTED_COST_FORMULAS",
]
