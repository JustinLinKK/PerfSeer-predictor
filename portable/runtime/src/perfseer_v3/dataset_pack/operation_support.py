"""Registry-derived V100 operation support planning contract."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from perfseer_v3.op_registry import OperationRegistry
from perfseer_v3.schema import build_feature_schema

from .fingerprints import canonical_sha256, canonical_value
from .coverage_config import (
    DEFAULT_COVERAGE_CONFIG_PATH,
    FROZEN_SHAPE_REGIMES,
    load_operation_coverage_config,
)


DEFAULT_SUPPORT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "registries" / "operation_support_v100.yaml"
)
DEFAULT_GENERATED_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "registries"
    / "operation_support_v100.generated.json"
)
SUPPORT_STATES = frozenset(
    {
        "required_measured",
        "structural_only",
        "hardware_unsupported",
        "out_of_scope",
        "deprecated",
    }
)
FROZEN_DTYPES = ("float32", "float16")
FROZEN_LAYOUTS = ("contiguous", "non_contiguous")
FROZEN_COMPOSITE_CONTEXTS = (
    "sequential",
    "residual_or_branch",
    "saved_activation_or_alias",
)


class OperationSupportError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise OperationSupportError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OperationSupportError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise OperationSupportError(
            f"{context} keys differ from schema; missing={missing}, unknown={unknown}"
        )


def _string(value: Any, *, context: str) -> str:
    if type(value) is not str or not value:
        raise OperationSupportError(f"{context} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise OperationSupportError(f"{context} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise OperationSupportError(f"{context} must not contain duplicates")
    return tuple(value)


def _identifier(prefix: str, value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"{prefix}:{slug}"


@dataclass(frozen=True)
class FamilySupportPolicy:
    family: str
    support_state: str
    reason: str
    required_phases: tuple[str, ...]
    required_backends: tuple[str, ...]

    def validate(self) -> None:
        if self.support_state not in SUPPORT_STATES:
            raise OperationSupportError(f"invalid support state {self.support_state!r}")
        if not self.reason:
            raise OperationSupportError(f"family {self.family!r} requires a written reason")
        if self.support_state == "required_measured":
            if not self.required_phases or not self.required_backends:
                raise OperationSupportError(
                    f"required family {self.family!r} needs phases and backends"
                )
        elif self.required_phases or self.required_backends:
            raise OperationSupportError(
                f"non-measured family {self.family!r} cannot declare measurement dimensions"
            )


@dataclass(frozen=True)
class OperationSupportEntry:
    canonical_id: str
    raw_target: str
    aliases: tuple[str, ...]
    family: str
    exact_id: int
    support_state: str
    reason: str
    structural_path: str
    structurally_encodable: bool
    exactly_covered: bool
    accuracy_validated: bool
    required_phases: tuple[str, ...]
    required_dtypes: tuple[str, ...]
    required_backends: tuple[str, ...]
    required_layouts: tuple[str, ...]
    required_shape_regimes: tuple[str, ...]
    microbenchmark_generator_ids: tuple[str, ...]
    composite_block_ids: tuple[str, ...]
    golden_test_ids: tuple[str, ...]
    mapping_status: str


@dataclass(frozen=True)
class OperationSupportContract:
    version: str
    target_hardware_id: str
    contract_status: str
    training_approved: bool
    registry_version: str
    registry_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str
    policy_sha256: str
    coverage_config_sha256: str
    generator_mapping_status: str
    generator_registry_sha256: str | None
    families: tuple[FamilySupportPolicy, ...]
    operations: tuple[OperationSupportEntry, ...]

    def unhashed_payload(self) -> dict[str, Any]:
        return canonical_value(asdict(self))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.unhashed_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.unhashed_payload()
        payload["contract_sha256"] = self.sha256
        return payload

    def validate(
        self,
        registry: OperationRegistry,
        *,
        policy_payload: Mapping[str, Any] | None = None,
        family_policies: tuple[FamilySupportPolicy, ...] | None = None,
        coverage_config: Any | None = None,
    ) -> None:
        if self.version != "perfseer_v3_v100_operation_support_contract_v1":
            raise OperationSupportError("operation support contract version mismatch")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise OperationSupportError("operation support contract must target NRP V100")
        if (
            self.contract_status != "planning"
            or type(self.training_approved) is not bool
            or self.training_approved
        ):
            raise OperationSupportError("local support contract must remain planning/unapproved")
        if self.generator_mapping_status != "declared_pending_phase2_implementation":
            raise OperationSupportError("Phase 1 generator mappings must remain declared and pending")
        if self.registry_version != registry.version or self.registry_sha256 != registry.sha256:
            raise OperationSupportError("support contract registry identity mismatch")
        if policy_payload is None:
            policy_payload, loaded_policies = _load_policy(DEFAULT_SUPPORT_POLICY_PATH, registry)
            family_policies = family_policies or loaded_policies
        if self.policy_sha256 != canonical_sha256(policy_payload):
            raise OperationSupportError("support contract policy hash mismatch")
        coverage_config = coverage_config or load_operation_coverage_config()
        if (
            self.coverage_config_sha256 != coverage_config.sha256
            or coverage_config.target_hardware_id != self.target_hardware_id
        ):
            raise OperationSupportError("support contract coverage-config identity mismatch")
        if self.generator_registry_sha256 is not None:
            raise OperationSupportError("Phase 1 cannot claim an implemented generator registry hash")
        feature_schema = build_feature_schema(registry)
        if (
            self.feature_schema_version != feature_schema["feature_schema_version"]
            or self.feature_schema_sha256 != feature_schema["feature_schema_sha256"]
        ):
            raise OperationSupportError("support contract feature-schema identity mismatch")
        if family_policies is not None and self.families != family_policies:
            raise OperationSupportError("support contract family policies differ from policy source")
        if tuple(policy.family for policy in self.families) != registry.families:
            raise OperationSupportError("support policy must explicitly cover every registry family")
        for policy in self.families:
            policy.validate()
        if len(self.operations) != len(registry.rules):
            raise OperationSupportError("support contract operation count differs from registry")
        expected_ids = tuple(rule.canonical_id for rule in registry.rules)
        actual_ids = tuple(entry.canonical_id for entry in self.operations)
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise OperationSupportError("support entries must map one-to-one in registry order")
        generator_ids: set[str] = set()
        golden_ids: set[str] = set()
        family_by_name = {policy.family: policy for policy in self.families}
        for rule, entry in zip(registry.rules, self.operations):
            policy = family_by_name[rule.family]
            if entry.family != rule.family or entry.raw_target != rule.raw:
                raise OperationSupportError(f"support identity mismatch for {rule.canonical_id}")
            if entry.aliases != rule.aliases or entry.exact_id != rule.exact_id:
                raise OperationSupportError(f"support aliases/exact ID mismatch for {rule.canonical_id}")
            if entry.support_state != policy.support_state or entry.reason != policy.reason:
                raise OperationSupportError(f"support state mismatch for {rule.canonical_id}")
            expected_path = "exact" if rule.exact_id > 0 else "family_hash_custom"
            if entry.structurally_encodable is not True or entry.structural_path != expected_path:
                raise OperationSupportError(f"missing structural path for {rule.canonical_id}")
            if entry.exactly_covered is not False or entry.accuracy_validated is not False:
                raise OperationSupportError("planning entries cannot claim measured/accuracy coverage")
            if entry.support_state == "required_measured":
                expected_micro = (_identifier("v100_microbenchmark", rule.canonical_id),)
                expected_golden = (_identifier("v100_golden", rule.canonical_id),)
                expected_composites = tuple(
                    _identifier("v100_composite", f"{rule.family}_{context}")
                    for context in FROZEN_COMPOSITE_CONTEXTS
                )
                if (
                    entry.required_phases != policy.required_phases
                    or entry.required_dtypes != FROZEN_DTYPES
                    or entry.required_backends != policy.required_backends
                    or entry.required_layouts != FROZEN_LAYOUTS
                    or entry.required_shape_regimes != FROZEN_SHAPE_REGIMES
                    or entry.microbenchmark_generator_ids != expected_micro
                    or entry.golden_test_ids != expected_golden
                    or entry.composite_block_ids != expected_composites
                    or entry.mapping_status != self.generator_mapping_status
                ):
                    raise OperationSupportError(
                        f"required operation {entry.canonical_id} differs from frozen mappings/dimensions"
                    )
                if not entry.microbenchmark_generator_ids or not entry.golden_test_ids:
                    raise OperationSupportError(
                        f"required operation {entry.canonical_id} lacks generator/golden mapping"
                    )
                for identifier, seen in (
                    (entry.microbenchmark_generator_ids[0], generator_ids),
                    (entry.golden_test_ids[0], golden_ids),
                ):
                    if identifier in seen:
                        raise OperationSupportError(f"duplicate declared mapping ID {identifier}")
                    seen.add(identifier)
            elif entry.microbenchmark_generator_ids or entry.golden_test_ids:
                raise OperationSupportError("non-measured operations cannot declare measurement mappings")
            elif any(
                (
                    entry.required_phases,
                    entry.required_dtypes,
                    entry.required_backends,
                    entry.required_layouts,
                    entry.required_shape_regimes,
                    entry.composite_block_ids,
                )
            ):
                raise OperationSupportError("non-measured operations cannot declare coverage dimensions")
            elif entry.mapping_status != self.generator_mapping_status:
                raise OperationSupportError("entry mapping status differs from contract")


_POLICY_KEYS = {
    "version",
    "target_hardware_id",
    "registry_version",
    "registry_sha256",
    "expected_operation_count",
    "contract_status",
    "training_approved",
    "generator_mapping_status",
    "defaults",
    "families",
}


def _load_policy(
    path: str | Path,
    registry: OperationRegistry,
) -> tuple[Mapping[str, Any], tuple[FamilySupportPolicy, ...]]:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _mapping(raw, context="support policy root")
    _exact_keys(root, _POLICY_KEYS, context="support policy root")
    if root["registry_version"] != registry.version or root["registry_sha256"] != registry.sha256:
        raise OperationSupportError(
            "support policy is stale for the current encoder registry; update it explicitly"
        )
    if type(root["expected_operation_count"]) is not int or root["expected_operation_count"] != len(
        registry.rules
    ):
        raise OperationSupportError("support policy expected operation count is stale")
    if type(root["training_approved"]) is not bool or root["training_approved"]:
        raise OperationSupportError("planning support policy must be explicitly unapproved")
    if _string(root["version"], context="version") != "perfseer_v3_v100_operation_support_policy_v1":
        raise OperationSupportError("support policy version mismatch")
    if _string(root["target_hardware_id"], context="target_hardware_id") != "nvidia_tesla_v100_sxm2_32gb_nrp":
        raise OperationSupportError("support policy must target NRP V100")
    if _string(root["contract_status"], context="contract_status") != "planning":
        raise OperationSupportError("support policy must remain in planning status")
    if _string(root["generator_mapping_status"], context="generator_mapping_status") != "declared_pending_phase2_implementation":
        raise OperationSupportError("generator mappings cannot claim implementation in Phase 1")
    defaults = _mapping(root["defaults"], context="support defaults")
    _exact_keys(
        defaults,
        {
            "required_dtypes",
            "required_layouts",
            "required_shape_regimes",
            "composite_contexts",
        },
        context="support defaults",
    )
    for name in defaults:
        _string_tuple(defaults[name], context=f"support defaults.{name}")
    families = _mapping(root["families"], context="support families")
    _exact_keys(families, set(registry.families), context="support families")
    policies = []
    for family in registry.families:
        entry = _mapping(families[family], context=f"family {family}")
        _exact_keys(
            entry,
            {"support_state", "reason", "required_phases", "required_backends"},
            context=f"family {family}",
        )
        policy = FamilySupportPolicy(
            family=family,
            support_state=_string(entry["support_state"], context=f"{family}.support_state"),
            reason=_string(entry["reason"], context=f"{family}.reason"),
            required_phases=_string_tuple(entry["required_phases"], context=f"{family}.required_phases"),
            required_backends=_string_tuple(entry["required_backends"], context=f"{family}.required_backends"),
        )
        policy.validate()
        policies.append(policy)
    return root, tuple(policies)


def build_operation_support_contract(
    *,
    registry: OperationRegistry | None = None,
    policy_path: str | Path = DEFAULT_SUPPORT_POLICY_PATH,
    coverage_config_path: str | Path = DEFAULT_COVERAGE_CONFIG_PATH,
) -> OperationSupportContract:
    registry = registry or OperationRegistry.load()
    root, policies = _load_policy(policy_path, registry)
    defaults = _mapping(root["defaults"], context="support defaults")
    dtypes = _string_tuple(defaults["required_dtypes"], context="required_dtypes")
    layouts = _string_tuple(defaults["required_layouts"], context="required_layouts")
    shapes = _string_tuple(defaults["required_shape_regimes"], context="required_shape_regimes")
    contexts = _string_tuple(defaults["composite_contexts"], context="composite_contexts")
    coverage_config = load_operation_coverage_config(coverage_config_path)
    if dtypes != FROZEN_DTYPES:
        raise OperationSupportError("support policy dtypes differ from the frozen V100 set")
    if layouts != FROZEN_LAYOUTS:
        raise OperationSupportError("support policy layouts differ from the frozen set")
    if shapes != coverage_config.required_shape_regimes:
        raise OperationSupportError("support policy shapes differ from the coverage config")
    if contexts != FROZEN_COMPOSITE_CONTEXTS or len(contexts) < (
        coverage_config.minimum_composite_contexts_when_semantically_possible
    ):
        raise OperationSupportError("support policy must declare the frozen three contexts")
    policy_by_family = {policy.family: policy for policy in policies}
    operations = []
    for rule in registry.rules:
        policy = policy_by_family[rule.family]
        required = policy.support_state == "required_measured"
        operations.append(
            OperationSupportEntry(
                canonical_id=rule.canonical_id,
                raw_target=rule.raw,
                aliases=rule.aliases,
                family=rule.family,
                exact_id=rule.exact_id,
                support_state=policy.support_state,
                reason=policy.reason,
                structural_path="exact" if rule.exact_id > 0 else "family_hash_custom",
                structurally_encodable=True,
                exactly_covered=False,
                accuracy_validated=False,
                required_phases=policy.required_phases if required else (),
                required_dtypes=dtypes if required else (),
                required_backends=policy.required_backends if required else (),
                required_layouts=layouts if required else (),
                required_shape_regimes=shapes if required else (),
                microbenchmark_generator_ids=(
                    (_identifier("v100_microbenchmark", rule.canonical_id),) if required else ()
                ),
                composite_block_ids=(
                    tuple(_identifier("v100_composite", f"{rule.family}_{context}") for context in contexts)
                    if required
                    else ()
                ),
                golden_test_ids=(
                    (_identifier("v100_golden", rule.canonical_id),) if required else ()
                ),
                mapping_status=str(root["generator_mapping_status"]),
            )
        )
    feature_schema = build_feature_schema(registry)
    contract = OperationSupportContract(
        version="perfseer_v3_v100_operation_support_contract_v1",
        target_hardware_id=str(root["target_hardware_id"]),
        contract_status=str(root["contract_status"]),
        training_approved=bool(root["training_approved"]),
        registry_version=registry.version,
        registry_sha256=registry.sha256,
        feature_schema_version=str(feature_schema["feature_schema_version"]),
        feature_schema_sha256=str(feature_schema["feature_schema_sha256"]),
        policy_sha256=canonical_sha256(root),
        coverage_config_sha256=coverage_config.sha256,
        generator_mapping_status=str(root["generator_mapping_status"]),
        generator_registry_sha256=None,
        families=policies,
        operations=tuple(operations),
    )
    contract.validate(
        registry,
        policy_payload=root,
        family_policies=policies,
        coverage_config=coverage_config,
    )
    return contract


def write_operation_support_contract(
    path: str | Path = DEFAULT_GENERATED_CONTRACT_PATH,
    *,
    registry: OperationRegistry | None = None,
    policy_path: str | Path = DEFAULT_SUPPORT_POLICY_PATH,
    coverage_config_path: str | Path = DEFAULT_COVERAGE_CONFIG_PATH,
) -> Path:
    contract = build_operation_support_contract(
        registry=registry,
        policy_path=policy_path,
        coverage_config_path=coverage_config_path,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


__all__ = [
    "DEFAULT_GENERATED_CONTRACT_PATH",
    "DEFAULT_SUPPORT_POLICY_PATH",
    "FROZEN_COMPOSITE_CONTEXTS",
    "FROZEN_DTYPES",
    "FROZEN_LAYOUTS",
    "FamilySupportPolicy",
    "OperationSupportContract",
    "OperationSupportEntry",
    "OperationSupportError",
    "SUPPORT_STATES",
    "build_operation_support_contract",
    "write_operation_support_contract",
]
