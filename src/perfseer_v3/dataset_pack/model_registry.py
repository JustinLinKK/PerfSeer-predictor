"""Versioned, quota-bound planning registry for the 35 model families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from perfseer_v3.op_registry import OperationRegistry

from .contracts import MODEL_REGISTRY_VERSION, ModelFamilyDefinition
from .fingerprints import canonical_sha256, canonical_value
from .quota import FROZEN_QUOTA_CELLS, QuotaPlan, load_quota_plan
from .task_registry import TaskRegistry, load_task_registry


DEFAULT_MODEL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "registries" / "a10g_model_registry.yaml"
)
_ROOT_KEYS = {
    "version",
    "target_hardware_id",
    "training_approved",
    "factory_status",
    "entries",
}
_SOURCE_ENTRY_KEYS = {
    "family_id",
    "modality",
    "adapter_ids",
    "assertions",
    "fields",
}
_SERIALIZED_ENTRY_KEYS = set(ModelFamilyDefinition.__dataclass_fields__)
_SERIALIZED_REGISTRY_KEYS = {
    "version",
    "target_hardware_id",
    "training_approved",
    "factory_status",
    "quota_plan_sha256",
    "task_registry_sha256",
    "entries",
    "registry_sha256",
}


class ModelRegistryError(ValueError):
    """Raised when a model-family declaration is stale or overclaims implementation."""


def _source_file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ModelRegistryError(f"implemented model source is missing: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implemented_source_sha256(
    family_id: str,
    definition_payload: Mapping[str, Any],
) -> str:
    models = Path(__file__).resolve().parent / "models"
    source_inputs = {
            "definition": definition_payload,
            "lineage_module_sha256": _source_file_sha256(models / f"{family_id}.py"),
            "module_api_sha256": _source_file_sha256(models / "module_api.py"),
            "model_contract_sha256": _source_file_sha256(models / "base.py"),
            "shared_factory_sha256": _source_file_sha256(models / "factory.py"),
            "architecture_binding_sha256": _source_file_sha256(models / "architecture.py"),
    }
    if family_id == "independent_generated":
        source_inputs["generated_lineage_registry_sha256"] = _source_file_sha256(
            models.parent / "generated_lineages.py"
        )
    return canonical_sha256(source_inputs)


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
            raise ModelRegistryError(f"duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ModelRegistryError(f"{context} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise ModelRegistryError(
            f"{context} schema differs; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value:
        raise ModelRegistryError(f"{context} must be a non-empty string")
    return value


def _strings(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ModelRegistryError(f"{context} must be a non-empty sequence")
    if any(type(item) is not str or not item for item in value):
        raise ModelRegistryError(f"{context} must contain non-empty strings")
    return tuple(value)


def model_entry_from_dict(value: Mapping[str, Any]) -> ModelFamilyDefinition:
    raw = _mapping(value, context="serialized model entry")
    _exact_keys(raw, _SERIALIZED_ENTRY_KEYS, context="serialized model entry")
    entry = ModelFamilyDefinition(
        registry_version=_text(raw["registry_version"], context="registry_version"),
        family_id=_text(raw["family_id"], context="family_id"),
        modality=_text(raw["modality"], context="modality"),
        factory_id=_text(raw["factory_id"], context="factory_id"),
        adapter_ids=_strings(raw["adapter_ids"], context="adapter_ids"),
        source_lineage=_text(raw["source_lineage"], context="source_lineage"),
        source_sha256=_text(raw["source_sha256"], context="source_sha256"),
        faithful_operation_assertions=_strings(
            raw["faithful_operation_assertions"], context="faithful_operation_assertions"
        ),
        adjustable_architecture_fields=_strings(
            raw["adjustable_architecture_fields"], context="adjustable_architecture_fields"
        ),
    )
    entry.validate()
    return entry


@dataclass(frozen=True)
class ModelRegistry:
    version: str
    target_hardware_id: str
    training_approved: bool
    factory_status: str
    quota_plan_sha256: str
    task_registry_sha256: str
    entries: tuple[ModelFamilyDefinition, ...]

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_value(asdict(self))
        payload["registry_sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRegistry":
        raw = _mapping(value, context="serialized model registry")
        _exact_keys(raw, _SERIALIZED_REGISTRY_KEYS, context="serialized model registry")
        entries = raw["entries"]
        if not isinstance(entries, (list, tuple)):
            raise ModelRegistryError("serialized model entries must be a sequence")
        registry = cls(
            version=_text(raw["version"], context="version"),
            target_hardware_id=_text(raw["target_hardware_id"], context="target_hardware_id"),
            training_approved=raw["training_approved"],
            factory_status=_text(raw["factory_status"], context="factory_status"),
            quota_plan_sha256=_text(raw["quota_plan_sha256"], context="quota_plan_sha256"),
            task_registry_sha256=_text(
                raw["task_registry_sha256"], context="task_registry_sha256"
            ),
            entries=tuple(model_entry_from_dict(entry) for entry in entries),
        )
        registry.validate()
        if raw["registry_sha256"] != registry.sha256:
            raise ModelRegistryError("serialized model registry hash mismatch")
        return registry

    def validate(
        self,
        *,
        quota: QuotaPlan | None = None,
        tasks: TaskRegistry | None = None,
    ) -> None:
        quota = quota or load_quota_plan()
        tasks = tasks or load_task_registry()
        if self.version != MODEL_REGISTRY_VERSION:
            raise ModelRegistryError("model registry version mismatch")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise ModelRegistryError("model registry must target AWS A10G")
        if type(self.training_approved) is not bool or self.training_approved:
            raise ModelRegistryError("local model registry must remain explicitly unapproved")
        if self.factory_status != "implemented_phase3_factories_v1":
            raise ModelRegistryError("model factory implementation status is not the frozen Phase 3 version")
        if self.quota_plan_sha256 != quota.sha256 or self.task_registry_sha256 != tasks.sha256:
            raise ModelRegistryError("model registry quota/task identity mismatch")
        frozen = tuple((modality, family_id) for modality, family_id, _, _ in FROZEN_QUOTA_CELLS)
        actual = tuple((entry.modality, entry.family_id) for entry in self.entries)
        if actual != frozen:
            raise ModelRegistryError("model registry must exactly follow the ordered 35-cell quota")
        if len({entry.family_id for entry in self.entries}) != len(self.entries):
            raise ModelRegistryError("model family IDs must be unique")
        task_by_id = {entry.task_id: entry for entry in tasks.entries}
        operation_ids = {rule.canonical_id for rule in OperationRegistry.load().rules}
        referenced_tasks: set[str] = set()
        for entry in self.entries:
            entry.validate()
            if entry.registry_version != self.version:
                raise ModelRegistryError("model entry registry version mismatch")
            expected_factory = f"perfseer_v3.dataset_pack.models.{entry.family_id}"
            expected_lineage = f"model_family:{entry.family_id}:v1"
            if entry.factory_id != expected_factory:
                raise ModelRegistryError("model factory declaration differs from its family identity")
            if entry.source_lineage != expected_lineage:
                raise ModelRegistryError("model source lineage differs from its family identity")
            if len(set(entry.adapter_ids)) != len(entry.adapter_ids):
                raise ModelRegistryError("model adapter IDs must be unique")
            if len(set(entry.faithful_operation_assertions)) != len(
                entry.faithful_operation_assertions
            ):
                raise ModelRegistryError("model operation assertions must be unique")
            if len(set(entry.adjustable_architecture_fields)) != len(
                entry.adjustable_architecture_fields
            ):
                raise ModelRegistryError("model adjustable fields must be unique")
            for assertion in entry.faithful_operation_assertions:
                if assertion not in operation_ids and not assertion.startswith("structural:"):
                    raise ModelRegistryError(
                        f"model {entry.family_id} has an unregistered operation assertion {assertion}"
                    )
                if assertion == "structural:":
                    raise ModelRegistryError("structural operation assertions require an identity")
            definition_payload = {
                "family_id": entry.family_id,
                "modality": entry.modality,
                "adapter_ids": entry.adapter_ids,
                "faithful_operation_assertions": entry.faithful_operation_assertions,
                "adjustable_architecture_fields": entry.adjustable_architecture_fields,
                "factory_status": self.factory_status,
            }
            expected_source_sha256 = _implemented_source_sha256(
                entry.family_id, definition_payload
            )
            if entry.source_sha256 != expected_source_sha256:
                raise ModelRegistryError("implemented model source fingerprint does not match its definition")
            try:
                factory_module = importlib.import_module(entry.factory_id)
            except ImportError as error:
                raise ModelRegistryError(
                    f"implemented model factory {entry.factory_id!r} cannot be imported"
                ) from error
            if not callable(getattr(factory_module, "build_model", None)):
                raise ModelRegistryError(
                    f"implemented model factory {entry.factory_id!r} has no build_model callable"
                )
            for adapter_id in entry.adapter_ids:
                task = task_by_id.get(adapter_id)
                if task is None:
                    raise ModelRegistryError(f"model {entry.family_id} references unknown task {adapter_id}")
                if entry.modality not in {task.modality, "generated"}:
                    raise ModelRegistryError(
                        f"model {entry.family_id} uses incompatible {task.modality} adapter"
                    )
                referenced_tasks.add(adapter_id)
        if referenced_tasks != set(task_by_id):
            raise ModelRegistryError(
                "every task registry entry must be consumed by at least one compatible model family"
            )


def load_model_registry(
    path: str | Path = DEFAULT_MODEL_REGISTRY_PATH,
    *,
    quota: QuotaPlan | None = None,
    tasks: TaskRegistry | None = None,
) -> ModelRegistry:
    quota = quota or load_quota_plan()
    tasks = tasks or load_task_registry()
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    root = _mapping(raw, context="model registry root")
    _exact_keys(root, _ROOT_KEYS, context="model registry root")
    entries = root["entries"]
    if not isinstance(entries, list):
        raise ModelRegistryError("model registry entries must be a list")
    expanded = []
    for index, value in enumerate(entries):
        source = _mapping(value, context=f"model source entry {index}")
        _exact_keys(source, _SOURCE_ENTRY_KEYS, context=f"model source entry {index}")
        family_id = _text(source["family_id"], context="family_id")
        modality = _text(source["modality"], context="modality")
        definition_payload = {
            "family_id": family_id,
            "modality": modality,
            "adapter_ids": _strings(source["adapter_ids"], context="adapter_ids"),
            "faithful_operation_assertions": _strings(
                source["assertions"], context="assertions"
            ),
            "adjustable_architecture_fields": _strings(source["fields"], context="fields"),
            "factory_status": root["factory_status"],
        }
        expanded.append(
            ModelFamilyDefinition(
                registry_version=MODEL_REGISTRY_VERSION,
                family_id=family_id,
                modality=modality,
                factory_id=f"perfseer_v3.dataset_pack.models.{family_id}",
                adapter_ids=definition_payload["adapter_ids"],
                source_lineage=f"model_family:{family_id}:v1",
                source_sha256=_implemented_source_sha256(family_id, definition_payload),
                faithful_operation_assertions=definition_payload[
                    "faithful_operation_assertions"
                ],
                adjustable_architecture_fields=definition_payload[
                    "adjustable_architecture_fields"
                ],
            )
        )
    registry = ModelRegistry(
        version=_text(root["version"], context="version"),
        target_hardware_id=_text(root["target_hardware_id"], context="target_hardware_id"),
        training_approved=root["training_approved"],
        factory_status=_text(root["factory_status"], context="factory_status"),
        quota_plan_sha256=quota.sha256,
        task_registry_sha256=tasks.sha256,
        entries=tuple(expanded),
    )
    registry.validate(quota=quota, tasks=tasks)
    return registry


__all__ = [
    "DEFAULT_MODEL_REGISTRY_PATH",
    "ModelRegistry",
    "ModelRegistryError",
    "load_model_registry",
    "model_entry_from_dict",
]
