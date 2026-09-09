"""Golden model/task audit and observed operation-cell contribution matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from typing import Any, Mapping

import torch

from ..op_registry import OperationRegistry
from .adapters import AdapterError, TaskAdapter, adapter_for_task, all_task_adapters
from .fingerprints import canonical_sha256, canonical_value
from .generated_lineages import build_generated_lineage_registry
from .model_registry import ModelRegistry, load_model_registry
from .models import GoldenValidationResult, golden_validate_model
from .models.factory import default_adapter_id
from .operation_support import build_operation_support_contract
from .specialized_steps import SpecializedStepResult, run_specialized_training_fixtures
from .task_registry import TaskRegistry, load_task_registry


class ModelGoldenAuditError(ValueError):
    """Raised when the Phase 3 golden matrix is incomplete or inconsistent."""


@dataclass(frozen=True)
class AdapterGoldenResult:
    version: str
    task_id: str
    modality: str
    task_kind: str
    fixture_fingerprint: str
    methods: tuple[str, ...]
    examples_per_epoch: int
    loss: float
    output_gradient_norm: float
    smoke_metric: float
    pyg_batch_verified: bool | None
    fixture_only: bool
    training_approved: bool

    def validate(self) -> None:
        if self.version != "perfseer_v3_adapter_golden_v1":
            raise ModelGoldenAuditError("adapter golden version mismatch")
        if not self.task_id or not self.modality or not self.task_kind:
            raise ModelGoldenAuditError("adapter golden identities must be non-empty")
        if len(self.fixture_fingerprint) != 64:
            raise ModelGoldenAuditError("adapter fixture fingerprint must be SHA-256")
        if self.methods != REQUIRED_ADAPTER_METHODS:
            raise ModelGoldenAuditError("adapter does not expose the exact required method contract")
        if self.examples_per_epoch < 1 or self.loss < 0.0 or self.output_gradient_norm <= 0.0:
            raise ModelGoldenAuditError("adapter golden loss/backward evidence is invalid")
        if not 0.0 <= self.smoke_metric <= 1.0:
            raise ModelGoldenAuditError("adapter golden metric is outside [0, 1]")
        if self.modality == "graph" and self.pyg_batch_verified is not True:
            raise ModelGoldenAuditError("graph adapter did not construct a PyG batch")
        if self.modality != "graph" and self.pyg_batch_verified is not None:
            raise ModelGoldenAuditError("non-graph adapter has unexpected PyG evidence")
        if self.fixture_only is not True or self.training_approved is not False:
            raise ModelGoldenAuditError("local adapter golden must remain fixture-only and unapproved")


REQUIRED_ADAPTER_METHODS = (
    "prepare_dataset",
    "dataset_fingerprint",
    "build_train_dataset",
    "build_validation_dataset",
    "build_collator",
    "infer_task_schema",
    "build_model_inputs",
    "build_targets",
    "build_loss",
    "validate_output_shape",
    "compute_smoke_metric",
    "examples_per_epoch",
)


def _adapter_output(adapter: TaskAdapter, target: torch.Tensor) -> torch.Tensor:
    kind = adapter.task_kind
    if kind == "single_label_classification":
        shape = (target.shape[0], adapter.target_width)
    elif kind == "multilabel_classification":
        shape = tuple(target.shape)
    elif kind in {"regression", "graph_regression", "image_restoration"}:
        shape = tuple(target.shape)
    elif kind == "teacher_forced_seq2seq":
        shape = (*target.shape, adapter.target_width)
    else:
        raise ModelGoldenAuditError(f"unsupported adapter audit kind {kind!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(adapter.dataset_fingerprint()[:16], 16) % (2**63 - 1))
    return torch.randn(shape, generator=generator, requires_grad=True)


def golden_validate_adapter(adapter: TaskAdapter) -> AdapterGoldenResult:
    prepared = adapter.prepare_dataset()
    batch = adapter.build_collator()(adapter.build_train_dataset())
    if not adapter.build_validation_dataset():
        raise ModelGoldenAuditError("adapter validation fixture is empty")
    adapter.build_model_inputs(batch)
    target = adapter.build_targets(batch)
    output = _adapter_output(adapter, target)
    adapter.validate_output_shape(output, target)
    loss = adapter.build_loss(output, target)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ModelGoldenAuditError("adapter loss is not a finite scalar")
    loss.backward()
    if output.grad is None:
        raise ModelGoldenAuditError("adapter loss produced no output gradient")
    metric = adapter.compute_smoke_metric(output.detach(), target)
    pyg_verified = None
    if adapter.entry.modality == "graph":
        pyg = batch["inputs"].get("pyg_batch")
        pyg_verified = bool(
            pyg is not None
            and getattr(pyg, "num_graphs", 0) == target.shape[0]
            and getattr(pyg, "edge_index", torch.empty(2, 0)).shape[0] == 2
        )
    result = AdapterGoldenResult(
        version="perfseer_v3_adapter_golden_v1",
        task_id=adapter.task_id,
        modality=adapter.entry.modality,
        task_kind=adapter.task_kind,
        fixture_fingerprint=prepared.fingerprint,
        methods=REQUIRED_ADAPTER_METHODS,
        examples_per_epoch=adapter.examples_per_epoch(),
        loss=float(loss.detach()),
        output_gradient_norm=float(output.grad.detach().norm()),
        smoke_metric=metric,
        pyg_batch_verified=pyg_verified,
        fixture_only=True,
        training_approved=False,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ModelFactoryAudit:
    version: str
    evidence_scope: str
    target_hardware_id: str
    observed_hardware_id: str
    requested_backend_id: str
    model_registry_sha256: str
    task_registry_sha256: str
    operation_registry_sha256: str
    support_contract_sha256: str
    model_results: tuple[GoldenValidationResult, ...]
    adapter_results: tuple[AdapterGoldenResult, ...]
    specialized_step_results: tuple[SpecializedStepResult, ...]
    specialized_contract_checks: Mapping[str, bool]
    source_sha256_by_family: Mapping[str, str]
    observed_operation_cells: tuple[Mapping[str, Any], ...]
    training_approved: bool

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = canonical_value(asdict(self))
        payload["audit_sha256"] = self.sha256
        return payload

    def validate(
        self,
        *,
        models: ModelRegistry | None = None,
        tasks: TaskRegistry | None = None,
    ) -> None:
        models = models or load_model_registry()
        tasks = tasks or load_task_registry()
        operations = OperationRegistry.load()
        support = build_operation_support_contract()
        if self.version != "perfseer_v3_model_factory_audit_v1":
            raise ModelGoldenAuditError("model factory audit version mismatch")
        if self.evidence_scope != "local_redistributable_fixture":
            raise ModelGoldenAuditError("model factory audit evidence scope mismatch")
        if self.target_hardware_id != "nvidia_a10g_24gb_aws_g5":
            raise ModelGoldenAuditError("model factory audit target hardware mismatch")
        if self.observed_hardware_id != "local_cpu_fixture" or self.requested_backend_id != "cpu_eager":
            raise ModelGoldenAuditError("local audit cannot claim an A10G observed backend")
        if (
            self.model_registry_sha256 != models.sha256
            or self.task_registry_sha256 != tasks.sha256
            or self.operation_registry_sha256 != operations.sha256
            or self.support_contract_sha256 != support.sha256
        ):
            raise ModelGoldenAuditError("model factory audit registry binding mismatch")
        if tuple(result.family_id for result in self.model_results) != tuple(
            entry.family_id for entry in models.entries
        ):
            raise ModelGoldenAuditError("model golden matrix differs from the ordered 35-family registry")
        if tuple(result.task_id for result in self.adapter_results) != tuple(
            entry.task_id for entry in tasks.entries
        ):
            raise ModelGoldenAuditError("adapter golden matrix differs from the ordered 22-task registry")
        model_by_family = {entry.family_id: entry for entry in models.entries}
        for result in self.model_results:
            result.validate()
            declared_fields = model_by_family[
                result.family_id
            ].adjustable_architecture_fields
            if tuple(result.architecture_parameters) != tuple(sorted(declared_fields)):
                raise ModelGoldenAuditError(
                    "golden architecture binding differs from the model registry"
                )
        for result in self.adapter_results:
            result.validate()
        if tuple(result.step_id for result in self.specialized_step_results) != (
            "volumetric_3d_convolution_training",
            "semantic_segmentation_training",
        ):
            raise ModelGoldenAuditError("specialized training-step matrix is incomplete")
        for result in self.specialized_step_results:
            result.validate()
        expected_checks = _specialized_contract_checks(
            self.model_results, self.adapter_results
        )
        if dict(self.specialized_contract_checks) != expected_checks or not all(
            expected_checks.values()
        ):
            raise ModelGoldenAuditError("specialized family training contract is incomplete")
        expected_sources = {entry.family_id: entry.source_sha256 for entry in models.entries}
        if dict(self.source_sha256_by_family) != expected_sources:
            raise ModelGoldenAuditError("model factory audit source hashes differ from the registry")
        expected_cells = []
        for result in self.model_results:
            for phase, operations_in_phase in result.phase_canonical_operations.items():
                for operation in operations_in_phase:
                    expected_cells.append(
                        {
                            "family_id": result.family_id,
                            "task_id": result.task_id,
                            "canonical_operation_id": operation,
                            "phase": phase,
                            "backend_id": result.requested_backend_id,
                            "fixture_fingerprint": result.fixture_fingerprint,
                            "accepted_a10g_measurement": False,
                        }
                    )
        expected_cells = tuple(
            sorted(
                expected_cells,
                key=lambda row: (
                    row["family_id"], row["phase"], row["canonical_operation_id"]
                ),
            )
        )
        if self.observed_operation_cells != expected_cells:
            raise ModelGoldenAuditError("observed operation-cell contribution matrix is not reproducible")
        if self.training_approved is not False:
            raise ModelGoldenAuditError("local model audit cannot approve production training")


def run_model_factory_audit(*, optimization_steps: int = 3) -> ModelFactoryAudit:
    models = load_model_registry()
    tasks = load_task_registry()
    operation_registry = OperationRegistry.load()
    support = build_operation_support_contract()
    model_results = []
    for entry in models.entries:
        adapter_id = default_adapter_id(entry.family_id, entry.adapter_ids)
        adapter = adapter_for_task(adapter_id)
        module = importlib.import_module(entry.factory_id)
        builder = getattr(module, "build_model", None)
        if not callable(builder):
            raise ModelGoldenAuditError(f"factory {entry.factory_id!r} has no build_model callable")
        architecture_parameters = None
        if entry.family_id == "independent_generated":
            lineage = build_generated_lineage_registry().lineages[0]
            architecture_parameters = {
                "source_lineage": lineage.lineage_id,
                "architecture_specification": lineage.architecture_specification,
                "modality": lineage.modality,
                "depth": lineage.depth,
                "width": lineage.width,
            }
        model = builder(
            output_width=adapter.target_width,
            task_kind=adapter.task_kind,
            seed=0,
            architecture_parameters=architecture_parameters,
        )
        if model.family_id != entry.family_id:
            raise ModelGoldenAuditError("factory returned a model from the wrong family")
        model_results.append(
            golden_validate_model(
                model=model,
                adapter=adapter,
                factory_id=entry.factory_id,
                assertions=entry.faithful_operation_assertions,
                backend_id="cpu_eager",
                optimization_steps=optimization_steps,
            )
        )
    adapter_results = tuple(golden_validate_adapter(adapter) for adapter in all_task_adapters())
    specialized_results = run_specialized_training_fixtures()
    cells = []
    for result in model_results:
        for phase, operations_in_phase in result.phase_canonical_operations.items():
            for operation in operations_in_phase:
                cells.append(
                    {
                        "family_id": result.family_id,
                        "task_id": result.task_id,
                        "canonical_operation_id": operation,
                        "phase": phase,
                        "backend_id": result.requested_backend_id,
                        "fixture_fingerprint": result.fixture_fingerprint,
                        "accepted_a10g_measurement": False,
                    }
                )
    audit = ModelFactoryAudit(
        version="perfseer_v3_model_factory_audit_v1",
        evidence_scope="local_redistributable_fixture",
        target_hardware_id=models.target_hardware_id,
        observed_hardware_id="local_cpu_fixture",
        requested_backend_id="cpu_eager",
        model_registry_sha256=models.sha256,
        task_registry_sha256=tasks.sha256,
        operation_registry_sha256=operation_registry.sha256,
        support_contract_sha256=support.sha256,
        model_results=tuple(model_results),
        adapter_results=adapter_results,
        specialized_step_results=specialized_results,
        specialized_contract_checks=_specialized_contract_checks(
            tuple(model_results), adapter_results
        ),
        source_sha256_by_family={entry.family_id: entry.source_sha256 for entry in models.entries},
        observed_operation_cells=tuple(
            sorted(
                cells,
                key=lambda row: (
                    row["family_id"], row["phase"], row["canonical_operation_id"]
                ),
            )
        ),
        training_approved=False,
    )
    audit.validate(models=models, tasks=tasks)
    return audit


def _specialized_contract_checks(
    model_results: tuple[GoldenValidationResult, ...],
    adapter_results: tuple[AdapterGoldenResult, ...],
) -> dict[str, bool]:
    by_family = {result.family_id: result for result in model_results}
    pix2pix = by_family["pix2pix"]
    distillation = by_family["distilbert_distillation"]
    crf = by_family["bilstm_crf"]
    panns = by_family["panns_cnn14"]
    m5 = by_family["m5_waveform_cnn"]
    cgcnn = by_family["cgcnn"]
    graph_adapter = next(result for result in adapter_results if result.task_id == "nomad2018")
    return {
        "pix2pix_generator_updated": any(
            name.startswith("generator_") for name in pix2pix.changed_parameter_names
        ),
        "pix2pix_discriminator_updated": any(
            name.startswith("discriminator") for name in pix2pix.changed_parameter_names
        ),
        "distillation_student_updated": any(
            not name.startswith("teacher_") for name in distillation.changed_parameter_names
        ),
        "distillation_teacher_frozen": not any(
            name.startswith("teacher_") for name in distillation.changed_parameter_names
        ),
        "bilstm_crf_transition_updated": "transitions" in crf.changed_parameter_names,
        "panns_stft_observed": "python::torch.functional.stft"
        in panns.phase_raw_targets["forward"],
        "panns_log_mel_matmul_observed": "python::torch._VariableFunctionsClass.matmul"
        in panns.phase_raw_targets["forward"],
        "raw_waveform_convolution_observed": "aten.convolution.1d" in m5.phase_canonical_operations["forward"],
        "cgcnn_message_passing_observed": "structural:aten.segment_reduce"
        in cgcnn.verified_assertions,
        "pyg_batch_constructed": graph_adapter.pyg_batch_verified is True,
        "cgcnn_pyg_batch_consumed": cgcnn.pyg_batch_consumed is True,
    }


__all__ = [
    "AdapterGoldenResult",
    "ModelFactoryAudit",
    "ModelGoldenAuditError",
    "REQUIRED_ADAPTER_METHODS",
    "golden_validate_adapter",
    "run_model_factory_audit",
]
