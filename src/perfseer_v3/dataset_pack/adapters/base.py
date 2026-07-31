"""Uniform adapters for the 22 pinned MLE-bench task identities.

The fixtures in this module are deliberately tiny and redistributable.  They
exercise real tensor formats and task losses, but they are never presented as
Kaggle measurements or as substitutes for validated shared prepared datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..fingerprints import canonical_sha256
from ..task_registry import FROZEN_TASK_SCHEMAS, TaskRegistryEntry, load_task_registry


class AdapterError(ValueError):
    """Raised when a task adapter or its fixture violates the frozen contract."""


@dataclass(frozen=True)
class PreparedTaskFixture:
    task_id: str
    modality: str
    task_kind: str
    samples: tuple[Mapping[str, Any], ...]
    fingerprint: str
    fixture_only: bool = True

    def validate(self) -> None:
        if not self.task_id or not self.modality or not self.task_kind:
            raise AdapterError("prepared fixture identities must be non-empty")
        if not self.samples:
            raise AdapterError("prepared fixture must contain samples")
        if len(self.fingerprint) != 64:
            raise AdapterError("prepared fixture fingerprint must be SHA-256")
        if self.fixture_only is not True:
            raise AdapterError("local prepared data must remain explicitly fixture-only")


_TASK_KINDS = {
    task_id: (str(schema["kind"]), int(schema["target_width"]))
    for task_id, schema in FROZEN_TASK_SCHEMAS.items()
}


def _tensor_digest(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if value is not None and all(hasattr(value, name) for name in ("x", "edge_index", "batch")):
        return {
            "kind": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "x": _tensor_digest(getattr(value, "x", None)),
            "edge_index": _tensor_digest(getattr(value, "edge_index", None)),
            "edge_attr": _tensor_digest(getattr(value, "edge_attr", None)),
            "batch": _tensor_digest(getattr(value, "batch", None)),
        }
    if isinstance(value, Mapping):
        return {str(key): _tensor_digest(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_tensor_digest(item) for item in value]
    return value


def _generator(task_id: str) -> torch.Generator:
    seed = int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
    result = torch.Generator(device="cpu")
    result.manual_seed(seed % (2**63 - 1))
    return result


def _vision_samples(task_id: str, kind: str, width: int) -> tuple[Mapping[str, Any], ...]:
    generator = _generator(task_id)
    image = torch.rand((4, 3, 16, 16), generator=generator)
    if kind == "image_restoration":
        clean = image.mul(0.8).add(0.1)
        target: torch.Tensor = clean
    elif kind == "multilabel_classification":
        target = torch.randint(0, 2, (4, width), generator=generator).float()
    else:
        target = torch.arange(4, dtype=torch.long).remainder(width)
    return ({"inputs": {"image": image}, "target": target},)


def _nlp_samples(task_id: str, kind: str, width: int) -> tuple[Mapping[str, Any], ...]:
    generator = _generator(task_id)
    token_ids = torch.randint(1, 48, (4, 8), generator=generator)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    if kind == "teacher_forced_seq2seq":
        target = token_ids.roll(-1, dims=1).remainder(width)
        decoder_ids = torch.cat((torch.ones((4, 1), dtype=torch.long), target[:, :-1]), dim=1)
        inputs = {"token_ids": token_ids, "attention_mask": mask, "decoder_ids": decoder_ids}
    elif kind == "multilabel_classification":
        target = torch.randint(0, 2, (4, width), generator=generator).float()
        inputs = {"token_ids": token_ids, "attention_mask": mask}
    else:
        target = torch.arange(4, dtype=torch.long).remainder(width)
        inputs = {"token_ids": token_ids, "attention_mask": mask}
    return ({"inputs": inputs, "target": target},)


def _audio_samples(task_id: str, kind: str, width: int) -> tuple[Mapping[str, Any], ...]:
    generator = _generator(task_id)
    waveform = torch.randn((4, 1, 256), generator=generator).mul(0.1)
    if kind == "multilabel_classification":
        target = torch.randint(0, 2, (4, width), generator=generator).float()
    else:
        target = torch.arange(4, dtype=torch.long).remainder(width)
    return ({"inputs": {"waveform": waveform, "sample_rate": 16_000}, "target": target},)


def _tabular_samples(task_id: str, kind: str, width: int) -> tuple[Mapping[str, Any], ...]:
    generator = _generator(task_id)
    dense = torch.randn((4, 8), generator=generator)
    categorical = torch.randint(0, 8, (4, 3), generator=generator)
    target = (
        dense[:, :1].mul(0.3).add(1.0)
        if kind == "regression"
        else torch.arange(4, dtype=torch.long).remainder(width)
    )
    return ({"inputs": {"dense": dense, "categorical": categorical}, "target": target},)


def _graph_samples(task_id: str) -> tuple[Mapping[str, Any], ...]:
    generator = _generator(task_id)
    nodes = torch.randn((8, 6), generator=generator)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 0, 2], [1, 2, 3, 0, 5, 6, 7, 4, 4, 6]],
        dtype=torch.long,
    )
    edge_features = torch.randn((edge_index.shape[1], 3), generator=generator)
    graph_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    target = torch.tensor([[0.25, 0.75], [0.5, 1.0]], dtype=torch.float32)
    inputs: dict[str, Any] = {
        "node_features": nodes,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "graph_index": graph_index,
        "graph_count": 2,
    }
    try:
        from torch_geometric.data import Batch, Data

        first = Data(x=nodes[:4], edge_index=edge_index[:, :4], edge_attr=edge_features[:4])
        second_edges = edge_index[:, 4:8] - 4
        second = Data(x=nodes[4:], edge_index=second_edges, edge_attr=edge_features[4:8])
        inputs["pyg_batch"] = Batch.from_data_list([first, second])
    except ImportError:
        inputs["pyg_batch"] = None
    return ({"inputs": inputs, "target": target},)


class TaskAdapter:
    """Concrete implementation of the eleven-method adapter contract."""

    def __init__(self, entry: TaskRegistryEntry) -> None:
        if entry.task_id not in _TASK_KINDS:
            raise AdapterError(f"task {entry.task_id!r} has no Phase 3 adapter")
        self.entry = entry
        self.task_kind, self.target_width = _TASK_KINDS[entry.task_id]
        if entry.target_schema.get("kind") != self.task_kind or entry.target_schema.get(
            "target_width"
        ) != self.target_width:
            raise AdapterError("task adapter executable schema differs from its registry entry")
        self._prepared: PreparedTaskFixture | None = None

    @property
    def task_id(self) -> str:
        return self.entry.task_id

    def prepare_dataset(self, *, fixture: bool = True) -> PreparedTaskFixture:
        if fixture is not True:
            raise AdapterError(
                "this method creates local fixtures only; real data must come from the "
                "validated shared prepared view"
            )
        builders: dict[str, Callable[[], tuple[Mapping[str, Any], ...]]] = {
            "vision": lambda: _vision_samples(self.task_id, self.task_kind, self.target_width),
            "nlp": lambda: _nlp_samples(self.task_id, self.task_kind, self.target_width),
            "audio": lambda: _audio_samples(self.task_id, self.task_kind, self.target_width),
            "tabular": lambda: _tabular_samples(self.task_id, self.task_kind, self.target_width),
            "graph": lambda: _graph_samples(self.task_id),
        }
        try:
            samples = builders[self.entry.modality]()
        except KeyError as error:
            raise AdapterError(f"unsupported adapter modality {self.entry.modality!r}") from error
        fingerprint = canonical_sha256(
            {
                "fixture_version": "perfseer_v3_phase3_fixture_v1",
                "task_id": self.task_id,
                "dataset_revision": self.entry.dataset_revision,
                "task_kind": self.task_kind,
                "samples": _tensor_digest(samples),
            }
        )
        self._prepared = PreparedTaskFixture(
            task_id=self.task_id,
            modality=self.entry.modality,
            task_kind=self.task_kind,
            samples=samples,
            fingerprint=fingerprint,
        )
        self._prepared.validate()
        return self._prepared

    def dataset_fingerprint(self) -> str:
        prepared = self._prepared or self.prepare_dataset()
        return prepared.fingerprint

    def build_train_dataset(self) -> tuple[Mapping[str, Any], ...]:
        prepared = self._prepared or self.prepare_dataset()
        return prepared.samples

    def build_validation_dataset(self) -> tuple[Mapping[str, Any], ...]:
        prepared = self._prepared or self.prepare_dataset()
        return tuple(
            {"inputs": dict(sample["inputs"]), "target": sample["target"].clone()}
            for sample in prepared.samples
        )

    def build_collator(self) -> Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]:
        def collate(samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
            if not samples:
                raise AdapterError("cannot collate an empty task batch")
            if len(samples) == 1:
                return samples[0]
            raise AdapterError("Phase 3 fixtures are already materialized as complete tiny batches")

        return collate

    def infer_task_schema(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "modality": self.entry.modality,
            "task_kind": self.task_kind,
            "target_width": self.target_width,
            "fixture_only": True,
        }

    def build_model_inputs(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        inputs = batch.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise AdapterError("adapter batch has no model inputs")
        return inputs

    def build_targets(self, batch: Mapping[str, Any]) -> torch.Tensor:
        target = batch.get("target")
        if not isinstance(target, torch.Tensor):
            raise AdapterError("adapter batch target must be a tensor")
        return target

    def build_loss(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        if self.task_kind == "single_label_classification":
            return F.cross_entropy(output, target.long())
        if self.task_kind == "multilabel_classification":
            return F.binary_cross_entropy_with_logits(output, target.float())
        if self.task_kind in {"regression", "graph_regression", "image_restoration"}:
            return F.mse_loss(output, target.float())
        if self.task_kind == "teacher_forced_seq2seq":
            return F.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1).long())
        raise AdapterError(f"unsupported loss kind {self.task_kind!r}")

    def validate_output_shape(self, output: Any, target: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise AdapterError("ordinary adapter output must be a tensor")
        if self.task_kind == "single_label_classification":
            valid = output.ndim == 2 and output.shape[0] == target.shape[0] and output.shape[1] >= self.target_width
        elif self.task_kind == "multilabel_classification":
            valid = output.shape == target.shape
        elif self.task_kind in {"regression", "graph_regression", "image_restoration"}:
            valid = output.shape == target.shape
        else:
            valid = output.shape[:-1] == target.shape and output.shape[-1] >= self.target_width
        if not valid:
            raise AdapterError(
                f"task {self.task_id} output shape {getattr(output, 'shape', None)} "
                f"is incompatible with target {tuple(target.shape)}"
            )

    def compute_smoke_metric(self, output: torch.Tensor, target: torch.Tensor) -> float:
        if self.task_kind == "single_label_classification":
            value = (output.argmax(dim=-1) == target).float().mean()
        elif self.task_kind == "multilabel_classification":
            value = ((output.sigmoid() >= 0.5) == target.bool()).float().mean()
        elif self.task_kind == "teacher_forced_seq2seq":
            value = (output.argmax(dim=-1) == target).float().mean()
        else:
            value = 1.0 / (1.0 + F.mse_loss(output.float(), target.float()))
        result = float(value.detach())
        if not 0.0 <= result <= 1.0:
            raise AdapterError("smoke metric must be finite and bounded")
        return result

    def examples_per_epoch(self) -> int:
        target = self.build_targets(self.build_train_dataset()[0])
        return int(target.shape[0])

    def build_verified_real_batch(
        self,
        public_directory: str,
        prepared_directory: str,
        *,
        archive_sha256: str,
        indices: Sequence[int],
    ) -> tuple[Any, Mapping[str, Any]]:
        """Decode a batch only after the shared view and selected bytes pass integrity."""

        from ..real_data import build_verified_real_batch

        return build_verified_real_batch(
            self,
            public_directory,
            prepared_directory,
            archive_sha256=archive_sha256,
            indices=indices,
        )


def adapter_for_task(task_id: str) -> TaskAdapter:
    registry = load_task_registry()
    try:
        entry = next(entry for entry in registry.entries if entry.task_id == task_id)
    except StopIteration as error:
        raise AdapterError(f"unknown task adapter {task_id!r}") from error
    return TaskAdapter(entry)


def all_task_adapters() -> tuple[TaskAdapter, ...]:
    return tuple(TaskAdapter(entry) for entry in load_task_registry().entries)


__all__ = [
    "AdapterError",
    "PreparedTaskFixture",
    "TaskAdapter",
    "adapter_for_task",
    "all_task_adapters",
]
