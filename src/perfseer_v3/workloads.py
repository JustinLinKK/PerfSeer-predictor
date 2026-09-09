"""Coverage-driven microbenchmark, composite, and external workload contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import canonical_json
from .graph_ir_v3 import GraphIRV3


SUPPORTED_PRECISIONS = ("float32", "float16", "bfloat16")
SHAPE_REGIMES = ("tiny", "small", "medium", "large", "boundary")
_REGIME_SIZE = {"tiny": 4, "small": 8, "medium": 16, "large": 32, "boundary": 33}


@dataclass(frozen=True)
class WorkloadDescriptor:
    workload_id: str
    data_layer: str
    source_group: str
    source_fingerprint: str
    family: str
    modality: str
    factory_id: str
    declared_operations: tuple[str, ...]
    shape_regime: str
    dtype: str
    phase: str
    optimizer: str | None
    batch_size: int
    gradient_accumulation_steps: int = 1
    config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.data_layer not in {"microbenchmark", "composite", "real", "generated"}:
            raise ValueError(f"invalid data layer {self.data_layer!r}")
        if not self.workload_id or not self.source_group or not self.source_fingerprint:
            raise ValueError("workload ID, source group, and source fingerprint are required")
        if not self.declared_operations:
            raise ValueError("declared_operations cannot be empty")
        if self.shape_regime not in SHAPE_REGIMES:
            raise ValueError(f"invalid shape regime {self.shape_regime!r}")
        if self.dtype not in SUPPORTED_PRECISIONS:
            raise ValueError(f"unsupported dtype {self.dtype!r}")
        if self.phase not in {"forward", "training"}:
            raise ValueError(f"invalid workload phase {self.phase!r}")
        if (
            self.phase == "training"
            and self.config.get("parameterized", True)
            and not self.optimizer
        ):
            raise ValueError("training workload requires an optimizer")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch and gradient accumulation must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class WorkloadInstance:
    model: nn.Module
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    target: Any | None = None
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None


class _Microbenchmark(nn.Module):
    def __init__(self, operation: str, size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.operation = operation
        self.size = size
        self.dtype = dtype
        if operation == "aten::linear":
            self.weight = nn.Parameter(torch.randn(size, size, dtype=dtype))
            self.bias = nn.Parameter(torch.randn(size, dtype=dtype))
        elif operation == "aten::conv2d":
            self.weight = nn.Parameter(torch.randn(size, 3, 3, 5, dtype=dtype))
            self.bias = nn.Parameter(torch.randn(size, dtype=dtype))
        elif operation == "aten::batch_norm":
            self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))
            self.register_buffer("running_mean", torch.zeros(size, dtype=dtype))
            self.register_buffer("running_var", torch.ones(size, dtype=dtype))
        elif operation == "aten::layer_norm":
            self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(size, dtype=dtype))
        elif operation == "aten::embedding":
            self.weight = nn.Parameter(torch.randn(max(64, size * 4), size, dtype=dtype))

    def forward(self, *args: torch.Tensor) -> Any:
        op = self.operation
        if op == "aten::linear":
            return F.linear(args[0], self.weight, self.bias)
        if op == "aten::matmul":
            return torch.matmul(args[0], args[1])
        if op == "aten::bmm":
            return torch.bmm(args[0], args[1])
        if op == "aten::conv2d":
            return F.conv2d(args[0], self.weight, self.bias, padding=(1, 2))
        if op == "aten::add.Tensor":
            return torch.add(args[0], args[1])
        if op == "aten::batch_norm":
            return F.batch_norm(
                args[0],
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=self.training,
            )
        if op == "aten::layer_norm":
            return F.layer_norm(args[0], (self.size,), self.weight, self.bias)
        if op == "aten::silu":
            return F.silu(args[0])
        if op == "aten::embedding":
            return F.embedding(args[0], self.weight)
        if op == "aten::transpose.int":
            return torch.transpose(args[0], -2, -1)
        if op == "aten::reshape":
            return torch.reshape(args[0], (args[0].shape[0], -1))
        if op == "aten::div.Tensor":
            return torch.div(args[0], args[1])
        if op == "aten::softmax.int":
            return torch.softmax(args[0], dim=-1)
        if op == "aten::gather":
            return torch.gather(args[0], 1, args[1])
        if op == "aten::topk":
            return torch.topk(args[0], k=min(4, args[0].shape[-1]), dim=-1)
        if op == "aten::clamp_min":
            return torch.clamp_min(args[0], 0.0)
        if op == "aten::logsumexp":
            return torch.logsumexp(args[0], dim=-1)
        if op == "aten::scaled_dot_product_attention":
            return F.scaled_dot_product_attention(args[0], args[1], args[2])
        raise ValueError(f"unsupported exact microbenchmark operation {op!r}")


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _micro_inputs(
    operation: str,
    size: int,
    batch_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    if operation == "aten::linear":
        return (torch.randn(batch_size, max(2, size // 2), size, dtype=dtype),)
    if operation == "aten::matmul":
        return (
            torch.randn(batch_size, size, size + 1, dtype=dtype),
            torch.randn(batch_size, size + 1, max(2, size - 1), dtype=dtype),
        )
    if operation == "aten::bmm":
        return (
            torch.randn(batch_size, size, size + 1, dtype=dtype),
            torch.randn(batch_size, size + 1, size, dtype=dtype),
        )
    if operation == "aten::conv2d":
        return (torch.randn(batch_size, 3, size, size + 2, dtype=dtype),)
    if operation == "aten::add.Tensor":
        return (
            torch.randn(batch_size, size, size, dtype=dtype),
            torch.randn(1, size, 1, dtype=dtype),
        )
    if operation == "aten::batch_norm":
        return (torch.randn(batch_size, size, max(2, size // 2), max(2, size // 2), dtype=dtype),)
    if operation in {"aten::layer_norm", "aten::silu", "aten::transpose.int", "aten::reshape"}:
        return (torch.randn(batch_size, size, size, dtype=dtype),)
    if operation == "aten::embedding":
        return (torch.randint(0, max(64, size * 4), (batch_size, size), dtype=torch.int64),)
    if operation == "aten::div.Tensor":
        return (
            torch.randn(batch_size, size, size, dtype=dtype),
            torch.rand(1, size, 1, dtype=dtype).clamp_min(0.1),
        )
    if operation in {
        "aten::softmax.int",
        "aten::topk",
        "aten::clamp_min",
        "aten::logsumexp",
    }:
        return (torch.randn(batch_size, size, size, dtype=dtype),)
    if operation == "aten::gather":
        source = torch.randn(batch_size, size, size, dtype=dtype)
        index = torch.randint(0, size, source.shape, dtype=torch.int64)
        return source, index
    if operation == "aten::scaled_dot_product_attention":
        head_dim = max(4, size // 2)
        shape = (batch_size, 2, size, head_dim)
        return (
            torch.randn(shape, dtype=dtype),
            torch.randn(shape, dtype=dtype),
            torch.randn(shape, dtype=dtype),
        )
    raise ValueError(f"unsupported operation {operation!r}")


def build_microbenchmark(descriptor: WorkloadDescriptor) -> WorkloadInstance:
    descriptor.validate()
    operation = str(descriptor.config["operation"])
    size = int(descriptor.config["size"])
    dtype = _dtype(descriptor.dtype)
    model = _Microbenchmark(operation, size, dtype)
    args = _micro_inputs(operation, size, descriptor.batch_size, dtype)
    if descriptor.phase == "training":
        model.train()
        args = tuple(
            value.detach().requires_grad_(True)
            if value.is_floating_point()
            else value
            for value in args
        )

        def loss_fn(output: Any, target: Any) -> torch.Tensor:
            tensor = output[0] if isinstance(output, tuple) else output
            return tensor.float().square().mean()

        optimizer_factories = {
            "sgd": lambda parameters: torch.optim.SGD(parameters, lr=1e-3, momentum=0.9),
            "adam": lambda parameters: torch.optim.Adam(parameters, lr=1e-3),
            "adamw": lambda parameters: torch.optim.AdamW(parameters, lr=1e-3),
        }
        optimizer_factory = (
            optimizer_factories[str(descriptor.optimizer)]
            if descriptor.optimizer is not None
            else None
        )
        return WorkloadInstance(
            model,
            args,
            {},
            target=torch.zeros(()),
            loss_fn=loss_fn,
            optimizer_factory=optimizer_factory,
        )
    model.eval()
    return WorkloadInstance(model, args, {})


class _ResidualComposite(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x))) + x


class _AttentionComposite(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
        ).squeeze(1)
        return self.norm(self.proj(attended) + x)


class _IndexComposite(nn.Module):
    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        selected = torch.gather(x, 1, index)
        values, _ = torch.topk(selected, min(4, selected.shape[-1]), dim=-1)
        return torch.logsumexp(values.clamp_min(1e-5), dim=-1)


class _VisionComposite(nn.Module):
    def __init__(self, channels: int, *, mobile: bool) -> None:
        super().__init__()
        groups = channels if mobile else 1
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=groups)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm(self.spatial(x)))
        return F.silu(self.pointwise(hidden) + x)


class _SequenceCoverageComposite(nn.Module):
    def __init__(self, dim: int, *, vocabulary: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("temperature", torch.tensor(float(max(1, dim)) ** 0.5))

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        gather_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_embedding = self.embedding(query_tokens)
        context_embedding = self.embedding(context_tokens)
        query = self.query(query_embedding)
        key = self.key(context_embedding)
        value = self.value(context_embedding)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value)
        normalized = self.norm(self.projection(attended) + query_embedding)
        expanded_index = gather_index.unsqueeze(-1).expand(
            -1,
            -1,
            normalized.shape[-1],
        )
        selected = torch.gather(normalized, 1, expanded_index)
        values, _ = torch.topk(selected, min(4, selected.shape[-1]), dim=-1)
        score = torch.logsumexp(values.clamp_min(1e-5), dim=-1)
        return score, probabilities


def build_composite(descriptor: WorkloadDescriptor) -> WorkloadInstance:
    descriptor.validate()
    size = int(descriptor.config["size"])
    dtype = _dtype(descriptor.dtype)
    if descriptor.factory_id == "composite.residual":
        channels = max(2, size // 2)
        model: nn.Module = _ResidualComposite(channels).to(dtype=dtype)
        args = (torch.randn(descriptor.batch_size, channels, size, size, dtype=dtype),)
    elif descriptor.factory_id == "composite.attention":
        model = _AttentionComposite(size).to(dtype=dtype)
        args = (torch.randn(descriptor.batch_size, max(2, size // 2), size, dtype=dtype),)
    elif descriptor.factory_id == "composite.index":
        model = _IndexComposite()
        source = torch.randn(descriptor.batch_size, size, max(4, size // 2), dtype=dtype)
        index = torch.randint(0, size, source.shape, dtype=torch.int64)
        args = (source, index)
    elif descriptor.factory_id in {
        "composite.vision_fusion",
        "composite.mobile_inverted",
    }:
        channels = max(4, size // 2)
        model = _VisionComposite(
            channels,
            mobile=descriptor.factory_id == "composite.mobile_inverted",
        ).to(dtype=dtype)
        args = (
            torch.randn(
                descriptor.batch_size,
                channels,
                size,
                size + 2,
                dtype=dtype,
            ),
        )
    elif descriptor.factory_id in {
        "composite.sequence_encoder",
        "composite.cross_attention",
        "composite.recommender_ranking",
    }:
        model = _SequenceCoverageComposite(size).to(dtype=dtype)
        sequence_length = max(4, size // 2)
        query_tokens = torch.randint(
            0,
            128,
            (descriptor.batch_size, sequence_length),
            dtype=torch.int64,
        )
        context_tokens = torch.randint(
            0,
            128,
            (descriptor.batch_size, sequence_length),
            dtype=torch.int64,
        )
        gather_index = torch.arange(
            sequence_length - 1,
            -1,
            -1,
            dtype=torch.int64,
        ).unsqueeze(0).expand(descriptor.batch_size, -1)
        args = (query_tokens, context_tokens, gather_index)
    else:
        raise ValueError(f"unknown composite factory {descriptor.factory_id!r}")
    model.eval()
    return WorkloadInstance(model, args, {})


_BOOTSTRAP_EXACT_OPERATIONS = (
    "aten::linear",
    "aten::matmul",
    "aten::conv2d",
    "aten::add.Tensor",
    "aten::batch_norm",
    "aten::layer_norm",
    "aten::silu",
    "aten::embedding",
    "aten::transpose.int",
    "aten::div.Tensor",
    "aten::softmax.int",
    "aten::gather",
    "aten::topk",
    "aten::clamp_min",
    "aten::logsumexp",
)
_PARAMETERIZED_EXACT_OPERATIONS = {
    "aten::linear",
    "aten::conv2d",
    "aten::batch_norm",
    "aten::layer_norm",
    "aten::embedding",
}


def _source_fingerprint(source_group: str) -> str:
    return hashlib.sha256(source_group.encode("utf-8")).hexdigest()


def default_microbenchmarks(
    *,
    dtypes: tuple[str, ...] = SUPPORTED_PRECISIONS,
    phases: tuple[str, ...] = ("forward", "training"),
    batch_sizes: tuple[int, ...] = (1, 2, 8),
    optimizers: tuple[str, ...] = ("sgd", "adam", "adamw"),
) -> tuple[WorkloadDescriptor, ...]:
    descriptors: list[WorkloadDescriptor] = []
    for operation in _BOOTSTRAP_EXACT_OPERATIONS:
        operation_slug = operation.replace("::", "_").replace(".", "_")
        source_group = f"micro:{operation}"
        for regime in SHAPE_REGIMES:
            size = _REGIME_SIZE[regime]
            for dtype in dtypes:
                for phase in phases:
                    parameterized = operation in _PARAMETERIZED_EXACT_OPERATIONS
                    phase_optimizers: tuple[str | None, ...]
                    if phase == "training" and parameterized:
                        phase_optimizers = optimizers
                    else:
                        phase_optimizers = (None,)
                    for batch_size in batch_sizes:
                        for optimizer in phase_optimizers:
                            optimizer_slug = optimizer or "none"
                            workload_id = (
                                f"micro_{operation_slug}_{regime}_{dtype}_{phase}"
                                f"_bs{batch_size}_{optimizer_slug}"
                            )
                            descriptor = WorkloadDescriptor(
                                workload_id=workload_id,
                                data_layer="microbenchmark",
                                source_group=source_group,
                                source_fingerprint=_source_fingerprint(source_group),
                                family="exact_operation",
                                modality="synthetic",
                                factory_id="microbenchmark.exact",
                                declared_operations=(operation,),
                                shape_regime=regime,
                                dtype=dtype,
                                phase=phase,
                                optimizer=optimizer,
                                batch_size=batch_size,
                                config={
                                    "operation": operation,
                                    "size": size,
                                    "parameterized": parameterized,
                                },
                            )
                            descriptor.validate()
                            descriptors.append(descriptor)
    return tuple(descriptors)


def default_composites(
    *,
    dtype: str = "float32",
) -> tuple[WorkloadDescriptor, ...]:
    specifications = (
        (
            "residual",
            "composite.residual",
            "residual_cnn",
            "image",
            ("aten::conv2d", "aten::batch_norm", "aten::silu", "aten::add.Tensor"),
        ),
        (
            "attention",
            "composite.attention",
            "transformer_block",
            "text",
            ("aten::linear", "aten::scaled_dot_product_attention", "aten::layer_norm"),
        ),
        (
            "index",
            "composite.index",
            "index_reduction",
            "tabular",
            ("aten::gather", "aten::topk", "aten::clamp_min", "aten::logsumexp"),
        ),
        (
            "vision_fusion",
            "composite.vision_fusion",
            "encoder_decoder_vision",
            "image",
            ("aten::conv2d", "aten::batch_norm", "aten::silu", "aten::add.Tensor"),
        ),
        (
            "mobile_inverted",
            "composite.mobile_inverted",
            "mobile_inverted_bottleneck",
            "image",
            ("aten::conv2d", "aten::batch_norm", "aten::silu", "aten::add.Tensor"),
        ),
        *(
            (
                name,
                f"composite.{name}",
                family,
                modality,
                (
                    "aten::embedding",
                    "aten::linear",
                    "aten::matmul",
                    "aten::transpose.int",
                    "aten::div.Tensor",
                    "aten::softmax.int",
                    "aten::add.Tensor",
                    "aten::layer_norm",
                    "aten::gather",
                    "aten::topk",
                    "aten::clamp_min",
                    "aten::logsumexp",
                ),
            )
            for name, family, modality in (
                ("sequence_encoder", "transformer_encoder", "text"),
                ("cross_attention", "transformer_cross_attention", "multimodal"),
                ("recommender_ranking", "embedding_recommender", "tabular"),
            )
        ),
    )
    descriptors: list[WorkloadDescriptor] = []
    for name, factory_id, family, modality, operations in specifications:
        source_group = f"composite:{name}"
        for regime in ("small", "medium", "large"):
            descriptors.append(
                WorkloadDescriptor(
                    workload_id=f"composite_{name}_{regime}_{dtype}",
                    data_layer="composite",
                    source_group=source_group,
                    source_fingerprint=_source_fingerprint(source_group),
                    family=family,
                    modality=modality,
                    factory_id=factory_id,
                    declared_operations=operations,
                    shape_regime=regime,
                    dtype=dtype,
                    phase="forward",
                    optimizer=None,
                    batch_size=2,
                    config={"size": _REGIME_SIZE[regime]},
                )
            )
    for descriptor in descriptors:
        descriptor.validate()
    return tuple(descriptors)


def default_source_workloads() -> tuple[WorkloadDescriptor, ...]:
    """Representative real-library sources; generated substitutes are excluded."""

    specifications = (
        (
            "source_torchvision_resnet18",
            "residual_cnn",
            "image",
            (
                "aten::conv2d",
                "aten::batch_norm",
                "aten::add_.Tensor",
                "aten::adaptive_avg_pool2d",
                "aten::linear",
            ),
        ),
        (
            "source_torchvision_mobilenet_v3_small",
            "mobile_cnn",
            "image",
            (
                "aten::conv2d",
                "aten::batch_norm",
                "aten::hardswish_",
                "aten::adaptive_avg_pool2d",
                "aten::linear",
            ),
        ),
        (
            "source_transformers_small_bert",
            "transformer_encoder",
            "text",
            (
                "aten::embedding",
                "aten::linear",
                "aten::layer_norm",
                "aten::scaled_dot_product_attention",
            ),
        ),
        (
            "source_pyg_small_gcn",
            "graph_convolution",
            "graph",
            (
                "aten::linear",
                "aten::index_select",
                "aten::scatter_add_",
            ),
        ),
    )
    result = []
    for case_id, family, modality, operations in specifications:
        source_group = f"real:{case_id}"
        result.append(
            WorkloadDescriptor(
                workload_id=case_id,
                data_layer="real",
                source_group=source_group,
                source_fingerprint=_source_fingerprint(source_group),
                family=family,
                modality=modality,
                factory_id=f"coverage_case.{case_id}",
                declared_operations=operations,
                shape_regime="small",
                dtype="float32",
                phase="forward",
                optimizer=None,
                batch_size=1,
                config={
                    "source_kind": "real_library",
                    "weights": "random_or_none",
                },
            )
        )
    return tuple(result)


def build_workload(descriptor: WorkloadDescriptor) -> WorkloadInstance:
    if descriptor.factory_id == "microbenchmark.exact":
        return build_microbenchmark(descriptor)
    if descriptor.factory_id.startswith("composite."):
        return build_composite(descriptor)
    if descriptor.factory_id.startswith("coverage_case."):
        from .coverage_corpus import representative_source_cases

        case_id = descriptor.factory_id.removeprefix("coverage_case.")
        matches = [
            case
            for case in representative_source_cases()
            if case.case_id == case_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown representative source case {case_id!r}")
        case = matches[0]
        model, args, kwargs = case.build()
        model.train(case.training)
        return WorkloadInstance(model, args, kwargs)
    raise ValueError(
        f"external workload factory {descriptor.factory_id!r} must be registered by its adapter"
    )


def validate_declared_operations(graph: GraphIRV3, descriptor: WorkloadDescriptor) -> None:
    semantic_summary = graph.metadata.get(
        "semantic_summary_pre_decomposition",
        {},
    )
    raw_histogram = semantic_summary.get("raw_target_histogram", {})
    captured = (
        set(raw_histogram)
        if isinstance(raw_histogram, Mapping)
        else {node.raw_target for node in graph.nodes}
    )
    missing = set(descriptor.declared_operations) - captured
    if missing:
        raise ValueError(
            f"workload {descriptor.workload_id!r} did not capture declared operation(s): "
            + ", ".join(sorted(missing))
        )


def manifest_payload(descriptors: Iterable[WorkloadDescriptor]) -> dict[str, Any]:
    rows = sorted((descriptor.to_dict() for descriptor in descriptors), key=lambda row: row["workload_id"])
    payload = {
        "manifest_version": "perfseer_v3_workloads_v1",
        "workloads": rows,
    }
    payload["sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def coverage_matrix(
    descriptors: Iterable[WorkloadDescriptor],
) -> dict[tuple[str, str, str, str], int]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for descriptor in descriptors:
        for operation in descriptor.declared_operations:
            counts[(operation, descriptor.shape_regime, descriptor.dtype, descriptor.phase)] += 1
    return dict(sorted(counts.items()))


def select_coverage_gaps(
    candidates: Iterable[WorkloadDescriptor],
    observed: Iterable[WorkloadDescriptor],
    *,
    limit: int,
    error_by_operation: Mapping[str, float] | None = None,
) -> tuple[WorkloadDescriptor, ...]:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    observed_counts = coverage_matrix(observed)
    errors = {str(key): float(value) for key, value in (error_by_operation or {}).items()}

    def score(descriptor: WorkloadDescriptor) -> tuple[float, str]:
        cells = [
            (operation, descriptor.shape_regime, descriptor.dtype, descriptor.phase)
            for operation in descriptor.declared_operations
        ]
        deficit = sum(1.0 / (1.0 + observed_counts.get(cell, 0)) for cell in cells)
        error = sum(errors.get(operation, 0.0) for operation in descriptor.declared_operations)
        expensive_rare_bonus = error / max(1, len(descriptor.declared_operations))
        return (-(deficit + expensive_rare_bonus), descriptor.workload_id)

    unique = {descriptor.workload_id: descriptor for descriptor in candidates}
    ranked = sorted(unique.values(), key=score)
    return tuple(ranked[:limit])


__all__ = [
    "SHAPE_REGIMES",
    "SUPPORTED_PRECISIONS",
    "WorkloadDescriptor",
    "WorkloadInstance",
    "build_composite",
    "build_microbenchmark",
    "build_workload",
    "coverage_matrix",
    "default_composites",
    "default_microbenchmarks",
    "default_source_workloads",
    "manifest_payload",
    "select_coverage_gaps",
    "validate_declared_operations",
]
