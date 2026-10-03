"""Public source-file capture entrypoint for PerfSeer v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .capture_export import CaptureOptions, CaptureResult, ReplayInputFactory, capture_export
from .inputs import TensorInputSpec, build_inputs
from .load import load_model_entry, source_sha256
from .op_registry import OperationRegistry


@dataclass(frozen=True)
class SourceModelSpecV3:
    source_path: str | Path
    entry: str
    positional_inputs: tuple[Any, ...]
    keyword_inputs: Mapping[str, Any] = field(default_factory=dict)
    constructor_args: tuple[Any, ...] = ()
    constructor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    dynamic_shapes: Any | None = None
    seed: int = 0

    @classmethod
    def from_shapes(
        cls,
        source_path: str | Path,
        entry: str,
        input_shapes: tuple[tuple[int, ...], ...],
        *,
        input_dtypes: tuple[str, ...] = ("float32",),
        constructor_args: tuple[Any, ...] = (),
        constructor_kwargs: Mapping[str, Any] | None = None,
        dynamic_shapes: Any | None = None,
    ) -> "SourceModelSpecV3":
        if len(input_dtypes) == 1 and len(input_shapes) > 1:
            input_dtypes = input_dtypes * len(input_shapes)
        if len(input_shapes) != len(input_dtypes):
            raise ValueError("input_shapes and input_dtypes must have the same length")
        return cls(
            source_path=source_path,
            entry=entry,
            positional_inputs=tuple(
                TensorInputSpec(tuple(shape), dtype=dtype)
                for shape, dtype in zip(input_shapes, input_dtypes)
            ),
            constructor_args=constructor_args,
            constructor_kwargs=dict(constructor_kwargs or {}),
            dynamic_shapes=dynamic_shapes,
        )


def capture_source(
    spec: SourceModelSpecV3,
    *,
    registry: OperationRegistry | None = None,
    options: CaptureOptions | None = None,
    replay_input_factory: ReplayInputFactory | None = None,
) -> CaptureResult:
    model = load_model_entry(
        spec.source_path,
        spec.entry,
        constructor_args=spec.constructor_args,
        constructor_kwargs=dict(spec.constructor_kwargs),
    )
    args, kwargs = build_inputs(spec.positional_inputs, spec.keyword_inputs, seed=spec.seed)
    return capture_export(
        model,
        args,
        kwargs,
        dynamic_shapes=spec.dynamic_shapes,
        registry=registry,
        options=options,
        source_fingerprint=source_sha256(spec.source_path),
        replay_input_factory=replay_input_factory,
    )


__all__ = ["SourceModelSpecV3", "capture_source"]

