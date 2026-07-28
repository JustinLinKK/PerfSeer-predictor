"""Recursive tensor metadata and pytree comparison helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from .diagnostics import ReplayValidationError


@dataclass(frozen=True)
class TensorMetadataV3:
    shape: tuple[int | str, ...]
    rank: int
    dtype: str
    element_width_bytes: int
    numel: int | None
    tensor_bytes: int | None
    stride: tuple[int | str, ...]
    memory_format: str


def symbolic_value(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError, RuntimeError):
        return str(value)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def memory_format_class(value: Any) -> str:
    try:
        if value.dim() == 4 and value.is_contiguous(memory_format=torch.channels_last):
            return "channels_last"
        if value.dim() == 5 and value.is_contiguous(memory_format=torch.channels_last_3d):
            return "channels_last_3d"
        if value.is_contiguous():
            return "contiguous"
        if getattr(value, "layout", torch.strided) == torch.sparse_coo:
            return "sparse_coo"
        if getattr(value, "layout", torch.strided) == torch.sparse_csr:
            return "sparse_csr"
    except (AttributeError, RuntimeError):
        pass
    return "strided" if hasattr(value, "stride") else "unknown"


def tensor_metadata(value: Any) -> TensorMetadataV3:
    shape = tuple(symbolic_value(dim) for dim in value.shape)
    concrete = all(isinstance(dim, int) for dim in shape)
    numel = math.prod(shape) if concrete else None
    dtype = value.dtype
    width = torch.empty((), dtype=dtype).element_size()
    try:
        stride = tuple(symbolic_value(item) for item in value.stride())
    except (AttributeError, RuntimeError):
        stride = ()
    return TensorMetadataV3(
        shape=shape,
        rank=len(shape),
        dtype=dtype_name(dtype),
        element_width_bytes=width,
        numel=numel,
        tensor_bytes=None if numel is None else numel * width,
        stride=stride,
        memory_format=memory_format_class(value),
    )


def flatten_tensor_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (value,)
    if isinstance(value, Mapping):
        flattened: list[Any] = []
        for key in sorted(value, key=str):
            flattened.extend(flatten_tensor_values(value[key]))
        return tuple(flattened)
    if isinstance(value, (tuple, list)):
        flattened = []
        for item in value:
            flattened.extend(flatten_tensor_values(item))
        return tuple(flattened)
    return ()


def tensor_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        metadata = tensor_metadata(value)
        return {
            "kind": "tensor",
            "shape": list(metadata.shape),
            "dtype": metadata.dtype,
            "stride": list(metadata.stride),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [tensor_signature(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [tensor_signature(item) for item in value]}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                {"key": str(key), "value": tensor_signature(value[key])}
                for key in sorted(value, key=str)
            ],
        }
    return {"kind": "value", "type": type(value).__name__, "repr": repr(value)}


def clone_inputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        cloned.requires_grad_(value.requires_grad)
        return cloned
    if isinstance(value, tuple):
        return tuple(clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [clone_inputs(item) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, clone_inputs(item)) for key, item in value.items())
    return value


def randomized_like(value: Any, *, generator: torch.Generator) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return torch.randint(0, 2, value.shape, generator=generator, device=value.device).bool()
        if value.dtype.is_floating_point:
            result = torch.randn(value.shape, dtype=value.dtype, generator=generator, device=value.device)
            result.requires_grad_(value.requires_grad)
            return result
        if value.dtype.is_complex:
            real = torch.randn(value.shape, generator=generator, device=value.device)
            imag = torch.randn(value.shape, generator=generator, device=value.device)
            return torch.complex(real, imag).to(value.dtype)
        low = int(value.min().item()) if value.numel() else 0
        high = int(value.max().item()) + 1 if value.numel() else 2
        if high <= low:
            high = low + 1
        return torch.randint(low, high, value.shape, dtype=value.dtype, generator=generator, device=value.device)
    if isinstance(value, tuple):
        return tuple(randomized_like(item, generator=generator) for item in value)
    if isinstance(value, list):
        return [randomized_like(item, generator=generator) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, randomized_like(item, generator=generator)) for key, item in value.items())
    return value


def _compare_leaf(eager: Any, exported: Any, path: str, rtol: float, atol: float) -> None:
    if isinstance(eager, torch.Tensor) or isinstance(exported, torch.Tensor):
        if not isinstance(eager, torch.Tensor) or not isinstance(exported, torch.Tensor):
            raise ReplayValidationError(f"{path}: output tensor structure differs")
        if eager.shape != exported.shape:
            raise ReplayValidationError(f"{path}: shape {eager.shape} != {exported.shape}")
        if eager.dtype != exported.dtype:
            raise ReplayValidationError(f"{path}: dtype {eager.dtype} != {exported.dtype}")
        try:
            torch.testing.assert_close(exported, eager, rtol=rtol, atol=atol, equal_nan=True)
        except AssertionError as exc:
            raise ReplayValidationError(f"{path}: values differ: {exc}") from exc
        return
    if type(eager) is not type(exported):
        raise ReplayValidationError(f"{path}: output types differ: {type(eager)} != {type(exported)}")
    if eager != exported:
        raise ReplayValidationError(f"{path}: output values differ: {eager!r} != {exported!r}")


def compare_output_pytrees(
    eager: Any,
    exported: Any,
    *,
    rtol: float,
    atol: float,
    path: str = "output",
) -> None:
    if isinstance(eager, Mapping):
        if not isinstance(exported, Mapping) or list(eager.keys()) != list(exported.keys()):
            raise ReplayValidationError(f"{path}: mapping structure differs")
        for key in eager:
            compare_output_pytrees(eager[key], exported[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
        return
    if isinstance(eager, (tuple, list)):
        if not isinstance(exported, type(eager)) or len(eager) != len(exported):
            raise ReplayValidationError(f"{path}: sequence structure differs")
        for index, (left, right) in enumerate(zip(eager, exported)):
            compare_output_pytrees(left, right, rtol=rtol, atol=atol, path=f"{path}[{index}]")
        return
    _compare_leaf(eager, exported, path, rtol, atol)


def flatten_nodes(value: Any) -> tuple[torch.fx.Node, ...]:
    if isinstance(value, torch.fx.Node):
        return (value,)
    if isinstance(value, Mapping):
        result: list[torch.fx.Node] = []
        for key in sorted(value, key=str):
            result.extend(flatten_nodes(value[key]))
        return tuple(result)
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(flatten_nodes(item))
        return tuple(result)
    return ()


__all__ = [
    "TensorMetadataV3",
    "clone_inputs",
    "compare_output_pytrees",
    "dtype_name",
    "flatten_nodes",
    "flatten_tensor_values",
    "memory_format_class",
    "randomized_like",
    "symbolic_value",
    "tensor_metadata",
    "tensor_signature",
]

