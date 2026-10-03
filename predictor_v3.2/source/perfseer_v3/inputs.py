"""Example-input construction with positional, keyword, dtype, and pytree support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .tensor_metadata import tensor_signature


DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


@dataclass(frozen=True)
class TensorInputSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    requires_grad: bool = False
    integer_low: int = 0
    integer_high: int = 128

    def build(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.dtype not in DTYPE_ALIASES:
            raise ValueError(f"unsupported input dtype alias {self.dtype!r}")
        dtype = DTYPE_ALIASES[self.dtype]
        if dtype == torch.bool:
            tensor = torch.randint(0, 2, self.shape, generator=generator).bool()
        elif dtype.is_floating_point:
            tensor = torch.randn(self.shape, dtype=dtype, generator=generator)
        elif dtype.is_complex:
            real = torch.randn(self.shape, generator=generator)
            imag = torch.randn(self.shape, generator=generator)
            tensor = torch.complex(real, imag).to(dtype)
        else:
            tensor = torch.randint(
                self.integer_low,
                self.integer_high,
                self.shape,
                dtype=dtype,
                generator=generator,
            )
        if self.requires_grad:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise ValueError("requires_grad is valid only for floating/complex tensors")
            tensor.requires_grad_(True)
        return tensor


def build_inputs(
    positional: Sequence[Any],
    keyword: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)

    def build(value: Any) -> Any:
        if isinstance(value, TensorInputSpec):
            return value.build(generator=generator)
        if isinstance(value, tuple):
            return tuple(build(item) for item in value)
        if isinstance(value, list):
            return [build(item) for item in value]
        if isinstance(value, Mapping):
            return type(value)((key, build(item)) for key, item in value.items())
        return value

    return tuple(build(item) for item in positional), {
        str(key): build(value) for key, value in (keyword or {}).items()
    }


def input_signature(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "args": tensor_signature(args),
        "kwargs": tensor_signature(dict(kwargs)),
    }


__all__ = ["DTYPE_ALIASES", "TensorInputSpec", "build_inputs", "input_signature"]

