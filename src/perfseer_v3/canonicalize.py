"""Canonical target, argument, source-stack, and graph-constraint helpers."""

from __future__ import annotations

import operator
from typing import Any, Mapping

import torch


def raw_target_name(target: Any) -> str:
    if target is operator.getitem:
        return "prim::getitem"
    if hasattr(target, "name"):
        try:
            return str(target.name())
        except TypeError:
            pass
    return str(target)


def normalize_argument(value: Any) -> Any:
    if isinstance(value, torch.fx.Node):
        return {"tensor_ref": value.name}
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.layout):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.memory_format):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): normalize_argument(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [normalize_argument(item) for item in value]
    if isinstance(value, list):
        return [normalize_argument(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def normalized_node_arguments(node: torch.fx.Node) -> dict[str, Any]:
    return {
        "args": normalize_argument(node.args),
        "kwargs": normalize_argument(node.kwargs),
    }


def source_module_stack(node: torch.fx.Node) -> tuple[str, ...]:
    raw = node.meta.get("nn_module_stack")
    if not isinstance(raw, Mapping):
        return ()
    paths: list[str] = []
    for value in raw.values():
        if isinstance(value, (tuple, list)) and value:
            path = str(value[0])
        else:
            path = str(value)
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


def normalize_constraints(exported_program: Any) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for symbol, value_range in sorted(exported_program.range_constraints.items(), key=lambda item: str(item[0])):
        constraints[str(symbol)] = str(value_range)
    return constraints


__all__ = [
    "normalize_argument",
    "normalize_constraints",
    "normalized_node_arguments",
    "raw_target_name",
    "source_module_stack",
]

