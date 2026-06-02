"""Convert trusted PyTorch source files into PerfSeer compute graphs."""

from __future__ import annotations

import importlib.util
import math
import operator
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


ARG_KEYS: tuple[str, ...] = (
    "conv_kernel_size",
    "conv_stride",
    "conv_padding",
    "conv_dilation",
    "conv_groups",
    "conv_bias",
    "linear_in_features",
    "linear_out_features",
    "linear_bias",
    "pool_kernel_size",
    "pool_stride",
    "pool_padding",
    "pool_ceil_mode",
)

DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class SourceModelSpec:
    source_path: str | Path
    entry: str
    input_shapes: Sequence[Sequence[int]]
    constructor_args: tuple[Any, ...] = ()
    constructor_kwargs: dict[str, Any] = field(default_factory=dict)
    input_dtypes: tuple[str, ...] = ("float32",)


class UnsupportedOpError(RuntimeError):
    """Raised when a traced source model uses an op outside v1 coverage."""


def convert_source_to_networkx(spec: SourceModelSpec) -> nx.DiGraph:
    """Trace ``spec`` and return a feature-bearing PerfSeer compute graph."""

    model = _load_source_model(spec)
    example_inputs = _example_inputs(spec)
    try:
        traced = symbolic_trace(model)
        ShapeProp(traced).propagate(*example_inputs)
    except Exception as exc:
        raise RuntimeError(f"failed to trace and shape-propagate {spec.entry!r}: {exc}") from exc
    return _fx_to_networkx(traced)


def convert_source_to_pyg_data(
    spec: SourceModelSpec,
    ckpt_path: str | Path | None = None,
    norm_stats: dict[str, Any] | None = None,
    feature_config: Any | None = None,
):
    """Convert source to the optimized PerfSeer predictor's PyG data object."""

    from perfseer_optimized.data import FeatureConfig, build_pyg_inference_data

    if ckpt_path is not None:
        ckpt_stats, ckpt_feature_config = _checkpoint_context(ckpt_path)
        if norm_stats is None:
            norm_stats = ckpt_stats
        if feature_config is None:
            feature_config = ckpt_feature_config

    if feature_config is None:
        feature_config = FeatureConfig()
    elif isinstance(feature_config, dict):
        feature_config = FeatureConfig.from_dict(feature_config)

    graph = convert_source_to_networkx(spec)
    return build_pyg_inference_data(graph, norm_stats=norm_stats, feature_config=feature_config)


def _checkpoint_context(path: str | Path):
    from perfseer_optimized.data import FeatureConfig

    try:
        ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=torch.device("cpu"))

    meta = ckpt.get("metadata", {})
    raw_stats = meta.get("norm_stats")
    if not raw_stats:
        raise KeyError(f"checkpoint {path} does not include metadata.norm_stats")
    norm_stats = {key: torch.as_tensor(value).cpu().numpy() for key, value in raw_stats.items()}
    feature_config = FeatureConfig.from_dict(meta.get("feature_config") or meta.get("config", {}).get("features"))
    return norm_stats, feature_config


def _load_source_model(spec: SourceModelSpec) -> nn.Module:
    source = Path(spec.source_path).expanduser().resolve()
    module = _import_source(source)
    entry = _resolve_attr(module, spec.entry)

    if isinstance(entry, nn.Module):
        model = entry
    elif isinstance(entry, type) and issubclass(entry, nn.Module):
        model = entry(*spec.constructor_args, **spec.constructor_kwargs)
    elif callable(entry):
        model = entry(*spec.constructor_args, **spec.constructor_kwargs)
    else:
        raise TypeError(f"entry {spec.entry!r} is not an nn.Module class, instance, or factory")

    if not isinstance(model, nn.Module):
        raise TypeError(f"entry {spec.entry!r} returned {type(model).__name__}, not torch.nn.Module")
    return model.cpu().eval()


def _import_source(source: Path) -> ModuleType:
    if not source.exists():
        raise FileNotFoundError(source)
    module_name = f"_perfseer_source_{source.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source file {source}")

    module = importlib.util.module_from_spec(spec)
    source_dir = str(source.parent)
    added_path = False
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
        added_path = True
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if added_path:
            try:
                sys.path.remove(source_dir)
            except ValueError:
                pass
    return module


def _resolve_attr(module: ModuleType, dotted: str) -> Any:
    value: Any = module
    for part in dotted.split("."):
        if not part:
            raise AttributeError(f"empty component in entry {dotted!r}")
        value = getattr(value, part)
    return value


def _example_inputs(spec: SourceModelSpec) -> tuple[torch.Tensor, ...]:
    if not spec.input_shapes:
        raise ValueError("SourceModelSpec.input_shapes must contain at least one shape")
    dtypes = spec.input_dtypes or ("float32",)
    if len(dtypes) == 1 and len(spec.input_shapes) > 1:
        dtypes = tuple(dtypes[0] for _ in spec.input_shapes)
    if len(dtypes) != len(spec.input_shapes):
        raise ValueError("input_dtypes must have length 1 or match input_shapes")

    tensors = []
    for shape, dtype_name in zip(spec.input_shapes, dtypes):
        dtype = DTYPE_ALIASES.get(str(dtype_name).lower())
        if dtype is None:
            raise ValueError(f"unsupported input dtype {dtype_name!r}")
        dims = tuple(int(dim) for dim in shape)
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"input shape dimensions must be positive: {shape!r}")
        tensors.append(torch.zeros(dims, dtype=dtype, device=torch.device("cpu")))
    return tuple(tensors)


def _fx_to_networkx(gm: GraphModule) -> nx.DiGraph:
    graph = nx.DiGraph()
    modules = dict(gm.named_modules())
    fx_to_graph_id: dict[Node, int] = {}

    for node in gm.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if _tensor_meta(node) is None:
            continue

        op_type = _classify_node(node, modules)
        if op_type is None:
            raise UnsupportedOpError(f"unsupported op at FX node {node.name!r}: {node.op} target={node.target!r}")

        node_id = len(graph)
        feature = _feature_for_node(node, modules, op_type)
        graph.add_node(node_id, feature=feature)
        fx_to_graph_id[node] = node_id

        for dep in _dependency_nodes(node):
            src_id = fx_to_graph_id.get(dep)
            if src_id is not None:
                graph.add_edge(src_id, node_id)

    if graph.number_of_nodes() == 0:
        raise UnsupportedOpError("source model produced no supported tensor operations")
    return graph


def _classify_node(node: Node, modules: dict[str, nn.Module]) -> str | None:
    output_meta = _tensor_meta(node)
    output_shape = _shape(output_meta)

    if node.op == "call_module":
        module = modules[str(node.target)]
        if isinstance(module, nn.Conv2d):
            return "Conv"
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            return "Relu"
        if isinstance(module, nn.BatchNorm2d):
            return "BatchNormalization"
        if isinstance(module, nn.AvgPool2d):
            return "AveragePool"
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return "GlobalAveragePool" if _is_global_pool_output(output_shape) else "AveragePool"
        if isinstance(module, nn.MaxPool2d):
            return "MaxPool"
        if isinstance(module, nn.Flatten):
            return "Flatten"
        if isinstance(module, nn.Linear):
            return "Gemm"
        return None

    if node.op == "call_function":
        target = node.target
        name = _target_name(target)
        if target in {torch.relu, F.relu} or name in {"relu", "relu_"}:
            return "Relu"
        if target is torch.cat or name == "cat":
            return "Concat"
        if target is torch.flatten or name == "flatten":
            return "Flatten"
        if target in {operator.add, torch.add} or name == "add":
            return "Add"
        if target is F.avg_pool2d or name == "avg_pool2d":
            return "AveragePool"
        if target is F.adaptive_avg_pool2d or name == "adaptive_avg_pool2d":
            return "GlobalAveragePool" if _is_global_pool_output(output_shape) else "AveragePool"
        if target is F.max_pool2d or name == "max_pool2d":
            return "MaxPool"
        return None

    if node.op == "call_method":
        name = str(node.target)
        if name in {"relu", "relu_"}:
            return "Relu"
        if name in {"flatten"}:
            return "Flatten"
        if name in {"view", "reshape"} and _looks_like_flatten(node):
            return "Flatten"
        if name in {"add", "add_"}:
            return "Add"
        return None

    return None


def _feature_for_node(node: Node, modules: dict[str, nn.Module], op_type: str) -> dict[str, Any]:
    output_meta = _require_tensor_meta(node)
    input_metas = _input_tensor_metas((node.args, node.kwargs))
    if not input_metas:
        raise UnsupportedOpError(f"op {node.name!r} has no tensor input metadata")

    output_shape = _shape(output_meta)
    output_size = _tensor_nbytes(output_meta)
    elem_size = _element_size(output_meta)
    input_size = sum(_tensor_nbytes(meta) for meta in input_metas)
    weight_size = _weight_size(node, modules, op_type, input_metas, output_meta)
    input_size_with_weight = int(input_size / elem_size + weight_size) if elem_size else int(weight_size)
    total_bytes = int(input_size + output_size + weight_size * elem_size)
    args = _args_for_node(node, modules, op_type, input_metas, output_meta)
    flops = int(_flops_for_node(node, modules, op_type, input_metas, output_meta, args))

    memory_info = _memory_info(
        input_metas=input_metas,
        output_meta=output_meta,
        input_size=input_size,
        output_size=output_size,
        weight_size=weight_size,
        input_size_with_weight=input_size_with_weight,
        total_bytes=total_bytes,
    )

    return {
        "type": op_type,
        "args": args,
        "memory_info": memory_info,
        "flops": flops,
        "arith_intensity": _safe_div(float(flops), _safe_div(float(total_bytes), float(elem_size))),
    }


def _args_for_node(
    node: Node,
    modules: dict[str, nn.Module],
    op_type: str,
    input_metas: Sequence[Any],
    output_meta: Any,
) -> dict[str, int]:
    args = {key: 0 for key in ARG_KEYS}
    module = modules.get(str(node.target)) if node.op == "call_module" else None

    if op_type == "Conv":
        conv = module if isinstance(module, nn.Conv2d) else None
        if conv is None:
            return args
        args.update(
            {
                "conv_kernel_size": _first_int(conv.kernel_size),
                "conv_stride": _first_int(conv.stride),
                "conv_padding": _first_int(conv.padding),
                "conv_dilation": _first_int(conv.dilation),
                "conv_groups": int(conv.groups),
                "conv_bias": int(conv.bias is not None),
            }
        )
    elif op_type in {"AveragePool", "GlobalAveragePool", "MaxPool"}:
        if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
            stride = module.stride if module.stride is not None else module.kernel_size
            args.update(
                {
                    "pool_kernel_size": _first_int(module.kernel_size),
                    "pool_stride": _first_int(stride),
                    "pool_padding": _first_int(module.padding),
                    "pool_ceil_mode": int(bool(module.ceil_mode)),
                }
            )
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            in_shape = _shape(input_metas[0])
            kernel = int(in_shape[2]) if len(in_shape) >= 4 else 1
            args.update({"pool_kernel_size": kernel, "pool_stride": kernel, "pool_padding": 0})
        else:
            kernel = _node_arg(node, 1, "kernel_size", default=0)
            stride = _node_arg(node, 2, "stride", default=kernel)
            padding = _node_arg(node, 3, "padding", default=0)
            ceil_mode = _node_arg(node, 5 if op_type == "MaxPool" else 4, "ceil_mode", default=False)
            if stride is None:
                stride = kernel
            args.update(
                {
                    "pool_kernel_size": _first_int(kernel),
                    "pool_stride": _first_int(stride),
                    "pool_padding": _first_int(padding),
                    "pool_ceil_mode": int(bool(ceil_mode)),
                }
            )
    elif op_type == "Gemm":
        linear = module if isinstance(module, nn.Linear) else None
        if linear is not None:
            args.update(
                {
                    "linear_in_features": int(linear.in_features),
                    "linear_out_features": int(linear.out_features),
                    "linear_bias": int(linear.bias is not None),
                }
            )
    return args


def _flops_for_node(
    node: Node,
    modules: dict[str, nn.Module],
    op_type: str,
    input_metas: Sequence[Any],
    output_meta: Any,
    args: dict[str, int],
) -> int:
    output_shape = _shape(output_meta)
    output_elems = _numel(output_shape)
    module = modules.get(str(node.target)) if node.op == "call_module" else None

    if op_type == "Conv":
        in_shape = _shape(input_metas[0])
        batch = output_shape[0] if output_shape else 1
        out_channels = output_shape[1] if len(output_shape) >= 2 else 1
        out_h = output_shape[2] if len(output_shape) >= 4 else 1
        out_w = output_shape[3] if len(output_shape) >= 4 else 1
        in_channels = in_shape[1] if len(in_shape) >= 2 else 1
        groups = max(1, int(args.get("conv_groups", 1)))
        kernel = int(args.get("conv_kernel_size", 1))
        macs = batch * out_channels * out_h * out_w * (in_channels // groups) * kernel * kernel
        bias_cost = 2 * output_elems if args.get("conv_bias") else 0
        return int(2 * macs + bias_cost)
    if op_type == "Relu":
        return int(output_elems)
    if op_type == "BatchNormalization":
        return int(2 * output_elems)
    if op_type in {"AveragePool", "MaxPool"}:
        kernel = max(1, int(args.get("pool_kernel_size", 1)))
        return int(output_elems * kernel * kernel)
    if op_type == "GlobalAveragePool":
        return int(_numel(_shape(input_metas[0])))
    if op_type == "Flatten":
        return 0
    if op_type == "Gemm":
        in_features = int(args.get("linear_in_features", 0))
        out_features = int(args.get("linear_out_features", 0))
        batch = output_shape[0] if output_shape else 1
        if isinstance(module, nn.Linear):
            in_features = int(module.in_features)
            out_features = int(module.out_features)
        return int(2 * batch * in_features * out_features)
    if op_type == "Concat":
        return 0
    if op_type == "Add":
        return int(max(1, len(input_metas) - 1) * output_elems)
    return 0


def _weight_size(
    node: Node,
    modules: dict[str, nn.Module],
    op_type: str,
    input_metas: Sequence[Any],
    output_meta: Any,
) -> int:
    module = modules.get(str(node.target)) if node.op == "call_module" else None
    if isinstance(module, nn.Conv2d):
        return int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0))
    if isinstance(module, nn.BatchNorm2d):
        return int(4 * module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0))
    if op_type == "BatchNormalization":
        shape = _shape(output_meta)
        return int(4 * shape[1]) if len(shape) >= 2 else 0
    return 0


def _memory_info(
    *,
    input_metas: Sequence[Any],
    output_meta: Any,
    input_size: int,
    output_size: int,
    weight_size: int,
    input_size_with_weight: int,
    total_bytes: int,
) -> dict[str, int | float]:
    in_shapes = [_shape(meta) for meta in input_metas]
    out_shape = _shape(output_meta)
    batch_size, input_channels, input_h, input_w = _aggregate_input_dims(in_shapes)
    _, output_channels, output_h, output_w = _tensor_dims(out_shape)
    return {
        "bytes": int(total_bytes),
        "weight_size": int(weight_size),
        "batch_size": int(batch_size),
        "input_size_with_weight": int(input_size_with_weight),
        "input_size": int(input_size),
        "input_channels": input_channels,
        "input_w": input_w,
        "input_h": input_h,
        "output_size": int(output_size),
        "output_channels": output_channels,
        "output_w": output_w,
        "output_h": output_h,
    }


def _aggregate_input_dims(shapes: Sequence[tuple[int, ...]]) -> tuple[int, float, float, float]:
    dims = [_tensor_dims(shape) for shape in shapes if shape]
    if not dims:
        return 0, 0.0, 0.0, 0.0
    batch = int(dims[0][0])
    channels = float(sum(dim[1] for dim in dims) / len(dims))
    h = float(sum(dim[2] for dim in dims) / len(dims))
    w = float(sum(dim[3] for dim in dims) / len(dims))
    return batch, channels, h, w


def _tensor_dims(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) >= 4:
        return int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2]), 0
    if len(shape) == 2:
        return int(shape[0]), int(shape[1]), 0, 0
    if len(shape) == 1:
        return 1, int(shape[0]), 0, 0
    return 0, 0, 0, 0


def _looks_like_flatten(node: Node) -> bool:
    input_metas = _input_tensor_metas((node.args[:1], {}))
    output_meta = _tensor_meta(node)
    if not input_metas or output_meta is None:
        return False
    in_shape = _shape(input_metas[0])
    out_shape = _shape(output_meta)
    return len(in_shape) > 2 and len(out_shape) <= 2 and _numel(in_shape) == _numel(out_shape)


def _dependency_nodes(node: Node) -> list[Node]:
    deps: list[Node] = []
    seen: set[Node] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Node):
            if value not in seen and _tensor_meta(value) is not None:
                seen.add(value)
                deps.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(node.args)
    visit(node.kwargs)
    return deps


def _input_tensor_metas(value: Any) -> list[Any]:
    metas: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, Node):
            meta = _tensor_meta(item)
            if meta is not None:
                metas.append(meta)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    return metas


def _tensor_meta(node: Node) -> Any | None:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    if hasattr(meta, "shape"):
        return meta
    if isinstance(meta, (list, tuple)):
        for item in meta:
            if hasattr(item, "shape"):
                return item
    return None


def _require_tensor_meta(node: Node) -> Any:
    meta = _tensor_meta(node)
    if meta is None:
        raise UnsupportedOpError(f"FX node {node.name!r} has no tensor metadata")
    return meta


def _shape(meta: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in meta.shape)


def _tensor_nbytes(meta: Any) -> int:
    return int(_numel(_shape(meta)) * _element_size(meta))


def _element_size(meta: Any) -> int:
    dtype = getattr(meta, "dtype", torch.float32)
    return int(torch.empty((), dtype=dtype).element_size())


def _numel(shape: Iterable[int]) -> int:
    return int(math.prod(int(dim) for dim in shape))


def _first_int(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    return int(value)


def _node_arg(node: Node, pos: int, name: str, default: Any = None) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > pos:
        return node.args[pos]
    return default


def _is_global_pool_output(shape: tuple[int, ...]) -> bool:
    return len(shape) >= 4 and int(shape[2]) == 1 and int(shape[3]) == 1


def _target_name(target: Any) -> str:
    return str(getattr(target, "__name__", target))


def _safe_div(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den):
        return 0.0
    out = num / den
    return out if math.isfinite(out) else 0.0
