"""Family-routed executable recipes for every required registry operation."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operation_sampler import OperationCandidate, OperationGeneratorSpec
from .base import BenchmarkRecipe, CallableModule, OperationBenchmarkError


_SIZE = {"tiny": 4, "small": 8, "medium": 12, "large": 16, "boundary": 17}
_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _tensor(
    candidate: OperationCandidate,
    *shape: int,
    positive: bool = False,
    boolean: bool = False,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(candidate.variant_seed % (2**63 - 1))
    if boolean:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).bool()
    dtype = _DTYPE[candidate.dtype]
    if candidate.layout == "non_contiguous" and shape and shape[-1] > 1:
        expanded = (*shape[:-1], shape[-1] * 2)
        value = torch.randn(expanded, generator=generator, dtype=dtype)[..., ::2]
    else:
        value = torch.randn(shape, generator=generator, dtype=dtype)
    return value.abs().add_(0.25) if positive else value


def _integer(candidate: OperationCandidate, high: int, *shape: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((candidate.variant_seed + 17) % (2**63 - 1))
    return torch.randint(0, high, shape, generator=generator, dtype=torch.int64)


def _call(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    identity_source: str = "dispatcher_trace",
    semantic_exception: str | None = None,
) -> BenchmarkRecipe:
    return BenchmarkRecipe(
        module=CallableModule(function),
        args=args,
        kwargs={},
        identity_source=identity_source,
        semantic_exception=semantic_exception,
    )


def _unavailable(reason: str) -> BenchmarkRecipe:
    return BenchmarkRecipe(
        module=nn.Identity(),
        args=(torch.zeros(1),),
        kwargs={},
        identity_source="dispatcher_trace",
        portable=False,
        unsupported_reason=reason,
    )


def _dense(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = candidate.batch_size
    if operation == "aten.linear":
        return _call(torch.ops.aten.linear.default, (_tensor(candidate, b, n, n), _tensor(candidate, n, n), _tensor(candidate, n)))
    if operation == "aten.matmul":
        return _call(torch.ops.aten.matmul.default, (_tensor(candidate, b, n, n + 1), _tensor(candidate, b, n + 1, n)))
    if operation == "aten.addmm":
        return _call(torch.addmm, (_tensor(candidate, n, n), _tensor(candidate, n, n + 1), _tensor(candidate, n + 1, n)))
    if operation == "aten.mm":
        return _call(torch.mm, (_tensor(candidate, n, n + 1), _tensor(candidate, n + 1, n)))
    if operation == "aten.bmm":
        return _call(torch.bmm, (_tensor(candidate, b, n, n + 1), _tensor(candidate, b, n + 1, n)))
    if operation == "aten.einsum":
        return _call(lambda left, right: torch.ops.aten.einsum.default("bij,bjk->bik", [left, right]), (_tensor(candidate, b, n, n), _tensor(candidate, b, n, n)))
    if operation == "aten.bilinear":
        return _call(torch.ops.aten.bilinear.default, (_tensor(candidate, b, n), _tensor(candidate, b, n), _tensor(candidate, n, n, n), _tensor(candidate, n)))
    if operation == "aten.mv":
        return _call(torch.mv, (_tensor(candidate, n, n), _tensor(candidate, n)))
    if operation == "aten.trilinear":
        return _call(
            lambda a, b_, c: torch.ops.aten._trilinear.default(
                a, b_, c, [], [], [], [1], 1
            ),
            (_tensor(candidate, n, n),) * 3,
        )
    raise OperationBenchmarkError(f"missing dense recipe for {operation}")


def _convolution(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = candidate.batch_size
    if operation == "aten.convolution.1d":
        return _call(torch.ops.aten.conv1d.default, (_tensor(candidate, b, 2, n), _tensor(candidate, 3, 2, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution.2d":
        return _call(torch.ops.aten.conv2d.default, (_tensor(candidate, b, 2, n, n + 1), _tensor(candidate, 3, 2, 3, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution.3d":
        return _call(torch.ops.aten.conv3d.default, (_tensor(candidate, b, 2, max(4, n // 2), n, n), _tensor(candidate, 3, 2, 3, 3, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution_transpose.1d":
        return _call(torch.ops.aten.conv_transpose1d.default, (_tensor(candidate, b, 2, n), _tensor(candidate, 2, 3, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution_transpose.2d":
        return _call(torch.ops.aten.conv_transpose2d.input, (_tensor(candidate, b, 2, n, n), _tensor(candidate, 2, 3, 3, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution_transpose.3d":
        return _call(torch.ops.aten.conv_transpose3d.input, (_tensor(candidate, b, 2, max(4, n // 2), n, n), _tensor(candidate, 2, 3, 3, 3, 3), _tensor(candidate, 3)))
    if operation == "aten.convolution.generic":
        return _call(
            lambda x, weight, bias: torch.ops.aten.convolution.default(
                x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
            ),
            (_tensor(candidate, b, 2, n, n), _tensor(candidate, 3, 2, 3, 3), _tensor(candidate, 3)),
        )
    raise OperationBenchmarkError(f"missing convolution recipe for {operation}")


def _attention(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = candidate.batch_size
    d = max(4, n)
    shape = (b, 2, n, d)
    if operation == "aten.scaled_dot_product_attention":
        return _call(torch.ops.aten.scaled_dot_product_attention.default, tuple(_tensor(candidate, *shape) for _ in range(3)))
    if operation in {
        "aten.scaled_dot_product_attention.flash",
        "aten.scaled_dot_product_attention.efficient",
    }:
        return _unavailable(
            "specialized SDPA kernel requires pinned CUDA V100 backend qualification"
        )
    if operation == "aten.native_multi_head_attention":
        q = _tensor(candidate, b, n, d)
        return _call(
            lambda query, qkv_weight, qkv_bias, proj_weight, proj_bias: torch.ops.aten._native_multi_head_attention.default(
                query,
                query,
                query,
                d,
                2,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                None,
                True,
                True,
                None,
            ),
            (q, _tensor(candidate, 3 * d, d), _tensor(candidate, 3 * d), _tensor(candidate, d, d), _tensor(candidate, d)),
        )
    raise OperationBenchmarkError(f"missing attention recipe for {operation}")


def _normalization(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = max(2, candidate.batch_size)
    x = _tensor(candidate, b, n, max(2, n // 2), max(2, n // 2))
    weight = torch.ones(n, dtype=_DTYPE[candidate.dtype])
    bias = torch.zeros(n, dtype=_DTYPE[candidate.dtype])
    mean = torch.zeros(n, dtype=_DTYPE[candidate.dtype])
    var = torch.ones(n, dtype=_DTYPE[candidate.dtype])
    if operation == "aten.batch_norm":
        return _call(lambda value, w, b_, m, v: torch.ops.aten.batch_norm.default(value, w, b_, m, v, True, 0.1, 1e-5, False), (x, weight, bias, mean, var))
    if operation == "aten.layer_norm":
        y = _tensor(candidate, b, n, n)
        return _call(lambda value, w, b_: torch.ops.aten.layer_norm.default(value, [n], w, b_, 1e-5, False), (y, weight, bias))
    if operation == "aten.native_batch_norm":
        return _call(lambda value, w, b_, m, v: torch.ops.aten.native_batch_norm.default(value, w, b_, m, v, True, 0.1, 1e-5), (x, weight, bias, mean, var))
    if operation == "aten.native_layer_norm":
        y = _tensor(candidate, b, n, n)
        return _call(lambda value, w, b_: torch.ops.aten.native_layer_norm.default(value, [n], w, b_, 1e-5), (y, weight, bias))
    if operation == "aten.group_norm":
        return _call(lambda value, w, b_: torch.ops.aten.group_norm.default(value, 1, w, b_, 1e-5, False), (x, weight, bias))
    if operation == "aten.instance_norm":
        return _call(lambda value, w, b_: torch.ops.aten.instance_norm.default(value, w, b_, None, None, True, 0.1, 1e-5, False), (x, weight, bias))
    if operation == "aten.rms_norm":
        y = _tensor(candidate, b, n, n)
        return _call(lambda value, w: torch.ops.aten.rms_norm.default(value, [n], w, None), (y, weight))
    raise OperationBenchmarkError(f"missing normalization recipe for {operation}")


def _elementwise(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    x = _tensor(candidate, candidate.batch_size, n, n, positive=True)
    y = _tensor(candidate, 1, n, 1, positive=True)
    recipes: dict[str, Callable[[], BenchmarkRecipe]] = {
        "aten.add.Tensor": lambda: _call(torch.add, (x, y)),
        "aten.div.Tensor": lambda: _call(torch.div, (x, y)),
        "aten.sub.Tensor": lambda: _call(torch.sub, (x, y)),
        "aten.mul.Tensor": lambda: _call(torch.mul, (x, y)),
        "aten.pow.Tensor": lambda: _call(torch.pow, (x, y)),
        "aten.where": lambda: _call(torch.where, (x > 0.5, x, y)),
        "aten.maximum": lambda: _call(torch.maximum, (x, y)),
        "aten.minimum": lambda: _call(torch.minimum, (x, y)),
        "aten.remainder.Tensor": lambda: _call(torch.remainder, (x, y)),
        "aten.bitwise_and.Tensor": lambda: _call(torch.ops.aten.__and__.Tensor, (_integer(candidate, 8, candidate.batch_size, n), _integer(candidate, 8, candidate.batch_size, n))),
        "aten.eq.Scalar": lambda: _call(lambda value: torch.eq(value, 1.0), (x,)),
        "aten.ge.Scalar": lambda: _call(lambda value: torch.ge(value, 1.0), (x,)),
        "aten.ne.Tensor": lambda: _call(torch.ne, (x, y)),
        "aten.pow.Scalar": lambda: _call(lambda value: torch.ops.aten.pow_.Scalar(value.clone(), 2.0), (x,)),
        "aten.xlogy.Tensor": lambda: _call(torch.xlogy, (x, y)),
    }
    if operation not in recipes:
        raise OperationBenchmarkError(f"missing elementwise recipe for {operation}")
    return recipes[operation]()


def _activation(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    x = _tensor(candidate, candidate.batch_size, n, n, positive=operation in {"aten.log", "aten.sqrt", "aten.rsqrt", "aten.reciprocal"})
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "aten.silu": F.silu,
        "aten.clamp_min": lambda value: torch.clamp_min(value, 0.0),
        "aten.relu": F.relu,
        "aten.gelu": F.gelu,
        "aten.sigmoid": torch.sigmoid,
        "aten.tanh": torch.tanh,
        "aten.abs": torch.abs,
        "aten.clamp": lambda value: torch.clamp(value, -1.0, 1.0),
        "aten.cos": torch.cos,
        "aten.elu": F.elu,
        "aten.erf": torch.erf,
        "aten.exp": torch.exp,
        "aten.leaky_relu": F.leaky_relu,
        "aten.log": torch.log,
        "aten.mish": F.mish,
        "aten.rsqrt": torch.rsqrt,
        "aten.selu": torch.ops.aten.selu.default,
        "aten.sin": torch.sin,
        "aten.softplus": F.softplus,
        "aten.sqrt": torch.sqrt,
        "aten.hardsigmoid": F.hardsigmoid,
        "aten.hardswish": F.hardswish,
    }
    if operation not in functions:
        raise OperationBenchmarkError(f"missing activation recipe for {operation}")
    return _call(functions[operation], (x,))


def _reduction(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    x = _tensor(candidate, candidate.batch_size, n, n, positive=True)
    functions: dict[str, Callable[[torch.Tensor], Any]] = {
        "aten.logsumexp": lambda value: torch.logsumexp(value, dim=-1),
        "aten.sum": lambda value: torch.sum(value, dim=-1),
        "aten.mean": lambda value: torch.mean(value, dim=-1),
        "aten.amax": lambda value: torch.amax(value, dim=-1),
        "aten.amin": lambda value: torch.amin(value, dim=-1),
        "aten.argmax": lambda value: torch.argmax(value, dim=-1),
        "aten.argmin": lambda value: torch.argmin(value, dim=-1),
        "aten.linalg_vector_norm": lambda value: torch.linalg.vector_norm(value, dim=-1),
        "aten.prod": lambda value: torch.prod(value, dim=-1),
        "aten.std": lambda value: torch.std(value, dim=-1),
        "aten.var": lambda value: torch.var(value, dim=-1),
        "aten.all": torch.ops.aten.all.default,
    }
    if operation not in functions:
        raise OperationBenchmarkError(f"missing reduction recipe for {operation}")
    return _call(functions[operation], (x,))


def _loss(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = max(4, _SIZE[candidate.shape_regime])
    b = max(2, candidate.batch_size)
    logits = _tensor(candidate, b, n)
    target = _integer(candidate, n, b)
    if operation == "aten.softmax":
        return _call(lambda value: torch.softmax(value, dim=-1), (logits,))
    if operation == "aten.mse_loss":
        return _call(F.mse_loss, (logits, _tensor(candidate, b, n)))
    if operation == "aten.nll_loss":
        return _call(lambda value, labels: torch.ops.aten.nll_loss_forward.default(F.log_softmax(value, -1), labels, None, 1, -100), (logits, target))
    if operation == "perfseer.loss.generic":
        recipe = _call(F.cross_entropy, (logits, target), identity_source="perfseer_phase_annotation")
        return recipe
    if operation == "aten.binary_cross_entropy_with_logits":
        labels = _tensor(candidate, b, n).sigmoid()
        return _call(F.binary_cross_entropy_with_logits, (logits, labels))
    if operation == "aten.cross_entropy_loss":
        return _call(
            lambda value, labels: torch.ops.aten.cross_entropy_loss.default(
                value, labels, None, 1, -100, 0.0
            ),
            (logits, target),
        )
    if operation == "aten.kl_div":
        return _call(
            lambda value, other: torch.ops.aten.kl_div.default(value, other, 1, log_target=False),
            (F.log_softmax(logits, -1), F.softmax(_tensor(candidate, b, n), -1)),
        )
    if operation == "aten.log_softmax":
        return _call(lambda value: F.log_softmax(value, dim=-1), (logits,))
    raise OperationBenchmarkError(f"missing loss recipe for {operation}")


def _pool(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = max(6, _SIZE[candidate.shape_regime])
    b = candidate.batch_size
    x1 = _tensor(candidate, b, 2, n)
    x2 = _tensor(candidate, b, 2, n, n)
    x3 = _tensor(candidate, b, 2, max(4, n // 2), n, n)
    recipes: dict[str, Callable[[], BenchmarkRecipe]] = {
        "aten.max_pool.2d": lambda: _call(lambda value: F.max_pool2d(value, 2, return_indices=True), (x2,)),
        "aten.avg_pool.2d": lambda: _call(lambda value: F.avg_pool2d(value, 2), (x2,)),
        "aten.adaptive_avg_pool.2d": lambda: _call(lambda value: F.adaptive_avg_pool2d(value, (2, 2)), (x2,)),
        "aten.upsample.bilinear2d": lambda: _call(lambda value: F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False), (x2,)),
        "aten.adaptive_avg_pool1d": lambda: _call(lambda value: torch.ops.aten.adaptive_avg_pool1d.default(value, [2]), (x1,)),
        "aten.adaptive_avg_pool3d": lambda: _call(lambda value: F.adaptive_avg_pool3d(value, (2, 2, 2)), (x3,)),
        "aten.avg_pool1d": lambda: _call(lambda value: torch.ops.aten.avg_pool1d.default(value, [2]), (x1,)),
        "aten.avg_pool3d": lambda: _call(lambda value: torch.ops.aten.avg_pool3d.default(value, [2, 2, 2]), (x3,)),
        "aten.grid_sampler": lambda: _call(lambda value, grid: F.grid_sample(value, grid, align_corners=False), (x2, _tensor(candidate, b, n, n, 2).tanh())),
        "aten.max_pool1d": lambda: _call(lambda value: torch.ops.aten.max_pool1d.default(value, [2]), (x1,)),
        "aten.max_pool2d": lambda: _call(lambda value: torch.ops.aten.max_pool2d.default(value, [2, 2]), (x2,)),
        "aten.max_pool3d": lambda: _call(lambda value: F.max_pool3d(value, 2), (x3,)),
        "aten.pad": lambda: _call(lambda value: torch.ops.aten.pad.default(value, [1, 1, 2, 2], "constant", 0.0), (x2,)),
        "aten.upsample_bilinear2d.vec": lambda: _call(lambda value: torch.ops.aten.upsample_bilinear2d.vec(value, [n + 3, n + 3], False, None), (x2,)),
        "aten.constant_pad_nd": lambda: _call(lambda value: torch.ops.aten.constant_pad_nd.default(value, [1, 1, 1, 1], 0.25), (x2,)),
    }
    if operation not in recipes:
        raise OperationBenchmarkError(f"missing pool recipe for {operation}")
    return recipes[operation]()


def _layout(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = candidate.batch_size
    x = _tensor(candidate, b, n, n)
    source = (
        "capture_semantic_summary"
        if operation in {"prim.getitem", "operator.getitem", "aten.sym_size"}
        else "dispatcher_trace"
    )
    prim_getitem = torch.jit.CompilationUnit(
        "def getitem(x: Tensor, index: int) -> Tensor:\n"
        "    values = (x, x + 1)\n"
        "    return values[index]\n"
    ).getitem
    recipes: dict[str, Callable[[], BenchmarkRecipe]] = {
        "aten.transpose": lambda: _call(lambda value: value.transpose(-1, -2), (x,)),
        "prim.getitem": lambda: _call(prim_getitem, (x, 0), identity_source=source),
        "aten.view": lambda: _call(lambda value: value.view(b, -1), (x.contiguous(),)),
        "aten.reshape": lambda: _call(lambda value: torch.ops.aten.reshape.default(value, [b, -1]), (x,)),
        "aten.permute": lambda: _call(lambda value: value.permute(0, 2, 1), (x,)),
        "aten.cat": lambda: _call(lambda value: torch.cat((value, value), dim=-1), (x,)),
        "aten.stack": lambda: _call(lambda value: torch.stack((value, value), dim=0), (x,)),
        "aten.clone": lambda: _call(torch.clone, (x,)),
        "aten.expand": lambda: _call(lambda value: value.expand(b, n, n), (_tensor(candidate, 1, n, n),)),
        "aten.repeat": lambda: _call(lambda value: value.repeat(1, 1, 2), (x,)),
        "aten.to_copy": lambda: _call(lambda value: torch.ops.aten._to_copy.default(value, dtype=torch.float64), (x,)),
        "operator.getitem": lambda: _call(lambda value: value[0], (x,), identity_source=source),
        "aten.arange": lambda: _call(lambda value: torch.arange(value, dtype=_DTYPE[candidate.dtype]), (n,)),
        "aten.broadcast_tensors": lambda: _call(lambda left, right: torch.ops.aten.broadcast_tensors.default([left, right]), (_tensor(candidate, b, n, 1), _tensor(candidate, 1, n))),
        "aten.chunk": lambda: _call(lambda value: torch.ops.aten.chunk.default(value, 2, -1), (x,)),
        "aten.contiguous": lambda: _call(lambda value: torch.ops.aten.contiguous.default(value), (x.transpose(-1, -2),)),
        "aten.flatten": lambda: _call(lambda value: torch.ops.aten.flatten.using_ints(value, 1, -1), (x,)),
        "aten.ones_like": lambda: _call(torch.ones_like, (x,)),
        "aten.split": lambda: _call(lambda value: torch.split(value, max(1, n // 2), dim=-1), (x,)),
        "aten.squeeze": lambda: _call(lambda value: value.squeeze(1), (_tensor(candidate, b, 1, n),)),
        "aten.sym_size": lambda: _call(lambda value: torch.ops.aten.sym_size.int(value, -1), (x,), identity_source=source, semantic_exception="symbolic size is metadata, not a tensor-producing operation"),
        "aten.tile": lambda: _call(lambda value: torch.ops.aten.tile.default(value, [1, 1, 2]), (x,)),
        "aten.unbind": lambda: _call(lambda value: torch.unbind(value, dim=0), (x,)),
        "aten.unflatten": lambda: _call(lambda value: torch.ops.aten.unflatten.int(value, -1, [n, n]), (_tensor(candidate, b, n * n),)),
        "aten.unsqueeze": lambda: _call(lambda value: value.unsqueeze(1), (x,)),
        "aten.zeros": lambda: _call(lambda value: torch.zeros((value, value), dtype=_DTYPE[candidate.dtype]), (n,)),
        "aten.zeros_like": lambda: _call(torch.zeros_like, (x,)),
        "aten.new_ones": lambda: _call(lambda value: value.new_ones((b, n)), (x,)),
        "aten.new_zeros": lambda: _call(lambda value: value.new_zeros((b, n)), (x,)),
        "aten.ones": lambda: _call(lambda value: torch.ones((value, value), dtype=_DTYPE[candidate.dtype]), (n,)),
        "aten.t": lambda: _call(torch.t, (_tensor(candidate, n, n),)),
        "aten.local_scalar_dense": lambda: _call(lambda value: torch.ops.aten._local_scalar_dense.default(value), (_tensor(candidate, 1),), semantic_exception="local scalar extraction intentionally returns a scalar"),
        "aten.detach": lambda: _call(torch.detach, (x,)),
        "aten.empty": lambda: _call(lambda value: torch.empty((value, value), dtype=_DTYPE[candidate.dtype]), (n,)),
        "aten.set.source_storage": lambda: _call(lambda value, source_tensor: torch.ops.aten.set_.source_Storage(value, source_tensor.untyped_storage()), (torch.empty(0, dtype=_DTYPE[candidate.dtype]), x), semantic_exception="storage rebinding is an internal mutation fixture"),
    }
    if operation not in recipes:
        raise OperationBenchmarkError(f"missing layout recipe for {operation}")
    return recipes[operation]()


def _index(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    b = candidate.batch_size
    x = _tensor(candidate, b, n, n)
    index = _integer(candidate, n, b, n, n)
    recipes: dict[str, Callable[[], BenchmarkRecipe]] = {
        "aten.gather": lambda: _call(lambda value, idx: torch.gather(value, 1, idx), (x, index)),
        "aten.topk": lambda: _call(lambda value: torch.topk(value, min(3, n), dim=-1), (x,)),
        "aten.index": lambda: _call(lambda value, idx: value[idx], (x, _integer(candidate, b, max(1, b)))),
        "aten.scatter": lambda: _call(lambda value, idx, src: torch.scatter(value, 1, idx, src), (x, index, _tensor(candidate, b, n, n))),
        "aten.sort": lambda: _call(lambda value: torch.sort(value, dim=-1), (x,)),
        "aten.index_put": lambda: _call(lambda value, idx, src: torch.index_put(value, (idx,), src), (x, _integer(candidate, b, max(1, b)), _tensor(candidate, max(1, b), n, n))),
        "aten.index_select": lambda: _call(lambda value, idx: torch.index_select(value, 1, idx), (x, _integer(candidate, n, max(1, n // 2)))),
        "aten.masked_fill": lambda: _call(lambda value, mask: value.masked_fill(mask, 0.0), (x, _tensor(candidate, b, n, n, boolean=True))),
        "aten.masked_select": lambda: _call(torch.masked_select, (x, _tensor(candidate, b, n, n, boolean=True))),
        "aten.narrow": lambda: _call(lambda value: torch.ops.aten.narrow.default(value, 1, 0, max(1, n // 2)), (x,)),
        "aten.scatter_reduce": lambda: _call(lambda value, idx, src: torch.scatter_reduce(value, 1, idx, src, reduce="sum"), (x, index, _tensor(candidate, b, n, n))),
        "aten.select": lambda: _call(lambda value: torch.select(value, 1, 0), (x,)),
        "aten.slice": lambda: _call(lambda value: torch.ops.aten.slice.Tensor(value, 1, 0, max(1, n // 2), 1), (x,)),
        "aten.scatter_add": lambda: _call(lambda value, idx, src: torch.scatter_add(value, 1, idx, src), (x, index, _tensor(candidate, b, n, n))),
    }
    if operation not in recipes:
        raise OperationBenchmarkError(f"missing index recipe for {operation}")
    return recipes[operation]()


def _sequence(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = max(4, _SIZE[candidate.shape_regime])
    b = candidate.batch_size
    if operation == "aten.embedding":
        return _call(F.embedding, (_integer(candidate, n * 2, b, n), _tensor(candidate, n * 2, n)))
    if operation == "aten.embedding_bag":
        indices = _integer(candidate, n * 2, b * n)
        offsets = torch.arange(0, b * n, n, dtype=torch.int64)
        return _call(
            lambda idx, weight, off: torch.ops.aten.embedding_bag.padding_idx(
                weight, idx, off, False, 0, False, None, False, None
            ),
            (indices, _tensor(candidate, n * 2, n), offsets),
        )
    if operation == "aten.gru":
        module = nn.GRU(n, n, batch_first=True).to(dtype=_DTYPE[candidate.dtype])
        params = tuple(value for value in module._flat_weights if value is not None)
        return _call(
            lambda value, hidden: torch.ops.aten.gru.input(
                value, hidden, list(params), True, 1, 0.0, True, False, True
            ),
            (_tensor(candidate, b, n, n), _tensor(candidate, 1, b, n)),
        )
    if operation == "aten.lstm":
        module = nn.LSTM(n, n, batch_first=True).to(dtype=_DTYPE[candidate.dtype])
        params = tuple(value for value in module._flat_weights if value is not None)
        return _call(
            lambda value, hidden, cell: torch.ops.aten.lstm.input(
                value, [hidden, cell], list(params), True, 1, 0.0, True, False, True
            ),
            (
                _tensor(candidate, b, n, n),
                _tensor(candidate, 1, b, n),
                _tensor(candidate, 1, b, n),
            ),
        )
    if operation == "aten.rnn_tanh":
        module = nn.RNN(n, n, nonlinearity="tanh", batch_first=True).to(dtype=_DTYPE[candidate.dtype])
        params = tuple(value for value in module._flat_weights if value is not None)
        return _call(
            lambda value, hidden: torch.ops.aten.rnn_tanh.input(
                value, hidden, list(params), True, 1, 0.0, True, False, True
            ),
            (_tensor(candidate, b, n, n), _tensor(candidate, 1, b, n)),
        )
    if operation == "aten.cudnn_rnn":
        return _unavailable("cuDNN RNN requires pinned CUDA V100 backend qualification")
    raise OperationBenchmarkError(f"missing sequence recipe for {operation}")


def _random(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    x = _tensor(candidate, candidate.batch_size, n, n)
    recipes = {
        "aten.dropout": lambda: _call(lambda value: torch.ops.aten.dropout.default(value, 0.25, True), (x,)),
        "aten.bernoulli": lambda: _call(torch.bernoulli, (x.sigmoid(),)),
        "aten.rand_like": lambda: _call(torch.rand_like, (x,)),
        "aten.native_dropout": lambda: _call(lambda value: torch.ops.aten.native_dropout.default(value, 0.25, True), (x,)),
    }
    if operation not in recipes:
        raise OperationBenchmarkError(f"missing random recipe for {operation}")
    return recipes[operation]()


def _training(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    n = _SIZE[candidate.shape_regime]
    x = _tensor(candidate, candidate.batch_size, n, n)
    if operation == "perfseer.training.analytical_backward":
        return _call(lambda value: value.square().mean(), (x,), identity_source="perfseer_phase_annotation")
    if operation == "aten.native_dropout_backward":
        return _call(lambda grad, mask: torch.ops.aten.native_dropout_backward.default(grad, mask, 4.0 / 3.0), (x, _tensor(candidate, candidate.batch_size, n, n, boolean=True)))
    if operation == "aten.sigmoid_backward":
        return _call(lambda grad, output: torch.ops.aten.sigmoid_backward.default(grad, output), (x, x.sigmoid()))
    raise OperationBenchmarkError(f"missing training recipe for {operation}")


def _manual_optimizer_step(name: str, candidate: OperationCandidate) -> Callable[[], torch.Tensor]:
    n = _SIZE[candidate.shape_regime]
    dtype = _DTYPE[candidate.dtype]
    parameter = nn.Parameter(_tensor(candidate, n, n).contiguous())
    x = _tensor(candidate, n, n).contiguous()
    if name == "sparse_adam":
        embedding = nn.Embedding(n * 2, n, sparse=True, dtype=dtype)
        optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=1e-3)

        def sparse_step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = embedding(_integer(candidate, n * 2, n)).square().mean()
            loss.backward()
            optimizer.step()
            return embedding.weight.detach().clone()

        return sparse_step

    builtins: dict[str, type[torch.optim.Optimizer]] = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "asgd": torch.optim.ASGD,
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adamax": torch.optim.Adamax,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
    }
    if hasattr(torch.optim, "Adafactor"):
        builtins["adafactor"] = torch.optim.Adafactor
    if name in builtins:
        optimizer = builtins[name]([parameter], lr=1e-3)

        def builtin_step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter @ x).square().mean()
            loss.backward()
            optimizer.step()
            return parameter.detach().clone()

        return builtin_step
    if name == "lbfgs":
        optimizer = torch.optim.LBFGS([parameter], lr=0.1, max_iter=2)

        def lbfgs_step() -> torch.Tensor:
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss = (parameter @ x).square().mean()
                loss.backward()
                return loss

            optimizer.step(closure)
            return parameter.detach().clone()

        return lbfgs_step

    state = torch.zeros_like(parameter)
    second = torch.zeros_like(parameter)

    def custom_step() -> torch.Tensor:
        if parameter.grad is not None:
            parameter.grad = None
        loss = (parameter @ x).square().mean()
        loss.backward()
        assert parameter.grad is not None
        grad = parameter.grad
        with torch.no_grad():
            if name == "lion":
                update = state.mul(0.9).add(grad, alpha=0.1).sign()
                parameter.add_(update, alpha=-1e-3)
                state.mul_(0.99).add_(grad, alpha=0.01)
            elif name == "lars":
                trust = parameter.norm() / (grad.norm() + 1e-8)
                state.mul_(0.9).add_(grad, alpha=float(trust))
                parameter.add_(state, alpha=-1e-3)
            elif name == "lamb":
                state.mul_(0.9).add_(grad, alpha=0.1)
                second.mul_(0.999).addcmul_(grad, grad, value=0.001)
                update = state / (second.sqrt() + 1e-6)
                trust = parameter.norm() / (update.norm() + 1e-8)
                parameter.add_(update, alpha=-1e-3 * float(trust))
            elif name == "muon":
                state.mul_(0.95).add_(grad, alpha=0.05)
                orthogonal, _ = torch.linalg.qr(state.float())
                parameter.add_(orthogonal.to(dtype=parameter.dtype), alpha=-1e-3)
            else:
                raise OperationBenchmarkError(f"optimizer {name!r} has no faithful step")
        return parameter.detach().clone()

    return custom_step


def _optimizer(operation: str, candidate: OperationCandidate) -> BenchmarkRecipe:
    name = operation.removeprefix("perfseer.optimizer.")
    step = _manual_optimizer_step(name, candidate)
    return BenchmarkRecipe(
        module=nn.Identity(),
        args=(torch.zeros(1),),
        kwargs={},
        identity_source="perfseer_phase_annotation",
        optimizer_step=step,
    )


_BUILDERS = {
    "dense_matrix": _dense,
    "convolution": _convolution,
    "attention": _attention,
    "normalization": _normalization,
    "elementwise": _elementwise,
    "activation_unary": _activation,
    "reduction": _reduction,
    "loss_probability": _loss,
    "pool_resample": _pool,
    "layout_shape": _layout,
    "index_scatter": _index,
    "embedding_sequence": _sequence,
    "random_regularization": _random,
    "training": _training,
    "optimizer": _optimizer,
}


def build_family_benchmark(
    family: str,
    generator: OperationGeneratorSpec,
    candidate: OperationCandidate,
) -> BenchmarkRecipe:
    if generator.family != family or candidate.family != family:
        raise OperationBenchmarkError("operation benchmark family routing mismatch")
    builder = _BUILDERS.get(family)
    if builder is None:
        raise OperationBenchmarkError(f"no operation benchmark family builder for {family}")
    recipe = builder(generator.canonical_operation_id, candidate)
    recipe.validate()
    return recipe


__all__ = ["build_family_benchmark"]
