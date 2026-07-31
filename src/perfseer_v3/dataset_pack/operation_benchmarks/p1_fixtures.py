"""Explicit P1 structural/measurement fixtures required by the A10G plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@torch.library.custom_op("perfseer_a10g::scale", mutates_args=())
def custom_scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    """Small custom training op used to verify schema/fake/autograd registration."""

    return value * factor


@custom_scale.register_fake
def _custom_scale_fake(value: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.empty_like(value)


def _custom_scale_setup(ctx, inputs, output) -> None:
    ctx.factor = inputs[1]


def _custom_scale_backward(context, gradient: torch.Tensor):
    return gradient * context.factor, None


custom_scale.register_autograd(_custom_scale_backward, setup_context=_custom_scale_setup)


@dataclass(frozen=True)
class P1FixtureResult:
    fixture_id: str
    family: str
    status: str
    observed_targets: tuple[str, ...]
    reason: str | None


def _run_custom() -> P1FixtureResult:
    value = torch.randn(3, 5, requires_grad=True)
    checks = torch.library.opcheck(custom_scale, (value, 1.5))
    if set(checks.values()) != {"SUCCESS"}:
        raise RuntimeError(f"custom-op opcheck failed: {checks}")
    custom_scale(value, 1.5).square().mean().backward()
    return P1FixtureResult(
        "p1:custom_torch_library_scale",
        "custom_fused",
        "local_structural_verified",
        ("perfseer_a10g::scale",),
        None,
    )


def _run_sparse_graph() -> P1FixtureResult:
    indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    values = torch.randn(4, requires_grad=True)
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = torch.sparse_coo_tensor(
            indices, values, (3, 3), check_invariants=True
        ).coalesce()
    dense = torch.randn(3, 4, requires_grad=True)
    output = torch.sparse.mm(sparse, dense)
    scatter_index = torch.tensor([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=torch.int64)
    scattered = torch.zeros(2, 3).scatter_reduce(
        1,
        scatter_index,
        torch.ones(2, 4),
        reduce="sum",
    )
    (output.square().mean() + scattered.mean()).backward()
    return P1FixtureResult(
        "p1:sparse_mm_scatter_reduce",
        "sparse_graph",
        "local_structural_verified",
        ("aten::_sparse_mm", "aten::scatter_reduce"),
        None,
    )


def _run_spectral() -> P1FixtureResult:
    waveform = torch.randn(2, 64, requires_grad=True)
    spectrum = torch.stft(
        waveform,
        n_fft=16,
        hop_length=8,
        window=torch.hann_window(16),
        return_complex=True,
    )
    transformed = torch.fft.fft(waveform)
    (spectrum.abs().mean() + transformed.abs().mean()).backward()
    return P1FixtureResult(
        "p1:stft_fft_training",
        "spectral_linalg",
        "local_structural_verified",
        ("aten::stft", "aten::_fft_r2c"),
        None,
    )


def _run_quantized() -> P1FixtureResult:
    value = torch.randn(4, 8, requires_grad=True)
    fake = torch.fake_quantize_per_tensor_affine(value, 0.05, 0, -128, 127)
    fake.square().mean().backward()
    return P1FixtureResult(
        "p1:qat_fake_quant_training",
        "quantized_low_precision",
        "local_structural_verified",
        ("aten::fake_quantize_per_tensor_affine",),
        None,
    )


def _run_specialized_vision() -> P1FixtureResult:
    value = torch.randn(2, 8, 6, 6, requires_grad=True)
    shuffled = F.pixel_shuffle(value, 2)
    F.pixel_unshuffle(shuffled, 2).square().mean().backward()
    return P1FixtureResult(
        "p1:pixel_shuffle_unshuffle",
        "specialized_vision",
        "local_structural_verified",
        ("aten::pixel_shuffle", "aten::pixel_unshuffle"),
        None,
    )


def run_p1_fixture_suite() -> tuple[P1FixtureResult, ...]:
    """Run redistributable CPU structural checks and retain A10G-only blockers."""

    results = [
        _run_custom(),
        _run_sparse_graph(),
        _run_spectral(),
        _run_quantized(),
        _run_specialized_vision(),
    ]
    results.append(
        P1FixtureResult(
            "p1:triton_cuda_fused_training",
            "custom_fused",
            "a10g_environment_qualification_required",
            (),
            "Triton/CUDA forward-backward identity and timing require the pinned AWS A10G stack",
        )
    )
    return tuple(results)


__all__ = ["P1FixtureResult", "custom_scale", "run_p1_fixture_suite"]
