"""Executable composite training blocks for operation-context coverage."""

from .base import (
    CompositeBlockError,
    CompositeBlockInstance,
    CompositeExecutionResult,
    CompositeVerificationResult,
    build_composite_block,
    verify_composite_block,
)

__all__ = [
    "CompositeBlockError",
    "CompositeBlockInstance",
    "CompositeExecutionResult",
    "CompositeVerificationResult",
    "build_composite_block",
    "verify_composite_block",
]
