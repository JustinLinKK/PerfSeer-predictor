"""Structured capture diagnostics; capture failures are data, not omissions."""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureFailureV3:
    backend: str
    mode: str
    stage: str
    exception_type: str
    message: str
    traceback_tail: tuple[str, ...]
    retryable: bool
    details: dict[str, Any]

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        backend: str,
        mode: str,
        stage: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> "CaptureFailureV3":
        return cls(
            backend=backend,
            mode=mode,
            stage=stage,
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback_tail=tuple(traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:]),
            retryable=retryable,
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayValidationError(RuntimeError):
    """Raised when eager and non-strict exported programs diverge."""


__all__ = ["CaptureFailureV3", "ReplayValidationError"]

