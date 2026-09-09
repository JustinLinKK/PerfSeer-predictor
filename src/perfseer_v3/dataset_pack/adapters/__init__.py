"""Task-faithful adapters and redistributable Phase 3 fixture views."""

from .base import (
    AdapterError,
    PreparedTaskFixture,
    TaskAdapter,
    adapter_for_task,
    all_task_adapters,
)

__all__ = [
    "AdapterError",
    "PreparedTaskFixture",
    "TaskAdapter",
    "adapter_for_task",
    "all_task_adapters",
]
