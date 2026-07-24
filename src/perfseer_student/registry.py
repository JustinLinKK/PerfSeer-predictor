"""Hardware registry and artifact integrity checks for student predictors."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """Raised when no compatible, intact predictor model is available."""


def normalize_gpu_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


@dataclass(frozen=True)
class HardwareInfo:
    name: str
    compute_capability: str
    vram_mb: int


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    artifact_path: Path
    sha256: str
    aliases: tuple[str, ...]
    compute_capability: str
    vram_min_mb: int
    vram_max_mb: int
    node_dim: int
    edge_dim: int
    global_dim: int
    targets: tuple[str, ...]
    train_mem_index: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any], registry_dir: Path) -> "ModelRecord":
        hardware = payload["hardware"]
        schema = payload["schema"]
        outputs = payload["outputs"]
        artifact = payload["artifact"]
        return cls(
            model_id=str(payload["id"]),
            artifact_path=(registry_dir / artifact["path"]).resolve(),
            sha256=str(artifact["sha256"]).lower(),
            aliases=tuple(normalize_gpu_name(value) for value in hardware["aliases"]),
            compute_capability=str(hardware["compute_capability"]),
            vram_min_mb=int(hardware["vram_mb"]["min"]),
            vram_max_mb=int(hardware["vram_mb"]["max"]),
            node_dim=int(schema["node_dim"]),
            edge_dim=int(schema["edge_dim"]),
            global_dim=int(schema["global_dim"]),
            targets=tuple(outputs["targets"]),
            train_mem_index=int(outputs["train_mem_index"]),
        )

    def matches(self, hardware: HardwareInfo) -> bool:
        name = normalize_gpu_name(hardware.name)
        return (
            any(alias == name or name.endswith(f" {alias}") for alias in self.aliases)
            and self.compute_capability == str(hardware.compute_capability)
            and self.vram_min_mb <= hardware.vram_mb <= self.vram_max_mb
        )

    def verify(self) -> None:
        if not self.artifact_path.is_file():
            raise ModelUnavailableError(f"predictor artifact is missing: {self.artifact_path}")
        digest = hashlib.sha256(self.artifact_path.read_bytes()).hexdigest()
        if digest != self.sha256:
            raise ModelUnavailableError(
                f"predictor artifact hash mismatch for {self.model_id}: expected {self.sha256}, got {digest}"
            )


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = tuple(
            ModelRecord.from_dict(record, self.path.parent) for record in payload.get("models", ())
        )

    def select(self, hardware: HardwareInfo) -> ModelRecord:
        for record in self.records:
            if record.matches(hardware):
                record.verify()
                return record
        raise ModelUnavailableError(
            f"no predictor for {hardware.name!r}, compute capability "
            f"{hardware.compute_capability}, {hardware.vram_mb} MiB"
        )
