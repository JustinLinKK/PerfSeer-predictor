"""Executable loader for integrity-verified MLE-bench prepared views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import zipfile

import torch
import torch.nn.functional as F

from .adapters import TaskAdapter
from .fingerprints import canonical_sha256, canonical_value
from .prepared_view import load_and_verify_prepared_view
from .task_registry import TaskRegistryEntry


class RealPreparedDataError(RuntimeError):
    """Raised when a verified row cannot be decoded into its task tensor format."""


def _stable_bucket(value: str, width: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % width


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return _stable_bucket(str(value), 10_000) / 10_000.0


def _pad_1d(rows: Sequence[torch.Tensor], *, value: int | float = 0) -> torch.Tensor:
    width = max(row.numel() for row in rows)
    return torch.stack([F.pad(row, (0, width - row.numel()), value=value) for row in rows])


@dataclass
class VerifiedPreparedDataset:
    """The exact 4,096-row view, with selected source bytes verified before use."""

    entry: TaskRegistryEntry
    public_directory: Path
    prepared_directory: Path
    archive_sha256: str

    def __post_init__(self) -> None:
        self.public_directory = Path(self.public_directory).resolve()
        self.prepared_directory = Path(self.prepared_directory).resolve()
        self.manifest = load_and_verify_prepared_view(
            self.entry,
            self.public_directory,
            self.prepared_directory,
            archive_sha256=self.archive_sha256,
            # The parent materializer verifies every selected source once. Each
            # decoded source is rehashed below, avoiding a 4,096-file prepass in
            # every fresh configuration process.
            verify_selected_source_bytes=False,
        )
        try:
            self.rows = tuple(
                json.loads(line)
                for line in (self.prepared_directory / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RealPreparedDataError("prepared rows cannot be loaded") from error
        # The verifier and consumer read independently. Bind the exact in-memory
        # rows used for training back to the signed view manifest to close that
        # small read/verify race.
        if (
            len(self.rows) != self.manifest.prepared_example_count
            or canonical_sha256(self.rows) != self.manifest.samples_sha256
        ):
            raise RealPreparedDataError("training rows differ from the verified view")
        unique_targets = {
            json.dumps(canonical_value(row["target"]), sort_keys=True, separators=(",", ":"))
            for row in self.rows
        }
        self._category_index = {
            value: index for index, value in enumerate(sorted(unique_targets))
        }
        width = int(self.entry.target_schema["target_width"])
        if (
            self.entry.target_schema["target_encoding"] == "categorical_index"
            and len(self._category_index) > width
        ):
            raise RealPreparedDataError("prepared categories exceed the frozen target width")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def dataset_fingerprint(self) -> str:
        return self.manifest.dataset_fingerprint

    def _reference_bytes(self, reference: Mapping[str, Any]) -> bytes:
        kind = reference.get("kind")
        relative = str(reference.get("path", ""))
        path = self.public_directory.joinpath(*PurePosixPath(relative).parts)
        try:
            if kind == "file":
                payload = path.read_bytes()
            elif kind == "zip_member":
                with zipfile.ZipFile(path) as archive:
                    payload = archive.read(str(reference["member"]))
            else:
                raise RealPreparedDataError("row contains an unsupported source reference")
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise RealPreparedDataError("cannot read a bound prepared source") from error
        if (
            len(payload) != reference.get("size_bytes")
            or hashlib.sha256(payload).hexdigest() != reference.get("sha256")
        ):
            raise RealPreparedDataError("prepared source changed between verification and decode")
        return payload

    def _image(self, reference: Mapping[str, Any]) -> torch.Tensor:
        try:
            from PIL import Image
            import numpy as np

            with Image.open(io.BytesIO(self._reference_bytes(reference))) as image:
                array = np.asarray(image.convert("RGB"), dtype="float32").copy()
        except Exception as error:
            raise RealPreparedDataError("prepared image cannot be decoded as RGB") from error
        return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)

    def _audio(self, reference: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
        payload = self._reference_bytes(reference)
        try:
            import soundfile

            samples, sample_rate = soundfile.read(
                io.BytesIO(payload), dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(samples.copy()).transpose(0, 1)
        except (ImportError, OSError, RuntimeError):
            # WAV fallback keeps local verification usable in a minimal PyTorch
            # environment. The production image includes soundfile for WAV/AIF.
            try:
                import wave
                import numpy as np

                with wave.open(io.BytesIO(payload), "rb") as stream:
                    channels = stream.getnchannels()
                    sample_rate = stream.getframerate()
                    sample_width = stream.getsampwidth()
                    frames = stream.readframes(stream.getnframes())
                dtypes = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
                if sample_width not in dtypes:
                    raise RealPreparedDataError("WAV sample width is unsupported")
                array = np.frombuffer(frames, dtype=dtypes[sample_width]).astype("float32")
                if sample_width == 1:
                    array = (array - 128.0) / 128.0
                else:
                    array /= float(2 ** (8 * sample_width - 1))
                waveform = torch.from_numpy(array.reshape(-1, channels).copy()).transpose(0, 1)
            except (OSError, EOFError, wave.Error, ValueError) as error:
                raise RealPreparedDataError("prepared audio cannot be decoded") from error
        if waveform.ndim != 2 or waveform.shape[-1] < 1 or sample_rate < 1:
            raise RealPreparedDataError("decoded audio shape/sample rate is invalid")
        return waveform.mean(dim=0, keepdim=True).float(), int(sample_rate)

    def _categorical_target(self, value: Any) -> torch.Tensor:
        key = json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"))
        return torch.tensor(self._category_index[key], dtype=torch.long)

    def _target(self, value: Any) -> torch.Tensor:
        encoding = self.entry.target_schema["target_encoding"]
        if encoding == "categorical_index":
            return self._categorical_target(value)
        if encoding in {"binary_vector", "float_vector"}:
            values = value if isinstance(value, list) else [value]
            return torch.tensor([_numeric(item) for item in values], dtype=torch.float32)
        if encoding == "stable_token_bucket_32_v1":
            return torch.tensor(
                [_stable_bucket(str(item), 32) for item in value], dtype=torch.long
            )
        if encoding == "paired_image":
            return self._image(value["media"])
        raise RealPreparedDataError(f"unsupported target encoding {encoding!r}")

    def decode(self, index: int) -> Mapping[str, Any]:
        row = self.rows[index]
        raw_inputs = row["inputs"]
        modality = self.entry.modality
        if modality == "vision":
            inputs = {"image": self._image(raw_inputs["media"])}
        elif modality == "audio":
            waveform, sample_rate = self._audio(raw_inputs["media"])
            inputs = {"waveform": waveform, "sample_rate": sample_rate}
        elif modality == "nlp":
            raw_tokens = raw_inputs.get("tokens", raw_inputs.get("text", []))
            words = [word for value in raw_tokens for word in str(value).split()]
            if not words:
                words = [""]
            tokens = torch.tensor([1 + _stable_bucket(word, 63) for word in words], dtype=torch.long)
            inputs = {"token_ids": tokens, "attention_mask": torch.ones_like(tokens, dtype=torch.bool)}
            if self.entry.target_schema["kind"] == "teacher_forced_seq2seq":
                target = self._target(row["target"])
                inputs["decoder_ids"] = torch.cat((torch.ones(1, dtype=torch.long), target[:-1]))
                return {"inputs": inputs, "target": target}
        elif modality == "tabular":
            features = raw_inputs["features"]
            ordered = [features[name] for name in sorted(features)]
            inputs = {
                "dense": torch.tensor([_numeric(value) for value in ordered], dtype=torch.float32),
                "categorical": torch.tensor(
                    [_stable_bucket(str(value), 8) for value in ordered[:3]], dtype=torch.long
                ),
            }
        elif modality == "graph":
            lines = self._reference_bytes(raw_inputs["geometry"]).decode("utf-8").splitlines()
            try:
                atom_count = int(lines[0])
                atoms = [line.split() for line in lines[2 : 2 + atom_count]]
                if len(atoms) != atom_count:
                    raise ValueError
                nodes = torch.tensor(
                    [
                        [_stable_bucket(atom[0], 32) / 31.0, *map(float, atom[1:4]), 0.0, 1.0]
                        for atom in atoms
                    ],
                    dtype=torch.float32,
                )
            except (ValueError, IndexError) as error:
                raise RealPreparedDataError("prepared geometry.xyz cannot be decoded") from error
            edge_ids = torch.arange(max(1, atom_count), dtype=torch.long)
            inputs = {
                "node_features": nodes,
                "edge_index": torch.stack((edge_ids, (edge_ids + 1).remainder(atom_count))),
                "edge_features": torch.ones((len(edge_ids), 3), dtype=torch.float32),
            }
        else:
            raise RealPreparedDataError(f"unsupported modality {modality!r}")
        return {"inputs": inputs, "target": self._target(row["target"])}

    def build_batch(self, indices: Sequence[int]) -> Mapping[str, Any]:
        if not indices:
            raise RealPreparedDataError("cannot build an empty real-data batch")
        samples = [self.decode(index % len(self)) for index in indices]
        modality = self.entry.modality
        targets = [sample["target"] for sample in samples]
        if modality == "vision":
            images = [
                F.interpolate(sample["inputs"]["image"].unsqueeze(0), (32, 32), mode="bilinear", align_corners=False).squeeze(0)
                for sample in samples
            ]
            inputs: dict[str, Any] = {"image": torch.stack(images)}
            if self.entry.target_schema["kind"] == "image_restoration":
                targets = [
                    F.interpolate(target.unsqueeze(0), (32, 32), mode="bilinear", align_corners=False).squeeze(0)
                    for target in targets
                ]
        elif modality == "nlp":
            inputs = {
                "token_ids": _pad_1d([sample["inputs"]["token_ids"] for sample in samples]),
                "attention_mask": _pad_1d(
                    [sample["inputs"]["attention_mask"] for sample in samples], value=False
                ),
            }
            if "decoder_ids" in samples[0]["inputs"]:
                inputs["decoder_ids"] = _pad_1d(
                    [sample["inputs"]["decoder_ids"] for sample in samples]
                )
                target = _pad_1d(targets)
            else:
                target = torch.stack(targets)
            return {"inputs": inputs, "target": target}
        elif modality == "audio":
            waves = [
                F.interpolate(sample["inputs"]["waveform"].unsqueeze(0), 256, mode="linear", align_corners=False).squeeze(0)
                for sample in samples
            ]
            inputs = {"waveform": torch.stack(waves), "sample_rate": samples[0]["inputs"]["sample_rate"]}
        elif modality == "tabular":
            inputs = {
                "dense": _pad_1d([sample["inputs"]["dense"] for sample in samples]),
                "categorical": _pad_1d([sample["inputs"]["categorical"] for sample in samples]),
            }
        elif modality == "graph":
            nodes, edges, edge_features, graph_index = [], [], [], []
            offset = 0
            for graph_id, sample in enumerate(samples):
                item = sample["inputs"]
                nodes.append(item["node_features"])
                edges.append(item["edge_index"] + offset)
                edge_features.append(item["edge_features"])
                graph_index.append(torch.full((item["node_features"].shape[0],), graph_id, dtype=torch.long))
                offset += item["node_features"].shape[0]
            try:
                from torch_geometric.data import Batch, Data

                pyg_batch = Batch.from_data_list(
                    [
                        Data(
                            x=sample["inputs"]["node_features"],
                            edge_index=sample["inputs"]["edge_index"],
                            edge_attr=sample["inputs"]["edge_features"],
                        )
                        for sample in samples
                    ]
                )
            except (ImportError, RuntimeError, TypeError, ValueError) as error:
                raise RealPreparedDataError("cannot construct the required PyG graph batch") from error
            inputs = {
                "node_features": torch.cat(nodes),
                "edge_index": torch.cat(edges, dim=1),
                "edge_features": torch.cat(edge_features),
                "graph_index": torch.cat(graph_index),
                "graph_count": len(samples),
                "pyg_batch": pyg_batch,
            }
        else:  # pragma: no cover - decode already rejects this
            raise RealPreparedDataError("unsupported batch modality")
        return {"inputs": inputs, "target": torch.stack(targets)}


def build_verified_real_batch(
    adapter: TaskAdapter,
    public_directory: str | Path,
    prepared_directory: str | Path,
    *,
    archive_sha256: str,
    indices: Sequence[int],
) -> tuple[VerifiedPreparedDataset, Mapping[str, Any]]:
    dataset = VerifiedPreparedDataset(
        adapter.entry,
        Path(public_directory),
        Path(prepared_directory),
        archive_sha256,
    )
    return dataset, dataset.build_batch(indices)


__all__ = [
    "RealPreparedDataError",
    "VerifiedPreparedDataset",
    "build_verified_real_batch",
]
