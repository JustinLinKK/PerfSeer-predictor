"""Tiny real-format task files used only by the local trainability gate."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import wave
import zipfile

from .adapters import TaskAdapter
from .fingerprints import canonical_sha256
from .prepared_view import (
    _JIGSAW_TARGETS,
    _RANZCR_TARGETS,
    build_shared_prepared_view,
)
from .real_data import VerifiedPreparedDataset


class LocalTaskFixtureError(RuntimeError):
    """Raised when a frozen adapter has no executable real-format fixture."""


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as error:
        raise LocalTaskFixtureError("Pillow cannot create a local image fixture") from error


def _wave_bytes(sample_rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * 64)
    return buffer.getvalue()


def _populate(task_id: str, public: Path) -> None:
    public.mkdir(parents=True)
    image = _image_bytes((10, 20, 30))
    clean = _image_bytes((30, 20, 10))
    if task_id == "histopathologic-cancer":
        _write_csv(public / "train_labels.csv", [{"id": "a", "label": 1}])
        _write(public / "train/a.tif", image)
    elif task_id == "dogs-vs-cats":
        _write_zip(public / "train.zip", {"train/cat.0.jpg": image})
    elif task_id == "dog-breed":
        _write_csv(public / "labels.csv", [{"id": "a", "breed": "beagle"}])
        _write(public / "train/a.jpg", image)
    elif task_id == "siim-isic-melanoma":
        _write_csv(public / "train.csv", [{"image_name": "a", "target": 0}])
        _write(public / "jpeg/train/a.jpg", image)
    elif task_id == "aptos2019":
        _write_csv(public / "train.csv", [{"id_code": "a", "diagnosis": 2}])
        _write(public / "train_images/a.png", image)
    elif task_id == "aerial-cactus":
        _write_csv(public / "train.csv", [{"id": "a.jpg", "has_cactus": 1}])
        _write_zip(public / "train.zip", {"train/a.jpg": image})
    elif task_id == "plant-pathology":
        _write_csv(
            public / "train.csv",
            [
                {
                    "image_id": "a",
                    "healthy": 1,
                    "multiple_diseases": 0,
                    "rust": 0,
                    "scab": 0,
                }
            ],
        )
        _write(public / "images/a.jpg", image)
    elif task_id == "ranzcr-clip":
        row: dict[str, object] = {"StudyInstanceUID": "a"}
        row.update(
            {name: int(index == 0) for index, name in enumerate(_RANZCR_TARGETS)}
        )
        _write_csv(public / "train.csv", [row])
        _write(public / "train/a.jpg", image)
    elif task_id == "leaf-classification":
        _write_csv(
            public / "train.csv",
            [{"id": 1, "species": "Acer", "margin1": 0.5}],
        )
        _write(public / "images/1.jpg", image)
    elif task_id == "denoising-dirty-documents":
        _write(public / "train/a.png", image)
        _write(public / "train_cleaned/a.png", clean)
    elif task_id == "jigsaw-toxic":
        row = {"id": "a", "comment_text": "hello"}
        row.update(
            {name: int(index == 0) for index, name in enumerate(_JIGSAW_TARGETS)}
        )
        _write_csv(public / "train.csv", [row])
    elif task_id == "detecting-insults":
        _write_csv(
            public / "train.csv",
            [{"Insult": 0, "Date": "x", "Comment": "hello"}],
        )
    elif task_id == "disaster-tweets":
        _write_csv(
            public / "train.csv",
            [
                {
                    "id": index,
                    "keyword": "fire" if index % 2 else "",
                    "location": "test" if index % 3 else "",
                    "text": f"fixture tweet {index}",
                    "target": index % 2,
                }
                for index in range(7_613)
            ],
        )
    elif task_id == "spooky-author":
        _write_csv(
            public / "train.csv",
            [{"id": "a", "text": "hello", "author": "EAP"}],
        )
    elif task_id == "random-acts-of-pizza":
        (public / "train.json").write_text(
            json.dumps(
                [
                    {
                        "request_id": "a",
                        "request_text": "pizza",
                        "requester_received_pizza": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
    elif task_id in {"text-normalization-english", "text-normalization-russian"}:
        prefix = "en" if task_id.endswith("english") else "ru"
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["sentence_id", "token_id", "class", "before", "after"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sentence_id": 0,
                "token_id": 0,
                "class": "PLAIN",
                "before": "one",
                "after": "one",
            }
        )
        _write_zip(
            public / f"{prefix}_train.csv.zip",
            {f"{prefix}_train.csv": buffer.getvalue().encode("utf-8")},
        )
    elif task_id == "mlsp-2013-birds":
        _write_csv(
            public / "essential_data/CVfolds_2.txt",
            [{"rec_id": 0, "fold": 0}],
        )
        _write_csv(
            public / "essential_data/rec_id2filename.txt",
            [{"rec_id": 0, "filename": "bird"}],
        )
        _write(
            public / "essential_data/rec_labels_test_hidden.txt",
            b"rec_id,[labels]\n0,1,3\n",
        )
        _write(public / "essential_data/src_wavs/bird.wav", _wave_bytes())
    elif task_id == "icml-2013-whale":
        _write_zip(
            public / "train2.zip",
            {"train2/20130101_x_TRAIN0_1.aif": _wave_bytes()},
        )
    elif task_id == "tensorflow-speech-yes-no":
        payload = _wave_bytes(16_000)
        for label in ("no", "yes"):
            for index in range(2_048):
                _write(
                    public / "train" / "audio" / label / f"speaker_{index:04d}.wav",
                    payload,
                )
    elif task_id == "nyc-taxi-fare":
        _write_csv(
            public / "labels.csv",
            [{"key": "a", "fare_amount": 4.5, "passenger_count": 1}],
        )
    elif task_id == "nomad2018":
        _write_csv(
            public / "train.csv",
            [
                {
                    "id": 1,
                    "spacegroup": 2,
                    "formation_energy_ev_natom": 0.2,
                    "bandgap_energy_ev": 1.2,
                }
            ],
        )
        _write(public / "train/1/geometry.xyz", b"1\nfixture\nH 0 0 0\n")
    elif task_id == "tabular-playground-dec-2021":
        _write_csv(
            public / "train.csv",
            [{"Id": 1, "feature": 0.5, "Cover_Type": 2}],
        )
    elif task_id == "tabular-playground-may-2022":
        _write_csv(
            public / "train.csv",
            [{"id": 1, "f_00": 0.5, "target": 1}],
        )
    else:
        raise LocalTaskFixtureError(f"task {task_id!r} has no local real-format fixture")


def build_local_real_format_batch(
    adapter: TaskAdapter,
) -> tuple[str, Mapping[str, Any]]:
    """Build one decoded batch through the same byte-bound loader used on NRP."""

    archive_sha256 = canonical_sha256(
        {
            "scope": "local_real_format_fixture_only",
            "task_id": adapter.task_id,
            "dataset_revision": adapter.entry.dataset_revision,
        }
    )
    with tempfile.TemporaryDirectory(prefix="perfseer-v3-real-fixture-") as directory:
        root = Path(directory)
        public = root / "public"
        prepared = root / "prepared"
        _populate(adapter.task_id, public)
        build_shared_prepared_view(
            adapter.entry,
            public,
            prepared,
            archive_sha256=archive_sha256,
        )
        dataset = VerifiedPreparedDataset(
            adapter.entry,
            public,
            prepared,
            archive_sha256,
        )
        batch = dataset.build_batch((0,))
        return dataset.dataset_fingerprint, batch


__all__ = [
    "LocalTaskFixtureError",
    "build_local_real_format_batch",
]
