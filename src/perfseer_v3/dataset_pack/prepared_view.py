"""Deterministic 4,096-example shared views over pinned MLE-bench outputs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import wave
import zipfile

from .fingerprints import canonical_sha256, canonical_value
from .labeler_profile import PROFILE
from .speech_substitution import SPEECH_TASK_ID, load_speech_substitution_contract
from .storage import atomic_write_json, atomic_write_jsonl
from .task_registry import MLEBENCH_METADATA_REVISION, TaskRegistryEntry


PREPARED_VIEW_VERSION = (
    "perfseer_v3_nrp_a10_speech_prepared_view_v2"
    if PROFILE.uses_speech_v2
    else "perfseer_v3_v100_prepared_view_v2"
)
PREPARED_EXAMPLE_COUNT = 4_096


class PreparedViewError(RuntimeError):
    """Raised when an MLE-bench public training view is incomplete or malformed."""


def _safe_relative(value: str, *, context: str) -> str:
    if not value or "\\" in value or value.startswith("/"):
        raise PreparedViewError(f"{context} path is unsafe")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PreparedViewError(f"{context} path traverses its root")
    return path.as_posix()


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise PreparedViewError("prepared table contains a non-finite number")
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        integer = int(text)
        if str(integer) == text or text.startswith("+") and str(integer) == text[1:]:
            return integer
    except ValueError:
        pass
    try:
        number = float(text)
        if math.isfinite(number):
            return number
    except ValueError:
        pass
    return text


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise PreparedViewError(f"prepared training table is missing: {path.name!r}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows:
        raise PreparedViewError(f"prepared training table is empty: {path.name!r}")
    return rows


def _read_csv_from_zip(path: Path, member: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise PreparedViewError(f"prepared ZIP table is missing: {path.name!r}")
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                rows = [dict(row) for row in csv.DictReader(text)]
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise PreparedViewError(f"cannot read {member!r} from {path.name!r}") from error
    if not rows:
        raise PreparedViewError(f"prepared ZIP table is empty: {member!r}")
    return rows


def _require_columns(rows: Sequence[Mapping[str, Any]], columns: Iterable[str]) -> None:
    required = set(columns)
    if not rows or not required <= set(rows[0]):
        raise PreparedViewError(f"prepared table is missing columns {sorted(required)!r}")


def _file(public: Path, relative: str) -> dict[str, str]:
    normalized = _safe_relative(relative, context="prepared file")
    path = public.joinpath(*PurePosixPath(normalized).parts)
    if not path.is_file() or path.is_symlink():
        raise PreparedViewError(f"prepared source file is missing: {normalized!r}")
    return {"kind": "file", "path": normalized}


def _zip_members(public: Path, relative: str) -> tuple[Path, set[str]]:
    normalized = _safe_relative(relative, context="prepared ZIP")
    path = public.joinpath(*PurePosixPath(normalized).parts)
    if not path.is_file() or path.is_symlink():
        raise PreparedViewError(f"prepared ZIP is missing: {normalized!r}")
    try:
        with zipfile.ZipFile(path) as archive:
            members = {
                _safe_relative(info.filename.rstrip("/"), context="prepared ZIP member")
                for info in archive.infolist()
                if not info.is_dir()
            }
    except (OSError, zipfile.BadZipFile) as error:
        raise PreparedViewError(f"prepared ZIP is unreadable: {normalized!r}") from error
    if not members:
        raise PreparedViewError(f"prepared ZIP is empty: {normalized!r}")
    return path, members


def _zip_ref(zip_path: str, member: str, members: set[str]) -> dict[str, str]:
    normalized = _safe_relative(member, context="prepared ZIP member")
    if normalized not in members:
        raise PreparedViewError(f"prepared ZIP member is missing: {normalized!r}")
    return {"kind": "zip_member", "path": zip_path, "member": normalized}


@dataclass(frozen=True)
class SourceSample:
    source_sample_id: str
    inputs: Mapping[str, Any]
    target: Any

    def validate(self, target_schema: Mapping[str, Any] | None = None) -> None:
        if not self.source_sample_id or not isinstance(self.inputs, Mapping) or not self.inputs:
            raise PreparedViewError("prepared source sample is incomplete")
        canonical_value(self.inputs)
        canonical_value(self.target)
        if target_schema is not None:
            _validate_target(self.target, target_schema)


def _validate_target(target: Any, schema: Mapping[str, Any]) -> None:
    """Reject prepared labels that cannot satisfy the frozen executable schema."""

    encoding = schema.get("target_encoding")
    width = schema.get("target_width")
    if type(width) is not int or width < 1:
        raise PreparedViewError("task target width is invalid")
    if encoding == "categorical_index":
        if isinstance(target, (list, tuple, Mapping)) or target is None:
            raise PreparedViewError("categorical target must be one scalar value")
        return
    if encoding in {"binary_vector", "float_vector"}:
        values = [target] if width == 1 and not isinstance(target, (list, tuple)) else target
        if not isinstance(values, (list, tuple)) or len(values) != width:
            raise PreparedViewError(f"{encoding} target width differs from the task schema")
        for value in values:
            if isinstance(value, bool):
                number = int(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                number = float(value)
            else:
                raise PreparedViewError(f"{encoding} target contains a non-numeric value")
            if encoding == "binary_vector" and number not in {0, 1}:
                raise PreparedViewError("binary target contains a value outside {0, 1}")
        return
    if encoding == "paired_image":
        if not isinstance(target, Mapping) or set(target) != {"media"}:
            raise PreparedViewError("paired-image target must contain exactly one media reference")
        return
    if encoding == "stable_token_bucket_32_v1":
        if not isinstance(target, (list, tuple)) or not target or any(
            not isinstance(value, str) for value in target
        ):
            raise PreparedViewError("sequence target must be a non-empty string sequence")
        return
    raise PreparedViewError(f"unsupported frozen target encoding {encoding!r}")


def _sha256_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        digest.update(block)
    return digest.hexdigest(), size


def _bind_input_content(
    public: Path,
    value: Any,
    cache: dict[tuple[str, ...], tuple[str, int]],
) -> Any:
    """Bind only selected media/geometry bytes, deduplicating repeated view rows."""

    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "file":
            if set(value) != {"kind", "path"}:
                raise PreparedViewError("prepared file reference schema is invalid")
            relative = _safe_relative(str(value["path"]), context="prepared file")
            key = ("file", relative)
            if key not in cache:
                path = public.joinpath(*PurePosixPath(relative).parts)
                if not path.is_file() or path.is_symlink():
                    raise PreparedViewError(f"prepared source file is missing: {relative!r}")
                with path.open("rb") as stream:
                    cache[key] = _sha256_stream(stream)
            digest, size = cache[key]
            return {"kind": "file", "path": relative, "sha256": digest, "size_bytes": size}
        if kind == "zip_member":
            if set(value) != {"kind", "path", "member"}:
                raise PreparedViewError("prepared ZIP-member reference schema is invalid")
            relative = _safe_relative(str(value["path"]), context="prepared ZIP")
            member = _safe_relative(str(value["member"]), context="prepared ZIP member")
            key = ("zip_member", relative, member)
            if key not in cache:
                path = public.joinpath(*PurePosixPath(relative).parts)
                if not path.is_file() or path.is_symlink():
                    raise PreparedViewError(f"prepared ZIP is missing: {relative!r}")
                try:
                    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
                        cache[key] = _sha256_stream(stream)
                except (KeyError, OSError, zipfile.BadZipFile) as error:
                    raise PreparedViewError(
                        f"cannot bind {member!r} from prepared ZIP {relative!r}"
                    ) from error
            digest, size = cache[key]
            return {
                "kind": "zip_member",
                "path": relative,
                "member": member,
                "sha256": digest,
                "size_bytes": size,
            }
        return {str(key): _bind_input_content(public, item, cache) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bind_input_content(public, item, cache) for item in value]
    return value


def _verify_bound_input_content(
    public: Path,
    value: Any,
    cache: dict[tuple[str, ...], tuple[str, int]],
) -> None:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind in {"file", "zip_member"}:
            expected_keys = {"kind", "path", "sha256", "size_bytes"}
            if kind == "zip_member":
                expected_keys.add("member")
            if set(value) != expected_keys:
                raise PreparedViewError("bound prepared reference schema is invalid")
            unbound = {key: value[key] for key in expected_keys if key not in {"sha256", "size_bytes"}}
            actual = _bind_input_content(public, unbound, cache)
            if actual != value:
                raise PreparedViewError("prepared source bytes changed after view creation")
            return
        for item in value.values():
            _verify_bound_input_content(public, item, cache)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _verify_bound_input_content(public, item, cache)


def _image_table(
    public: Path,
    *,
    table: str,
    id_column: str,
    target_columns: Sequence[str],
    image_template: str,
    extra_input_columns: Sequence[str] = (),
) -> list[SourceSample]:
    rows = _read_csv(public / table)
    _require_columns(rows, [id_column, *target_columns, *extra_input_columns])
    result = []
    for row in rows:
        identity = str(row[id_column])
        target: Any = (
            _cell(row[target_columns[0]])
            if len(target_columns) == 1
            else [_cell(row[name]) for name in target_columns]
        )
        inputs: dict[str, Any] = {
            "media": _file(public, image_template.format(id=identity)),
        }
        if extra_input_columns:
            inputs["tabular"] = {name: _cell(row[name]) for name in extra_input_columns}
        result.append(SourceSample(identity, inputs, target))
    return result


def _zip_image_table(
    public: Path,
    *,
    table: str,
    zip_path: str,
    id_column: str,
    target_column: str,
    member_template: str,
) -> list[SourceSample]:
    rows = _read_csv(public / table)
    _require_columns(rows, [id_column, target_column])
    _, members = _zip_members(public, zip_path)
    return [
        SourceSample(
            str(row[id_column]),
            {
                "media": _zip_ref(
                    zip_path,
                    member_template.format(id=str(row[id_column])),
                    members,
                )
            },
            _cell(row[target_column]),
        )
        for row in rows
    ]


def _text_table(
    public: Path,
    *,
    table: str,
    id_column: str | None,
    text_columns: Sequence[str],
    target_columns: Sequence[str],
) -> list[SourceSample]:
    rows = _read_csv(public / table)
    required = [*text_columns, *target_columns]
    if id_column:
        required.append(id_column)
    _require_columns(rows, required)
    result = []
    for index, row in enumerate(rows):
        identity = str(row[id_column]) if id_column else canonical_sha256(
            {"index": index, "text": [row[name] for name in text_columns]}
        )
        target: Any = (
            _cell(row[target_columns[0]])
            if len(target_columns) == 1
            else [_cell(row[name]) for name in target_columns]
        )
        result.append(
            SourceSample(
                identity,
                {"text": [str(row[name] or "") for name in text_columns]},
                target,
            )
        )
    return result


def _tabular_table(
    public: Path,
    *,
    table: str,
    id_column: str,
    target_columns: Sequence[str],
) -> list[SourceSample]:
    rows = _read_csv(public / table)
    _require_columns(rows, [id_column, *target_columns])
    feature_columns = [name for name in rows[0] if name not in {id_column, *target_columns}]
    if not feature_columns:
        raise PreparedViewError("tabular task has no feature columns")
    result = []
    for row in rows:
        target: Any = (
            _cell(row[target_columns[0]])
            if len(target_columns) == 1
            else [_cell(row[name]) for name in target_columns]
        )
        result.append(
            SourceSample(
                str(row[id_column]),
                {"features": {name: _cell(row[name]) for name in feature_columns}},
                target,
            )
        )
    return result


_RANZCR_TARGETS = (
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
)
_JIGSAW_TARGETS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)


def _source_samples(entry: TaskRegistryEntry, public: Path) -> list[SourceSample]:
    task = entry.task_id
    if task == "histopathologic-cancer":
        return _image_table(public, table="train_labels.csv", id_column="id", target_columns=("label",), image_template="train/{id}.tif")
    if task == "dogs-vs-cats":
        _, members = _zip_members(public, "train.zip")
        result = []
        for member in sorted(members):
            name = PurePosixPath(member).name
            if name.startswith("cat."):
                target = 0
            elif name.startswith("dog."):
                target = 1
            else:
                continue
            result.append(SourceSample(name, {"media": _zip_ref("train.zip", member, members)}, target))
        return result
    if task == "dog-breed":
        return _image_table(public, table="labels.csv", id_column="id", target_columns=("breed",), image_template="train/{id}.jpg")
    if task == "siim-isic-melanoma":
        return _image_table(public, table="train.csv", id_column="image_name", target_columns=("target",), image_template="jpeg/train/{id}.jpg")
    if task == "aptos2019":
        return _image_table(public, table="train.csv", id_column="id_code", target_columns=("diagnosis",), image_template="train_images/{id}.png")
    if task == "aerial-cactus":
        return _zip_image_table(public, table="train.csv", zip_path="train.zip", id_column="id", target_column="has_cactus", member_template="train/{id}")
    if task == "plant-pathology":
        return _image_table(public, table="train.csv", id_column="image_id", target_columns=("healthy", "multiple_diseases", "rust", "scab"), image_template="images/{id}.jpg")
    if task == "ranzcr-clip":
        return _image_table(public, table="train.csv", id_column="StudyInstanceUID", target_columns=_RANZCR_TARGETS, image_template="train/{id}.jpg")
    if task == "leaf-classification":
        rows = _read_csv(public / "train.csv")
        _require_columns(rows, ("id", "species"))
        extra = tuple(name for name in rows[0] if name not in {"id", "species"})
        return _image_table(public, table="train.csv", id_column="id", target_columns=("species",), image_template="images/{id}.jpg", extra_input_columns=extra)
    if task == "denoising-dirty-documents":
        dirty = {path.name: path for path in (public / "train").glob("*.png")}
        clean = {path.name: path for path in (public / "train_cleaned").glob("*.png")}
        if not dirty or set(dirty) != set(clean):
            raise PreparedViewError("dirty-document train/clean pairs are incomplete")
        return [
            SourceSample(name, {"media": _file(public, f"train/{name}")}, {"media": _file(public, f"train_cleaned/{name}")})
            for name in sorted(dirty)
        ]
    if task == "jigsaw-toxic":
        return _text_table(public, table="train.csv", id_column="id", text_columns=("comment_text",), target_columns=_JIGSAW_TARGETS)
    if task == "detecting-insults":
        return _text_table(public, table="train.csv", id_column=None, text_columns=("Date", "Comment"), target_columns=("Insult",))
    if task == "spooky-author":
        return _text_table(public, table="train.csv", id_column="id", text_columns=("text",), target_columns=("author",))
    if task == "random-acts-of-pizza":
        path = public / "train.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PreparedViewError("pizza training JSON is unreadable") from error
        if not isinstance(rows, list) or not rows:
            raise PreparedViewError("pizza training JSON is empty")
        result = []
        for row in rows:
            if not isinstance(row, Mapping) or "request_id" not in row or "requester_received_pizza" not in row:
                raise PreparedViewError("pizza training record schema is invalid")
            text = [str(value) for key, value in sorted(row.items()) if key not in {"request_id", "requester_received_pizza"} and isinstance(value, (str, int, float, bool))]
            result.append(SourceSample(str(row["request_id"]), {"text": text}, _cell(row["requester_received_pizza"])))
        return result
    if task in {"text-normalization-english", "text-normalization-russian"}:
        prefix = "en" if task.endswith("english") else "ru"
        rows = _read_csv_from_zip(public / f"{prefix}_train.csv.zip", f"{prefix}_train.csv")
        _require_columns(rows, ("sentence_id", "token_id", "before", "after"))
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["sentence_id"]), []).append(row)
        return [
            SourceSample(
                sentence,
                {"tokens": [str(row["before"]) for row in sorted(values, key=lambda row: int(row["token_id"]))]},
                [str(row["after"]) for row in sorted(values, key=lambda row: int(row["token_id"]))],
            )
            for sentence, values in grouped.items()
        ]
    if task == "mlsp-2013-birds":
        folds = _read_csv(public / "essential_data/CVfolds_2.txt")
        names = _read_csv(public / "essential_data/rec_id2filename.txt")
        fold_by_id = {str(row["rec_id"]): int(row["fold"]) for row in folds}
        filename_by_id = {str(row["rec_id"]): str(row["filename"]) for row in names}
        label_path = public / "essential_data/rec_labels_test_hidden.txt"
        labels: dict[str, list[int]] = {}
        for line in label_path.read_text(encoding="utf-8").splitlines()[1:]:
            pieces = line.split(",")
            labels[pieces[0]] = [int(value) for value in pieces[1:] if value and value != "?"]
        train_ids = sorted(identity for identity, fold in fold_by_id.items() if fold == 0)
        return [
            SourceSample(identity, {"media": _file(public, f"essential_data/src_wavs/{filename_by_id[identity]}.wav")}, [int(index in labels.get(identity, ())) for index in range(19)])
            for identity in train_ids
        ]
    if task == "icml-2013-whale":
        _, members = _zip_members(public, "train2.zip")
        result = []
        for member in sorted(members):
            name = PurePosixPath(member).name
            if not name.lower().endswith(".aif") or "_TRAIN" not in name:
                continue
            stem = Path(name).stem
            label_text = stem.rsplit("_", 1)[-1]
            if label_text not in {"0", "1"}:
                raise PreparedViewError("whale training filename has no binary target")
            result.append(SourceSample(name, {"media": _zip_ref("train2.zip", member, members)}, int(label_text)))
        return result
    if task == SPEECH_TASK_ID:
        return _speech_yes_no_samples(public)
    if task == "nyc-taxi-fare":
        return _tabular_table(public, table="labels.csv", id_column="key", target_columns=("fare_amount",))
    if task == "nomad2018":
        rows = _read_csv(public / "train.csv")
        targets = ("formation_energy_ev_natom", "bandgap_energy_ev")
        _require_columns(rows, ("id", *targets))
        feature_columns = [name for name in rows[0] if name not in {"id", *targets}]
        return [
            SourceSample(
                str(row["id"]),
                {"geometry": _file(public, f"train/{row['id']}/geometry.xyz"), "features": {name: _cell(row[name]) for name in feature_columns}},
                [_cell(row[name]) for name in targets],
            )
            for row in rows
        ]
    if task == "tabular-playground-dec-2021":
        return _tabular_table(public, table="train.csv", id_column="Id", target_columns=("Cover_Type",))
    if task == "tabular-playground-may-2022":
        return _tabular_table(public, table="train.csv", id_column="id", target_columns=("target",))
    raise PreparedViewError(f"task {task!r} has no real prepared-view recipe")


def _speech_yes_no_samples(public: Path) -> list[SourceSample]:
    """Select an exact balanced view without consulting MLE-bench's test split."""

    contract = load_speech_substitution_contract().payload["new_task"]
    audio_root = public / str(contract["public_training_root"])
    if not audio_root.is_dir() or audio_root.is_symlink():
        raise PreparedViewError("MLE-bench public speech training root is missing")
    required = int(contract["required_valid_wavs_per_class"])
    sample_rate = int(contract["required_sample_rate_hz"])
    selected: list[SourceSample] = []
    for class_name, target in (("no", 0), ("yes", 1)):
        class_root = audio_root / class_name
        if not class_root.is_dir() or class_root.is_symlink():
            raise PreparedViewError(f"speech class {class_name!r} is missing")
        qualified: list[tuple[str, str]] = []
        for path in sorted(class_root.rglob("*.wav")):
            if not path.is_file() or path.is_symlink():
                raise PreparedViewError("speech training view contains a non-regular WAV")
            relative = path.relative_to(public).as_posix()
            try:
                with wave.open(str(path), "rb") as stream:
                    actual_sample_rate = stream.getframerate()
                    frames = stream.getnframes()
                    channels = stream.getnchannels()
            except (OSError, EOFError, wave.Error) as error:
                raise PreparedViewError(f"speech WAV is corrupt: {relative!r}") from error
            if (
                actual_sample_rate != sample_rate
                or frames < 1
                or channels < 1
            ):
                raise PreparedViewError(
                    f"speech WAV violates the 16 kHz non-empty audio contract: {relative!r}"
                )
            qualified.append((hashlib.sha256(relative.encode("utf-8")).hexdigest(), relative))
        if len(qualified) < required:
            raise PreparedViewError(
                f"speech class {class_name!r} has {len(qualified)} valid WAVs; "
                f"at least {required} are required"
            )
        for _, relative in sorted(qualified)[:required]:
            selected.append(
                SourceSample(relative, {"media": _file(public, relative)}, target)
            )
    if len(selected) != PREPARED_EXAMPLE_COUNT:
        raise PreparedViewError("speech binary view is not exactly 2,048/2,048")
    return selected


@dataclass(frozen=True)
class PreparedViewManifest:
    version: str
    task_id: str
    kaggle_slug: str
    modality: str
    mlebench_revision: str
    task_schema_sha256: str
    archive_sha256: str
    source_example_count: int
    prepared_example_count: int
    sampling_with_replacement: bool
    recipe_sha256: str
    samples_sha256: str
    dataset_fingerprint: str

    def validate(self) -> None:
        if self.version != PREPARED_VIEW_VERSION or not self.task_id or not self.kaggle_slug:
            raise PreparedViewError("prepared-view identity is invalid")
        if self.mlebench_revision != MLEBENCH_METADATA_REVISION:
            raise PreparedViewError("prepared view uses the wrong MLE-bench revision")
        if self.source_example_count < 1 or self.prepared_example_count != PREPARED_EXAMPLE_COUNT:
            raise PreparedViewError("prepared-view example counts are invalid")
        if self.sampling_with_replacement != (self.source_example_count < PREPARED_EXAMPLE_COUNT):
            raise PreparedViewError("prepared-view replacement flag is inconsistent")
        for name in (
            "task_schema_sha256",
            "archive_sha256",
            "recipe_sha256",
            "samples_sha256",
            "dataset_fingerprint",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise PreparedViewError(f"prepared-view {name} is not a SHA-256 digest")
        expected = canonical_sha256(
            {
                "version": self.version,
                "task_id": self.task_id,
                "kaggle_slug": self.kaggle_slug,
                "modality": self.modality,
                "mlebench_revision": self.mlebench_revision,
                "task_schema_sha256": self.task_schema_sha256,
                "archive_sha256": self.archive_sha256,
                "source_example_count": self.source_example_count,
                "prepared_example_count": self.prepared_example_count,
                "sampling_with_replacement": self.sampling_with_replacement,
                "recipe_sha256": self.recipe_sha256,
                "samples_sha256": self.samples_sha256,
            }
        )
        if self.dataset_fingerprint != expected:
            raise PreparedViewError("prepared-view dataset fingerprint is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return canonical_value(asdict(self))


def build_shared_prepared_view(
    entry: TaskRegistryEntry,
    public_directory: str | Path,
    destination: str | Path,
    *,
    archive_sha256: str,
) -> PreparedViewManifest:
    public = Path(public_directory).resolve()
    target = Path(destination)
    if not public.is_dir() or target.exists() or target.is_symlink():
        raise PreparedViewError("prepared-view source/destination state is invalid")
    if len(archive_sha256) != 64:
        raise PreparedViewError("prepared-view archive SHA-256 is invalid")
    sources = _source_samples(entry, public)
    if not sources:
        raise PreparedViewError("prepared-view recipe found no training examples")
    for source in sources:
        source.validate(entry.target_schema)
    if len({source.source_sample_id for source in sources}) != len(sources):
        raise PreparedViewError("prepared source sample IDs are not unique")
    speech_view = entry.task_id == SPEECH_TASK_ID
    if speech_view:
        selected = sources
    else:
        ordered = sorted(
            sources,
            key=lambda source: canonical_sha256(
                {"task_id": entry.task_id, "source_sample_id": source.source_sample_id}
            ),
        )
        selected = [ordered[index % len(ordered)] for index in range(PREPARED_EXAMPLE_COUNT)]
    occurrences: dict[str, int] = {}
    rows = []
    content_cache: dict[tuple[str, ...], tuple[str, int]] = {}
    for index, source in enumerate(selected):
        occurrence = occurrences.get(source.source_sample_id, 0)
        occurrences[source.source_sample_id] = occurrence + 1
        rows.append(
            {
                "view_index": index,
                "source_sample_id": source.source_sample_id,
                "source_occurrence": occurrence,
                "inputs": _bind_input_content(public, source.inputs, content_cache),
                "target": _bind_input_content(public, source.target, content_cache),
            }
        )
    recipe: dict[str, Any] = {
        "version": PREPARED_VIEW_VERSION,
        "task_id": entry.task_id,
        "mlebench_revision": MLEBENCH_METADATA_REVISION,
        "prepared_example_count": PREPARED_EXAMPLE_COUNT,
        "selection": "sha256_order_then_deterministic_cycle_v1",
    }
    if speech_view:
        substitution = load_speech_substitution_contract()
        recipe.update(
            {
                "selection": "sha256_relative_path_then_relative_path_v1",
                "classes": {"no": 0, "yes": 1},
                "valid_wavs_per_class": 2_048,
                "sample_rate_hz": 16_000,
                "public_training_root": "train/audio",
                "ignore_mlebench_test_split": True,
                "substitution_contract_sha256": substitution.sha256,
            }
        )
    recipe_sha256 = canonical_sha256(recipe)
    samples_sha256 = canonical_sha256(rows)
    task_schema_sha256 = canonical_sha256(entry.target_schema)
    fingerprint = canonical_sha256(
        {
            "version": PREPARED_VIEW_VERSION,
            "task_id": entry.task_id,
            "kaggle_slug": entry.kaggle_slug,
            "modality": entry.modality,
            "mlebench_revision": MLEBENCH_METADATA_REVISION,
            "task_schema_sha256": task_schema_sha256,
            "archive_sha256": archive_sha256,
            "source_example_count": len(sources),
            "prepared_example_count": len(rows),
            "sampling_with_replacement": False
            if speech_view
            else len(sources) < PREPARED_EXAMPLE_COUNT,
            "recipe_sha256": recipe_sha256,
            "samples_sha256": samples_sha256,
        }
    )
    manifest = PreparedViewManifest(
        version=PREPARED_VIEW_VERSION,
        task_id=entry.task_id,
        kaggle_slug=entry.kaggle_slug,
        modality=entry.modality,
        mlebench_revision=MLEBENCH_METADATA_REVISION,
        task_schema_sha256=task_schema_sha256,
        archive_sha256=archive_sha256,
        source_example_count=len(sources),
        prepared_example_count=len(rows),
        sampling_with_replacement=False
        if speech_view
        else len(sources) < PREPARED_EXAMPLE_COUNT,
        recipe_sha256=recipe_sha256,
        samples_sha256=samples_sha256,
        dataset_fingerprint=fingerprint,
    )
    manifest.validate()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        atomic_write_jsonl(staging / "samples.jsonl", rows)
        atomic_write_json(staging / "manifest.json", manifest.to_dict())
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def load_and_verify_prepared_view(
    entry: TaskRegistryEntry,
    public_directory: str | Path,
    prepared_directory: str | Path,
    *,
    archive_sha256: str,
    verify_selected_source_bytes: bool = True,
) -> PreparedViewManifest:
    """Re-read the view and every selected source byte before resume or training."""

    public = Path(public_directory).resolve()
    prepared = Path(prepared_directory).resolve()
    try:
        payload = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        manifest = PreparedViewManifest(**payload)
        lines = (prepared / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise PreparedViewError("prepared view cannot be loaded") from error
    manifest.validate()
    if (
        manifest.task_id != entry.task_id
        or manifest.kaggle_slug != entry.kaggle_slug
        or manifest.modality != entry.modality
        or manifest.task_schema_sha256 != canonical_sha256(entry.target_schema)
        or manifest.archive_sha256 != archive_sha256
    ):
        raise PreparedViewError("prepared view does not match the frozen task")
    if len(lines) != PREPARED_EXAMPLE_COUNT:
        raise PreparedViewError("prepared view row count changed")
    rows: list[Mapping[str, Any]] = []
    content_cache: dict[tuple[str, ...], tuple[str, int]] = {}
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PreparedViewError("prepared view contains malformed JSONL") from error
        if not isinstance(row, Mapping) or set(row) != {
            "view_index", "source_sample_id", "source_occurrence", "inputs", "target"
        }:
            raise PreparedViewError("prepared view row schema changed")
        if row["view_index"] != index or type(row["source_occurrence"]) is not int or row["source_occurrence"] < 0:
            raise PreparedViewError("prepared view row identity changed")
        SourceSample(str(row["source_sample_id"]), row["inputs"], row["target"]).validate(
            entry.target_schema
        )
        if verify_selected_source_bytes:
            _verify_bound_input_content(public, row["inputs"], content_cache)
            _verify_bound_input_content(public, row["target"], content_cache)
        rows.append(row)
    if canonical_sha256(rows) != manifest.samples_sha256:
        raise PreparedViewError("prepared view samples hash changed")
    return manifest


__all__ = [
    "PREPARED_EXAMPLE_COUNT",
    "PREPARED_VIEW_VERSION",
    "PreparedViewError",
    "PreparedViewManifest",
    "SourceSample",
    "build_shared_prepared_view",
    "load_and_verify_prepared_view",
]
