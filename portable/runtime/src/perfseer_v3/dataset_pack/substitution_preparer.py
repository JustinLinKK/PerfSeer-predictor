"""Preparation dispatch for narrowly scoped non-MLE-bench substitutions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from .disaster_substitution import (
    DISASTER_TASK_ID,
    EXPECTED_INVENTORY,
    load_disaster_substitution_contract,
)
from .mlebench_bridge import PinnedMleBenchPreparer
from .task_registry import TaskRegistryEntry


class SubstitutionPreparationError(RuntimeError):
    """Raised when a custom substitution source violates its frozen schema."""


def _prepare_disaster_tweets(
    entry: TaskRegistryEntry,
    raw: Path,
    public: Path,
    private: Path,
) -> None:
    contract = load_disaster_substitution_contract()
    if entry.task_id != DISASTER_TASK_ID or entry.kaggle_slug != "nlp-getting-started":
        raise SubstitutionPreparationError("Disaster preparer received another task")
    if not raw.is_dir() or public.exists() or private.exists():
        raise SubstitutionPreparationError("Disaster preparer paths are invalid")
    actual_files = tuple(
        sorted(
            (path.relative_to(raw).as_posix(), path.stat().st_size)
            for path in raw.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    )
    if actual_files != tuple(sorted(EXPECTED_INVENTORY)):
        raise SubstitutionPreparationError("Disaster extracted inventory differs")
    if any(path.is_symlink() for path in raw.rglob("*")):
        raise SubstitutionPreparationError("Disaster source contains a symlink")
    train = raw / "train.csv"
    try:
        with train.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "keyword", "location", "text", "target"]:
                raise SubstitutionPreparationError("Disaster CSV columns differ")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise SubstitutionPreparationError("Disaster training CSV is unreadable") from error
    if len(rows) != int(contract.payload["new_task"]["source_row_count"]):
        raise SubstitutionPreparationError("Disaster source row count differs")
    identities = [row["id"] for row in rows]
    if any(not value for value in identities) or len(set(identities)) != len(identities):
        raise SubstitutionPreparationError("Disaster source IDs are empty or duplicated")
    if any(not row["text"].strip() for row in rows):
        raise SubstitutionPreparationError("Disaster source contains empty text")
    if any(row["target"] not in {"0", "1"} for row in rows):
        raise SubstitutionPreparationError("Disaster source contains an invalid target")
    if {row["target"] for row in rows} != {"0", "1"}:
        raise SubstitutionPreparationError("Disaster source is missing a target class")
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    for name, _ in EXPECTED_INVENTORY:
        shutil.copyfile(raw / name, public / name)
    (private / "UNUSED.json").write_text(
        json.dumps(
            {
                "task_id": DISASTER_TASK_ID,
                "reason": "competition public training view is independently contracted",
                "substitution_contract_sha256": contract.sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class SubstitutionAwarePreparer:
    """Dispatch one exact custom task and delegate every other task to MLE-bench."""

    def __init__(self, mlebench_checkout: str | Path) -> None:
        self._mlebench = PinnedMleBenchPreparer(Path(mlebench_checkout))

    def prepare(
        self,
        entry: TaskRegistryEntry,
        raw: str | Path,
        public: str | Path,
        private: str | Path,
    ) -> None:
        if entry.task_id == DISASTER_TASK_ID:
            _prepare_disaster_tweets(
                entry,
                Path(raw).resolve(),
                Path(public).resolve(),
                Path(private).resolve(),
            )
            return
        self._mlebench.prepare(entry, raw, public, private)


__all__ = ["SubstitutionAwarePreparer", "SubstitutionPreparationError"]
