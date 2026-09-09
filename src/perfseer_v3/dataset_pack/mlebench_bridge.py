"""Pinned MLE-bench preparation bridge for already downloaded task data.

MLE-bench's public CLI owns downloading and extraction.  PerfSeer must instead
call the audited per-competition ``prepare(raw, public, private)`` function
after its own disk and archive gates.  This bridge verifies the checkout commit
and invokes only that function in a child process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence

from .task_registry import MLEBENCH_METADATA_REVISION, TaskRegistryEntry


class MleBenchPreparationError(RuntimeError):
    """Raised when the pinned preparer checkout or invocation is invalid."""


def _git(checkout: Path, arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise MleBenchPreparationError("cannot inspect the MLE-bench checkout") from error
    return result.stdout.strip()


def validate_mlebench_checkout(checkout: str | Path) -> Path:
    root = Path(checkout).resolve()
    if not root.is_dir():
        raise MleBenchPreparationError("pinned MLE-bench checkout is missing")
    if _git(root, ["rev-parse", "HEAD"]) != MLEBENCH_METADATA_REVISION:
        raise MleBenchPreparationError("MLE-bench checkout is not at the frozen revision")
    if _git(root, ["status", "--porcelain", "--untracked-files=all"]):
        raise MleBenchPreparationError("MLE-bench checkout contains modified or untracked files")
    if not (root / "mlebench" / "registry.py").is_file():
        raise MleBenchPreparationError("MLE-bench checkout has no registry implementation")
    return root


def verify_frozen_preparers(
    checkout: str | Path,
    entries: Sequence[TaskRegistryEntry],
) -> None:
    root = validate_mlebench_checkout(checkout)
    missing = [
        entry.kaggle_slug
        for entry in entries
        if not (
            root / "mlebench" / "competitions" / entry.kaggle_slug / "prepare.py"
        ).is_file()
    ]
    if missing:
        raise MleBenchPreparationError(
            f"frozen MLE-bench checkout lacks preparers for {sorted(missing)!r}"
        )
    environment = os.environ.copy()
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        environment.pop(name, None)
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )
    with tempfile.TemporaryDirectory(prefix=".perfseer-mlebench-preflight-") as sandbox:
        os.chmod(sandbox, 0o700)
        environment["KAGGLE_CONFIG_DIR"] = sandbox
        command = [
            sys.executable,
            "-m",
            "perfseer_v3.dataset_pack.mlebench_bridge",
            "--preflight",
            "--checkout",
            str(root),
        ]
        for entry in entries:
            command.extend(("--competition", entry.kaggle_slug))
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
    if result.returncode != 0:
        raise MleBenchPreparationError(
            "pinned MLE-bench runtime cannot import every frozen preparer"
        )


@dataclass(frozen=True)
class PinnedMleBenchPreparer:
    checkout: Path
    timeout_seconds: int = 6 * 60 * 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkout", validate_mlebench_checkout(self.checkout))
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise MleBenchPreparationError("preparer timeout must be positive")

    def prepare(
        self,
        entry: TaskRegistryEntry,
        raw: str | Path,
        public: str | Path,
        private: str | Path,
    ) -> None:
        preparer_source = (
            self.checkout
            / "mlebench"
            / "competitions"
            / entry.kaggle_slug
            / "prepare.py"
        )
        if not preparer_source.is_file():
            raise MleBenchPreparationError(
                f"pinned MLE-bench has no preparer for {entry.kaggle_slug!r}"
            )
        raw_path = Path(raw).resolve()
        public_path = Path(public).resolve()
        private_path = Path(private).resolve()
        if not raw_path.is_dir():
            raise MleBenchPreparationError("extracted raw task directory is missing")
        if public_path.exists() or private_path.exists():
            raise MleBenchPreparationError("prepared output directories must not already exist")
        environment = os.environ.copy()
        for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            environment.pop(name, None)
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(self.checkout) + (
            os.pathsep + prior_pythonpath if prior_pythonpath else ""
        )
        with tempfile.TemporaryDirectory(
            prefix=".perfseer-credential-free-kaggle-",
            dir=raw_path.parent,
        ) as credential_free_directory:
            os.chmod(credential_free_directory, 0o700)
            environment["KAGGLE_CONFIG_DIR"] = credential_free_directory
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "perfseer_v3.dataset_pack.mlebench_bridge",
                        "--invoke",
                        "--checkout",
                        str(self.checkout),
                        "--competition",
                        entry.kaggle_slug,
                        "--raw",
                        str(raw_path),
                        "--public",
                        str(public_path),
                        "--private",
                        str(private_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise MleBenchPreparationError("pinned MLE-bench preparer timed out") from error
        if result.returncode != 0:
            raise MleBenchPreparationError(
                f"pinned MLE-bench preparer failed for {entry.task_id!r}"
            )
        if not public_path.is_dir() or not any(public_path.iterdir()):
            raise MleBenchPreparationError("MLE-bench public prepared directory is empty")
        if not private_path.is_dir() or not any(private_path.iterdir()):
            raise MleBenchPreparationError("MLE-bench private prepared directory is empty")


def _invoke(
    *,
    checkout: Path,
    competition: str,
    raw: Path,
    public: Path,
    private: Path,
) -> None:
    """Child-only invocation path; intentionally contains no download call."""

    root = validate_mlebench_checkout(checkout)
    sys.path.insert(0, str(root))
    from mlebench.registry import registry

    selected = registry.get_competition(competition)
    if selected.id != competition:
        raise MleBenchPreparationError("MLE-bench registry returned the wrong competition")
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    selected.prepare_fn(raw=raw, public=public, private=private)


def _preflight(*, checkout: Path, competitions: Sequence[str]) -> None:
    """Child-only import check; it never downloads or prepares task data."""

    root = validate_mlebench_checkout(checkout)
    sys.path.insert(0, str(root))
    from mlebench.registry import registry

    if not competitions:
        raise MleBenchPreparationError("preflight requires frozen competitions")
    for competition in competitions:
        selected = registry.get_competition(competition)
        if selected.id != competition or not callable(selected.prepare_fn):
            raise MleBenchPreparationError("MLE-bench preparer import differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--invoke", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--competition", action="append", default=[])
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--private", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.preflight:
        _preflight(
            checkout=arguments.checkout,
            competitions=tuple(arguments.competition),
        )
        return 0
    if (
        len(arguments.competition) != 1
        or arguments.raw is None
        or arguments.public is None
        or arguments.private is None
    ):
        raise MleBenchPreparationError("invoke requires one competition and all paths")
    _invoke(
        checkout=arguments.checkout,
        competition=arguments.competition[0],
        raw=arguments.raw,
        public=arguments.public,
        private=arguments.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MleBenchPreparationError",
    "PinnedMleBenchPreparer",
    "validate_mlebench_checkout",
    "verify_frozen_preparers",
]
