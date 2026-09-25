#!/usr/bin/env python
"""Package NRP source models and profiler labels without large graph PKLs."""

from __future__ import annotations

import argparse
import io
import json
import socket
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXCLUDED_SUFFIXES = {".pkl", ".pt", ".pth", ".ckpt", ".pyc"}
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}
REPLAY_FILES = (
    "README.md",
    "nrp_calibration_pack/GOLDEN_DATA_GUIDE.md",
    "nrp_calibration_pack/profile/generated_model_runtime.py",
    "nrp_calibration_pack/profile/make_profile_datasets.py",
    "nrp_calibration_pack/profile/run_profile.py",
    "nrp_calibration_pack/package_source_tar.py",
    "scripts/rebuild_source_tar_dataset.py",
    "src/perfseer_source_converter/__init__.py",
    "src/perfseer_source_converter/converter.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a source-only NRP labels tarball.")
    parser.add_argument("--pack-dir", required=True, help="Generated pack directory containing models/, manifest/, and profile_datasets/.")
    parser.add_argument("--results-dir", action="append", default=[], help="Profiler output directory. May be repeated.")
    parser.add_argument("--out", required=True, help="Output .tar.gz path.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], help="Repository root for replay scripts.")
    parser.add_argument("--note", default="", help="Optional provenance note stored in package_manifest.json.")
    return parser.parse_args(argv)


def should_include(path: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in path.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if should_include(path):
            yield path


def add_tree(tar: tarfile.TarFile, root: Path, arc_root: str) -> int:
    count = 0
    for path in iter_files(root):
        tar.add(path, arcname=str(Path(arc_root) / path.relative_to(root)))
        count += 1
    return count


def add_replay_files(tar: tarfile.TarFile, repo_root: Path) -> int:
    count = 0
    for rel in REPLAY_FILES:
        path = repo_root / rel
        if path.exists() and should_include(path):
            tar.add(path, arcname=str(Path("replay") / rel))
            count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pack_dir = Path(args.pack_dir).resolve()
    result_dirs = [Path(item).resolve() for item in args.results_dir]
    repo_root = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pack_dir": str(pack_dir),
        "results_dir": [str(path) for path in result_dirs],
        "repo_root": str(repo_root),
        "note": args.note,
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "layout": {
            "pack": "Generated Python model sources, manifests, profile specs, and reports.",
            "results/<name>": "Profiler result rows, hardware metadata, and label files.",
            "replay": "Minimal repo scripts needed to regenerate PKLs or replay labels from the source package.",
        },
    }

    with tarfile.open(out, "w:gz") as tar:
        pack_count = add_tree(tar, pack_dir, "pack")
        result_counts = {}
        for result_dir in result_dirs:
            arc_name = f"results/{result_dir.name}"
            result_counts[result_dir.name] = add_tree(tar, result_dir, arc_name)
        replay_count = add_replay_files(tar, repo_root)
        manifest["pack_file_count"] = pack_count
        manifest["result_file_counts"] = result_counts
        manifest["replay_file_count"] = replay_count
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("package_manifest.json")
        info.size = len(payload)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        tar.addfile(info, fileobj=io.BytesIO(payload))

    print(f"wrote source-only package to {out}", flush=True)


if __name__ == "__main__":
    main()
