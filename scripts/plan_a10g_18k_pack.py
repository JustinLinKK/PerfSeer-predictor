#!/usr/bin/env python3
"""Build the deterministic source-only AWS A10G candidate planning bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Any

from perfseer_v3.dataset_pack.fingerprints import canonical_value, file_sha256
from perfseer_v3.dataset_pack.planning_bundle import (
    build_dataset_planning_bundle,
)


def _atomic_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(canonical_value(row), sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_lines(path, (payload,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write compact JSONL manifests and a summary under this directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and print identities without writing files",
    )
    args = parser.parse_args()
    if args.output_dir is None and not args.check:
        parser.error("either --output-dir or --check is required")

    bundle = build_dataset_planning_bundle()
    summary = dict(bundle.to_summary())
    summary["evidence_scope"] = "source_only_unmeasured_aws_targets"
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        target_path = output / "a10g_18k_target_manifest.jsonl"
        _atomic_lines(
            target_path, (asdict(row) for row in bundle.target_manifest.candidates)
        )
        summary["files"] = {
            path.name: {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                target_path,
            )
        }
        summary_path = output / "a10g_planning_summary.json"
        _atomic_json(summary_path, summary)
        summary["summary_path"] = str(summary_path)
    print(json.dumps(canonical_value(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
