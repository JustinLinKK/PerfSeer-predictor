#!/usr/bin/env python3
"""Aggregate integrity-checked target worker attempts into a training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.label_worker import WORKER_ENVELOPE_VERSION
from perfseer_v3.dataset_pack.transfer_labeling import (
    TransferLabelAttemptV3,
    build_target_training_manifest,
)


def _attempt_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if value.get("version") != WORKER_ENVELOPE_VERSION:
        return value
    raw = dict(value)
    declared = str(raw.pop("envelope_sha256", ""))
    if declared != canonical_sha256(raw):
        raise ValueError("target worker envelope content hash mismatch")
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or "attempt_version" not in payload:
        raise ValueError("worker envelope does not contain a transfer attempt")
    return payload


def _read_attempts(path: Path) -> list[TransferLabelAttemptV3]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values: list[Any] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    attempts = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} contains a non-object attempt")
        attempts.append(TransferLabelAttemptV3.from_dict(_attempt_payload(value)))
    return attempts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    subset = json.loads(args.subset.read_text(encoding="utf-8"))
    if not isinstance(subset, Mapping):
        raise ValueError("transfer subset root must be an object")
    attempts = [attempt for path in args.attempt for attempt in _read_attempts(path)]
    manifest = build_target_training_manifest(subset, attempts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "attempt_count": len(attempts),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
