#!/usr/bin/env python3
"""Evaluate v3 prediction JSON/JSONL with the six-target and OOM contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import canonical_json
from perfseer_v3.evaluation import (
    AblationResult,
    PredictionRecord,
    evaluate_oom_calibration,
    evaluate_predictions,
    evaluate_slices,
    validate_ablation_matrix,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("prediction input must be a JSON list/object or JSONL rows")
    return payload


def _record(row: dict[str, Any]) -> PredictionRecord:
    raw = dict(row)
    raw["prediction"] = tuple(float(value) for value in raw["prediction"])
    raw["target"] = tuple(float(value) for value in raw["target"])
    if raw.get("log_variance") is not None:
        raw["log_variance"] = tuple(float(value) for value in raw["log_variance"])
    return PredictionRecord(**raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablations", type=Path)
    parser.add_argument("--near-zero-epsilon", type=float, default=1e-6)
    parser.add_argument("--oom-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.near_zero_epsilon <= 0:
        parser.error("near-zero-epsilon must be positive")
    records = [_record(row) for row in _rows(args.predictions)]
    ablations = None
    if args.ablations is not None:
        raw_ablations = json.loads(args.ablations.read_text(encoding="utf-8"))
        if isinstance(raw_ablations, dict):
            raw_ablations = raw_ablations.get("ablations")
        ablations = [AblationResult(**row) for row in raw_ablations]
        validate_ablation_matrix(ablations)
    report: dict[str, Any] = {
        "report_version": "perfseer_v3_prediction_evaluation_v1",
        "prediction_source": str(args.predictions.resolve()),
        "prediction_source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "record_count": len(records),
        "near_zero_policy": {
            "metric": "MAPE excludes abs(target) below epsilon",
            "epsilon": args.near_zero_epsilon,
        },
        "overall": {
            name: asdict(metrics)
            for name, metrics in evaluate_predictions(
                records, near_zero_epsilon=args.near_zero_epsilon
            ).items()
        },
        "slices": {
            field: {
                value: {
                    target: asdict(metrics)
                    for target, metrics in target_metrics.items()
                }
                for value, target_metrics in groups.items()
            }
            for field, groups in evaluate_slices(
                records, near_zero_epsilon=args.near_zero_epsilon
            ).items()
        },
        "oom": evaluate_oom_calibration(records, threshold=args.oom_threshold),
        "ablations": None if ablations is None else [asdict(row) for row in ablations],
        "acceptance_gates_evaluated": False,
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json(report).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"records={len(records)} oom={report['oom']['available']} "
        f"sha256={report['report_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
