"""Generate the reproducible v2 baseline and v3 export coverage smoke report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.baseline import collect_v2_baseline, write_baseline
from perfseer_v3.coverage import audit_corpus, write_coverage_reports
from perfseer_v3.coverage_corpus import (
    frontier_source_cases,
    p0_cases,
    representative_source_cases,
    smoke_cases,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--corpus",
        choices=("smoke", "p0", "supported", "frontier"),
        default="supported",
    )
    parser.add_argument(
        "--gpu-time-json",
        type=Path,
        help="JSON operation->milliseconds mapping, or a report containing profiler_time_by_operation",
    )
    args = parser.parse_args(argv)

    if not args.skip_baseline:
        snapshot = collect_v2_baseline(args.repo_root)
        write_baseline(snapshot, args.output_dir / "v2_baseline.json")
    gpu_times = None
    if args.gpu_time_json is not None:
        payload = json.loads(args.gpu_time_json.read_text(encoding="utf-8"))
        if "profiler_time_by_operation" in payload:
            payload = payload["profiler_time_by_operation"]
        if not isinstance(payload, dict):
            raise ValueError("GPU-time JSON must contain an operation-to-time mapping")
        gpu_times = {str(key): float(value) for key, value in payload.items()}
    cases_by_name = {
        "smoke": smoke_cases,
        "p0": p0_cases,
        "supported": lambda: (*p0_cases(), *representative_source_cases()),
        "frontier": lambda: (
            *p0_cases(),
            *representative_source_cases(),
            *frontier_source_cases(),
        ),
    }
    cases = cases_by_name[args.corpus]()
    report, failures = audit_corpus(cases, gpu_time_by_operation=gpu_times)
    write_coverage_reports(report, failures, args.output_dir)
    print(
        f"strict_export={report['strict_export_success_rate']:.1%} "
        f"complete={report['complete_graph_success_rate']:.1%} "
        f"tensor_nodes={report['encoded_tensor_nodes']}/{report['tensor_nodes']} "
        f"failures={len(failures)}"
    )
    return 0 if report["complete_graph_success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
