#!/usr/bin/env python
"""Validate generated model structures before large label jobs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perfseer.architecture_schema import ARCHITECTURE_FAMILIES, VARIANT_KINDS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and smoke-test representative generated model structures.")
    parser.add_argument("--output-dir", help="Defaults to record/generated_model_structure_validation_<timestamp>.")
    parser.add_argument("--subset-size", type=int, default=75)
    parser.add_argument("--full", action="store_true", help="Run the heavy 10,005-base-model verifier.")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--generation-workers", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-if-available", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(cmd: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": time.time() - start,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def require_ok(status: dict[str, Any], phase: str) -> None:
    if int(status.get("returncode", 1)) != 0:
        message = [f"{phase} failed with exit code {status.get('returncode')}"]
        if status.get("stdout_tail"):
            message.extend(["stdout tail:", str(status["stdout_tail"])])
        if status.get("stderr_tail"):
            message.extend(["stderr tail:", str(status["stderr_tail"])])
        raise SystemExit("\n".join(message))


def verify_manifest_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = {str(row.get("architecture_family", "unknown")) for row in rows}
    variants = {str(row.get("variant_kind", "unknown")) for row in rows}
    missing_families = sorted(set(ARCHITECTURE_FAMILIES) - families)
    missing_variants = sorted(set(VARIANT_KINDS) - variants)
    if missing_families or missing_variants:
        raise SystemExit(
            "coverage verifier failed: "
            f"missing_families={missing_families} missing_variant_kinds={missing_variants}"
        )
    return {
        "rows": len(rows),
        "families": sorted(families),
        "variant_kinds": sorted(variants),
    }


def smoke_profile(
    python: str,
    pack_dir: Path,
    output_dir: Path,
    device: str,
    manifest_by_model_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cmd = [
        python,
        str(ROOT / "nrp_calibration_pack" / "profile" / "run_profile.py"),
        "--manifest",
        str(pack_dir / "manifest" / "subset_manifest.jsonl"),
        "--models-dir",
        str(pack_dir / "models"),
        "--output-dir",
        str(output_dir),
        "--hardware-id",
        device,
        "--device",
        device,
        "--warmup",
        "0",
        "--infer-repeats",
        "1",
        "--train-repeats",
        "1",
        "--sm-occupancy-source",
        "nvml_proxy",
        "--resource-profile-mode",
        "sustained",
        "--label-time-mode",
        "step_extrapolated",
        "--no-resume",
    ]
    status = run(cmd)
    require_ok(status, f"{device} smoke profile")
    rows = load_jsonl(output_dir / "results_shard0.jsonl")
    failed = []
    for row in rows:
        if row.get("status") == "ok":
            continue
        manifest = manifest_by_model_id.get(str(row.get("model_id")), {})
        failed.append(
            {
                "profile_point_id": row.get("profile_point_id"),
                "model_id": row.get("model_id"),
                "architecture_family": manifest.get("architecture_family"),
                "variant_kind": manifest.get("variant_kind"),
                "precision_config": row.get("precision_config"),
                "status": row.get("status"),
                "error": row.get("error"),
            }
        )
    if failed:
        raise SystemExit(f"{device} smoke profile produced non-ok rows:\n{json.dumps(failed, indent=2, sort_keys=True)}")
    status["profiled_rows"] = len(rows)
    return status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subset_size = 10005 if args.full else args.subset_size
    if subset_size <= 0:
        raise SystemExit("--subset-size must be > 0")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "record" / f"generated_model_structure_validation_{time.strftime('%Y%m%d_%H%M%S')}"
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = output_dir / "pack"
    cmd = [
        args.python,
        str(ROOT / "nrp_calibration_pack" / "generate_model_sources.py"),
        "--out-dir",
        str(pack_dir),
        "--catalog-mode",
        "template",
        "--subset-size",
        str(subset_size),
        "--seed",
        str(args.seed),
        "--precision-sweep",
        "fp32_ieee",
        "--validation-mode",
        "construct",
        "--generation-workers",
        str(args.generation_workers),
        "--force",
    ]
    generate_status = run(cmd)
    require_ok(generate_status, "generate representative pack")
    manifest_rows = load_jsonl(pack_dir / "manifest" / "subset_manifest.jsonl")
    manifest_by_model_id = {str(row.get("model_id")): row for row in manifest_rows}
    coverage = verify_manifest_coverage(manifest_rows)
    cpu_status = smoke_profile(args.python, pack_dir, output_dir / "cpu_profile", "cpu", manifest_by_model_id)
    cuda_status: dict[str, Any] | None = None
    if args.cuda_if_available:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            cuda_status = smoke_profile(args.python, pack_dir, output_dir / "cuda_profile", "cuda", manifest_by_model_id)
        else:
            cuda_status = {"skipped": True, "reason": "cuda unavailable"}
    summary = {
        "subset_size": subset_size,
        "full": bool(args.full),
        "coverage": coverage,
        "generate": generate_status,
        "cpu_profile": cpu_status,
        "cuda_profile": cuda_status,
        "passed": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Generated Model Structure Validation",
                "",
                f"- Subset size: {subset_size}",
                f"- Manifest rows: {coverage['rows']}",
                f"- Families covered: {len(coverage['families'])}/{len(ARCHITECTURE_FAMILIES)}",
                f"- Variant kinds covered: {len(coverage['variant_kinds'])}/{len(VARIANT_KINDS)}",
                f"- CPU profiled rows: {cpu_status.get('profiled_rows')}",
                f"- CUDA profile: {('skipped' if cuda_status and cuda_status.get('skipped') else 'not requested' if cuda_status is None else str(cuda_status.get('profiled_rows')) + ' rows')}",
                "- Passed: True",
            ]
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "passed": True, "rows": coverage["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
