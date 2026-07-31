#!/usr/bin/env python3
"""Build the 18K local gate and execute its representatives in fresh workers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validated_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    from perfseer_v3.dataset_pack.local_runtime import LocalExecutionResult

    result: dict[str, Any] = {}
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid result JSONL line {line_number}") from error
        signature_id = row.get("validation_signature_id")
        result_sha256 = row.get("result_sha256")
        payload = {key: value for key, value in row.items() if key != "result_sha256"}
        try:
            validated = LocalExecutionResult(**payload)
            validated.validate()
        except Exception as error:
            raise ValueError(f"invalid local result line {line_number}") from error
        if (
            type(signature_id) is not str
            or len(signature_id) != 64
            or result_sha256 != validated.sha256
            or signature_id in result
        ):
            raise ValueError(f"invalid or duplicate result line {line_number}")
        result[signature_id] = validated
    return result


def _completed(path: Path) -> set[str]:
    return set(_validated_results(path))


def _failure_signature_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for line_number, text_value in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(text_value)
        payload = {key: value for key, value in row.items() if key != "failure_sha256"}
        signature_id = row.get("validation_signature_id")
        if (
            not isinstance(row, Mapping)
            or row.get("failure_sha256") != _sha256(payload)
            or type(signature_id) is not str
            or len(signature_id) != 64
        ):
            raise ValueError(f"invalid failure line {line_number}")
        result.add(signature_id)
    return result


def _finalize_gate(args: argparse.Namespace, *, partial_requested: bool) -> bool:
    from perfseer_v3.dataset_pack.fingerprints import canonical_value
    from perfseer_v3.dataset_pack.local_validation import build_local_validation_plan
    from perfseer_v3.dataset_pack.local_provenance import (
        validation_environment_sha256,
        validation_harness_sha256,
    )
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    plan = json.loads(args.plan_output.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or "plan_sha256" not in plan:
        raise ValueError("local validation plan schema differs")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != _sha256(plan_payload):
        raise ValueError("local validation plan hash differs")
    current_manifest = build_target_manifest()
    current_plan = build_local_validation_plan(current_manifest)
    current_plan_sha256 = current_plan.sha256
    if (
        plan["plan_sha256"] != current_plan_sha256
        or plan_payload != canonical_value(asdict(current_plan))
    ):
        raise ValueError("persisted local plan differs from the current source manifest/gate")
    queue = _queue_rows(args.queue)
    current_harness_sha256 = validation_harness_sha256()
    if (
        plan.get("validation_harness_sha256") != current_harness_sha256
        or any(
            row["validation_harness_sha256"] != current_harness_sha256
            for row in queue
        )
    ):
        raise ValueError("plan/worker queue targets another validation harness")
    selected = {row["signature"]["signature_id"] for row in queue}
    plan_selected = {
        row["signature_id"] for row in plan.get("representative_signatures", ())
    }
    if selected != plan_selected:
        raise ValueError("worker queue differs from selected plan signatures")
    result_records = _validated_results(args.results)
    passed = set(result_records)
    if not passed <= selected:
        raise ValueError("results contain a signature outside the current plan")
    queue_by_signature = {
        row["signature"]["signature_id"]: row for row in queue
    }
    if any(
        result_records[signature_id].representative_configuration_id
        != queue_by_signature[signature_id]["candidate"]["candidate_id"]
        for signature_id in passed
    ):
        raise ValueError("result representative differs from the current worker queue")
    current_environment_sha256 = validation_environment_sha256(
        args.device, args.compile_backend
    )
    if any(
        result.validation_harness_sha256 != current_harness_sha256
        or result.validation_environment_sha256 != current_environment_sha256
        or result.compile_backend != args.compile_backend
        for result in result_records.values()
    ):
        raise ValueError("results target another validation harness/environment")
    unresolved_failures = _failure_signature_ids(args.failures) - passed
    mappings = plan.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != 18_000:
        raise ValueError("local plan does not contain all 18K mappings")
    mappings_covered = all(
        set(row.get("validation_evidence_signature_ids", ())) <= passed
        for row in mappings
    )
    missing = selected - passed
    complete = not missing and not unresolved_failures and mappings_covered
    gate_passed = complete and not partial_requested
    summary = {
        "version": "perfseer_v3_local_gate_summary_v1",
        "plan_sha256": plan["plan_sha256"],
        "validation_harness_sha256": current_harness_sha256,
        "validation_environment_sha256": current_environment_sha256,
        "selected_signature_count": len(selected),
        "passing_signature_count": len(passed),
        "missing_signature_ids": sorted(missing),
        "unresolved_failure_signature_ids": sorted(unresolved_failures),
        "mapped_configuration_count": len(mappings),
        "all_mappings_covered_by_passing_results": mappings_covered,
        "partial_requested": partial_requested,
        "complete": complete,
        "gate_passed": gate_passed,
        "accepted_a10g_measurement": False,
    }
    summary["summary_sha256"] = _sha256(summary)
    _atomic_json(args.summary, summary)
    print(
        json.dumps(
            {
                "plan_sha256": summary["plan_sha256"],
                "passing_signatures": summary["passing_signature_count"],
                "selected_signatures": summary["selected_signature_count"],
                "unresolved_failures": len(unresolved_failures),
                "all_mappings_covered": mappings_covered,
                "partial_requested": partial_requested,
                "gate_passed": gate_passed,
                "summary": str(args.summary),
                "summary_sha256": summary["summary_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return gate_passed


def _queue_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"local validation worker queue is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(text)
        if not isinstance(raw, Mapping) or set(raw) != {
            "candidate",
            "signature",
            "validation_harness_sha256",
            "payload_sha256",
        }:
            raise ValueError(f"worker queue schema differs at line {line_number}")
        payload = {
            "candidate": raw["candidate"],
            "signature": raw["signature"],
            "validation_harness_sha256": raw["validation_harness_sha256"],
        }
        signature_id = raw["signature"].get("signature_id")
        if (
            raw["payload_sha256"] != _sha256(payload)
            or type(signature_id) is not str
            or len(signature_id) != 64
            or signature_id in identities
        ):
            raise ValueError(f"worker queue identity/hash differs at line {line_number}")
        identities.add(signature_id)
        rows.append(raw)
    if not rows:
        raise ValueError("local validation worker queue is empty")
    return rows


def _worker(args: argparse.Namespace) -> int:
    from perfseer_v3.dataset_pack.fingerprints import canonical_value
    from perfseer_v3.dataset_pack.local_runtime import validate_candidate_execution
    from perfseer_v3.dataset_pack.local_validation import validation_signature_from_dict
    from perfseer_v3.dataset_pack.local_provenance import validation_harness_sha256
    from perfseer_v3.dataset_pack.sampler import target_candidate_from_dict

    raw = json.load(sys.stdin)
    if not isinstance(raw, Mapping) or set(raw) != {
        "candidate",
        "signature",
        "validation_harness_sha256",
    }:
        raise ValueError("worker input schema differs")
    if raw["validation_harness_sha256"] != validation_harness_sha256():
        raise ValueError("worker input targets another validation harness")
    candidate = target_candidate_from_dict(raw["candidate"])
    signature = validation_signature_from_dict(raw["signature"])
    result = validate_candidate_execution(
        candidate,
        signature,
        device=args.device,
        compile_backend=args.compile_backend,
    )
    payload = canonical_value(asdict(result))
    payload["result_sha256"] = result.sha256
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _build_plan_queue(args: argparse.Namespace) -> int:
    from perfseer_v3.dataset_pack.local_validation import build_local_validation_plan
    from perfseer_v3.dataset_pack.sampler import build_target_manifest

    manifest = build_target_manifest()
    plan = build_local_validation_plan(manifest)
    plan.validate(manifest)
    plan_sha256 = plan.sha256
    harness_sha256 = plan.validation_harness_sha256
    gc.collect()
    plan_payload = asdict(plan)
    plan_payload["plan_sha256"] = plan_sha256
    _atomic_json(args.plan_output, plan_payload)
    candidates = {row.candidate_id: row for row in manifest.candidates}

    def queue_values() -> Any:
        for signature in plan.representative_signatures:
            candidate = candidates[signature.representative_candidate_id]
            payload = {
                "candidate": candidate.to_dict(),
                "signature": signature.to_dict(),
                "validation_harness_sha256": harness_sha256,
            }
            yield {**payload, "payload_sha256": _sha256(payload)}

    _atomic_jsonl(args.queue, queue_values())
    print(
        json.dumps(
            {
                "manifest_sha256": manifest._sha256_unchecked(),
                "plan_sha256": plan_sha256,
                "static_rows": plan.static_row_count,
                "representatives": len(plan.representative_signatures),
                "accepted_a10g_measurement": False,
                "plan_output": str(args.plan_output),
                "worker_queue": str(args.queue),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-output", type=Path, default=ROOT / "reports" / "a10g_18k_local_validation_plan.json")
    parser.add_argument("--queue", type=Path, default=ROOT / "reports" / "a10g_18k_local_validation_workers.jsonl")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "a10g_18k_local_execution_results.jsonl")
    parser.add_argument("--failures", type=Path, default=ROOT / "reports" / "a10g_18k_local_execution_failures.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports" / "a10g_18k_local_gate_summary.json")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--worker-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-plan-queue", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker_json:
        return _worker(args)
    if args.build_plan_queue:
        return _build_plan_queue(args)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.verify_only:
        return 0 if _finalize_gate(args, partial_requested=False) else 1
    if not args.resume:
        if args.results.exists():
            raise FileExistsError(f"result file exists; pass --resume: {args.results}")
        planner = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--build-plan-queue",
            "--plan-output",
            str(args.plan_output),
            "--queue",
            str(args.queue),
        ]
        completed_plan = subprocess.run(
            planner,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC)},
            check=False,
        )
        if completed_plan.returncode != 0:
            return completed_plan.returncode
    elif not args.plan_output.is_file():
        raise FileNotFoundError(f"resume plan is missing: {args.plan_output}")
    if args.plan_only:
        return 0
    completed = _completed(args.results) if args.resume else set()
    pending = [
        row
        for row in _queue_rows(args.queue)
        if row["signature"]["signature_id"] not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    encountered_failure = False
    for ordinal, row in enumerate(pending, 1):
        candidate = row["candidate"]
        signature = row["signature"]
        payload = {
            "candidate": candidate,
            "signature": signature,
            "validation_harness_sha256": row["validation_harness_sha256"],
        }
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-json",
            "--device",
            args.device,
            "--compile-backend",
            args.compile_backend,
        ]
        print(
            f"[{ordinal}/{len(pending)}] {candidate['family_id']} {candidate['task_id']} "
            f"{candidate['optimizer']['name']} {candidate['scheduler']['name']} "
            f"{signature['signature_id'][:12]}",
            flush=True,
        )
        try:
            completed_process = subprocess.run(
                command,
                input=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                check=False,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(SRC),
                    "TORCHINDUCTOR_CACHE_DIR": os.environ.get(
                        "TORCHINDUCTOR_CACHE_DIR",
                        "/tmp/perfseer_v3_local_inductor",
                    ),
                },
            )
            if completed_process.returncode != 0:
                raise RuntimeError(
                    f"worker exited {completed_process.returncode}: "
                    f"{completed_process.stderr[-4000:]}"
                )
            rows = completed_process.stdout.strip().splitlines()
            if len(rows) != 1:
                raise RuntimeError("worker did not emit exactly one result record")
            result = json.loads(rows[0])
            if (
                result.get("validation_signature_id") != signature["signature_id"]
                or result.get("representative_configuration_id") != candidate["candidate_id"]
                or result.get("accepted_a10g_measurement") is not False
            ):
                raise RuntimeError("worker result identity/scope mismatch")
            if args.device.startswith("cuda") and result.get("observed_hardware_id") != "NVIDIA GeForce RTX 5090":
                raise RuntimeError("local CUDA gate did not run on the RTX 5090")
            _append_jsonl(args.results, result)
        except Exception as error:
            failure = {
                "validation_signature_id": signature["signature_id"],
                "representative_configuration_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "task_id": candidate["task_id"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failure["failure_sha256"] = _sha256(failure)
            _append_jsonl(args.failures, failure)
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
            encountered_failure = True
            if not args.keep_going:
                break
    gate_passed = _finalize_gate(
        args,
        partial_requested=args.limit is not None,
    )
    if encountered_failure:
        return 1
    if args.limit is not None:
        return 0
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
