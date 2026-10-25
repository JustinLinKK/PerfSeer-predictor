"""CPU-only preparation, validation selection, and sealed-test calibration CLI."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .calibration import feature_values, load_adapter, source_identity
from .calibration_contracts import (CONTRACT_VERSION, ENVIRONMENT_VERSION, MIB, TARGET_SCHEMA, audit_records,
                                    environment_identity, measurement, workload_identity)
from .calibration_evaluation import evaluate_test, promotion_gate, select_experiments
from .capture import graphs_from_design
from .inference import predict, predict_calibrated
from .memory_baseline import graph_baseline
from .version import TARGET_NAMES

REVIEWED_COMMIT = "747ccdd936362133c4b91a70b282b11d8a298a2f"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "predictor_v3.2/weights/predictor_v3.2_student.pt"
DEFAULT_CAMPAIGN = ROOT / "record/perfseer-v32/transfer-rtx5090"
DEFAULT_DATASET = ROOT / "record/perfseer-v32/transfer-training-rtx5090/dataset"


def checked_path(root, relative):
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("artifact path escapes its root")
    return path


def write_new(path, value, *, compress=False):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"refusing to replace an existing experiment artifact: {path}")
        return
    atomic_write(path, value, compress=compress)
    if read_json(path) != value:
        raise ValueError("artifact write verification failed")


def provenance(source, repository=ROOT):
    repository, source = Path(repository), Path(source)
    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, text=True).stdout.strip()
    reviewed_paths = ("records/2026-09-20-a100-v32-transfer.md", "records/2026-09-24-aba-transfer-model-local-package.md")
    records = {path: git("show", f"{REVIEWED_COMMIT}:{path}") for path in reviewed_paths}
    missing = [name for name in (
        "predictor_v3.2/weights/predictor_v3.2_transfer_student_justin_batch1_20260922.pt",
        "predictor_v3.2/weights/teacher_v3.2_transfer_justin_batch1_20260922.pt",
        "predictor_v3.2/weights/transfer-student-final-report-20260922.json",
        "predictor_v3.2/weights/transfer-teacher-final-report-20260922.json",
    ) if not (repository / name).exists()]
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    teacher = repository / "predictor_v3.2/teacher_model/teacher_v3.2_epoch26.pt"
    return {"version": CONTRACT_VERSION, "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "reviewed_commit": REVIEWED_COMMIT, "reviewed_records": records,
            "package_import": str(Path(__file__).parent), "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": file_sha256(source), "source_identity": source_identity(artifact),
            "source_role": artifact["role"], "teacher_checkpoint": str(teacher),
            "teacher_checkpoint_sha256": file_sha256(teacher) if teacher.exists() else None,
            "missing_reviewed_artifacts": missing,
            "additional_reproducibility_gaps": ["teammate failing-run command, exact target labels and query predictions",
                                                "/data1/yufan/perfseer_v32_a100_bs_training_20260908/ready_for_train on ABA"],
            "exact_reviewed_failure_reproduced": False, "source_weights_modified": False,
            "baseline_command": "python scripts/run_perfseer_v32_calibration.py prepare --output record/time-memory-calibration/data"}


def campaign_environment(execution, hardware_id):
    return {"version": ENVIRONMENT_VERSION, "hardware_id": hardware_id, "gpu_model": execution["gpu_name"],
            "capacity_bytes": execution["memory_total_bytes"], "partition": "unknown_not_recorded",
            "driver": execution["driver"], "cuda": execution["cuda"], "framework": execution["torch"],
            "libraries": {"cudnn": execution["cudnn"], "python": execution["python"]},
            "allocator": "unknown_not_recorded", "backend_policy": "cuda_eager_with_declared_precision_controls",
            "host_context": {key: execution[key] for key in ("platform", "cpu_threads", "omp_threads", "mkl_threads")}}


def _write_splits(output, records):
    audit = audit_records(records)
    split_files = {}
    for split in sorted({row["split"] for row in records}):
        path = Path(output) / f"{split}.json.gz"
        write_new(path, sorted((row for row in records if row["split"] == split), key=lambda row: row["sample_id"]), compress=True)
        split_files[split] = {"path": path.name, "sha256": file_sha256(path)}
    return audit, split_files


def _prepare_shard(arguments):
    torch.set_num_threads(1)
    source, campaign, dataset, output, limit, shard = arguments
    return prepare(source, campaign, dataset, output, limit_per_split=limit, shard=shard)


def prepare(source, campaign, dataset, output, *, limit_per_split=None, workers=1, shard=None):
    """Recover repeat means from verified native attempts without rewriting labels."""
    from .transfer_labeling import resolve_attempt_execution
    from .transfer_verification import verify_attempt

    source, campaign, dataset, output = map(Path, (source, campaign, dataset, output))
    if output.resolve() in {campaign.resolve(), dataset.resolve()}:
        raise ValueError("calibration output must be separate from native datasets")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("preparation limit must be positive")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("preparation workers must be between one and eight")
    if limit_per_split is not None and limit_per_split < workers:
        raise ValueError("preparation limit must be at least the worker count")
    if workers > 1:
        arguments = [(source, campaign, dataset, output / "shards" / str(index), limit_per_split, (index, workers))
                     for index in range(workers)]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            manifests = list(pool.map(_prepare_shard, arguments))
        records = [row for arguments in arguments for split in ("train", "validation", "test")
                   for row in load_split(arguments[3], split)]
        audit, splits = _write_splits(output, records)
        result = {**manifests[0], "audit": audit, "split_files": splits,
                  "raw_attempt_hashes": {key: value for item in manifests for key, value in item["raw_attempt_hashes"].items()},
                  "repeatability": [row for item in manifests for row in item["repeatability"]],
                  "cpu_prediction_seconds": sum(item["cpu_prediction_seconds"] for item in manifests),
                  "shards": [{"path": str(arguments[3].relative_to(output)), "fingerprint": item["fingerprint"]}
                             for arguments, item in zip(arguments, manifests, strict=True)]}
        result["fingerprint"] = fingerprint({key: value for key, value in result.items() if key != "fingerprint"})
        write_new(output / "manifest.json", result)
        return result
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    identity = source_identity(artifact)
    labels_path = campaign / "labels-24h.json.gz"
    labels = {row["profile_point_id"]: row for row in read_json(labels_path)}
    configurations = {row["configuration_id"]: row for row in read_json(campaign / "configurations.json.gz")}
    anchors = {row["anchor_id"]: row for row in read_json(campaign / "anchors.json.gz")}
    execution, manifest = (read_json(campaign / name) for name in ("execution.json", "manifest.json"))
    prepared = read_json(dataset / "dataset_manifest.json")
    if prepared["source_labels_sha256"] != file_sha256(labels_path) or prepared["fingerprint"] != fingerprint({key: value for key, value in prepared.items() if key != "fingerprint"}):
        raise ValueError("prepared dataset or native labels fingerprint differs")
    records, timings, attempt_hashes, repeat_rows = [], [], {}, []
    for split, meta in sorted(prepared["split_files"].items()):
        path = checked_path(dataset, meta["path"])
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("native split file hash differs")
        split_rows = read_json(path)
        if len(split_rows) != meta["rows"]:
            raise ValueError("native split row count differs")
        if limit_per_split:
            split_rows = sorted(split_rows, key=lambda row: fingerprint(row["sample_id"]))[:limit_per_split]
        if shard:
            split_rows = split_rows[shard[0]::shard[1]]
        for start in range(0, len(split_rows), 32):
            items, designs = split_rows[start:start + 32], []
            for item in items:
                design_path = checked_path(dataset, item["input_path"])
                if file_sha256(design_path) != item["input_sha256"]:
                    raise ValueError("paired graph hash differs")
                designs.append(read_json(design_path))
            before = time.perf_counter()
            predictions = predict(artifact, designs, amp=False, microbatch=1)
            timings.append({"rows": len(items), "cpu_prediction_seconds": time.perf_counter() - before})
            for item, design, prediction in zip(items, designs, predictions, strict=True):
                native = labels[item["sample_id"]]
                configuration = configurations[item["sample_id"]]
                anchor = anchors[native["anchor_id"]]
                if (split != native["split"] or configuration["split"] != split or item["group_id"] != native["group_id"] or
                        item["targets"] != {name: native["targets"][name] for name in TARGET_NAMES} or
                        item["targets"] != {name: item["native_targets"][name] for name in TARGET_NAMES}):
                    raise ValueError("prepared labels differ from independent native observations")
                graph = graphs_from_design(design)["training"]
                if graph.training_config != native["training"] or design["training"] != read_json(checked_path(campaign, native["graph_path"])):
                    raise ValueError("executed workload configuration differs from paired graph")
                attempts, run_ids, domains = [], [], []
                count = len(native["repeat_quality"]["train_step_wall_ms"]["values"])
                for repetition in range(count):
                    attempt_path = campaign / "attempts" / f"{item['sample_id']}.{repetition}.json.gz"
                    attempt = read_json(attempt_path)
                    verify_attempt(attempt, configuration, manifest, execution, campaign)
                    resolved = resolve_attempt_execution(campaign, execution, attempt["execution_fingerprint"])
                    domains.append(campaign_environment(resolved, native["hardware_id"]))
                    attempts.append(attempt)
                    run_ids.append(f"{manifest['fingerprint']}:{item['sample_id']}:{repetition}")
                    attempt_hashes[str(attempt_path.relative_to(campaign))] = file_sha256(attempt_path)
                if not domains or any(domain != domains[0] for domain in domains):
                    raise ValueError("repeat runs have different environments")
                observations = {}
                for target, schema in TARGET_SCHEMA.items():
                    name = schema["api_name"]
                    values = [attempt["targets"][name] for attempt in attempts]
                    if values != native["repeat_quality"][name]["values"] or attempts[0]["targets"][name] != native["targets"][name]:
                        raise ValueError("exported repeat values differ from raw measurements")
                    if target == "memory_bytes":
                        values = [value * MIB for value in values]
                    protocol = {"warmup": "one_epoch_excluded_with_optimizer_initialization",
                                "boundaries": "synchronize_at_epoch_window_end; NVML_10ms_samples_include_warm_window_boundary",
                                "step_unit": "logical_optimizer_step", "data_loading": "local_subset_tensor_generation_and_transfer_included",
                                "source": "verified_raw_attempts", "cuda_event_diagnostic": "elapsed_stream_interval_not_active_kernel_sum"}
                    observations[target] = measurement(values, run_ids, target=target, protocol=protocol)
                for attempt in attempts:
                    if not math.isclose(attempt["targets"]["train_epoch_ms"], attempt["targets"]["train_step_wall_ms"] * native["training"]["steps_per_epoch"], rel_tol=1e-10):
                        raise ValueError("epoch is not the declared count times mean logical step")
                inputs = [{"shape": list(edge.shape), "dtype": edge.dtype, "source_name": edge.source_name}
                          for edge in graph.tensor_edges if edge.tensor_role == "model_input"]
                source_hash = fingerprint(anchor["model"])
                workload = workload_identity(source_hash, inputs, native["training"], anchor["identity"])
                baseline = graph_baseline(design)
                features = feature_values(design, prediction)
                features["memory_bytes"]["log_analytic_bytes"] = math.log1p(baseline["peak_bytes"])
                record = {"version": CONTRACT_VERSION, "sample_id": item["sample_id"], "split": split,
                          "group_id": native["group_id"], "anchor_id": native["anchor_id"], "workload_id": workload,
                          "source_sha256": source_hash, "source_identity": identity, "environment": domains[0],
                          "domain_fingerprint": environment_identity(domains[0]), "measurements": observations,
                          "source_prediction": prediction, "features": features, "memory_baseline": baseline,
                          "family": anchor["modality"], "precision": native["training"]["precision"],
                          "batch": native["training"]["microbatch_size"], "training": native["training"], "executed_inputs": inputs,
                          "source_panel": "unknown_source_training_membership", "paired_source_ids": native["paired_source_ids"],
                          "design_path": str((dataset / item["input_path"]).resolve()), "design_sha256": item["input_sha256"],
                          "status": "ok"}
                records.append(record)
                if count > 1:
                    repeat_rows.append({"sample_id": record["sample_id"], "group_id": record["group_id"], "family": record["family"],
                                        "precision": record["precision"], "batch": record["batch"], "run_count": count,
                                        "time_cv": observations["time_ms"]["std"] / observations["time_ms"]["value"],
                                        "memory_std_mib": observations["memory_bytes"]["std"] / MIB,
                                        "peak_vram_mib": observations["memory_bytes"]["value"] / MIB})
            print(f"prepared {split}: {min(start + 32, len(split_rows))}/{len(split_rows)}", flush=True)
    audit, split_files = _write_splits(output, records)
    result = {"version": CONTRACT_VERSION, "source_checkpoint_sha256": file_sha256(source), "source_identity": identity,
              "labels_sha256": file_sha256(labels_path), "paired_dataset_fingerprint": prepared["fingerprint"],
              "split_files": split_files, "audit": audit, "target_schema": TARGET_SCHEMA,
              "raw_attempt_hashes": attempt_hashes, "repeatability": repeat_rows,
              "cpu_prediction_seconds": sum(item["cpu_prediction_seconds"] for item in timings),
              "new_gpu_runs": 0, "limit_per_split": limit_per_split,
              "limitations": ["NVML is whole-device sampled usage, not process or allocator peak",
                              "allocator and partition were not recorded in the native campaign",
                              "source training membership not verified for either evaluation panel",
                              "static memory baseline is approximate; empirical trace gate unverified"]}
    result["fingerprint"] = fingerprint(result)
    write_new(output / "manifest.json", result)
    return result


def load_split(root, split):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest["fingerprint"] != fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}):
        raise ValueError("calibration dataset manifest differs")
    meta = manifest["split_files"][split]
    path = checked_path(root, meta["path"])
    if file_sha256(path) != meta["sha256"]:
        raise ValueError("calibration split hash differs")
    rows = read_json(path)
    if any(row["split"] != split for row in rows):
        raise ValueError("calibration split membership differs")
    audit_records(rows)
    return rows


def benchmark(source, adapter_path, design_path, *, repeats=20, intended_host=False):
    if repeats < 5:
        raise ValueError("benchmark needs at least five warm queries")
    start = time.perf_counter()
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    adapter = load_adapter(adapter_path)
    load_ms = (time.perf_counter() - start) * 1000
    design = read_json(design_path)
    start = time.perf_counter()
    predict_calibrated(artifact, [design], adapter, adapter["environment"])
    cold_ms = (time.perf_counter() - start) * 1000
    samples = {"source": [], "candidate": []}
    for index in range(repeats):
        for name in ("source", "candidate") if index % 2 == 0 else ("candidate", "source"):
            start = time.perf_counter()
            if name == "source":
                predict(artifact, [design], amp=False, microbatch=1)
            else:
                predict_calibrated(artifact, [design], adapter, adapter["environment"])
            samples[name].append((time.perf_counter() - start) * 1000)
    return {"host": platform.platform(), "intended_host_verified": intended_host, "cpu_threads": torch.get_num_threads(),
            "adapter_fingerprint": adapter["fingerprint"], "design_sha256": file_sha256(design_path),
            "checkpoint_load_ms": load_ms, "cold_query_ms": cold_ms, "repeats": repeats, "concurrency": 1,
            "source_warm_p95_ms": float(np.quantile(samples["source"], .95)),
            "candidate_warm_p95_ms": float(np.quantile(samples["candidate"], .95)),
            "source_artifact_bytes": Path(source).stat().st_size, "adapter_bytes": Path(adapter_path).stat().st_size,
            "graph_extraction_ms": None, "graph_extraction_status": "precomputed_graph; benchmark_capture_separately",
            "scope": "public_API_including_model_restore_and_binding_checks"}


def package(source, selection_path, report_path, output, *, budget, seed):
    """Package an adapter only after its sealed held-out promotion gate passed."""
    selection, report = read_json(selection_path), read_json(report_path)
    if selection["fingerprint"] != fingerprint({key: value for key, value in selection.items() if key != "fingerprint"}):
        raise ValueError("selection fingerprint differs")
    if report["selection_fingerprint"] != selection["fingerprint"]:
        raise ValueError("promotion report does not match the sealed selection")
    if report.get("stage") != "final_test" or report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"}):
        raise ValueError("test report fingerprint differs")
    trials = [trial for trial in selection["trials"] if (trial["budget"], trial["seed"]) == (budget, seed)]
    results = [trial for trial in report["trials"] if (trial["budget"], trial["seed"]) == (budget, seed)]
    if len(trials) != 1 or len(results) != 1 or results[0]["promotion"]["status"] != "passed":
        raise ValueError("deployment requires a unique candidate with a passed promotion gate")
    trial = trials[0]
    if results[0]["candidate"] != trial["candidate"]:
        raise ValueError("candidate changed after validation selection")
    adapter = trial["models"][trial["candidate"]]
    verified_gate = promotion_gate(results[0]["paired_intervals"], selection["margins"],
                                   benchmark=results[0]["benchmark"],
                                   complete_comparisons=bool(trial.get("current_transfer_binding")),
                                   memory_baseline_verified=trial["candidate"] != "linear_analytic")
    if verified_gate["status"] != "passed":
        raise ValueError("promotion evidence does not pass the declared gates")
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    if source_identity(artifact) != adapter["source_identity"]:
        raise ValueError("deployment source checkpoint differs from calibrated source")
    output = Path(output)
    write_new(output / "adapter.json", adapter)
    load_adapter(output / "adapter.json")
    manifest = {"version": CONTRACT_VERSION, "source_checkpoint": str(Path(source).resolve()),
                "source_checkpoint_sha256": file_sha256(source), "adapter": "adapter.json",
                "adapter_sha256": file_sha256(output / "adapter.json"),
                "selection_sha256": file_sha256(selection_path), "test_report_sha256": file_sha256(report_path),
                "enable": "predict(..., calibration=adapter, environment=environment)",
                "rollback": "predict(..., calibration=None)", "source_checkpoint_preserved": True}
    write_new(output / "deployment.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("provenance")
    inspect.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    inspect.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    build.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--limit-per-split", type=int, help="bounded CPU smoke only; omit for full corpus")
    build.add_argument("--workers", type=int, default=1, help="independent CPU workers, each with one Torch thread (maximum 8)")
    audit = commands.add_parser("audit")
    audit.add_argument("--data", type=Path, required=True)
    select = commands.add_parser("select")
    select.add_argument("--data", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--margins", type=Path, required=True)
    select.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256])
    select.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    select.add_argument("--current-transfer", type=Path, help="equal-budget provenance and validation predictions")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap-draws", type=int, default=1000)
    evaluate.add_argument("--current-transfer", type=Path, help="same external checkpoints with test predictions")
    evaluate.add_argument("--benchmarks", type=Path, help="list of intended-host benchmark reports bound to adapters")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    bench.add_argument("--adapter", type=Path, required=True)
    bench.add_argument("--design", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--repeats", type=int, default=20)
    bench.add_argument("--intended-host", action="store_true")
    deploy = commands.add_parser("package")
    deploy.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    deploy.add_argument("--selection", type=Path, required=True)
    deploy.add_argument("--report", type=Path, required=True)
    deploy.add_argument("--output", type=Path, required=True)
    deploy.add_argument("--budget", type=int, required=True)
    deploy.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "provenance":
        result = provenance(args.source)
    elif args.command == "prepare":
        result = prepare(args.source, args.campaign, args.dataset, args.output, limit_per_split=args.limit_per_split, workers=args.workers)
        print(json.dumps({"audit": result["audit"], "manifest": str(args.output / "manifest.json")}, indent=2))
        return
    elif args.command == "audit":
        manifest = read_json(args.data / "manifest.json")
        result = audit_records([row for split in manifest["split_files"] for row in load_split(args.data, split)])
        print(json.dumps(result, indent=2))
        return
    elif args.command == "select":
        result = select_experiments(load_split(args.data, "train"), load_split(args.data, "validation"),
                                    budgets=args.budgets, seeds=args.seeds, margins=read_json(args.margins),
                                    current_transfer=read_json(args.current_transfer) if args.current_transfer else None)
        result["data_fingerprint"] = read_json(args.data / "manifest.json")["fingerprint"]
        result["fingerprint"] = fingerprint({key: value for key, value in result.items() if key != "fingerprint"})
        for trial in result["trials"]:
            for name, adapter in trial["models"].items():
                if adapter is not None:
                    path = args.output.parent / f"budget-{trial['budget']}-seed-{trial['seed']}-{name}.json"
                    write_new(path, adapter)
                    load_adapter(path)
    elif args.command == "evaluate":
        selection = read_json(args.selection)
        if selection["data_fingerprint"] != read_json(args.data / "manifest.json")["fingerprint"]:
            raise ValueError("test dataset differs from sealed validation selection")
        result = evaluate_test(selection, load_split(args.data, "test"), draws=args.bootstrap_draws,
                               current_transfer=read_json(args.current_transfer) if args.current_transfer else None,
                               benchmarks=read_json(args.benchmarks) if args.benchmarks else None)
    elif args.command == "package":
        package(args.source, args.selection, args.report, args.output, budget=args.budget, seed=args.seed)
        print(json.dumps({"deployment": str(args.output / "deployment.json")}, indent=2))
        return
    else:
        result = benchmark(args.source, args.adapter, args.design, repeats=args.repeats, intended_host=args.intended_host)
    write_new(args.output, result)
    print(json.dumps({"command": args.command, "output": str(args.output), "sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
