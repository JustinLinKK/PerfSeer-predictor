#!/usr/bin/env python
"""Run or print the precision-transfer training sequence."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_TEACHER_CONFIG = "src/perfseer-optimized/configs/train_precision_teacher/large_teacher.yaml"
DEFAULT_STUDENT_CONFIG = "src/perfseer-optimized/configs/train_deploy_model/precision_distill_student_128.yaml"
DEFAULT_EVAL_PROFILE = "src/perfseer-optimized/configs/eval_profiles/gpu_accuracy.yaml"
DEFAULT_DEPLOY_EVAL_PROFILE = "src/perfseer-optimized/configs/eval_profiles/cpu_torchscript_fp32.yaml"
STRUCTURAL_SPLIT_UNITS = {"graph_signature", "graph_family"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the precision-transfer source, transfer, distill, and eval flow.")
    p.add_argument("--python", default=sys.executable, help="Python executable to use for module commands.")
    p.add_argument("--source-data-root", default="dataset", help="Original source-domain dataset root.")
    p.add_argument("--precision-data-root", default="dataset_precision", help="Materialized precision dataset root.")
    p.add_argument("--pack-dir", default="nrp_calibration_pack", help="Calibration pack root for materialization.")
    p.add_argument("--results-dir", help="Profiler output directory containing results_shard*.jsonl.")
    p.add_argument("--hardware-id", help="Hardware id override for materialized precision rows.")
    p.add_argument("--base-mode", choices=("copy", "symlink"), default="symlink", help="How to include source-domain data during materialization.")
    p.add_argument("--source-precision-config", default="source_domain_unknown", help="Precision config assigned to source-domain labels.")
    p.add_argument("--source-hardware-id", default="source_domain_unknown", help="Hardware id assigned to source-domain labels.")
    p.add_argument("--source-hardware-features-json", default="{}", help="JSON object passed to the precision materializer.")
    p.add_argument("--source-precision-provenance", default="", help="Short note/path/URI proving the source labels' precision setup.")
    p.add_argument("--require-source-precision-provenance", action="store_true", help="Fail materialization unless source precision provenance is provided.")
    p.add_argument("--require-source-precision-confirmed", action="store_true", help="Require source precision to be confirmed during --check-results.")
    p.add_argument("--pseudo-precision-sweep", default="", help="Comma-separated precision configs for optional pseudo rows used during student distillation.")
    p.add_argument("--pseudo-hardware-id", help="Hardware id assigned to pseudo rows.")
    p.add_argument("--pseudo-hardware-features-json", default="{}", help="JSON object of hardware features for optional pseudo rows.")
    p.add_argument("--teacher-config", default=DEFAULT_TEACHER_CONFIG)
    p.add_argument("--student-config", default=DEFAULT_STUDENT_CONFIG)
    p.add_argument("--eval-profile", default=DEFAULT_EVAL_PROFILE)
    p.add_argument("--deploy-eval-profile", help=f"Optional deployment eval profile, for example {DEFAULT_DEPLOY_EVAL_PROFILE}.")
    p.add_argument("--out-dir", default="runs/optimized")
    p.add_argument("--results-path", default="runs/results.jsonl")
    p.add_argument("--materialization-report", help="precision_materialization_report.json path used by --check-results. Defaults under --precision-data-root.")
    p.add_argument("--source-run-id", default="precision_large_teacher_source")
    p.add_argument("--baseline-run-id", help="Optional baseline eval run id used to compare source-teacher accuracy before transfer.")
    p.add_argument("--baseline-data-root", help="Expected data root for --baseline-run-id. Defaults to --source-data-root in the checker.")
    p.add_argument("--transfer-run-id", default="precision_large_teacher_transfer")
    p.add_argument("--student-run-id", default="precision_distill_student_128")
    p.add_argument("--limit", type=int, help="Optional data limit for all train/eval commands.")
    p.add_argument("--split-unit", choices=("pair", "graph", "graph_signature", "graph_family"), help="Override data.split_unit for all train commands.")
    p.add_argument(
        "--structural-validation-splits",
        default="",
        help="Comma-separated structural split units to run as additional target-domain validation flows, e.g. graph_signature,graph_family.",
    )
    p.add_argument("--source-epochs", type=int, help="Override source pretraining epochs.")
    p.add_argument("--transfer-epochs", type=int, help="Override transfer fine-tuning epochs.")
    p.add_argument("--student-epochs", type=int, help="Override student distillation epochs.")
    p.add_argument("--check-results", action="store_true", help="After eval, validate required precision-transfer result rows.")
    p.add_argument("--required-precision", default="", help="Comma-separated precision slices required by --check-results.")
    p.add_argument("--min-eval-precision-count", action="append", default=[], help="Minimum held-out rows for a precision config, e.g. bf16_amp=20. Can be repeated or comma-separated.")
    p.add_argument("--required-label-domain", default="", help="Comma-separated label-domain slices required by --check-results, for example precision_profile.")
    p.add_argument("--min-eval-source-labels", type=int, help="Minimum source-domain held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-precision-labels", type=int, help="Minimum precision_profile held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-pseudo-labels", type=int, help="Minimum pseudo held-out labels required in precision teacher/student/deployment eval rows.")
    p.add_argument("--min-eval-hardware-count", action="append", default=[], help="Minimum held-out rows for a hardware id, e.g. a100=20. Can be repeated or comma-separated.")
    p.add_argument("--max-source-mean-mape", type=float, help="Optional source teacher MAPE threshold for --check-results.")
    p.add_argument("--max-source-baseline-mape-delta", type=float, help="Optional source-teacher minus baseline MAPE delta threshold for --check-results.")
    p.add_argument("--max-transfer-mean-mape", type=float, help="Optional precision teacher MAPE threshold for --check-results.")
    p.add_argument("--max-student-mean-mape", type=float, help="Optional precision student MAPE threshold for --check-results.")
    p.add_argument("--max-deploy-mean-mape", type=float, help="Optional deployment eval MAPE threshold for --check-results.")
    p.add_argument("--max-deploy-latency-p50", type=float, help="Optional deployment p50 latency threshold for --check-results.")
    p.add_argument("--expected-deploy-runtime-backend", help="Expected requested runtime backend for deployment eval validation.")
    p.add_argument("--expected-deploy-runtime-backend-actual", help="Expected actual runtime backend for deployment eval validation.")
    p.add_argument("--require-checkpoint-files", action="store_true", help="Require eval rows to reference existing checkpoint files during --check-results.")
    p.add_argument("--require-deployment-student-checkpoint", action="store_true", help="Require deployment eval to use the same checkpoint paths as held-out student eval. Automatically enabled with deployment eval plus --require-checkpoint-files.")
    p.add_argument("--require-train-events", action="store_true", help="Require train_complete rows during --check-results.")
    p.add_argument("--require-eval-train-checkpoints", action="store_true", help="Require source/transfer/student eval checkpoint paths to match train_complete checkpoint paths. Automatically enabled with --require-train-events plus --require-checkpoint-files.")
    p.add_argument("--required-train-label-domain", default="", help="Comma-separated label-domain slices required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-precision-count", action="append", default=[], help="Minimum train-split rows for a precision config, e.g. bf16_amp=20. Can be repeated or comma-separated.")
    p.add_argument("--min-train-source-labels", type=int, help="Minimum source-domain labels required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-precision-labels", type=int, help="Minimum measured precision-profile labels required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-pseudo-labels", type=int, help="Minimum pseudo-label rows required in precision teacher/student train split metadata.")
    p.add_argument("--min-train-hardware-count", action="append", default=[], help="Minimum train-split rows for a hardware id, e.g. a100=20. Can be repeated or comma-separated.")
    p.add_argument("--min-train-split-count", type=int, help="Minimum train split count required in train_complete metadata.")
    p.add_argument("--min-val-split-count", type=int, help="Minimum validation split count required in train_complete metadata.")
    p.add_argument("--min-train-test-count", type=int, help="Minimum test split count required in train_complete metadata.")
    p.add_argument("--require-unlimited-train-data", action="store_true", help="Require checked train runs to use the full dataset, not --limit.")
    p.add_argument("--require-train-checkpoint-metadata", action="store_true", help="Require train_complete rows to include saved-checkpoint metadata summaries during --check-results.")
    p.add_argument("--require-train-lineage", action="store_true", help="Require source->transfer->student checkpoint lineage during --check-results.")
    p.add_argument("--min-precision-slices", type=int, help="Minimum precision slices required by --check-results for transfer/student eval rows.")
    p.add_argument("--min-label-domain-slices", type=int, help="Minimum label-domain slices required by --check-results.")
    p.add_argument("--min-batch-size-slices", type=int, help="Minimum batch-size slices required by --check-results.")
    p.add_argument("--min-resource-regime-slices", type=int, help="Minimum resource-regime slices required by --check-results.")
    p.add_argument("--min-graph-signature-slices", type=int, help="Minimum graph-signature slices required by --check-results.")
    p.add_argument("--min-graph-family-slices", type=int, help="Minimum graph-family slices required by --check-results.")
    p.add_argument("--min-materialized-precision-labels", type=int, help="Minimum accepted precision labels required in the materialization report.")
    p.add_argument("--min-materialized-base-pairs", type=int, help="Minimum source dataset graph/label pairs required in the materialization report.")
    p.add_argument("--min-materialized-source-labels", type=int, help="Minimum source-domain labels required in the materialization report.")
    p.add_argument("--min-materialized-pseudo-labels", type=int, help="Minimum pseudo labels required in the materialization report.")
    p.add_argument("--skip-materialize", action="store_true")
    p.add_argument("--skip-source-pretrain", action="store_true")
    p.add_argument("--skip-source-eval", action="store_true")
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--skip-distill", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-deploy-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return p.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("+", quote_cmd(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_structural_validation_splits(raw: str) -> list[str]:
    splits: list[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        split = part.strip()
        if not split:
            continue
        if split not in STRUCTURAL_SPLIT_UNITS:
            allowed = ", ".join(sorted(STRUCTURAL_SPLIT_UNITS))
            raise ValueError(f"unsupported structural validation split {split!r}; expected one of: {allowed}")
        if split not in seen:
            seen.add(split)
            splits.append(split)
    return splits


def split_run_id(base: str, split_unit: str) -> str:
    return f"{base}_{split_unit}"


def train_cmd(
    args: argparse.Namespace,
    config: str,
    run_id: str,
    data_root: str,
    epochs: int | None,
    *,
    source_pretrain: bool = False,
    split_unit: str | None = None,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.train",
        "--config",
        config,
        "--run-id",
        run_id,
        "--data-root",
        data_root,
        "--out",
        args.out_dir,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    effective_split_unit = args.split_unit if split_unit is None else split_unit
    if effective_split_unit is not None:
        cmd += ["--split-unit", effective_split_unit]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if source_pretrain:
        cmd += ["--precision-config", args.source_precision_config, "--hardware-id", args.source_hardware_id]
        if args.source_precision_provenance:
            cmd += ["--source-precision-provenance", args.source_precision_provenance]
        if args.require_source_precision_provenance:
            cmd += ["--require-source-precision-provenance"]
    return cmd


def eval_cmd(args: argparse.Namespace, ckpt_dir: Path, run_label: str, data_root: str) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.eval",
        "--eval-profile",
        args.eval_profile,
        "--ckpt-dir",
        str(ckpt_dir),
        "--data-root",
        data_root,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    print(f"# evaluate {run_label}", flush=True)
    return cmd


def deploy_eval_cmd(args: argparse.Namespace, ckpt_dir: Path, data_root: str) -> list[str]:
    if not args.deploy_eval_profile:
        raise ValueError("--deploy-eval-profile is required for deployment eval")
    cmd = [
        args.python,
        "-m",
        "perfseer_optimized.eval_deploy",
        "--eval-profile",
        args.deploy_eval_profile,
        "--ckpt-dir",
        str(ckpt_dir),
        "--data-root",
        data_root,
        "--results-path",
        args.results_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    print("# evaluate deployment student", flush=True)
    return cmd


def check_results_cmd(
    args: argparse.Namespace,
    *,
    transfer_run_id: str | None = None,
    student_run_id: str | None = None,
    required_split_unit: str | None = None,
    skip_source: bool = False,
    include_baseline: bool = True,
) -> list[str]:
    cmd = [
        args.python,
        "scripts/check_precision_transfer_results.py",
        "--results",
        args.results_path,
        "--source-run-id",
        args.source_run_id,
        "--transfer-run-id",
        transfer_run_id or args.transfer_run_id,
        "--student-run-id",
        student_run_id or args.student_run_id,
        "--source-data-root",
        args.source_data_root,
        "--precision-data-root",
        args.precision_data_root,
    ]
    if include_baseline and args.baseline_run_id:
        cmd += ["--baseline-run-id", args.baseline_run_id]
    if include_baseline and args.baseline_data_root:
        cmd += ["--baseline-data-root", args.baseline_data_root]
    if args.required_precision:
        cmd += ["--required-precision", args.required_precision]
    for item in args.min_eval_precision_count:
        cmd += ["--min-eval-precision-count", item]
    if args.required_label_domain:
        cmd += ["--required-label-domain", args.required_label_domain]
    if args.min_eval_source_labels is not None:
        cmd += ["--min-eval-source-labels", str(args.min_eval_source_labels)]
    if args.min_eval_precision_labels is not None:
        cmd += ["--min-eval-precision-labels", str(args.min_eval_precision_labels)]
    if args.min_eval_pseudo_labels is not None:
        cmd += ["--min-eval-pseudo-labels", str(args.min_eval_pseudo_labels)]
    for item in args.min_eval_hardware_count:
        cmd += ["--min-eval-hardware-count", item]
    effective_required_split_unit = required_split_unit or args.split_unit
    if effective_required_split_unit is not None:
        cmd += ["--required-split-unit", effective_required_split_unit]
    materialization_report = args.materialization_report
    if materialization_report is None and (args.require_source_precision_provenance or args.require_source_precision_confirmed):
        materialization_report = str(Path(args.precision_data_root) / "precision_materialization_report.json")
    if materialization_report:
        cmd += ["--materialization-report", materialization_report]
    if args.require_source_precision_provenance:
        cmd += ["--require-source-precision-provenance"]
    if args.require_source_precision_confirmed:
        cmd += ["--require-source-precision-confirmed"]
    if args.require_source_precision_provenance or args.require_source_precision_confirmed:
        cmd += ["--expected-source-precision-config", args.source_precision_config]
        if args.source_precision_provenance:
            cmd += ["--expected-source-precision-provenance", args.source_precision_provenance]
    if args.max_source_mean_mape is not None:
        cmd += ["--max-source-mean-mape", str(args.max_source_mean_mape)]
    if include_baseline and args.max_source_baseline_mape_delta is not None:
        cmd += ["--max-source-baseline-mape-delta", str(args.max_source_baseline_mape_delta)]
    if args.max_transfer_mean_mape is not None:
        cmd += ["--max-transfer-mean-mape", str(args.max_transfer_mean_mape)]
    if args.max_student_mean_mape is not None:
        cmd += ["--max-student-mean-mape", str(args.max_student_mean_mape)]
    if args.deploy_eval_profile and not args.skip_deploy_eval:
        cmd += ["--require-deployment-eval", "--require-deployment-metadata"]
    if args.max_deploy_mean_mape is not None:
        cmd += ["--max-deploy-mean-mape", str(args.max_deploy_mean_mape)]
    if args.max_deploy_latency_p50 is not None:
        cmd += ["--max-deploy-latency-p50", str(args.max_deploy_latency_p50)]
    if args.expected_deploy_runtime_backend is not None:
        cmd += ["--expected-deploy-runtime-backend", args.expected_deploy_runtime_backend]
    if args.expected_deploy_runtime_backend_actual is not None:
        cmd += ["--expected-deploy-runtime-backend-actual", args.expected_deploy_runtime_backend_actual]
    if args.require_checkpoint_files:
        cmd += ["--require-checkpoint-files"]
    require_deployment_checkpoint = (
        args.require_deployment_student_checkpoint
        or (
            bool(args.deploy_eval_profile)
            and not args.skip_deploy_eval
            and not args.skip_distill
            and bool(args.require_checkpoint_files)
        )
    )
    if require_deployment_checkpoint:
        cmd += ["--require-deployment-student-checkpoint"]
    if args.require_train_events:
        cmd += ["--require-train-events"]
    require_eval_train_checkpoints = (
        args.require_eval_train_checkpoints
        or (bool(args.require_train_events) and bool(args.require_checkpoint_files))
    )
    if require_eval_train_checkpoints:
        cmd += ["--require-eval-train-checkpoints"]
    if args.required_train_label_domain:
        cmd += ["--required-train-label-domain", args.required_train_label_domain]
    for item in args.min_train_precision_count:
        cmd += ["--min-train-precision-count", item]
    if args.min_train_source_labels is not None:
        cmd += ["--min-train-source-labels", str(args.min_train_source_labels)]
    if args.min_train_precision_labels is not None:
        cmd += ["--min-train-precision-labels", str(args.min_train_precision_labels)]
    if args.min_train_pseudo_labels is not None:
        cmd += ["--min-train-pseudo-labels", str(args.min_train_pseudo_labels)]
    for item in args.min_train_hardware_count:
        cmd += ["--min-train-hardware-count", item]
    if args.min_train_split_count is not None:
        cmd += ["--min-train-split-count", str(args.min_train_split_count)]
    if args.min_val_split_count is not None:
        cmd += ["--min-val-split-count", str(args.min_val_split_count)]
    if args.min_train_test_count is not None:
        cmd += ["--min-train-test-count", str(args.min_train_test_count)]
    if args.require_unlimited_train_data:
        cmd += ["--require-unlimited-train-data"]
    if args.require_train_checkpoint_metadata:
        cmd += ["--require-train-checkpoint-metadata"]
    if args.require_train_lineage:
        cmd += ["--require-train-lineage"]
    if args.require_source_precision_provenance:
        cmd += ["--require-source-train-provenance"]
    if args.require_source_precision_confirmed:
        cmd += ["--require-source-train-precision-confirmed"]
    if args.min_precision_slices is not None:
        cmd += ["--min-precision-slices", str(args.min_precision_slices)]
    if args.min_label_domain_slices is not None:
        cmd += ["--min-label-domain-slices", str(args.min_label_domain_slices)]
    if args.min_batch_size_slices is not None:
        cmd += ["--min-batch-size-slices", str(args.min_batch_size_slices)]
    if args.min_resource_regime_slices is not None:
        cmd += ["--min-resource-regime-slices", str(args.min_resource_regime_slices)]
    if args.min_graph_signature_slices is not None:
        cmd += ["--min-graph-signature-slices", str(args.min_graph_signature_slices)]
    if args.min_graph_family_slices is not None:
        cmd += ["--min-graph-family-slices", str(args.min_graph_family_slices)]
    if args.min_materialized_precision_labels is not None:
        cmd += ["--min-materialized-precision-labels", str(args.min_materialized_precision_labels)]
    if args.min_materialized_base_pairs is not None:
        cmd += ["--min-materialized-base-pairs", str(args.min_materialized_base_pairs)]
    if args.min_materialized_source_labels is not None:
        cmd += ["--min-materialized-source-labels", str(args.min_materialized_source_labels)]
    if args.min_materialized_pseudo_labels is not None:
        cmd += ["--min-materialized-pseudo-labels", str(args.min_materialized_pseudo_labels)]
    if skip_source or args.skip_source_eval:
        cmd += ["--skip-source"]
    if args.skip_transfer:
        cmd += ["--skip-transfer"]
    if args.skip_distill:
        cmd += ["--skip-student"]
    return cmd


def main() -> None:
    args = parse_args()
    try:
        structural_splits = parse_structural_validation_splits(args.structural_validation_splits)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if structural_splits and (args.skip_transfer or args.skip_distill or args.skip_eval):
        raise SystemExit("--structural-validation-splits requires transfer, distill, and eval steps")
    out_root = Path(args.out_dir)
    source_dir = out_root / args.source_run_id
    source_ckpt = source_dir / "seernet_multi.pt"
    transfer_dir = out_root / args.transfer_run_id
    student_dir = out_root / args.student_run_id

    if not args.skip_materialize:
        if not args.results_dir:
            raise SystemExit("--results-dir is required unless --skip-materialize is set")
        cmd = [
            args.python,
            "scripts/materialize_precision_dataset.py",
            "--pack-dir",
            args.pack_dir,
            "--results-dir",
            args.results_dir,
            "--out-root",
            args.precision_data_root,
            "--base-data-root",
            args.source_data_root,
            "--base-mode",
            args.base_mode,
            "--source-precision-config",
            args.source_precision_config,
            "--source-hardware-id",
            args.source_hardware_id,
            "--source-hardware-features-json",
            args.source_hardware_features_json,
        ]
        if args.source_precision_provenance:
            cmd += ["--source-precision-provenance", args.source_precision_provenance]
        if args.require_source_precision_provenance:
            cmd += ["--require-source-precision-provenance"]
        if args.hardware_id:
            cmd += ["--hardware-id", args.hardware_id]
        if args.pseudo_precision_sweep:
            cmd += [
                "--pseudo-precision-sweep",
                args.pseudo_precision_sweep,
                "--pseudo-hardware-features-json",
                args.pseudo_hardware_features_json,
            ]
            if args.pseudo_hardware_id:
                cmd += ["--pseudo-hardware-id", args.pseudo_hardware_id]
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_source_pretrain:
        run_cmd(
            train_cmd(args, args.teacher_config, args.source_run_id, args.source_data_root, args.source_epochs, source_pretrain=True),
            dry_run=args.dry_run,
        )

    if not args.skip_eval and not args.skip_source_eval:
        run_cmd(eval_cmd(args, source_dir, "source teacher", args.source_data_root), dry_run=args.dry_run)

    if not args.skip_transfer:
        cmd = train_cmd(args, args.teacher_config, args.transfer_run_id, args.precision_data_root, args.transfer_epochs)
        cmd += ["--init-checkpoint", str(source_ckpt)]
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_distill:
        cmd = train_cmd(args, args.student_config, args.student_run_id, args.precision_data_root, args.student_epochs)
        cmd += ["--teacher-ckpt-dir", str(transfer_dir)]
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_eval:
        if not args.skip_transfer:
            run_cmd(eval_cmd(args, transfer_dir, "precision teacher", args.precision_data_root), dry_run=args.dry_run)
        if not args.skip_distill:
            run_cmd(eval_cmd(args, student_dir, "precision student", args.precision_data_root), dry_run=args.dry_run)
        if args.deploy_eval_profile and not args.skip_deploy_eval and not args.skip_distill:
            run_cmd(deploy_eval_cmd(args, student_dir, args.precision_data_root), dry_run=args.dry_run)
        if args.check_results:
            run_cmd(check_results_cmd(args), dry_run=args.dry_run)

    for split_unit in structural_splits:
        transfer_run_id = split_run_id(args.transfer_run_id, split_unit)
        student_run_id = split_run_id(args.student_run_id, split_unit)
        split_transfer_dir = out_root / transfer_run_id
        split_student_dir = out_root / student_run_id
        print(f"# structural validation split {split_unit}", flush=True)

        cmd = train_cmd(
            args,
            args.teacher_config,
            transfer_run_id,
            args.precision_data_root,
            args.transfer_epochs,
            split_unit=split_unit,
        )
        cmd += ["--init-checkpoint", str(source_ckpt)]
        run_cmd(cmd, dry_run=args.dry_run)

        cmd = train_cmd(
            args,
            args.student_config,
            student_run_id,
            args.precision_data_root,
            args.student_epochs,
            split_unit=split_unit,
        )
        cmd += ["--teacher-ckpt-dir", str(split_transfer_dir)]
        run_cmd(cmd, dry_run=args.dry_run)

        run_cmd(eval_cmd(args, split_transfer_dir, f"precision teacher {split_unit}", args.precision_data_root), dry_run=args.dry_run)
        run_cmd(eval_cmd(args, split_student_dir, f"precision student {split_unit}", args.precision_data_root), dry_run=args.dry_run)
        if args.deploy_eval_profile and not args.skip_deploy_eval:
            run_cmd(deploy_eval_cmd(args, split_student_dir, args.precision_data_root), dry_run=args.dry_run)
        if args.check_results:
            run_cmd(
                check_results_cmd(
                    args,
                    transfer_run_id=transfer_run_id,
                    student_run_id=student_run_id,
                    required_split_unit=split_unit,
                    skip_source=True,
                    include_baseline=False,
                ),
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
