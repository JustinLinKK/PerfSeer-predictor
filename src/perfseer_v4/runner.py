"""Training-only teacher/student training, strict resume, and gated distillation."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

from perfseer_v3.features import NormalizationBlock, NormalizationStatsV3, _transform, apply_normalization
from perfseer_v3.op_registry import OperationRegistry

from .dataset import DATA, verify
from .features import build_features, normalize_features
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .model import SeerNetV4, SeerNetV4Config
from .training import (T1, S1, build_optimizers, build_schedulers, checkpoint_payload,
                       evaluate, evaluation_config, quality_gate, restore_model, restore_rng, rng_state, selection_key, train_batch)
from .version import TARGET_NAMES, MODES, SM_INDICES, FEATURE_VERSION, NORMALIZATION_VERSION, METRIC_VERSION, SELECTION_VERSION


def code_fingerprint():
    package = Path(__file__).resolve().parent
    from perfseer_v32.runner import code_fingerprint as shared_fingerprint
    return fingerprint({"shared": shared_fingerprint(), "v4": {
        path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}})


def early_stopping_reason(metrics, epoch, best_key, *, patience=0, min_epochs=0, stop_on_gate=False):
    if stop_on_gate and metrics["gate_passed"]:
        return "validation_gate_passed"
    if patience and epoch >= min_epochs and epoch - int(best_key[-1]) >= patience:
        return f"no_validation_improvement_for_{patience}_epochs"
    return None


class Samples:
    def __init__(self, root, rows, normalization=None, cache_root=None, *, include_resources=False):
        self.root, self.rows, self.normalization = Path(root), rows, normalization
        self.include_resources = include_resources
        self.cache = OrderedDict()
        self.cache_root = Path(cache_root) if cache_root else None
        self.cache_identity = code_fingerprint() if cache_root else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        if tuple(row.get("target_names", ())) != TARGET_NAMES or set(row["targets"]) != set(TARGET_NAMES):
            raise ValueError("sample target contract differs")
        key = (row["input_path"], row["input_sha256"])
        if key not in self.cache:
            from .dataset import checked_path
            input_path = checked_path(self.root, row["input_path"])
            if file_sha256(input_path) != row["input_sha256"]:
                raise ValueError("sample input hash differs")
            cache_path = self.cache_root / (fingerprint([row["input_sha256"], self.cache_identity, FEATURE_VERSION, self.include_resources]) + ".pt") if self.cache_root else None
            if cache_path and cache_path.exists():
                features = torch.load(cache_path, map_location="cpu", weights_only=False)
                features.validate()
            else:
                features = build_features(read_json(input_path), include_resources=self.include_resources)
                if cache_path:
                    atomic_write(cache_path, features, checkpoint=True)
            self.cache[key] = normalize_features(features, self.normalization) if self.normalization else features
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return {"features": self.cache[key], "target": torch.tensor([row["targets"][name] for name in TARGET_NAMES], dtype=torch.float32),
                "sample_id": row["sample_id"], "group_id": row.get("group_id"),
                "workload_id": row["input_sha256"], "split": row.get("split")}


def _fit_mode_normalization(samples, dataset_fingerprint, mode):
    """Streaming full-training moments, with min/max clipping and bounded RAM."""
    layout = getattr(samples[0]["features"], mode).layout
    blocks = [("x_cont", layout.node_continuous_fields), ("edge_cont", layout.edge_continuous_fields),
              ("u_cont", layout.global_continuous_fields)]
    states = [[0, np.zeros(len(names)), np.zeros(len(names)), np.full(len(names), np.inf),
               np.full(len(names), -np.inf)] for _, names in blocks]
    for index in range(len(samples)):
        features = getattr(samples[index]["features"], mode)
        for (attribute, names), state in zip(blocks, states, strict=True):
            values = _transform(getattr(features, attribute).numpy(), names)
            if not len(values):
                continue
            count, mean, m2, low, high = state
            next_count = count + len(values)
            delta = values.mean(0) - mean
            state[:] = [next_count, mean + delta * len(values) / next_count,
                        m2 + ((values - values.mean(0)) ** 2).sum(0) + delta ** 2 * count * len(values) / next_count,
                        np.minimum(low, values.min(0)), np.maximum(high, values.max(0))]
        if index % 1000 == 0:
            print(f"{mode} normalization: {index}/{len(samples)} training rows", flush=True)
    fitted = []
    for count, mean, m2, low, high in states:
        if not count:
            raise ValueError("empty training feature block")
        std = np.sqrt(m2 / count)
        fitted.append(NormalizationBlock(tuple(mean), tuple(np.where(std < 1e-8, 1., std)), tuple(low), tuple(high)))
    return NormalizationStatsV3(layout.feature_schema_sha256, layout.operator_registry_sha256, layout.layout_sha256,
                                "train", dataset_fingerprint, (0., 1.), *fitted,
                                normalization_version=NORMALIZATION_VERSION)


def _mode_normalization_from_dict(value):
    return NormalizationStatsV3(**{**value, "quantiles": tuple(value["quantiles"]),
                                  **{name: NormalizationBlock(**{key: tuple(items) for key, items in value[name].items()})
                                     for name in ("node", "edge", "global_features")}})


@dataclass(frozen=True)
class TrainingNormalization:
    training: NormalizationStatsV3
    version: str = NORMALIZATION_VERSION

    def __post_init__(self):
        if self.version != NORMALIZATION_VERSION or any(getattr(self, mode).normalization_version != NORMALIZATION_VERSION or getattr(self, mode).split_name != "train" for mode in MODES):
            raise ValueError("normalization contract differs")

    @property
    def split_fingerprint(self):
        return self.training.split_fingerprint


def fit_training_normalization(samples, dataset_fingerprint):
    return TrainingNormalization(**{mode: _fit_mode_normalization(samples, dataset_fingerprint, mode) for mode in MODES})


def normalization_from_dict(value):
    if set(value) != {*MODES, "version"}:
        raise ValueError("expected training-only v4 normalization")
    return TrainingNormalization(version=value["version"], **{mode: _mode_normalization_from_dict(value[mode]) for mode in MODES})


def memory_probe(model, samples, lr, *, teacher=None, maximum=64):
    """Probe large graphs including optimizer-state allocation; discard all updates."""
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state = rng_state()
    # Large graphs are selected from input artifacts without reading any labels.
    sizes = sorted(range(len(samples)), key=lambda i: (samples.root / samples.rows[i]["input_path"]).stat().st_size, reverse=True)
    lower, upper = 0, min(maximum, len(samples))
    candidate = upper
    while lower < upper:
        optimizers, _ = build_optimizers(model, lr)
        try:
            model.train()
            _, actual = train_batch(model, [samples[i] for i in sizes[:candidate]], optimizers, candidate, teacher=teacher)
            lower = max(lower, actual)
            if actual < candidate:
                upper = candidate - 1
        except torch.cuda.OutOfMemoryError:
            upper = candidate - 1
        finally:
            model.load_state_dict(original)
            model.zero_grad(set_to_none=True)
            del optimizers
            restore_rng(state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        candidate = (lower + upper + 1) // 2
    if lower:
        return lower
    raise RuntimeError("a single graph cannot fit the training device")


def run_stage(role, train, validation, output, manifest, normalization, device, *, teacher=None, resume=False,
              early_stopping_patience=0, early_stopping_min_epochs=0, stop_on_gate=False,
              compatible_resume_code_fingerprints=(), epochs=None, effective_batch=256, maximum_microbatch=64,
              initial_checkpoint=None, learning_rate=None):
    default_epochs, warmup, lr = (600, 10, 5e-4) if role == "teacher" else (100, 5, 1e-3)
    lr = lr if learning_rate is None else learning_rate
    epochs = default_epochs if epochs is None else epochs
    warmup = min(warmup, epochs)
    latest, best = output / f"{role}-latest.pt", output / f"{role}-best.pt"
    teacher_sha256 = file_sha256(output / "teacher-best.pt") if role == "student" else None
    progress = None
    if resume and latest.exists():
        progress = torch.load(latest, map_location="cpu", weights_only=False)
        if progress.get("model_variant") != "v4.0":
            raise ValueError("teacher/student training requires a v4.0 checkpoint")
        checkpoint_fingerprint = progress.get("code_fingerprint")
        if progress["normalization"] != asdict(normalization) or progress["role"] != role:
            raise ValueError("resume normalization or role mismatch")
        if progress.get("planned_epochs") != epochs or progress["effective_batch"] != effective_batch:
            raise ValueError("resume epoch schedule or effective batch mismatch")
        if progress.get("teacher_checkpoint_sha256") != teacher_sha256:
            raise ValueError("resume distillation teacher checkpoint differs")
        if progress.get("transfer") != manifest.get("transfer") or (manifest.get("transfer") and progress.get("learning_rate") != lr):
            raise ValueError("resume transfer lineage or learning rate differs")
        if checkpoint_fingerprint not in {code_fingerprint(), *compatible_resume_code_fingerprints}:
            raise ValueError("resume code fingerprint mismatch")
        model = restore_model(progress, dataset_fingerprint=manifest["fingerprint"], device=device)
    else:
        if latest.exists() or best.exists():
            raise ValueError("output contains checkpoints; use --resume or a fresh output directory")
        if initial_checkpoint is not None:
            payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
            if payload.get("model_variant") != "v4.0":
                raise ValueError("teacher/student training requires a v4.0 checkpoint")
            if payload["role"] != role or file_sha256(initial_checkpoint) != manifest["transfer"]["base_checkpoint_sha256"]:
                raise ValueError("pretrained checkpoint role or hash differs")
            model = restore_model(payload, device=device)
            if role == "teacher":
                model.config = replace(model.config, checkpoint_blocks=True)
        else:
            targets = torch.tensor([[row["targets"][name] for name in TARGET_NAMES] for row in train.rows])
            medians = targets.median(0).values
            scales = medians.clone()
            scales[list(SM_INDICES)] = 100
            config = SeerNetV4Config.from_registry(OperationRegistry.load(), train[0]["features"].layout,
                                                   **(T1 if role == "teacher" else S1))
            model = SeerNetV4(config, scales, medians).to(device)
    microbatch = progress["microbatch"] if progress else memory_probe(model, train, lr, teacher=teacher,
                                                                   maximum=min(maximum_microbatch, effective_batch))
    optimizers, partition = build_optimizers(model, lr)
    steps_per_epoch = math.ceil(len(train) / effective_batch)
    schedulers = build_schedulers(optimizers, epochs=epochs, warmup_epochs=warmup, steps_per_epoch=steps_per_epoch)
    if progress:
        for optimizer, state in zip(optimizers, progress["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, progress["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        restore_rng(progress["rng"])
    start_epoch = progress["next_epoch"] if progress else 1
    cursor = progress["next_batch"] if progress else 0
    best_key = tuple(progress["best_key"]) if progress else (math.inf, math.inf, math.inf, math.inf, 0)
    atomic_write(output / f"{role}-optimizer-partition.json", partition)
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(path, epoch, next_epoch, next_batch, metrics=None):
        atomic_write(path, checkpoint_payload(model, role=role, epoch=epoch, normalization=asdict(normalization),
                     dataset_fingerprint=manifest["fingerprint"], optimizers=optimizers, schedulers=schedulers,
                     label_policy=manifest.get("label_policy"),
                     prediction_hardware=manifest.get("prediction_hardware", manifest.get("hardware_id")),
                     next_epoch=next_epoch, next_batch=next_batch, best_key=best_key,
                     microbatch=microbatch, effective_batch=effective_batch, planned_epochs=epochs,
                     train_rows=len(train), validation_rows=len(validation), code_fingerprint=code_fingerprint(),
                     **({"transfer": manifest["transfer"], "learning_rate": lr} if manifest.get("transfer") else {}),
                     teacher_checkpoint_sha256=teacher_sha256, validation=metrics), checkpoint=True)

    try:
        for epoch in range(start_epoch, epochs + 1):
            started = time.monotonic()
            order = list(range(len(train)))
            random.Random(42 + epoch).shuffle(order)
            model.train()
            for step in range(cursor, steps_per_epoch):
                logical = [train[i] for i in order[step * effective_batch:(step + 1) * effective_batch]]
                loss, microbatch = train_batch(model, logical, optimizers, microbatch, teacher=teacher)
                for scheduler in schedulers:
                    scheduler.step()
                print(f"{role} epoch={epoch}/{epochs} step={step + 1}/{steps_per_epoch} loss={loss:.6f} microbatch={microbatch}", flush=True)
                if stop[0]:
                    save(latest, epoch, epoch, step + 1)
                    raise InterruptedError("checkpoint saved at optimizer-step boundary")
            cursor = 0
            prediction_path = output / f"{role}-epoch-{epoch:04d}-predictions.json.gz"
            metrics = evaluate(model, validation, microbatch, prediction_path=prediction_path,
                               identity={"role": role, "epoch": epoch, "dataset_fingerprint": manifest["fingerprint"],
                                         "label_policy": manifest.get("label_policy"),
                                         "code_fingerprint": code_fingerprint()})
            metrics["predictions_sha256"] = file_sha256(prediction_path)
            key = selection_key(metrics, epoch)
            if key < best_key:
                best_key = key
                save(best, epoch, epoch + 1, 0, metrics)
            save(latest, epoch, epoch + 1, 0, metrics)
            atomic_write(output / f"{role}-epoch-{epoch:04d}.json", {
                "epoch": epoch, "validation": metrics, "best_key": best_key,
                "train_rows": len(order), "train_sample_ids_sha256": fingerprint(sorted(train.rows[i]["sample_id"] for i in order)),
                "optimizer_steps": steps_per_epoch, "validation_rows": len(validation),
                "effective_batch": effective_batch, "microbatch": microbatch, "seconds": time.monotonic() - started})
            print(f"{role} epoch={epoch} validation={metrics} best_epoch={best_key[-1]}", flush=True)
            reason = early_stopping_reason(metrics, epoch, best_key, patience=early_stopping_patience,
                                           min_epochs=early_stopping_min_epochs, stop_on_gate=stop_on_gate)
            if reason:
                print(f"{role} early_stop epoch={epoch} reason={reason} best_epoch={best_key[-1]}", flush=True)
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    del optimizers, schedulers, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = torch.load(best, map_location="cpu", weights_only=False)
    return restore_model(payload, dataset_fingerprint=manifest["fingerprint"], device=device), payload


def run(args):
    verify(args.dataset)
    if not args.resume and any((args.output / f"{role}-{kind}.pt").exists()
                               for role in ("teacher", "student") for kind in ("latest", "best")):
        raise ValueError("output contains checkpoints; use --resume or a fresh output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA device")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    manifest = read_json(args.dataset / "dataset_manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.output / "feature-cache" / fingerprint([manifest["fingerprint"], code_fingerprint(), FEATURE_VERSION])
    rows = {name: read_json(args.dataset / meta["path"]) for name, meta in manifest["split_files"].items() if name != "test"}
    norm_path = args.output / "normalization.json"
    if args.resume and norm_path.exists():
        normalization = normalization_from_dict(read_json(norm_path))
        if normalization.split_fingerprint != manifest["fingerprint"]:
            raise ValueError("normalization dataset fingerprint mismatch")
    else:
        normalization = fit_training_normalization(Samples(args.dataset, sorted(rows["train"], key=lambda row: row["input_path"]), cache_root=cache_root), manifest["fingerprint"])
        atomic_write(norm_path, asdict(normalization))
    train = Samples(args.dataset, rows["train"], normalization, cache_root)
    validation = Samples(args.dataset, rows["validation"], normalization, cache_root)
    def completed_stage(role, epochs):
        if not args.resume or not (args.output / f"{role}-gate.json").exists():
            return None
        payload = torch.load(args.output / f"{role}-best.pt", map_location="cpu", weights_only=False)
        if (payload.get("model_variant") != "v4.0" or payload["normalization"] != asdict(normalization) or payload["role"] != role
                or payload.get("planned_epochs") != epochs or payload.get("effective_batch") != args.effective_batch
                or payload.get("code_fingerprint") not in {code_fingerprint(), *args.compatible_resume_code_fingerprint}):
            raise ValueError("completed stage resume contract differs")
        if role == "student" and payload.get("teacher_checkpoint_sha256") != file_sha256(args.output / "teacher-best.pt"):
            raise ValueError("resume distillation teacher checkpoint differs")
        return restore_model(payload, dataset_fingerprint=manifest["fingerprint"], device="cuda"), payload

    teacher, selected = completed_stage("teacher", args.teacher_epochs) or run_stage(
        "teacher", train, validation, args.output, manifest, normalization, "cuda", resume=args.resume,
        early_stopping_patience=args.teacher_early_stopping_patience,
        early_stopping_min_epochs=args.teacher_early_stopping_min_epochs,
        stop_on_gate=args.teacher_stop_on_validation_gate,
        compatible_resume_code_fingerprints=args.compatible_resume_code_fingerprint,
        epochs=args.teacher_epochs, effective_batch=args.effective_batch, maximum_microbatch=args.microbatch,
    )
    from .inference import export_model
    export_model(selected, args.output / "teacher-export.pt")
    if args.local_validation:
        from .inference import predict
        artifact = torch.load(args.output / "teacher-export.pt", map_location="cpu", weights_only=False)
        design = read_json(args.dataset / validation.rows[0]["input_path"])
        restored_prediction = predict(artifact, [design], device="cuda")
        report = {"status": "passed", "mode": "full_epoch_execution_validation", "accuracy_gate_claimed": False,
                  "dataset_fingerprint": manifest["fingerprint"], "code_fingerprint": code_fingerprint(),
                  "device": torch.cuda.get_device_name(), "torch_version": torch.__version__, "precision": "bfloat16",
                  "model_config": selected["model_config"], "parameters": sum(p.numel() for p in teacher.parameters()),
                  "epochs": args.teacher_epochs, "train_rows": len(train), "validation_rows": len(validation),
                  "target_names": list(TARGET_NAMES), "validation": selected["validation"],
                  "checkpoint_reload_prediction": restored_prediction, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "checkpoint_sha256": file_sha256(args.output / "teacher-latest.pt")}
        atomic_write(args.output / "local-validation.json", report)
        print(f"Local full-epoch verification passed: {report}", flush=True)
        return
    gate = run_gate("teacher", teacher, selected, args.dataset, args.output, manifest, normalization)
    if not gate["passed"]:
        print("Teacher quality gate failed; distillation will not run.", flush=True)
        return
    teacher.eval().requires_grad_(False)
    student, selected_student = completed_stage("student", args.student_epochs) or run_stage(
        "student", train, validation, args.output, manifest, normalization, "cuda", teacher=teacher, resume=args.resume,
        compatible_resume_code_fingerprints=args.compatible_resume_code_fingerprint,
        epochs=args.student_epochs, effective_batch=args.effective_batch, maximum_microbatch=args.microbatch,
    )
    export_model(selected_student, args.output / "student-export.pt")
    run_gate("student", student, selected_student, args.dataset, args.output, manifest, normalization)


def run_gate(role, model, selected, dataset, output, manifest, normalization):
    identity = {"checkpoint_sha256": file_sha256(output / f"{role}-best.pt"), "role": role,
                "dataset_fingerprint": manifest["fingerprint"], "target_names": list(TARGET_NAMES),
                "label_policy": manifest.get("label_policy"),
                "metric_version": METRIC_VERSION, "selection_version": SELECTION_VERSION,
                "evaluation": evaluation_config(model, selected["microbatch"]),
                "normalization_sha256": fingerprint(asdict(normalization))}
    gate_path = output / f"{role}-gate.json"
    if gate_path.exists():
        gate = read_json(gate_path)
        if gate.get("identity") != identity or gate.get("validation") != selected["validation"]:
            raise ValueError("cached gate belongs to a different checkpoint, dataset, metric, or evaluation configuration")
        passed = quality_gate(gate["validation"]) and gate.get("test") is not None and quality_gate(gate["test"])
        if gate["passed"] != passed:
            raise ValueError("cached gate result disagrees with hit counts")
        if gate.get("test") is not None and file_sha256(output / f"{role}-test-predictions.json.gz") != gate["test_predictions_sha256"]:
            raise ValueError("cached gate prediction export differs")
        if gate.get("test") is not None:
            from .verification import verify_predictions
            verify_predictions(output / f"{role}-test-predictions.json.gz")
            if read_json(output / f"{role}-test-predictions.json.gz")["metrics"] != gate["test"]:
                raise ValueError("cached test metrics differ from verified export")
        return gate
    gate = {"identity": identity, "best_epoch": selected["epoch"], "validation": selected["validation"], "test": None, "passed": False}
    if quality_gate(selected["validation"]):
        test = Samples(dataset, read_json(dataset / manifest["split_files"]["test"]["path"]), normalization)
        prediction_path = output / f"{role}-test-predictions.json.gz"
        gate["test"] = evaluate(model, test, selected["microbatch"], prediction_path=prediction_path, identity=identity)
        from .verification import verify_predictions
        verify_predictions(prediction_path, test.rows)
        gate["test_predictions_sha256"] = file_sha256(prediction_path)
        gate["passed"] = quality_gate(gate["test"])
    atomic_write(gate_path, gate)
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA / "ready_for_train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-epochs", type=int, default=600)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=64, help="Maximum predictor microbatch; OOM retries preserve all rows")
    parser.add_argument("--local-validation", action="store_true", help="Verify complete teacher epochs and checkpoint export, without evaluating test or distilling")
    parser.add_argument("--teacher-early-stopping-patience", type=int, default=6)
    parser.add_argument("--teacher-early-stopping-min-epochs", type=int, default=30)
    parser.add_argument("--teacher-stop-on-validation-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compatible-resume-code-fingerprint", action="append", default=[])
    args = parser.parse_args(argv)
    if min(args.teacher_epochs, args.student_epochs, args.effective_batch, args.microbatch) < 1:
        parser.error("epochs and predictor batch sizes must be positive")
    if args.local_validation:
        args.teacher_stop_on_validation_gate = False
        args.teacher_early_stopping_patience = 0
    if args.teacher_early_stopping_patience < 0 or args.teacher_early_stopping_min_epochs < 1:
        parser.error("teacher early-stopping patience must be nonnegative and minimum epochs must be positive")
    class Tee:
        def __init__(self, console, log):
            self.console, self.log = console, log

        def write(self, value):
            self.log.write(value)
            return self.console.write(value)

        def flush(self):
            self.log.flush()
            self.console.flush()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.log").open("a", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args)
            except InterruptedError as error:
                print(error, flush=True)
                raise SystemExit(75)


if __name__ == "__main__":
    main()
