"""Fail-closed A10 teacher training and gated student distillation on A100."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict
from dataclasses import asdict
import math
from pathlib import Path
import random
import signal
import sys

import numpy as np
import torch

from perfseer_v3.features import NormalizationBlock, NormalizationStatsV3, _transform, apply_normalization
from perfseer_v3.op_registry import OperationRegistry

from .dataset import DATA, verify
from .features import build_features
from .io import atomic_write, file_sha256, fingerprint, read_json
from .model import SeerNetV31, SeerNetV31Config
from .training import (T1, S1, build_optimizers, build_schedulers, checkpoint_payload,
                       evaluate, restore_model, restore_rng, rng_state, selection_key, train_batch)
from .version import TARGET_NAMES


def code_fingerprint():
    package = Path(__file__).resolve().parent
    return fingerprint({name: file_sha256(package / name) for name in
                        ("model.py", "features.py", "features_core.py", "training.py", "runner.py", "version.py")})


def early_stopping_reason(metrics, epoch, best_key, *, patience=0, min_epochs=0, stop_on_gate=False):
    if stop_on_gate and metrics["gate_passed"]:
        return "validation_gate_passed"
    if patience and epoch >= min_epochs and epoch - int(best_key[2]) >= patience:
        return f"no_validation_improvement_for_{patience}_epochs"
    return None


class Samples:
    def __init__(self, root, rows, normalization=None, cache_root=None):
        self.root, self.rows, self.normalization = Path(root), rows, normalization
        self.cache = OrderedDict()
        self.cache_root = Path(cache_root) if cache_root else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        key = row["input_path"]
        if key not in self.cache:
            cache_path = self.cache_root / f"{row['input_sha256']}.pt" if self.cache_root else None
            if cache_path and cache_path.exists():
                features = torch.load(cache_path, map_location="cpu", weights_only=False)
                features.validate()
            else:
                features = build_features(read_json(self.root / key))
                if cache_path:
                    atomic_write(cache_path, features, checkpoint=True)
            self.cache[key] = apply_normalization(features, self.normalization) if self.normalization else features
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return {"features": self.cache[key], "target": torch.tensor([row["targets"][name] for name in TARGET_NAMES]),
                "sample_id": row["sample_id"]}


def fit_training_normalization(samples, dataset_fingerprint):
    """Streaming full-training moments, with min/max clipping and bounded RAM."""
    layout = samples[0]["features"].layout
    blocks = [("x_cont", layout.node_continuous_fields), ("edge_cont", layout.edge_continuous_fields),
              ("u_cont", layout.global_continuous_fields)]
    states = [[0, np.zeros(len(names)), np.zeros(len(names)), np.full(len(names), np.inf),
               np.full(len(names), -np.inf)] for _, names in blocks]
    for index in range(len(samples)):
        features = samples[index]["features"]
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
            print(f"normalization: {index}/{len(samples)} training rows", flush=True)
    fitted = []
    for count, mean, m2, low, high in states:
        if not count:
            raise ValueError("empty training feature block")
        std = np.sqrt(m2 / count)
        fitted.append(NormalizationBlock(tuple(mean), tuple(np.where(std < 1e-8, 1., std)), tuple(low), tuple(high)))
    return NormalizationStatsV3(layout.feature_schema_sha256, layout.operator_registry_sha256, layout.layout_sha256,
                                "train", dataset_fingerprint, (0., 1.), *fitted,
                                normalization_version="perfseer_v31_train_moments_v1")


def normalization_from_dict(value):
    return NormalizationStatsV3(**{**value, "quantiles": tuple(value["quantiles"]),
                                  **{name: NormalizationBlock(**{key: tuple(items) for key, items in value[name].items()})
                                     for name in ("node", "edge", "global_features")}})


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
              compatible_resume_code_fingerprints=()):
    epochs, warmup, lr = (600, 10, 5e-4) if role == "teacher" else (100, 5, 1e-3)
    latest, best = output / f"{role}-latest.pt", output / f"{role}-best.pt"
    progress = None
    if resume and latest.exists():
        progress = torch.load(latest, map_location="cpu", weights_only=False)
        checkpoint_fingerprint = progress.get("code_fingerprint")
        if progress["normalization"] != asdict(normalization) or progress["role"] != role:
            raise ValueError("resume normalization or role mismatch")
        if checkpoint_fingerprint not in {code_fingerprint(), *compatible_resume_code_fingerprints}:
            raise ValueError("resume code fingerprint mismatch")
        model = restore_model(progress, dataset_fingerprint=manifest["fingerprint"], device=device)
    else:
        if latest.exists() or best.exists():
            raise ValueError("output contains checkpoints; use --resume or a fresh output directory")
        targets = torch.tensor([[row["targets"][name] for name in TARGET_NAMES] for row in train.rows])
        scales = targets.median(0).values
        scales[1] = 100
        config = SeerNetV31Config.from_registry(OperationRegistry.load(), train[0]["features"].layout,
                                               **(T1 if role == "teacher" else S1))
        model = SeerNetV31(config, scales).to(device)
    microbatch = progress["microbatch"] if progress else memory_probe(model, train, lr, teacher=teacher)
    optimizers, partition = build_optimizers(model, lr)
    steps_per_epoch = math.ceil(len(train) / 256)
    schedulers = build_schedulers(optimizers, epochs=epochs, warmup_epochs=warmup, steps_per_epoch=steps_per_epoch)
    if progress:
        for optimizer, state in zip(optimizers, progress["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, progress["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        restore_rng(progress["rng"])
    start_epoch = progress["next_epoch"] if progress else 1
    cursor = progress["next_batch"] if progress else 0
    best_key = tuple(progress["best_key"]) if progress else (math.inf, math.inf, 0)
    atomic_write(output / f"{role}-optimizer-partition.json", partition)
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(path, epoch, next_epoch, next_batch, metrics=None):
        atomic_write(path, checkpoint_payload(model, role=role, epoch=epoch, normalization=asdict(normalization),
                     dataset_fingerprint=manifest["fingerprint"], optimizers=optimizers, schedulers=schedulers,
                     next_epoch=next_epoch, next_batch=next_batch, best_key=best_key,
                     microbatch=microbatch, effective_batch=256, code_fingerprint=code_fingerprint(),
                     validation=metrics), checkpoint=True)

    try:
        for epoch in range(start_epoch, epochs + 1):
            order = list(range(len(train)))
            random.Random(42 + epoch).shuffle(order)
            model.train()
            for step in range(cursor, steps_per_epoch):
                logical = [train[i] for i in order[step * 256:(step + 1) * 256]]
                loss, microbatch = train_batch(model, logical, optimizers, microbatch, teacher=teacher)
                for scheduler in schedulers:
                    scheduler.step()
                print(f"{role} epoch={epoch}/{epochs} step={step + 1}/{steps_per_epoch} loss={loss:.6f} microbatch={microbatch}", flush=True)
                if stop[0]:
                    save(latest, epoch, epoch, step + 1)
                    raise InterruptedError("checkpoint saved at optimizer-step boundary")
            cursor = 0
            metrics = evaluate(model, validation, microbatch)
            key = selection_key(metrics["errors"], epoch)
            if key < best_key:
                best_key = key
                save(best, epoch, epoch + 1, 0, metrics)
            save(latest, epoch, epoch + 1, 0, metrics)
            atomic_write(output / f"{role}-epoch-{epoch:04d}.json", {"epoch": epoch, "validation": metrics, "best_key": best_key})
            print(f"{role} epoch={epoch} validation={metrics} best_epoch={best_key[2]}", flush=True)
            reason = early_stopping_reason(metrics, epoch, best_key, patience=early_stopping_patience,
                                           min_epochs=early_stopping_min_epochs, stop_on_gate=stop_on_gate)
            if reason:
                print(f"{role} early_stop epoch={epoch} reason={reason} best_epoch={best_key[2]}", flush=True)
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
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("the production workflow requires an A100 CUDA device")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    manifest = read_json(args.dataset / "dataset_manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.output / "feature-cache" / fingerprint([manifest["fingerprint"], file_sha256(Path(__file__).with_name("features.py"))])
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
    teacher, selected = run_stage(
        "teacher", train, validation, args.output, manifest, normalization, "cuda", resume=args.resume,
        early_stopping_patience=args.teacher_early_stopping_patience,
        early_stopping_min_epochs=args.teacher_early_stopping_min_epochs,
        stop_on_gate=args.teacher_stop_on_validation_gate,
        compatible_resume_code_fingerprints=args.compatible_resume_code_fingerprint,
    )
    from .inference import export_model
    export_model(selected, args.output / "teacher-export.pt")
    gate_path = args.output / "teacher-gate.json"
    if gate_path.exists():
        gate = read_json(gate_path)
        if gate["checkpoint_sha256"] != file_sha256(args.output / "teacher-best.pt"):
            raise ValueError("test gate belongs to a different teacher checkpoint")
    else:
        gate = {"checkpoint_sha256": file_sha256(args.output / "teacher-best.pt"), "best_epoch": selected["epoch"],
                "validation": selected["validation"], "test": None, "passed": False}
        if selected["validation"]["gate_passed"]:
            test = Samples(args.dataset, read_json(args.dataset / manifest["split_files"]["test"]["path"]), normalization)
            gate["test"] = evaluate(teacher, test, selected["microbatch"])
            gate["passed"] = gate["test"]["gate_passed"]
        atomic_write(gate_path, gate)
    if not gate["passed"]:
        print("Teacher quality gate failed; distillation will not run.", flush=True)
        return
    teacher.eval().requires_grad_(False)
    student, selected_student = run_stage(
        "student", train, validation, args.output, manifest, normalization, "cuda", teacher=teacher, resume=args.resume,
        compatible_resume_code_fingerprints=args.compatible_resume_code_fingerprint,
    )
    export_model(selected_student, args.output / "student-export.pt")
    report = args.output / "student-test.json"
    if not report.exists():
        test = Samples(args.dataset, read_json(args.dataset / manifest["split_files"]["test"]["path"]), normalization)
        atomic_write(report, {"best_epoch": selected_student["epoch"], "test": evaluate(student, test, selected_student["microbatch"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA / "ready_for_train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--teacher-early-stopping-patience", type=int, default=6)
    parser.add_argument("--teacher-early-stopping-min-epochs", type=int, default=30)
    parser.add_argument("--teacher-stop-on-validation-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compatible-resume-code-fingerprint", action="append", default=[])
    args = parser.parse_args()
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
