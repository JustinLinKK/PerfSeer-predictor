"""Local CUDA baseline training for six-target v3.2 SeerBlock teachers."""

import argparse
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from perfseer_v3.features import batch_graph_features, feature_layout
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json

from .native_data import SOURCES, TARGET_NAMES
from .native_training import loss, metrics, verify_metrics
from .seerblock_data import seerblock_feature_layout
from .seerblock_model import VERSION, SeerBlockPredictor, model_config, restore
from .training import rng_state, restore_rng


def to_batch(features, indices, device):
    batch = batch_graph_features([features[int(index)] for index in indices])
    return replace(batch, **{field.name: getattr(batch, field.name).to(device)
                             for field in fields(batch) if isinstance(getattr(batch, field.name), torch.Tensor)})


def validate_data(data):
    targets = torch.as_tensor(data["targets"], dtype=torch.float32)
    splits = torch.as_tensor(data["splits"], dtype=torch.long).reshape(-1)
    size = len(targets)
    if (targets.shape != (size, 6) or not size or not torch.isfinite(targets).all()
            or (targets[:, [0, 1, 2, 4, 5]] <= 0).any()
            or ((targets[:, 3] < 0) | (targets[:, 3] > 100)).any()):
        raise ValueError("invalid six-target dataset")
    if (splits.shape != (size,) or set(splits.tolist()) != {0, 1}
            or any(len(data[key]) != size for key in ("features", "sample_ids", "configuration_groups"))
            or len(set(data["sample_ids"])) != size or not data["manifest"].get("fingerprint")):
        raise ValueError("invalid dataset identities or split coverage")
    layouts = (feature_layout(OperationRegistry.load()), seerblock_feature_layout())
    memberships = {}
    for index, (feature, group, split) in enumerate(zip(data["features"], data["configuration_groups"], splits.tolist(), strict=True)):
        if feature.layout not in layouts or not len(feature.x_cont):
            raise ValueError(f"invalid feature layout or empty graph at row {index}")
        if group in memberships and memberships[group] != split:
            raise ValueError("configuration group leaks across partitions")
        memberships[group] = split
    data["targets"], data["splits"] = targets, splits
    return data


def training_loss(output, sm_logit, sm_positive, target):
    if sm_logit is None:
        return loss(output.prediction, target, "balanced_v2")
    positive = (target[:, 3] > 0).float()
    sm_loss = .1 * F.binary_cross_entropy_with_logits(sm_logit, positive)
    sm_loss = sm_loss + (((sm_positive - target[:, 3]) / 100).square() * positive).sum() / positive.sum().clamp_min(1)
    physical_sm = ((output.prediction[:, 3] - target[:, 3]) / 100).square().mean()
    return loss(output.prediction, target, "balanced_v2") + (sm_loss - physical_sm) / 3


def stopping_progress(report, best_error, stale, min_relative_improvement):
    value = report["selection_error"]
    if value < best_error and value <= best_error * (1 - min_relative_improvement):
        return value, 0
    return best_error, stale + 1


def stopping_reason(report, epoch, stale, elapsed_seconds, *, min_epochs, patience, max_minutes):
    if report["gate_passed"]:
        return "validation_gate_passed"
    if elapsed_seconds >= max_minutes * 60:
        return "wall_clock_budget"
    if epoch >= min_epochs and stale >= patience:
        return "no_meaningful_validation_improvement"
    return None


def train_batch(model, optimizer, data, indices, microbatch, device="cuda"):
    state = rng_state()
    retried = False
    while True:
        restore_rng(state)
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        batch = target = output = sm_logit = sm_positive = objective = None
        try:
            for begin in range(0, len(indices), microbatch):
                subset = indices[begin:begin + microbatch]
                batch, target = to_batch(data["features"], subset, device), data["targets"][subset].to(device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    output, sm_logit, sm_positive = model(batch, return_aux=True)
                    objective = training_loss(output, sm_logit, sm_positive, target) * (len(subset) / len(indices))
                objective.backward()
                value += float(objective.detach())
                batch = target = output = sm_logit = sm_positive = objective = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            break
        except torch.cuda.OutOfMemoryError:
            batch = target = output = sm_logit = sm_positive = objective = None
            optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch = max(1, microbatch // 2)
            retried = True
            torch.cuda.empty_cache()
    optimizer.step()
    return value, microbatch, retried


@torch.no_grad()
def evaluate(model, data, indices, batch_size, device="cuda"):
    model.eval()
    predictions, begin = [], 0
    while begin < len(indices):
        batch = None
        try:
            subset = indices[begin:begin + batch_size]
            batch = to_batch(data["features"], subset, device)
            # FP32 evaluation makes checkpoint replay independent of AMP kernels.
            predictions.append(model(batch).prediction.float().cpu())
            begin += len(subset)
        except torch.cuda.OutOfMemoryError:
            batch = None
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
    prediction = torch.cat(predictions)
    report = metrics(prediction, data["targets"][indices], "balanced_v2")
    verify_metrics(prediction.numpy(), data["targets"][indices].numpy(), report)
    return prediction, report


def verify_stage(data_path, output, *, replay=True, data=None):
    output = Path(output)
    receipt = read_json(output / "teacher-complete.json")
    if (receipt["checkpoint_sha256"] != file_sha256(output / "teacher-best.pt")
            or receipt["dataset_sha256"] != file_sha256(data_path)):
        raise ValueError("SeerBlock checkpoint or dataset changed")
    if data is None:
        data = validate_data(torch.load(data_path, map_location="cpu", weights_only=False))
    payload = torch.load(output / "teacher-best.pt", map_location="cpu", weights_only=False)
    if (payload["contract"]["dataset_fingerprint"] != data["manifest"]["fingerprint"]
            or payload["contract"]["dataset_sha256"] != receipt["dataset_sha256"]
            or payload["contract"]["source"] != receipt["source"]
            or payload["epoch"] != receipt["best_epoch"]
            or payload["optimizer_steps"] <= 0):
        raise ValueError("SeerBlock checkpoint contract or optimizer progress differs")
    validation_prediction = None
    for number, split in ((1, "validation"),):
        path = output / f"teacher-{split}.npz"
        if split == "test" and receipt["test"] is None:
            if receipt["validation"]["gate_passed"] or path.exists():
                raise ValueError("test release does not match validation gate")
            continue
        if file_sha256(path) != receipt[split + "_sha256"]:
            raise ValueError("SeerBlock predictions changed")
        indices = torch.nonzero(data["splits"] == number).flatten()
        with np.load(path, allow_pickle=False) as exported:
            if (not np.array_equal(exported["sample_ids"], np.asarray(data["sample_ids"])[indices.numpy()])
                    or not np.array_equal(exported["targets"], data["targets"][indices].numpy())):
                raise ValueError("SeerBlock predictions differ from sealed partition")
            verify_metrics(exported["predictions"], exported["targets"], receipt[split])
            if split == "validation":
                validation_prediction = torch.from_numpy(exported["predictions"].copy())
                selected_prediction = payload["validation_predictions"].float().cpu()
                verify_metrics(selected_prediction.numpy(), exported["targets"], payload["validation"])
                torch.testing.assert_close(validation_prediction, selected_prediction, rtol=3e-4, atol=1e-5)
    passed = receipt["validation"]["gate_passed"] and receipt["test"] is not None and receipt["test"]["gate_passed"]
    if passed != receipt["quality_gate_passed"]:
        raise ValueError("SeerBlock quality gate receipt differs")
    if replay:
        model = restore(payload).eval()
        indices = torch.nonzero(data["splits"] == 1).flatten()[:8]
        prediction, _ = evaluate(model, data, indices, 1, "cpu")
        torch.testing.assert_close(prediction, validation_prediction[:len(indices)], rtol=3e-4, atol=1e-5)
    return receipt


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "teacher-complete.json").exists():
        if not args.resume:
            raise ValueError("completed teacher exists; use --resume")
        return verify_stage(args.data, output)
    if not torch.cuda.is_available():
        raise RuntimeError("SeerBlock baseline training requires CUDA")
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_per_process_memory_fraction(.80)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = validate_data(torch.load(args.data, map_location="cpu", weights_only=False))
    if data["features"][0].layout != seerblock_feature_layout():
        raise ValueError("new training requires the epoch-26 teacher's 110-field input contract")
    train_indices = torch.nonzero(data["splits"] == 0).flatten()
    validation_indices = torch.nonzero(data["splits"] == 1).flatten()
    shape = model_config(OperationRegistry.load(), data["features"][0].layout)
    contract = {"dataset_fingerprint": data["manifest"]["fingerprint"], "dataset_sha256": file_sha256(args.data),
                "source": args.source, "seed": args.seed, "effective_batch": args.effective_batch,
                "learning_rate": args.learning_rate, "objective": "balanced_v2", "model_config": shape.to_dict(),
                "stopping_policy": {key: getattr(args, key) for key in
                                    ("epochs", "min_epochs", "patience", "min_relative_improvement", "max_minutes")},
                "code_sha256": fingerprint({name: file_sha256(Path(__file__).with_name(name))
                                             for name in ("seerblock_model.py", "seerblock_training.py")})}
    model = SeerBlockPredictor(shape, data["targets"][train_indices].median(0).values).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, min_lr=1e-6)
    latest, best = output / "teacher-latest.pt", output / "teacher-best.pt"
    epoch, cursor, steps_completed, best_error, stale = 1, 0, 0, math.inf, 0
    stopping_best_error, elapsed_before_resume = math.inf, 0.
    microbatch, microbatch_limit = min(args.microbatch, args.effective_batch), args.effective_batch
    if latest.exists():
        if not args.resume:
            raise ValueError("teacher checkpoint exists; use --resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("SeerBlock resume contract changed")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, cursor, steps_completed, best_error, stale, microbatch, microbatch_limit = (
            state[key] for key in ("next_epoch", "next_batch", "optimizer_steps", "best_error", "stale", "microbatch", "microbatch_limit"))
        stopping_best_error = state["stopping_best_error"]
        elapsed_before_resume = state["elapsed_training_seconds"]
        restore_rng(state["rng"])
        del state
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    started, last_save = time.monotonic(), time.monotonic()

    def event(value):
        value = {"time": time.time(), "source": args.source, "role": "teacher", **value}
        atomic_write(output / "teacher-progress.json", value)
        with (output / "teacher-events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)

    def payload(current, validation=None, validation_predictions=None, epoch_complete=None):
        return {"version": VERSION, "target_names": list(TARGET_NAMES), "model_config": shape.to_dict(),
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "contract": contract, "epoch": current, "validation": validation,
                "validation_predictions": validation_predictions, "epoch_complete": epoch_complete,
                "optimizer_steps": steps_completed, "training_device": torch.cuda.get_device_name()}

    def save(next_epoch, next_batch):
        atomic_write(latest, {**payload(next_epoch), "next_epoch": next_epoch, "next_batch": next_batch,
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                             "best_error": best_error, "stale": stale, "microbatch": microbatch,
                             "stopping_best_error": stopping_best_error,
                             "elapsed_training_seconds": elapsed_before_resume + time.monotonic() - started,
                             "microbatch_limit": microbatch_limit}, checkpoint=True)

    event({"event": "training_started", "parameters": sum(parameter.numel() for parameter in model.parameters()),
           "training_rows": len(train_indices), "validation_rows": len(validation_indices), "microbatch": microbatch,
           "hidden": shape.hidden, "seerblocks": shape.num_blocks, "gpu": torch.cuda.get_device_name()})
    current = epoch - 1
    epochs_completed, stop_reason = current, "maximum_epochs"
    try:
        for current in range(epoch, args.epochs + 1):
            generator = torch.Generator().manual_seed(args.seed + current)
            order = train_indices[torch.randperm(len(train_indices), generator=generator)]
            steps = math.ceil(len(order) / args.effective_batch)
            model.train()
            completed_epoch = True
            for step in range(cursor, steps):
                if stop[0]:
                    save(current, step)
                    raise InterruptedError("SeerBlock teacher checkpointed at optimizer boundary")
                if elapsed_before_resume + time.monotonic() - started >= args.max_minutes * 60:
                    completed_epoch = False
                    save(current, step)
                    break
                indices = order[step * args.effective_batch:(step + 1) * args.effective_batch]
                began = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                value, microbatch, retried = train_batch(model, optimizer, data, indices, microbatch)
                steps_completed += 1
                free, total = torch.cuda.mem_get_info()
                peak = torch.cuda.max_memory_reserved()
                event({"event": "optimizer_step", "epoch": current, "step": step + 1, "steps": steps,
                       "optimizer_steps": steps_completed, "loss": value, "microbatch": microbatch,
                       "seconds": time.monotonic() - began, "gpu_used_bytes": total - free,
                       "gpu_total_bytes": total, "peak_reserved_bytes": peak, "oom_retried": retried})
                if retried:
                    microbatch_limit = microbatch
                elif (microbatch < microbatch_limit and total - free < total * .70
                      and peak * 1.7 < free + torch.cuda.memory_reserved() - 2 ** 30):
                    microbatch = min(microbatch * 2, microbatch_limit)
                if stop[0] or step == 0 or time.monotonic() - last_save >= args.checkpoint_seconds:
                    save(current, step + 1)
                    last_save = time.monotonic()
                if stop[0]:
                    raise InterruptedError("SeerBlock teacher checkpointed at optimizer boundary")
            cursor = 0
            prediction, report = evaluate(model, data, validation_indices, args.evaluation_batch)
            if report["gate_passed"] or report["selection_error"] < best_error:
                best_error = report["selection_error"]
                atomic_write(best, payload(current, report, prediction, completed_epoch), checkpoint=True)
            stopping_best_error, stale = stopping_progress(report, stopping_best_error, stale, args.min_relative_improvement)
            scheduler.step(report["selection_error"])
            if completed_epoch:
                epochs_completed = current
                save(current + 1, 0)
            elapsed = elapsed_before_resume + time.monotonic() - started
            reason = stopping_reason(report, current, stale, elapsed, min_epochs=args.min_epochs,
                                     patience=args.patience, max_minutes=args.max_minutes)
            if reason is None and current == args.epochs:
                reason = "maximum_epochs"
            event({"event": "epoch_complete" if completed_epoch else "epoch_partial", "epoch": current, "validation": report,
                   "best_error": best_error, "stopping_best_error": stopping_best_error, "stale": stale,
                   "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_training_seconds": elapsed,
                   "stop_reason": reason})
            if reason is not None:
                stop_reason = reason
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    del optimizer, scheduler
    selected = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    prediction, validation = evaluate(model, data, validation_indices, args.evaluation_batch)
    verify_metrics(selected["validation_predictions"].numpy(), data["targets"][validation_indices].numpy(), selected["validation"])
    torch.testing.assert_close(prediction, selected["validation_predictions"], rtol=3e-4, atol=1e-5)
    receipt = {"status": "complete", "source": args.source, "checkpoint_sha256": file_sha256(best),
               "dataset_sha256": contract["dataset_sha256"], "dataset_fingerprint": contract["dataset_fingerprint"],
               "best_epoch": selected["epoch"], "epochs_completed": epochs_completed, "optimizer_steps": steps_completed,
               "best_epoch_complete": selected["epoch_complete"],
               "seconds": elapsed_before_resume + time.monotonic() - started, "stop_reason": stop_reason,
               "stopping_policy": contract["stopping_policy"], "validation": validation, "test": None,
               "quality_gate_passed": False, "evaluation_precision": "float32"}
    for number, split in ((1, "validation"),):
        if number == 2 and not validation["gate_passed"]:
            continue
        indices = torch.nonzero(data["splits"] == number).flatten()
        if number == 2:
            prediction, report = evaluate(model, data, indices, args.evaluation_batch)
        else:
            report = validation
        path = output / f"teacher-{split}.npz"
        np.savez_compressed(path, predictions=prediction.numpy(), targets=data["targets"][indices].numpy(),
                            sample_ids=np.asarray(data["sample_ids"])[indices.numpy()])
        receipt[split], receipt[split + "_sha256"] = report, file_sha256(path)
    receipt["quality_gate_passed"] = validation["gate_passed"] and receipt["test"] is not None and receipt["test"]["gate_passed"]
    del model, selected
    torch.cuda.empty_cache()
    atomic_write(output / "teacher-complete.json", receipt)
    verify_stage(args.data, output, data=data)
    atomic_write(output / "student-readiness.json", {"status": "ready" if receipt["quality_gate_passed"] else "skipped_teacher_gate",
                 "teacher_sha256": receipt["checkpoint_sha256"], "quality_gate_passed": receipt["quality_gate_passed"],
                 "student_trained": False, "target_names": list(TARGET_NAMES)})
    event({"event": "teacher_complete", "quality_gate_passed": receipt["quality_gate_passed"], "best_epoch": receipt["best_epoch"]})
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-relative-improvement", type=float, default=.01)
    parser.add_argument("--max-minutes", type=float, default=45.)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--evaluation-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--checkpoint-seconds", type=float, default=120.)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if (any(getattr(args, key) < 1 for key in ("epochs", "min_epochs", "patience", "microbatch", "effective_batch", "evaluation_batch", "cpu_threads"))
            or args.min_epochs > args.epochs or not 0 <= args.min_relative_improvement < 1
            or not math.isfinite(args.max_minutes) or args.max_minutes <= 0
            or not 0 < args.learning_rate < 1 or not 0 < args.checkpoint_seconds <= 300):
        parser.error("positive training budgets, min_epochs <= epochs, improvement in [0,1), and checkpoint interval <=300 seconds required")
    result = verify_stage(args.data, args.output) if args.verify else run(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
