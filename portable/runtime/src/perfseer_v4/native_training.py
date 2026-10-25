"""Local, sequential native-data teachers and quality-gated students."""

import argparse
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .native_data import FEATURE_DIM, HEAD_GROUPS, SOURCES, SPLIT_POLICIES, TARGET_NAMES, prepare, verify
from .native_model import VERSION, PROPOSALS, NativePredictor, model_config, restore


def default_config(data, output, *, smoke=False):
    return {"version": "perfseer_v4_native_source_campaign_v1", "data": str(Path(data).resolve()),
            "output": str(Path(output).resolve()), "sources": list(SOURCES), "proposals": list(PROPOSALS),
            "device": "cuda", "smoke": smoke, "seed": 42, "effective_batch": 256, "model_revision": 2,
            "microbatch": 16, "evaluation_batch": 32, "adaptive_microbatch": True,
            "memory_fraction": .80, "checkpoint_seconds": 120, "cpu_threads": 4,
            "maximum_training_processes": 1, "teacher_gate": {"within_relative_error": .05,
            "minimum_fraction": .95, "splits": ["validation", "test"], "targets": list(TARGET_NAMES)},
            "stages": {"teacher": {"max_epochs": 600, "min_epochs": 30, "patience": 20, "learning_rate": .0005},
                       "resource_teacher": {"max_epochs": 300, "min_epochs": 20, "patience": 15, "learning_rate": .001},
                       "student": {"max_epochs": 300, "min_epochs": 20, "patience": 15, "learning_rate": .001}}}


def validate_config(config):
    reference = default_config(config["data"], config["output"], smoke=config.get("smoke", False))
    for key in ("version", "sources", "proposals", "device", "maximum_training_processes", "teacher_gate"):
        if config.get(key) != reference[key]:
            raise ValueError(f"native source campaign contract differs: {key}")
    for key in ("effective_batch", "microbatch", "evaluation_batch", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    if not 0 < config["memory_fraction"] <= .85 or not 0 < config["checkpoint_seconds"] <= 300:
        raise ValueError("invalid GPU headroom/checkpoint interval")
    if config.get("model_revision", 1) not in (1, 2):
        raise ValueError("unsupported native model revision")
    return config


def loss(prediction, target, objective="relative_v1"):
    if objective == "balanced_v2":
        # Log ratios treat relative over/under-prediction symmetrically. SM uses
        # its physical range so zero measurements cannot dominate the objective.
        timing = (prediction[:, :3].float().clamp_min(1e-6).log() - target[:, :3].float().log()).square().mean()
        memory = (prediction[:, 4:].float().clamp_min(1e-6).log() - target[:, 4:].float().log()).square().mean()
        sm = ((prediction[:, 3].float() - target[:, 3].float()) / 100).square().mean()
        return (timing + sm + memory) / 3
    denominator = target.float().clone()
    denominator[:, 3].clamp_min_(1.)
    errors = (prediction.float() - target.float()).abs() / denominator
    return torch.stack([errors[:, indices].mean() for indices in HEAD_GROUPS]).mean()


def metrics(prediction, target, objective="relative_v1"):
    prediction, target = prediction.detach().float().cpu(), target.detach().float().cpu()
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 6 or not len(target):
        raise ValueError("six-target nonempty predictions required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite native predictions/targets")
    relative = (prediction.double() - target.double()).abs() / target.double().abs().clamp_min(1e-6)
    counts = (relative <= .05).sum(0).tolist()
    return {**({"objective": objective} if objective != "relative_v1" else {}),
            "target_names": list(TARGET_NAMES), "rows": len(target), "within_5pct_count": counts,
            "within_5pct_accuracy": [n / len(target) for n in counts],
            "relative_mean_error": relative.mean(0).tolist(), "selection_error": float(loss(prediction, target, objective)),
            "gate_passed": all(20 * n >= 19 * len(target) for n in counts)}


def verify_metrics(prediction, target, report):
    """Recompute exported gates and the training loss independently with NumPy."""
    p, y = np.asarray(prediction, dtype=np.float32).astype(np.float64), np.asarray(target, dtype=np.float32).astype(np.float64)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 6 or not len(y) or not np.isfinite(p).all():
        raise ValueError("invalid native prediction report")
    relative = abs(p - y) / np.maximum(abs(y), 1e-6)
    counts = (relative <= .05).sum(0).tolist()
    denominator = y.copy(); denominator[:, 3] = np.maximum(denominator[:, 3], 1.)
    expected_loss = np.mean([np.mean((abs(p - y) / denominator)[:, indices]) for indices in HEAD_GROUPS])
    if report.get("objective") == "balanced_v2":
        expected_loss = np.mean([np.mean(np.log(np.maximum(p[:, :3], 1e-6) / y[:, :3]) ** 2),
                                 np.mean(((p[:, 3] - y[:, 3]) / 100) ** 2),
                                 np.mean(np.log(np.maximum(p[:, 4:], 1e-6) / y[:, 4:]) ** 2)])
    if (report["target_names"] != list(TARGET_NAMES) or report["rows"] != len(y)
            or report["within_5pct_count"] != counts
            or report["gate_passed"] != all(20 * n >= 19 * len(y) for n in counts)
            or not np.allclose(report["within_5pct_accuracy"], np.array(counts) / len(y), rtol=0, atol=1e-12)
            or not np.allclose(report["relative_mean_error"], relative.mean(0), rtol=1e-8, atol=1e-8)
            or not math.isclose(report["selection_error"], expected_loss, rel_tol=2e-6, abs_tol=1e-7)):
        raise ValueError("independent native prediction verification failed")


def split_indices(data, split, smoke):
    indices = np.flatnonzero(data["splits"] == split)
    return indices[:8] if smoke else indices


def stage_paths(config, source, proposal, role):
    root = Path(config["output"]) / source / proposal
    return root, root / f"{role}-best.pt", root / f"{role}-latest.pt", root / f"{role}-complete.json"


def code_identity():
    root = Path(__file__).parent
    return fingerprint({name: file_sha256(root / name) for name in ("native_data.py", "native_model.py", "native_training.py")})


def verify_stage(config, source, proposal, role, data=None):
    root, best, _, receipt_path = stage_paths(config, source, proposal, role)
    receipt = read_json(receipt_path)
    manifest = read_json(Path(config["data"]) / source / "dataset_manifest.json")
    if (receipt["config_sha256"] != fingerprint(config) or receipt["checkpoint_sha256"] != file_sha256(best)
            or receipt["dataset_fingerprint"] != manifest["fingerprint"]
            or (receipt["source"], receipt["proposal"], receipt["role"]) != (source, proposal, role)):
        raise ValueError("native stage identity/checksum differs")
    owned = data is None
    if owned:
        data = np.load(Path(config["data"]) / source / "dataset.npz", allow_pickle=False)
    try:
        for number, split in ((1, "validation"), (2, "test")):
            path = root / f"{role}-{split}.npz"
            if file_sha256(path) != receipt[split + "_sha256"]:
                raise ValueError("native prediction file changed")
            with np.load(path, allow_pickle=False) as report:
                indices = split_indices(data, number, config["smoke"])
                if (not np.array_equal(report["sample_ids"], data["sample_ids"][indices]) or
                        not np.array_equal(report["targets"], data["targets"][indices])):
                    raise ValueError("native predictions differ from sealed split")
                verify_metrics(report["predictions"], report["targets"], receipt[split])
        passed = all(receipt[split]["gate_passed"] for split in ("validation", "test"))
        if passed != receipt["quality_gate_passed"]:
            raise ValueError("teacher/student gate receipt differs")
        if role == "student" and receipt["teacher_sha256"] != file_sha256(root / "teacher-best.pt"):
            raise ValueError("student belongs to a different teacher")
    finally:
        if owned:
            data.close()
    return receipt


def teacher_gate(config, source, proposal, data=None):
    receipt = verify_stage(config, source, proposal, "teacher", data)
    return {"passed": receipt["quality_gate_passed"], "teacher_sha256": receipt["checkpoint_sha256"],
            "validation": receipt["validation"], "test": receipt["test"], "definition": config["teacher_gate"]}


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def train_batch(model, teacher, optimizer, x, y, indices, microbatch):
    state = _rng()
    retried = False
    while True:
        _restore_rng(state)
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        prediction = teaching = target = batch = objective = hard = soft = soft_errors = denominator = sm_logit = sm_positive = sm_loss = positive = None
        try:
            for begin in range(0, len(indices), microbatch):
                subset = indices[begin:begin + microbatch]
                batch, target = x[subset].cuda(), y[subset].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    revised = model.config.get("objective") == "balanced_v2"
                    if revised:
                        prediction, sm_logit, sm_positive = model(batch, return_aux=True)
                        # Train zero occurrence separately from utilization when
                        # active; the positive branch keeps gradients at zero output.
                        positive = (target[:, 3] > 0).float()
                        sm_loss = torch.nn.functional.binary_cross_entropy_with_logits(sm_logit, positive)
                        sm_loss = .1 * sm_loss + (((sm_positive - target[:, 3]) / 100).square() * positive).sum() / positive.sum().clamp_min(1)
                        hard = loss(prediction, target, "balanced_v2")
                        hard = hard + (sm_loss - ((prediction[:, 3] - target[:, 3]) / 100).square().mean()) / 3
                    else:
                        prediction = model(batch)
                        hard = loss(prediction, target)
                    if teacher is not None:
                        with torch.no_grad():
                            teaching = teacher(batch)
                        denominator = target.clone(); denominator[:, 3].clamp_min_(1.)
                        soft_errors = (prediction - teaching).abs() / denominator
                        soft = loss(prediction, teaching, "balanced_v2") if revised else torch.stack([soft_errors[:, group].mean() for group in HEAD_GROUPS]).mean()
                        objective = .6 * hard + .4 * soft
                    else:
                        objective = hard
                objective = objective * (len(subset) / len(indices))
                objective.backward()
                value += float(objective.detach())
                prediction = teaching = target = batch = objective = hard = soft = soft_errors = denominator = sm_logit = sm_positive = sm_loss = positive = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            break
        except torch.cuda.OutOfMemoryError:
            prediction = teaching = target = batch = objective = hard = soft = soft_errors = denominator = sm_logit = sm_positive = sm_loss = positive = None
            optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch = max(1, microbatch // 2)
            retried = True
            torch.cuda.empty_cache()
    optimizer.step()
    return value, microbatch, retried


@torch.no_grad()
def evaluate(model, x, y, indices, batch_size):
    model.eval()
    predictions = []
    for begin in range(0, len(indices), batch_size):
        batch = x[indices[begin:begin + batch_size]].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions.append(model(batch).float().cpu())
    prediction = torch.cat(predictions)
    report = metrics(prediction, y[indices], model.config.get("objective", "relative_v1"))
    verify_metrics(prediction.numpy(), y[indices].numpy(), report)
    return prediction, report


def run_stage(config, source, proposal, role, *, resume=False, validation_only=False):
    validate_config(config)
    if source not in SOURCES or proposal not in PROPOSALS or role not in {"teacher", "student"}:
        raise ValueError("invalid native source task")
    root, best, latest, complete = stage_paths(config, source, proposal, role)
    root.mkdir(parents=True, exist_ok=True)
    if complete.exists():
        if not resume:
            raise ValueError("completed native stage exists; use --resume")
        return verify_stage(config, source, proposal, role)
    manifest = verify(Path(config["data"]) / source)
    data = np.load(Path(config["data"]) / source / "dataset.npz", allow_pickle=False)
    gate = teacher_gate(config, source, proposal, data) if role == "student" else None
    if gate is not None and not gate["passed"]:
        result = {"status": "skipped_teacher_gate", "source": source, "proposal": proposal, "role": role,
                  "config_sha256": fingerprint(config), "gate": gate}
        atomic_write(root / "student-skipped.json", result)
        data.close()
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("native predictor training requires CUDA")
    torch.set_num_threads(config["cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    x, y = torch.from_numpy(data["features"]), torch.from_numpy(data["targets"])
    train_indices = split_indices(data, 0, config["smoke"])
    val_indices = split_indices(data, 1, config["smoke"])
    total = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    squares = torch.zeros_like(total)
    for begin in range(0, len(train_indices), 1024):
        values = x[train_indices[begin:begin + 1024]].double()
        total += values.sum(0); squares += values.square().sum(0)
    mean = (total / len(train_indices)).float()
    scale = (squares / len(train_indices) - (total / len(train_indices)).square()).clamp_min(0).sqrt().float()
    scale = torch.where(scale < 1e-6, 1., scale)
    medians = y[train_indices].median(0).values
    shape = model_config(proposal, role, smoke=config["smoke"], revision=config.get("model_revision", 1))
    contract = {"config_sha256": fingerprint(config), "dataset_fingerprint": manifest["fingerprint"],
                "source": source, "proposal": proposal, "role": role, "code_sha256": code_identity(),
                "teacher_sha256": gate["teacher_sha256"] if gate else None}
    model = NativePredictor(shape, mean, scale, medians).cuda()
    teacher = restore(torch.load(root / "teacher-best.pt", map_location="cpu", weights_only=False), "cuda").eval().requires_grad_(False) if gate else None
    schedule = config["stages"]["resource_teacher" if proposal == "v4.3" and role == "teacher" else role]
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5, min_lr=1e-6)
    epoch, cursor, best_error, anchor, stale = 1, 0, math.inf, math.inf, 0
    microbatch = min(config["microbatch"], config["effective_batch"])
    microbatch_limit = config["effective_batch"]
    if latest.exists():
        if not resume:
            raise ValueError("native checkpoint exists; use --resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("native resume contract changed")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, cursor, best_error, anchor, stale = (state[k] for k in ("next_epoch", "next_batch", "best_error", "anchor", "stale"))
        microbatch, microbatch_limit = state["microbatch"], state["microbatch_limit"]
        _restore_rng(state["rng"])
    stopped = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopped.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}
    last_save = time.monotonic()
    def event(value):
        value = {"time": time.time(), "source": source, "proposal": proposal, "role": role, **value}
        atomic_write(root / f"{role}-progress.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)
    def payload(current, validation=None):
        return {"version": VERSION, "target_names": list(TARGET_NAMES), "model_config": shape,
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "target_medians": medians, "contract": contract, "epoch": current,
                "validation": validation, "prediction_hardware": manifest["prediction_hardware"],
                "training_device": torch.cuda.get_device_name()}
    def save(next_epoch, next_batch):
        state = {**payload(next_epoch), "next_epoch": next_epoch, "next_batch": next_batch, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": _rng(), "best_error": best_error, "anchor": anchor,
                 "stale": stale, "microbatch": microbatch, "microbatch_limit": microbatch_limit}
        atomic_write(latest, state, checkpoint=True)
    event({"event": "training_started", "parameters": sum(p.numel() for p in model.parameters()),
           "training_rows": len(train_indices), "gpu": torch.cuda.get_device_name(), "microbatch": microbatch})
    max_epochs = 2 if config["smoke"] else schedule["max_epochs"]
    if epoch > schedule["min_epochs"] and stale >= schedule["patience"]:
        max_epochs = min(max_epochs, epoch - 1)
    current = epoch - 1
    try:
        for current in range(epoch, max_epochs + 1):
            generator = torch.Generator().manual_seed(config["seed"] + current)
            order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
            steps = math.ceil(len(order) / config["effective_batch"])
            model.train()
            for step in range(cursor, steps):
                if stopped[0]:
                    save(current, step)
                    raise InterruptedError("native training checkpointed before optimizer step")
                indices = order[step * config["effective_batch"]:(step + 1) * config["effective_batch"]]
                began = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                value, microbatch, retried = train_batch(model, teacher, optimizer, x, y, indices, microbatch)
                free, total_bytes = torch.cuda.mem_get_info()
                peak = torch.cuda.max_memory_reserved()
                event({"event": "optimizer_step", "epoch": current, "step": step + 1, "steps": steps,
                       "loss": value, "microbatch": microbatch, "seconds": time.monotonic() - began,
                       "gpu_used_bytes": total_bytes - free, "gpu_total_bytes": total_bytes,
                       "peak_reserved_bytes": peak, "oom_retried": retried})
                if retried:
                    microbatch_limit = microbatch
                elif (config["adaptive_microbatch"] and microbatch < microbatch_limit and
                      total_bytes - free < total_bytes * .70 and peak * 1.7 < free + torch.cuda.memory_reserved() - 2 ** 30):
                    microbatch = min(microbatch * 2, microbatch_limit)
                if stopped[0] or step == 0 or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                    save(current, step + 1); last_save = time.monotonic()
                if stopped[0]:
                    raise InterruptedError("native training checkpointed at optimizer boundary")
            cursor = 0
            _, report = evaluate(model, x, y, val_indices, config["evaluation_batch"])
            if report["selection_error"] < best_error:
                best_error = report["selection_error"]
                atomic_write(best, payload(current, report), checkpoint=True)
            if report["selection_error"] < anchor * .999:
                anchor, stale = report["selection_error"], 0
            else:
                stale += 1
            scheduler.step(report["selection_error"])
            save(current + 1, 0); last_save = time.monotonic()
            event({"event": "epoch_complete", "epoch": current, "validation": report, "stale_epochs": stale,
                   "learning_rate": optimizer.param_groups[0]["lr"]})
            if current >= schedule["min_epochs"] and stale >= schedule["patience"]:
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    del optimizer, scheduler, model, teacher
    torch.cuda.empty_cache()
    selected = torch.load(best, map_location="cpu", weights_only=False)
    model = restore(selected, "cuda")
    receipt = {"status": "complete", **contract, "checkpoint_sha256": file_sha256(best), "best_epoch": selected["epoch"],
               "epochs_completed": current, "gpu": torch.cuda.get_device_name(), "microbatch": microbatch}
    evaluation_splits = ((1, "validation"),) if validation_only else ((1, "validation"), (2, "test"))
    for number, split in evaluation_splits:
        indices = split_indices(data, number, config["smoke"])
        prediction, report = evaluate(model, x, y, indices, config["evaluation_batch"])
        path = root / f"{role}-{split}.npz"
        np.savez_compressed(path, predictions=prediction.numpy(), targets=y[indices].numpy(), sample_ids=data["sample_ids"][indices])
        receipt[split], receipt[split + "_sha256"] = report, file_sha256(path)
    if receipt["validation"] != selected["validation"]:
        raise ValueError("selected native checkpoint replay differs")
    if validation_only:
        receipt["status"] = "validation_complete"
        receipt["quality_gate_passed"] = False
        atomic_write(root / f"{role}-validation-complete.json", receipt)
        data.close()
        event({"event": "validation_complete", "best_epoch": selected["epoch"], "validation": receipt["validation"]})
        return receipt
    receipt["quality_gate_passed"] = all(receipt[split]["gate_passed"] for split in ("validation", "test"))
    atomic_write(complete, receipt)
    verify_stage(config, source, proposal, role, data)
    data.close()
    event({"event": "stage_complete", "quality_gate_passed": receipt["quality_gate_passed"], "best_epoch": selected["epoch"]})
    return receipt


def verify_campaign(config):
    validate_config(config)
    outcomes = []
    for source in SOURCES:
        verify(Path(config["data"]) / source)
        with np.load(Path(config["data"]) / source / "dataset.npz", allow_pickle=False) as data:
            for proposal in PROPOSALS:
                teacher = verify_stage(config, source, proposal, "teacher", data)
                gate = teacher_gate(config, source, proposal, data)
                root, _, _, _ = stage_paths(config, source, proposal, "student")
                if gate["passed"]:
                    student = verify_stage(config, source, proposal, "student", data)
                else:
                    student = read_json(root / "student-skipped.json")
                    if (student["status"] != "skipped_teacher_gate" or student["gate"] != gate
                            or student["config_sha256"] != fingerprint(config) or (root / "student-complete.json").exists()):
                        raise ValueError("student skip does not match teacher gate")
                outcomes.append({"source": source, "proposal": proposal, "teacher_gate_passed": gate["passed"],
                                 "teacher_sha256": teacher["checkpoint_sha256"], "student_status": student["status"]})
    return {"status": "smoke_complete" if config["smoke"] else "complete", "teacher_count": len(outcomes),
            "student_count": sum(r["student_status"] == "complete" for r in outcomes), "outcomes": outcomes,
            "transfer_performed": False, "deployment_approved": False}


def run(config, config_path, *, resume=False):
    import fcntl
    validate_config(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".native-campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher = Path(__file__).resolve().parents[2] / "scripts/run_perfseer_v4.py"
        for source in SOURCES:
            for proposal in PROPOSALS:
                for role in ("teacher", "student"):
                    command = [sys.executable, "-u", str(launcher), "native-stage", "--config", str(config_path),
                               "--source", source, "--proposal", proposal, "--role", role]
                    if resume:
                        command.append("--resume")
                    subprocess.run(command, check=True)
        result = verify_campaign(config)
        atomic_write(output / "completion.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare-native-data")
    prepare_command.add_argument("--repo", type=Path, default=Path.cwd())
    prepare_command.add_argument("--output", type=Path, required=True)
    prepare_command.add_argument("--static-modules", action="store_true")
    prepare_command.add_argument("--split-policy", choices=SPLIT_POLICIES, default="held-out-families")
    config_command = commands.add_parser("native-campaign-config")
    config_command.add_argument("--data", type=Path, required=True)
    config_command.add_argument("--output", type=Path, required=True)
    config_command.add_argument("--save", type=Path, required=True)
    config_command.add_argument("--smoke", action="store_true")
    for name in ("native-campaign", "native-stage", "verify-native-campaign"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "verify-native-campaign":
            command.add_argument("--resume", action="store_true")
        if name == "native-stage":
            command.add_argument("--validation-only", action="store_true")
            command.add_argument("--source", choices=SOURCES, required=True)
            command.add_argument("--proposal", choices=PROPOSALS, required=True)
            command.add_argument("--role", choices=("teacher", "student"), required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-native-data":
        result = prepare(args.repo, args.output, static_modules=args.static_modules, split_policy=args.split_policy)
    elif args.command == "native-campaign-config":
        result = default_config(args.data, args.output, smoke=args.smoke)
        atomic_write(args.save, validate_config(result))
    else:
        config = validate_config(read_json(args.config))
        if args.command == "native-stage":
            result = run_stage(config, args.source, args.proposal, args.role, resume=args.resume,
                               validation_only=args.validation_only)
        elif args.command == "verify-native-campaign":
            result = verify_campaign(config)
        else:
            result = run(config, args.config.resolve(), resume=args.resume)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result
