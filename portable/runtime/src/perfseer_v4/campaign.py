"""Four-stage independent teacher/student campaigns with sealed pair artifacts."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from perfseer_v3.op_registry import OperationRegistry
from perfseer_v31.io import atomic_write, file_sha256, fingerprint, read_json
from .campaign_data import CampaignSamples, LogicalBatch, load_rows, verify_dataset
from .campaign_objective import VERSION as OBJECTIVE_VERSION, head_weights, loss, distill, metrics, target_tensors, verify_prediction_report
from .model import SeerNetV4, SeerNetV4Config
from .resource_model import ResourceMLP, ResourceMLPConfig
from .runner import code_fingerprint, fit_training_normalization, normalization_from_dict, TrainingNormalization
from .training import T1, S1, checkpoint_payload, restore_model, rng_state, restore_rng, to_batch, build_optimizers, evaluation_config
from .version import TARGET_NAMES, HEAD_GROUPS

VERSION = "perfseer_v4_eight_pair_campaign_v1"
STAGES = ("source_teacher", "source_student", "target_teacher", "target_student")
PROPOSALS = ("v4.0", "v4.1", "v4.2", "v4.3")


def default_config(data, output, *, smoke=False):
    def schedule(epochs, minimum, patience, lr, warmup):
        return {"max_epochs": epochs, "min_epochs": minimum, "patience": patience, "learning_rate": lr,
                "warmup_epochs": warmup, "relative_min_delta": .001, "lr_patience": 5, "lr_factor": .5, "min_lr": 1e-6}
    result = {"version": VERSION, "data": str(data), "output": str(output), "smoke": smoke,
              "sources": ["A100", "A10"], "proposals": list(PROPOSALS), "device": "cuda", "source_seed": 42,
              "transfer_seed": 11, "effective_batch": 256, "transfer_batch": 64, "microbatch": 4,
              "checkpoint_seconds": 300, "cpu_threads": 2,
              "concurrency": {"default": 2, "maximum": 3, "memory_fraction": .85, "throughput_gain": .10, "pilot_steps": 8},
              "stages": {"source_teacher": schedule(600, 30, 20, 5e-4, 10),
                         "resource_teacher": schedule(300, 20, 15, 1e-3, 5),
                         "source_student": schedule(300, 20, 15, 1e-3, 5),
                         "target_teacher": schedule(200, 20, 15, 1e-4, 5),
                         "target_student": schedule(200, 20, 15, 1e-4, 5)}}
    if smoke:
        result.update(effective_batch=2, transfer_batch=2, microbatch=1, cpu_threads=1,
                      graph_teacher={"hidden": 32, "num_blocks": 1}, graph_student={"hidden": 16, "num_blocks": 1})
        result["stages"] = {key: {**value, "max_epochs": 2, "min_epochs": 1, "patience": 1, "warmup_epochs": 0} for key, value in result["stages"].items()}
    return result


def validate_config(config):
    if config.get("version") != VERSION or config.get("sources") != ["A100", "A10"] or config.get("proposals") != list(PROPOSALS):
        raise ValueError("campaign must contain all eight source/proposal combinations")
    if config["device"] != "cuda":
        raise ValueError("campaign execution requires CUDA")
    if not config["smoke"] and any(key in config for key in ("graph_teacher", "graph_student")):
        raise ValueError("reduced graph capacities are smoke-only")
    for key in ("effective_batch", "transfer_batch", "microbatch", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("positive integer batch/thread settings required")
    if not 0 < config["checkpoint_seconds"] <= 300:
        raise ValueError("checkpoint interval must be at most five minutes")
    for schedule in config["stages"].values():
        if not 1 <= schedule["min_epochs"] <= schedule["max_epochs"] or schedule["patience"] < 1 or schedule["learning_rate"] <= 0:
            raise ValueError("invalid adaptive schedule")
        if not (0 <= schedule["warmup_epochs"] <= schedule["max_epochs"] and 0 <= schedule["relative_min_delta"] < 1
                and schedule["lr_patience"] >= 1 and 0 < schedule["lr_factor"] < 1 and 0 < schedule["min_lr"] <= schedule["learning_rate"]):
            raise ValueError("invalid warmup or plateau schedule")
    admission = config["concurrency"]
    if not 1 <= admission["default"] <= admission["maximum"] <= 3 or not 0 < admission["memory_fraction"] <= .85:
        raise ValueError("invalid bounded GPU admission")
    return config


@dataclass
class EarlyStop:
    best: float = math.inf
    anchor: float = math.inf
    best_epoch: int = 0
    stale: int = 0

    def update(self, error, epoch, schedule):
        if not math.isfinite(error):
            raise ValueError("nonfinite validation score")
        improved = error < self.best
        if improved:
            self.best, self.best_epoch = error, epoch
        if error < self.anchor * (1 - schedule["relative_min_delta"]):
            self.anchor, self.stale = error, 0
        else:
            self.stale += 1
        return improved, epoch >= schedule["min_epochs"] and self.stale >= schedule["patience"]


def weights_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(fingerprint([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def restore_artifact(payload, device="cpu"):
    model = restore_model(payload, device=device)
    if "correction" in payload:
        from .campaign_residual import CorrectedTeacher
        binding = payload["correction"]["binding"]
        if binding["base_weights_sha256"] != weights_hash(model) or binding["normalization_sha256"] != payload["normalization_sha256"]:
            raise ValueError("correction is bound to different weights or normalization")
        if binding["hardware"] != payload["prediction_hardware"]:
            raise ValueError("correction hardware binding differs")
        model = CorrectedTeacher(model, payload["correction"])
    return model


def source_normalization(root):
    manifest = read_json(root / "dataset_manifest.json")
    path = root / "campaign-normalization.json"
    if path.exists():
        value = read_json(path)
        if value["dataset_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("source normalization data changed")
        return normalization_from_dict(value["normalization"]), value["resources"]
    samples = CampaignSamples(root, sorted(load_rows(root, "train"), key=lambda r: r["input_path"]))
    normalization = fit_training_normalization(samples, manifest["fingerprint"])
    resources = (torch.tensor([row["resource_inputs"]["resources"][0] for row in samples.rows], dtype=torch.float32)
                 if all("resource_inputs" in row for row in samples.rows)
                 else torch.cat([samples[i]["features"].resources for i in range(len(samples))]))
    mean, scale = resources.mean(0), resources.std(0, unbiased=False)
    resource = {"mean": mean.tolist(), "scale": torch.where(scale < 1e-8, 1., scale).tolist()}
    atomic_write(path, {"dataset_fingerprint": manifest["fingerprint"], "normalization": asdict(normalization), "resources": resource})
    return normalization, resource


def target_statistics(rows):
    target, mask = target_tensors(rows)
    medians = torch.tensor([target[:, i][mask[:, i]].median() if mask[:, i].any() else 64. for i in range(7)])
    scales = medians.clone()
    scales[3] = 100
    return medians, scales, mask.sum(0).tolist()


def initialize_new_outputs(model, rows, observed_counts):
    medians, scales, target_counts = target_statistics(rows)
    new = [i for i, count in enumerate(observed_counts) if not count and target_counts[i]]
    with torch.no_grad():
        for index in new:
            group = next(g for g, indices in enumerate(HEAD_GROUPS) if index in indices)
            local = HEAD_GROUPS[group].index(index)
            head = model.prediction_heads[group]
            last = head if isinstance(head, torch.nn.Linear) else head[-1]
            last.weight[local].zero_()
            model.target_scales[index] = scales[index]
            ratio = medians[index] / scales[index]
            last.bias[local] = torch.logit((medians[index] / 100).clamp(1e-6, 1 - 1e-6)) if index == 3 else ratio + torch.log(-torch.expm1(-ratio))
    return new


def stage_paths(config, source, proposal, stage):
    pair = Path(config["output"]) / source / proposal / ("source" if stage.startswith("source") else "rtx5090")
    role = stage.split("_")[1]
    return pair, role, pair / f"{role}-best.pt", pair / "resume" / f"{role}-latest.pt"


def _model_for_stage(config, source, proposal, stage, rows, raw, resource, device):
    target_stage = stage.startswith("target")
    role = stage.split("_")[1]
    medians, scales, observed = target_statistics(rows)
    parent = None
    if target_stage:
        _, _, source_path, _ = stage_paths(config, source, proposal, "source_" + role)
        parent = torch.load(source_path, map_location="cpu", weights_only=False)
        source_receipt = read_json(source_path.with_name(f"{role}-complete.json"))
        if source_receipt["checkpoint_sha256"] != file_sha256(source_path) or parent["epoch"] < 1 or parent["role"] != role:
            raise ValueError("transfer requires a verified trained source checkpoint")
        if (parent["campaign"]["source"], parent["campaign"]["proposal"], parent["campaign"]["stage"]) != (source, proposal, "source_" + role):
            raise ValueError("source pair belongs to a different task")
        model = restore_model(parent, device=device)
        new = initialize_new_outputs(model, rows, parent["campaign"]["observed_count"])
        if proposal == "v4.2":
            from .adapters import HardwareAdaptedV4
            # A genuinely trained head can remain constant when its observed
            # labels are constant. The completed-stage receipt proves training.
            model = HardwareAdaptedV4(model, new_outputs=new, require_trained_source=False).to(device)
        return model, observed, file_sha256(source_path)
    if proposal == "v4.3":
        model = ResourceMLP(ResourceMLPConfig(hidden=128 if role == "teacher" else 64), scales,
                            torch.tensor(resource["mean"]), torch.tensor(resource["scale"]), medians)
    else:
        shape = dict(T1 if role == "teacher" else S1)
        shape.update(config.get("graph_" + role, {}))
        model = SeerNetV4(SeerNetV4Config.from_registry(OperationRegistry.load(), raw[0]["features"].layout, **shape), scales, medians)
    return model.to(device), observed, None


def model_batch(model, samples, device):
    if isinstance(getattr(model, "base", model), ResourceMLP):
        return SimpleNamespace(resources=torch.cat([s["features"].resources for s in samples]).to(device),
            training=SimpleNamespace(**{key: torch.cat([getattr(s["features"].training, key) for s in samples]).to(device)
                                      for key in ("hardware_cont", "hardware_missing_mask")}))
    return to_batch(samples, device)


def train_step(model, samples, optimizers, microbatch, *, teacher=None, relation_weight=.05, amp=True, on_microbatch=None):
    device = next(model.parameters()).device
    all_masks = target_tensors(samples.rows, device)[1] if hasattr(samples, "rows") else torch.stack([s["mask"] for s in samples]).to(device)
    weights = head_weights(all_masks)
    initial = rng_state()
    while True:
        restore_rng(initial)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total = 0.
        batch = output = teaching = objective = target = None
        try:
            for begin in range(0, len(samples), microbatch):
                subset = samples[begin:begin + microbatch]
                batch = model_batch(model, subset, device)
                target = torch.stack([s["target"] for s in subset]).to(device)
                mask = all_masks[begin:begin + len(subset)]
                weight = weights[begin:begin + len(subset)]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                    output = model.predict_batch(batch)
                    if teacher is not None:
                        with torch.no_grad():
                            teaching = teacher.predict_batch(batch)
                objective = loss(output.prediction, target, mask, weights=weight) if teacher is None else distill(
                    output, teaching, target, mask, weights=weight, relation_weight=relation_weight * len(subset) / len(samples))
                objective.backward()
                total += float(objective.detach())
                batch = output = teaching = objective = target = None
                if on_microbatch is not None and on_microbatch(initial, microbatch):
                    # No optimizer has stepped. Roll partial accumulation back
                    # to its exact boundary so long batches can terminate safely.
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    restore_rng(initial)
                    raise InterruptedError("saved before the interrupted logical batch")
            if getattr(model, "model_variant", None) == "v4.2":
                model.displacement_penalty().backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            break
        except torch.cuda.OutOfMemoryError:
            batch = output = teaching = objective = target = None
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if microbatch == 1:
                raise
            microbatch = max(1, microbatch // 2)
            torch.cuda.empty_cache()
    for optimizer in optimizers:
        optimizer.step()
    return total, microbatch


@torch.no_grad()
def predictions(model, samples, microbatch, *, amp=False, runtime_out=None, on_batch=None):
    device = next(model.parameters()).device
    model.eval()
    if runtime_out is not None:
        runtime_out.update(configuration=evaluation_config(model, microbatch, amp=amp), batch_sizes=[], oom_retries=0)
    start, values = 0, []
    while start < len(samples):
        batch = output = None
        try:
            subset = samples[start:start + microbatch]
            batch = model_batch(model, subset, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                output = model.predict_batch(batch).prediction.detach().float().cpu()
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite evaluation prediction")
            values.append(output)
            if runtime_out is not None:
                runtime_out["batch_sizes"].append(len(subset))
            start += len(subset)
            if on_batch is not None:
                on_batch()
        except torch.cuda.OutOfMemoryError:
            batch = output = None
            if microbatch == 1:
                raise
            if runtime_out is not None:
                runtime_out["oom_retries"] += 1
            microbatch //= 2
            torch.cuda.empty_cache()
    return torch.cat(values)


def prediction_cache_digest(cache):
    return fingerprint({"identity": cache["identity"], "prediction_runtime": cache["prediction_runtime"],
                        "train": cache["train"].tolist(), "validation": cache["validation"].tolist()})


def evaluate(model, samples, microbatch, *, identity=None, on_batch=None):
    runtime = {}
    values = predictions(model, samples, microbatch, runtime_out=runtime, on_batch=on_batch)
    target, mask = target_tensors(samples.rows)
    report = {"identity": identity or {}, "metrics": metrics(values, target, mask),
              "prediction_runtime": runtime, "prediction_runtime_sha256": fingerprint(runtime),
              "rows": [{"sample_id": row["sample_id"], "prediction": values[i].tolist(), "target": target[i].tolist(),
                        "mask": mask[i].tolist()} for i, row in enumerate(samples.rows)]}
    verify_prediction_report(report, samples.rows)
    return report


def run_stage(config, source, proposal, stage, *, resume=False, probe=False, memory_fraction=None):
    validate_config(config)
    if source not in config["sources"] or proposal not in PROPOSALS or stage not in STAGES:
        raise ValueError("invalid campaign task")
    if not torch.cuda.is_available():
        raise RuntimeError("campaign requires CUDA; CPU fallback is disabled")
    torch.set_num_threads(config["cpu_threads"])
    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    seed = config["source_seed"] if stage.startswith("source") else config["transfer_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda"
    data = Path(config["data"])
    source_root = data / source
    target_stage = stage.startswith("target")
    root = data / ("RTX5090" if target_stage else source)
    manifest = read_json(root / "dataset_manifest.json")
    if bool(manifest["smoke_rows_per_split"]) != config["smoke"]:
        raise ValueError("smoke/production dataset mismatch")
    normalization, resource = source_normalization(source_root)
    if target_stage:
        normalization = TrainingNormalization(replace(normalization.training, split_fingerprint=manifest["fingerprint"]))
    rows = load_rows(root, "train")
    train = CampaignSamples(root, rows, normalization, resource_only=proposal == "v4.3")
    validation = CampaignSamples(root, load_rows(root, "validation"), normalization, resource_only=proposal == "v4.3")
    pair, role, best_path, latest = stage_paths(config, source, proposal, stage)
    teacher, teacher_sha = None, None
    if role == "student":
        teacher_path = pair / "teacher-best.pt"
        teacher_sha = file_sha256(teacher_path)
        teacher = restore_artifact(torch.load(teacher_path, map_location="cpu", weights_only=False), device).eval().requires_grad_(False)
    model, observed, parent_sha = _model_for_stage(config, source, proposal, stage, rows, train, resource, device)
    schedule_name = "resource_teacher" if proposal == "v4.3" and stage == "source_teacher" else stage
    schedule = config["stages"][schedule_name]
    effective_batch = config["transfer_batch"] if target_stage else config["effective_batch"]
    contract = {"version": VERSION, "objective_version": OBJECTIVE_VERSION, "source": source, "proposal": proposal,
                "stage": stage, "dataset_fingerprint": manifest["fingerprint"], "normalization_sha256": fingerprint(asdict(normalization)),
                "source_normalization_sha256": file_sha256(source_root / "campaign-normalization.json"), "schedule": schedule,
                "effective_batch": effective_batch, "seed": seed, "code_fingerprint": code_fingerprint(),
                "teacher_sha256": teacher_sha, "parent_sha256": parent_sha, "observed_count": observed,
                "smoke": config["smoke"], "model_config": model.config.to_dict()}
    complete_path = pair / f"{role}-complete.json"
    if complete_path.exists():
        receipt = read_json(complete_path)
        if receipt["contract"] != contract or receipt["checkpoint_sha256"] != file_sha256(best_path):
            raise ValueError("completed stage identity differs")
        if not resume and not probe:
            raise ValueError("completed output exists; use --resume")
        return receipt
    if not resume and not probe and (latest.exists() or best_path.exists()):
        raise ValueError("checkpoint exists; use --resume or fresh output")
    corrected = target_stage and role == "teacher" and proposal in {"v4.1", "v4.3"}
    graph_source = not target_stage and proposal != "v4.3"
    optimizers = list(build_optimizers(model, schedule["learning_rate"])[0]) if graph_source else [
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=schedule["learning_rate"],
                          weight_decay=1e-4 if not target_stage and proposal == "v4.3" else 0.)]
    microbatch = min(config["microbatch"], effective_batch)
    relation_weight = 0. if proposal == "v4.3" or (target_stage and proposal == "v4.2") else .05
    if probe:
        torch.cuda.reset_peak_memory_stats()
        largest = sorted(range(len(train)), key=lambda i: (rows[i]["graph_nodes"], rows[i]["graph_edges"]), reverse=True)
        started = time.monotonic()
        for _ in range(config["concurrency"]["pilot_steps"]):
            selected = LogicalBatch(train, largest[:min(effective_batch, len(train))])
            _, microbatch = train_step(model, selected, optimizers, microbatch, teacher=teacher, relation_weight=relation_weight)
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report = {"status": "probed", "contract": contract, "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "non_allocator_bytes": max(0, total_bytes - free_bytes - torch.cuda.memory_reserved()),
                  "microbatch": microbatch, "samples_per_second": len(selected) * config["concurrency"]["pilot_steps"] / (time.monotonic() - started),
                  "gpu": torch.cuda.get_device_name(), "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory}
        if corrected and proposal == "v4.3":
            report["peak_reserved_bytes"] += 30 * len(train) ** 2 * 8
        return report
    pair.mkdir(parents=True, exist_ok=True)
    def event(value):
        value = {"source": source, "proposal": proposal, "stage": stage, "timestamp": time.time(), **value}
        print(json.dumps(value, allow_nan=False), flush=True)
        atomic_write(pair / f"{role}-progress.json", value)
        with (pair / f"{role}-training.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
    def payload(epoch, validation_metrics, **extra):
        return checkpoint_payload(model, role=role, epoch=epoch, normalization=asdict(normalization),
                                  dataset_fingerprint=manifest["fingerprint"], prediction_hardware=manifest["prediction_hardware"],
                                  validation=validation_metrics, campaign=contract, evaluation=evaluation_config(model, microbatch, amp=False), **extra)
    if corrected:
        from .campaign_residual import fit_correction, CorrectedTeacher
        frozen_hash = weights_hash(model)
        runtime = evaluation_config(model, microbatch, amp=False)
        binding = {"base_weights_sha256": frozen_hash, "normalization_sha256": contract["normalization_sha256"],
                   "dataset_fingerprint": manifest["fingerprint"], "hardware": manifest["prediction_hardware"], "runtime": runtime,
                   "source_checkpoint_sha256": parent_sha, "prediction_code_sha256": contract["code_fingerprint"]}
        cache_path = pair / "teacher-base-predictions.pt"
        cache_identity = {**binding, "input_hashes": [r["input_sha256"] for r in [*rows, *validation.rows]],
                          "sample_ids": [r["sample_id"] for r in [*rows, *validation.rows]]}
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cache["identity"] != cache_identity or cache.get("content_sha256") != prediction_cache_digest(cache):
                raise ValueError("teacher prediction cache identity differs")
        else:
            traces = {"train": {}, "validation": {}}
            cache = {"identity": cache_identity, "train": predictions(model, train, microbatch, runtime_out=traces["train"]),
                     "validation": predictions(model, validation, microbatch, runtime_out=traces["validation"]),
                     "prediction_runtime": traces}
            cache["content_sha256"] = prediction_cache_digest(cache)
            atomic_write(cache_path, cache, checkpoint=True)
        binding["prediction_runtime_sha256"] = fingerprint(cache["prediction_runtime"])
        correction = fit_correction(train, validation, cache["train"], cache["validation"], variant=proposal,
                                    binding=binding, device=device, progress=event)
        wrapped = CorrectedTeacher(model, correction)
        report = evaluate(wrapped, validation, microbatch, identity=contract)
        if weights_hash(model) != frozen_hash:
            raise ValueError("correction fitting changed source weights")
        selected = payload(0, report["metrics"], correction=correction)
        atomic_write(best_path, selected, checkpoint=True)
        atomic_write(pair / f"{role}-validation.json.gz", report, compress=True)
        stopped, epochs_completed = "finite_validation_grid_complete", 0
    else:
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=schedule["lr_factor"],
                      patience=schedule["lr_patience"] - 1, threshold=schedule["relative_min_delta"], threshold_mode="rel",
                      min_lr=schedule["min_lr"]) for opt in optimizers]
        stopping, epoch, cursor = EarlyStop(), 1, 0
        validation_metrics = None
        if resume and latest.exists():
            progress = torch.load(latest, map_location="cpu", weights_only=False)
            if progress["campaign"] != contract:
                raise ValueError("resume contract, teacher, source, or schedule differs")
            model.load_state_dict(progress["model_state_dict"], strict=True)
            for optimizer, state in zip(optimizers, progress["optimizers"], strict=True):
                optimizer.load_state_dict(state)
            for scheduler, state in zip(schedulers, progress["schedulers"], strict=True):
                scheduler.load_state_dict(state)
            restore_rng(progress["rng"])
            stopping = EarlyStop(**progress["stopping"])
            epoch, cursor, microbatch = progress["next_epoch"], progress["next_batch"], progress["microbatch"]
            validation_metrics = progress.get("validation")
        stop_requested = [False]
        previous = {sig: signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}
        last_save = time.monotonic()
        def save(next_epoch, next_batch, rng_override=None):
            state = payload(max(0, next_epoch - int(next_batch == 0)), validation_metrics,
                         optimizers=optimizers, schedulers=schedulers, stopping=asdict(stopping),
                         next_epoch=next_epoch, next_batch=next_batch, microbatch=microbatch)
            if rng_override is not None:
                state["rng"] = rng_override
            atomic_write(latest, state, checkpoint=True)
        steps = math.ceil(len(train) / effective_batch)
        stopped = "maximum_epochs"
        epochs_completed = epoch - 1
        try:
            if epoch > schedule["max_epochs"] or (epoch - 1 >= schedule["min_epochs"] and stopping.stale >= schedule["patience"]):
                stopped = "resumed_completed_schedule"
            else:
                for current in range(epoch, schedule["max_epochs"] + 1):
                    order = list(range(len(train)))
                    random.Random(seed + current).shuffle(order)
                    if current <= schedule["warmup_epochs"]:
                        for optimizer in optimizers:
                            for group in optimizer.param_groups:
                                group["lr"] = schedule["learning_rate"] * current / schedule["warmup_epochs"]
                    model.train()
                    for step in range(cursor, steps):
                        logical = LogicalBatch(train, order[step * effective_batch:(step + 1) * effective_batch])
                        def checkpoint_boundary(before_batch_rng, active_microbatch):
                            nonlocal last_save, microbatch
                            microbatch = active_microbatch
                            if stop_requested[0] or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                                # Model weights and optimizer state still equal
                                # the pre-batch boundary; gradients are not saved.
                                save(current, step, rng_override=before_batch_rng)
                                last_save = time.monotonic()
                            return stop_requested[0]
                        value, microbatch = train_step(model, logical, optimizers, microbatch, teacher=teacher,
                                                      relation_weight=relation_weight, on_microbatch=checkpoint_boundary)
                        event({"event": "step", "epoch": current, "step": step + 1, "steps": steps, "loss": value, "microbatch": microbatch})
                        if stop_requested[0] or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                            save(current, step + 1); last_save = time.monotonic()
                        if stop_requested[0]:
                            raise InterruptedError("saved at optimizer-step boundary")
                    cursor = 0
                    def validation_boundary():
                        nonlocal last_save
                        if stop_requested[0] or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                            # All optimizer steps are complete; replay validation
                            # before updating selection/stopping/scheduler state.
                            save(current, steps)
                            last_save = time.monotonic()
                        if stop_requested[0]:
                            raise InterruptedError("saved before completing validation")
                    validation_boundary()
                    report = evaluate(model, validation, microbatch, identity=contract, on_batch=validation_boundary)
                    validation_metrics = report["metrics"]
                    improved, should_stop = stopping.update(validation_metrics["selection_error"], current, schedule)
                    if improved:
                        atomic_write(best_path, payload(current, validation_metrics), checkpoint=True)
                        atomic_write(pair / f"{role}-validation.json.gz", report, compress=True)
                    if current > schedule["warmup_epochs"]:
                        for scheduler in schedulers:
                            scheduler.step(validation_metrics["selection_error"])
                    save(current + 1, 0); last_save = time.monotonic()
                    epochs_completed = current
                    event({"event": "epoch", "epoch": current, "validation": validation_metrics,
                           "best_epoch": stopping.best_epoch, "stale_epochs": stopping.stale})
                    if should_stop:
                        stopped = "validation_plateau"
                        break
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    del optimizers
    if corrected:
        del wrapped
    else:
        del schedulers
    del teacher, model
    torch.cuda.empty_cache()
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    if selected.get("campaign") != contract:
        raise ValueError("selected checkpoint stage contract differs")
    restored = restore_artifact(selected, device)
    replay = evaluate(restored, validation, microbatch, identity=contract)
    if not math.isclose(replay["metrics"]["selection_error"], selected["validation"]["selection_error"], rel_tol=2e-5, abs_tol=1e-6):
        raise ValueError("selected checkpoint prediction replay differs")
    receipt = {"status": "complete", "contract": contract, "checkpoint_sha256": file_sha256(best_path),
               "validation_sha256": file_sha256(pair / f"{role}-validation.json.gz"), "best_epoch": selected["epoch"],
               "epochs_completed": epochs_completed, "stop_reason": stopped,
               "gpu": torch.cuda.get_device_name(), "parameters": sum(p.numel() for p in restored.parameters()),
               "test_labels_read": False, "production_gate_passed": selected["validation"]["gate_passed"]}
    atomic_write(complete_path, receipt)
    if role == "student":
        seal_pair(pair)
    event({"event": "stage_complete", "stop_reason": stopped, "best_epoch": selected["epoch"]})
    return receipt


def seal_pair(pair):
    files = {}
    for role in ("teacher", "student"):
        for suffix in ("best.pt", "complete.json", "validation.json.gz"):
            path = pair / f"{role}-{suffix}"
            files[path.name] = file_sha256(path)
    teacher = torch.load(pair / "teacher-best.pt", map_location="cpu", weights_only=False)
    student = torch.load(pair / "student-best.pt", map_location="cpu", weights_only=False)
    if student["campaign"]["teacher_sha256"] != files["teacher-best.pt"]:
        raise ValueError("student does not belong to selected teacher")
    bundle = {"version": VERSION, "status": "selected", "source": teacher["campaign"]["source"],
              "proposal": teacher["campaign"]["proposal"], "domain": pair.name,
              "dataset_fingerprint": teacher["dataset_fingerprint"], "files": files}
    bundle["fingerprint"] = fingerprint(bundle)
    atomic_write(pair / "pair.json", bundle)
    return verify_pair(pair)


def verify_pair(pair):
    pair = Path(pair)
    bundle = read_json(pair / "pair.json")
    if bundle["fingerprint"] != fingerprint({k: v for k, v in bundle.items() if k != "fingerprint"}):
        raise ValueError("pair manifest changed")
    for relative, sha in bundle["files"].items():
        from .dataset import checked_path
        if file_sha256(checked_path(pair, relative)) != sha:
            raise ValueError("pair file checksum differs")
    teacher_sha = file_sha256(pair / "teacher-best.pt")
    for role in ("teacher", "student"):
        payload = torch.load(pair / f"{role}-best.pt", map_location="cpu", weights_only=False)
        restore_artifact(payload)
        receipt = read_json(pair / f"{role}-complete.json")
        if (receipt["contract"] != payload["campaign"] or receipt["checkpoint_sha256"] != file_sha256(pair / f"{role}-best.pt")
                or receipt["validation_sha256"] != file_sha256(pair / f"{role}-validation.json.gz")):
            raise ValueError("selected stage receipt differs")
        if payload["role"] != role or payload["campaign"]["source"] != bundle["source"] or payload["campaign"]["proposal"] != bundle["proposal"]:
            raise ValueError("pair role or proposal differs")
        if role == "student" and payload["campaign"]["teacher_sha256"] != teacher_sha:
            raise ValueError("student teacher lineage differs")
        report = read_json(pair / f"{role}-validation.json.gz")
        if report["identity"] != payload["campaign"]:
            raise ValueError("prediction report stage identity differs")
        verify_prediction_report(report)
        if report["metrics"] != payload["validation"]:
            raise ValueError("selected checkpoint metrics differ")
    return bundle


def verify_campaign(config):
    bundles = []
    for source in config["sources"]:
        for proposal in config["proposals"]:
            for domain in ("source", "rtx5090"):
                bundle = verify_pair(Path(config["output"]) / source / proposal / domain)
                if (bundle["source"], bundle["proposal"], bundle["domain"]) != (source, proposal, domain):
                    raise ValueError("campaign pair identity differs")
                root = Path(config["data"]) / (source if domain == "source" else "RTX5090")
                manifest = read_json(root / "dataset_manifest.json")
                if bundle["dataset_fingerprint"] != manifest["fingerprint"]:
                    raise ValueError("selected pair uses a different dataset")
                pair = Path(config["output"]) / source / proposal / domain
                for role in ("teacher", "student"):
                    report = read_json(pair / f"{role}-validation.json.gz")
                    verify_prediction_report(report, load_rows(root, "validation"))
                    contract = report["identity"]
                    expected_stage = ("source_" if domain == "source" else "target_") + role
                    if contract["stage"] != expected_stage or contract["smoke"] != config["smoke"]:
                        raise ValueError("pair stage or smoke scope differs")
                    if domain == "rtx5090":
                        parent = pair.parent / "source" / f"{role}-best.pt"
                        if contract["parent_sha256"] != file_sha256(parent):
                            raise ValueError("transfer source lineage differs")
                bundles.append(bundle)
    if len(bundles) != 16:
        raise ValueError("campaign requires sixteen pairs")
    return {"version": VERSION, "status": "smoke_complete" if config["smoke"] else "selected_complete",
            "pairs": len(bundles), "selected_models": 2 * len(bundles), "pair_fingerprints": [b["fingerprint"] for b in bundles]}


def final_evaluate(config):
    torch.set_num_threads(config["cpu_threads"])
    receipt = verify_campaign(config)
    seal = Path(config["output"]) / "selection.json"
    if seal.exists() and read_json(seal) != receipt:
        raise ValueError("sealed campaign selection changed")
    atomic_write(seal, receipt)
    reports = []
    for source in config["sources"]:
        for proposal in config["proposals"]:
            for domain in ("source", "rtx5090", "zero_shot"):
                root = Path(config["data"]) / (source if domain == "source" else "RTX5090")
                rows = load_rows(root, "test")
                for role in ("teacher", "student"):
                    pair = Path(config["output"]) / source / proposal / ("source" if domain in {"source", "zero_shot"} else "rtx5090")
                    checkpoint = pair / f"{role}-best.pt"
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    normalization = normalization_from_dict(payload["normalization"])
                    samples = CampaignSamples(root, rows, normalization, resource_only=proposal == "v4.3")
                    model = restore_artifact(payload, "cuda")
                    identity = {"selection_sha256": file_sha256(seal), "checkpoint_sha256": file_sha256(checkpoint),
                                "source": source, "proposal": proposal, "domain": domain, "role": role}
                    report = evaluate(model, samples, config["microbatch"], identity=identity)
                    path = Path(config["output"]) / "evaluation" / f"{source}-{proposal}-{domain}-{role}.json.gz"
                    atomic_write(path, report, compress=True)
                    reports.append({**identity, "path": str(path.relative_to(config["output"])), "sha256": file_sha256(path), "metrics": report["metrics"]})
                    del model; torch.cuda.empty_cache()
    result = {**receipt, "status": "smoke_complete" if config["smoke"] else "complete", "evaluations": reports, "deployment_approved": False}
    atomic_write(Path(config["output"]) / "completion.json", result)
    return result
