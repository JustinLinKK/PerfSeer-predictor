"""Bounded few-label neural adaptation and compact source-model training."""

from dataclasses import asdict, replace
import hashlib
import random
import time

import numpy as np
import torch

from perfseer_v31.io import atomic_write, fingerprint, read_json
from .runner import Samples, TrainingNormalization, code_fingerprint, fit_training_normalization, normalization_from_dict
from .training import (checkpoint_payload, evaluate, normalized_errors, restore_model, rng_state,
                       restore_rng, selection_key, to_batch)
from .version import HEAD_GROUPS, SM_INDICES, TARGET_NAMES


def source_identity(payload):
    restore_model(payload)
    digest = hashlib.sha256()
    for name, tensor in sorted(payload["model_state_dict"].items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(fingerprint([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"weights_sha256": digest.hexdigest(), "model_variant": payload["model_variant"],
            "normalization_sha256": payload["normalization_sha256"],
            "model_config_sha256": fingerprint(payload["model_config"]),
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "prediction_hardware": payload.get("prediction_hardware"),
            "inference": {"device": "cpu", "amp": False, "microbatch": 1,
                          "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads()},
            "preprocessing_sha256": code_fingerprint()}


def _fit(model, train, validation, *, epochs, learning_rate, effective_batch, microbatch,
         weight_decay=0., seed=11, patience=10, min_epochs=20):
    if min(epochs, effective_batch, microbatch) < 1 or learning_rate <= 0 or not len(train) or not len(validation):
        raise ValueError("fitting requires nonempty splits and positive schedule values")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    device = next(model.parameters()).device
    best, best_metrics, best_epoch, best_key = None, None, None, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for begin in range(0, len(order), effective_batch):
            logical = [train[i] for i in order[begin:begin + effective_batch]]
            state = rng_state()
            batch = output = loss = target = None
            while True:
                restore_rng(state)
                optimizer.zero_grad(set_to_none=True)
                try:
                    for offset in range(0, len(logical), microbatch):
                        subset = logical[offset:offset + microbatch]
                        batch = to_batch(subset, device)
                        target = torch.stack([row["target"] for row in subset]).to(device)
                        output = model.predict_batch(batch).prediction
                        errors = normalized_errors(output, target)
                        loss = torch.stack([errors[:, group].mean() for group in HEAD_GROUPS]).mean()
                        (loss * len(subset) / len(logical)).backward()
                        batch = output = loss = target = None
                    if model.model_variant == "v4.2":
                        model.displacement_penalty().backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    batch = output = loss = target = None
                    optimizer.zero_grad(set_to_none=True)
                    if microbatch == 1:
                        raise
                    microbatch //= 2
                    torch.cuda.empty_cache()
            optimizer.step()
        metrics = evaluate(model, validation, microbatch, amp=False)
        key = selection_key(metrics, epoch)
        if best_key is None or key < best_key:
            best_key, best_metrics, best_epoch = key, metrics, epoch
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch >= min_epochs and epoch - best_epoch >= patience:
            break
    model.load_state_dict(best, strict=True)
    model.eval()
    return {"validation": best_metrics, "epoch": best_epoch, "epochs_completed": epoch,
            "fit_and_validation_seconds": time.perf_counter() - start,
            "trainable_parameters": sum(p.numel() for p in parameters), "microbatch": microbatch}


def fit_neural(source, train, validation, *, variant, dataset_fingerprint, hardware,
               environment, seed=11, device="cpu", epochs=100, effective_batch=64, microbatch=4):
    if source.get("model_variant") != "v4.0" or variant not in {"v4.0-finetune", "v4.2"}:
        raise ValueError("neural transfer requires a v4.0 source and a supported variant")
    from perfseer_v32.calibration_contracts import environment_identity
    domain = environment_identity(environment)
    if environment["hardware_id"] != hardware:
        raise ValueError("environment and target dataset hardware differ")
    identities, groups, workloads = set(), {}, {}
    for split, samples in (("train", train), ("validation", validation)):
        if not len(samples):
            raise ValueError("neural transfer requires nonempty fit and validation samples")
        for sample in samples:
            identity, group, workload = (sample.get(key) for key in ("sample_id", "group_id", "workload_id"))
            if not identity or not group or not workload or identity in identities:
                raise ValueError("neural transfer requires unique sample, group and workload identities")
            identities.add(identity)
            if sample.get("split") != split or groups.setdefault(group, split) != split or workloads.setdefault(workload, split) != split:
                raise ValueError("neural transfer split leakage")
            if sample["features"].training.metadata.get("hardware_id") != hardware:
                raise ValueError("neural transfer graph hardware differs")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = restore_model(source, device=device)
    initial_base = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if variant == "v4.2":
        from .adapters import HardwareAdaptedV4
        model = HardwareAdaptedV4(model).to(device)
    result = _fit(model, train, validation, epochs=epochs, learning_rate=1e-4,
                  effective_batch=effective_batch, microbatch=microbatch, seed=seed)
    if variant == "v4.2" and any(not torch.equal(value, model.base.state_dict()[key].cpu()) for key, value in initial_base.items()):
        raise ValueError("adapter fitting changed the frozen source")
    normalization = normalization_from_dict(source["normalization"])
    normalization = TrainingNormalization(replace(normalization.training, split_fingerprint=dataset_fingerprint))
    payload = checkpoint_payload(model, role=source["role"], epoch=result["epoch"],
                                 normalization=asdict(normalization), dataset_fingerprint=dataset_fingerprint,
                                 prediction_hardware=hardware, validation=result["validation"],
                                 transfer={"variant": variant, "source_identity": source_identity(source),
                                           "environment": environment, "domain_fingerprint": domain,
                                           "fit_ids": [row["sample_id"] for row in train],
                                           "validation_ids": [row["sample_id"] for row in validation],
                                           "experimental": True}, **{k: v for k, v in result.items() if k not in {"epoch", "validation"}})
    return payload, result


def train_resource(args):
    """Explicit source training; never invoked during transfer selection."""
    from .dataset import verify
    from .resource_model import ResourceMLP, ResourceMLPConfig
    manifest = verify(args.dataset)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("resource source training requires a fresh output directory")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rows = {split: read_json(args.dataset / manifest["split_files"][split]["path"]) for split in ("train", "validation")}
    raw = Samples(args.dataset, rows["train"], include_resources=True)
    normalization = fit_training_normalization(raw, manifest["fingerprint"])
    resources = torch.cat([raw[i]["features"].resources for i in range(len(raw))])
    mean, scale = resources.mean(0), resources.std(0, unbiased=False)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    targets = torch.tensor([[row["targets"][name] for name in TARGET_NAMES] for row in rows["train"]])
    medians = targets.median(0).values
    scales = medians.clone()
    scales[list(SM_INDICES)] = 100
    model = ResourceMLP(ResourceMLPConfig(), scales, mean, scale, medians).to(args.device)
    train = Samples(args.dataset, rows["train"], normalization, include_resources=True)
    validation = Samples(args.dataset, rows["validation"], normalization, include_resources=True)
    result = _fit(model, train, validation, epochs=args.epochs, learning_rate=1e-3, weight_decay=1e-4,
                  effective_batch=256, microbatch=args.microbatch, seed=args.seed)
    payload = checkpoint_payload(model, role="resource", epoch=result["epoch"], normalization=asdict(normalization),
                                 dataset_fingerprint=manifest["fingerprint"], prediction_hardware=manifest["prediction_hardware"],
                                 validation=result["validation"], label_policy=manifest.get("label_policy"),
                                 code_fingerprint=code_fingerprint())
    atomic_write(args.output / "resource-best.pt", payload, checkpoint=True)
    atomic_write(args.output / "training-report.json", {**result, "model_variant": "v4.3", "status": "source_trained_not_promoted", "test_evaluated": False})
    return result
