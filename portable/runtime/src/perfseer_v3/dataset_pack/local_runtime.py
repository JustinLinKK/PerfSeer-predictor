"""Short RTX-5090 trainability checks for selected validation signatures."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import importlib
import math
import warnings
from typing import Any, Callable, Iterable, Mapping

import torch
from torch.utils.checkpoint import checkpoint

from .adapters import TaskAdapter, adapter_for_task
from .fingerprints import canonical_sha256, canonical_value
from .local_validation import ValidationSignature
from .local_task_fixture import build_local_real_format_batch
from .local_provenance import (
    validation_environment_sha256,
    validation_harness_sha256,
)
from .labeler_profile import PROFILE
from .models.base import FamilyModel, tensor_outputs
from .sampler import TargetCandidate


LOCAL_EXECUTION_RESULT_VERSION = "perfseer_v3_local_execution_result_v3"


class LocalExecutionError(RuntimeError):
    """Raised when a representative is not executable on the local verifier."""


class _Lamb(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[torch.Tensor], *, lr: float, weight_decay: float) -> None:
        super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999), "eps": 1e-8})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise LocalExecutionError("LAMB does not accept sparse gradients")
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                state.setdefault("exp_avg", torch.zeros_like(parameter)).mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                state.setdefault("exp_avg_sq", torch.zeros_like(parameter)).mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                update = state["exp_avg"] / state["exp_avg_sq"].sqrt().add_(group["eps"])
                if group["weight_decay"]:
                    update = update.add(parameter, alpha=group["weight_decay"])
                weight_norm = parameter.norm()
                update_norm = update.norm()
                trust = torch.where(
                    (weight_norm > 0) & (update_norm > 0),
                    weight_norm / update_norm,
                    torch.ones_like(weight_norm),
                )
                parameter.add_(update, alpha=-group["lr"] * float(trust))
        return loss


class _Lars(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[torch.Tensor], *, lr: float, weight_decay: float) -> None:
        super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay, "momentum": 0.9, "trust": 1e-3})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise LocalExecutionError("LARS does not accept sparse gradients")
                if group["weight_decay"]:
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])
                weight_norm = parameter.norm()
                gradient_norm = gradient.norm()
                scale = torch.where(
                    (weight_norm > 0) & (gradient_norm > 0),
                    group["trust"] * weight_norm / gradient_norm,
                    torch.ones_like(weight_norm),
                )
                state = self.state[parameter]
                momentum = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                momentum.mul_(group["momentum"]).add_(gradient, alpha=float(scale))
                parameter.add_(momentum, alpha=-group["lr"])
        return loss


class _Lion(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[torch.Tensor], *, lr: float, weight_decay: float) -> None:
        super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.99)})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise LocalExecutionError("Lion does not accept sparse gradients")
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                momentum = self.state[parameter].setdefault("exp_avg", torch.zeros_like(parameter))
                update = momentum.mul(beta1).add(gradient, alpha=1.0 - beta1).sign()
                parameter.add_(update, alpha=-group["lr"])
                momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
        return loss


class _OptimizerBundle:
    def __init__(self, optimizers: tuple[torch.optim.Optimizer, ...]) -> None:
        if not optimizers:
            raise LocalExecutionError("optimizer bundle cannot be empty")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        if closure is not None:
            if len(self.optimizers) != 1:
                raise LocalExecutionError("closures require exactly one optimizer")
            return self.optimizers[0].step(closure)
        return tuple(optimizer.step() for optimizer in self.optimizers)

    def state_dict(self) -> Mapping[str, Any]:
        return {str(index): optimizer.state_dict() for index, optimizer in enumerate(self.optimizers)}


class _Scheduler:
    def __init__(
        self,
        scheduler: Any | None,
        *,
        step_unit: str,
        maximum_steps: int,
        metric_driven: bool = False,
        minimum_learning_rates: tuple[float, ...] = (),
    ) -> None:
        self.scheduler = scheduler
        self.step_unit = step_unit
        self.maximum_steps = maximum_steps
        self.steps_taken = 0
        self.metric_driven = metric_driven
        self.minimum_learning_rates = minimum_learning_rates
        self._clamp_learning_rates()

    def _clamp_learning_rates(self) -> None:
        if self.scheduler is None or not self.minimum_learning_rates:
            return
        groups = self.scheduler.optimizer.param_groups
        if len(groups) != len(self.minimum_learning_rates):
            raise LocalExecutionError("scheduler learning-rate floor shape drifted")
        for group, floor in zip(groups, self.minimum_learning_rates, strict=True):
            group["lr"] = max(float(group["lr"]), floor)
        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = [float(group["lr"]) for group in groups]

    def step(self, loss: torch.Tensor, *, unit: str) -> None:
        if (
            self.scheduler is None
            or unit != self.step_unit
            or self.steps_taken >= self.maximum_steps
        ):
            return
        if self.metric_driven:
            self.scheduler.step(float(loss.detach()))
        else:
            self.scheduler.step()
        # The manifest explicitly records a floor only for terminal-decay
        # schedules whose end state can otherwise make a training update a
        # no-op. Other scheduler implementations retain their native curve.
        self._clamp_learning_rates()
        self.steps_taken += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {} if self.scheduler is None else self.scheduler.state_dict()

    def fast_forward(self, progress: float, *, total_optimizer_steps: int) -> None:
        if self.scheduler is None or progress <= 0:
            return
        schedule_steps = total_optimizer_steps if self.step_unit == "optimizer_step" else 5
        completed = min(schedule_steps, math.floor(progress * schedule_steps))
        placeholder = torch.tensor(1.0)
        # This is state restoration, not a training update: the scheduler's
        # construction-time value represents step zero and the loop advances
        # to the manifest's declared progress before the first real optimizer
        # step. PyTorch warns about this ordering for ordinary training loops,
        # where it could skip a usable learning rate, but that warning is not
        # applicable to an intentional fast-forward with no historical model
        # updates in this process.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
                category=UserWarning,
            )
            for _ in range(completed):
                self.step(placeholder, unit=self.step_unit)


def _trainable_parameters(model: FamilyModel) -> tuple[torch.nn.Parameter, ...]:
    result = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not result:
        raise LocalExecutionError("model has no trainable parameters")
    return result


def _build_optimizer(candidate: TargetCandidate, model: FamilyModel) -> _OptimizerBundle:
    parameters = _trainable_parameters(model)
    dense = tuple(parameter for parameter in parameters if parameter.ndim == 2)
    auxiliary = tuple(parameter for parameter in parameters if parameter.ndim != 2)
    name = str(candidate.optimizer["name"])
    lr = float(candidate.optimizer["learning_rate"])
    weight_decay = float(candidate.optimizer["weight_decay"])
    common = {"lr": lr, "weight_decay": weight_decay}
    if name == "sgd":
        optimizers = (torch.optim.SGD(parameters, momentum=0.9, **common),)
    elif name == "asgd":
        optimizers = (torch.optim.ASGD(parameters, **common),)
    elif name == "adadelta":
        optimizers = (torch.optim.Adadelta(parameters, **common),)
    elif name == "adafactor":
        optimizers = (torch.optim.Adafactor(parameters, **common),)
    elif name == "adagrad":
        optimizers = (torch.optim.Adagrad(parameters, **common),)
    elif name == "adam":
        optimizers = (torch.optim.Adam(parameters, **common),)
    elif name == "adamax":
        optimizers = (torch.optim.Adamax(parameters, **common),)
    elif name == "adamw":
        optimizers = (torch.optim.AdamW(parameters, **common),)
    elif name == "lbfgs":
        optimizers = (torch.optim.LBFGS(parameters, lr=lr, max_iter=2, history_size=4),)
    elif name == "muon":
        if not dense:
            raise LocalExecutionError("Muon representative has no matrix parameters")
        parts: list[torch.optim.Optimizer] = [torch.optim.Muon(dense, **common)]
        if auxiliary:
            parts.append(torch.optim.AdamW(auxiliary, **common))
        optimizers = tuple(parts)
    elif name == "nadam":
        optimizers = (torch.optim.NAdam(parameters, **common),)
    elif name == "radam":
        optimizers = (torch.optim.RAdam(parameters, **common),)
    elif name == "rmsprop":
        optimizers = (torch.optim.RMSprop(parameters, momentum=0.9, **common),)
    elif name == "rprop":
        optimizers = (torch.optim.Rprop(parameters, lr=lr),)
    elif name == "sparse_adam":
        optimizers = (torch.optim.SparseAdam(parameters, lr=lr),)
    elif name == "lamb":
        optimizers = (_Lamb(parameters, **common),)
    elif name == "lars":
        optimizers = (_Lars(parameters, **common),)
    elif name == "lion":
        optimizers = (_Lion(parameters, **common),)
    else:
        raise LocalExecutionError(f"unsupported deployment optimizer {name!r}")
    return _OptimizerBundle(optimizers)


def _build_scheduler(
    candidate: TargetCandidate,
    optimizer: _OptimizerBundle,
    *,
    total_optimizer_steps: int = 4,
) -> _Scheduler:
    name = str(candidate.scheduler["name"])
    step_unit = str(candidate.scheduler["step_unit"])
    if total_optimizer_steps < 1:
        raise LocalExecutionError("scheduler total optimizer steps must be positive")
    primary = optimizer.optimizers[0]
    minimum_lr_ratio = float(candidate.scheduler["minimum_lr_ratio"])
    minimum_learning_rates = tuple(
        float(candidate.optimizer["learning_rate"]) * minimum_lr_ratio
        for _ in primary.param_groups
    ) if minimum_lr_ratio > 0.0 else ()
    schedulers = torch.optim.lr_scheduler
    maximum_steps = total_optimizer_steps if step_unit == "optimizer_step" else 5
    if name == "none":
        return _Scheduler(None, step_unit=step_unit, maximum_steps=0)
    if name == "constant":
        value = schedulers.ConstantLR(primary, factor=1.0, total_iters=1)
    elif name == "constant_with_warmup":
        value = schedulers.LambdaLR(primary, lambda step: min(1.0, (step + 1) / 2.0))
    elif name == "linear":
        value = schedulers.LinearLR(primary, start_factor=1.0, end_factor=0.5, total_iters=total_optimizer_steps)
    elif name == "linear_with_warmup":
        value = schedulers.LambdaLR(primary, lambda step: min(1.0, (step + 1) / 2.0) * max(0.1, 1.0 - step / 8.0))
    elif name == "step":
        value = schedulers.StepLR(primary, step_size=1, gamma=0.9)
    elif name == "multi_step":
        value = schedulers.MultiStepLR(primary, milestones=[1, 3], gamma=0.9)
    elif name == "exponential":
        value = schedulers.ExponentialLR(primary, gamma=0.95)
    elif name == "cosine":
        value = schedulers.CosineAnnealingLR(primary, T_max=max(1, 5 if step_unit == "epoch" else total_optimizer_steps))
    elif name == "cosine_warm_restarts":
        value = schedulers.CosineAnnealingWarmRestarts(primary, T_0=2)
    elif name == "cosine_with_warmup":
        value = schedulers.LambdaLR(primary, lambda step: min(1.0, (step + 1) / 2.0) * (0.5 + 0.5 * math.cos(math.pi * min(step, 8) / 8.0)))
    elif name == "polynomial":
        value = schedulers.PolynomialLR(primary, total_iters=total_optimizer_steps, power=2.0)
    elif name == "one_cycle":
        value = schedulers.OneCycleLR(primary, max_lr=max(group["lr"] for group in primary.param_groups), total_steps=total_optimizer_steps, cycle_momentum=False)
    elif name == "cyclic":
        base = min(group["lr"] for group in primary.param_groups)
        value = schedulers.CyclicLR(primary, base_lr=base * 0.5, max_lr=base, step_size_up=1, cycle_momentum=False)
    elif name == "reduce_on_plateau":
        value = schedulers.ReduceLROnPlateau(primary, factor=0.5, patience=0)
        result = _Scheduler(
            value,
            step_unit=step_unit,
            maximum_steps=maximum_steps,
            metric_driven=True,
            minimum_learning_rates=minimum_learning_rates,
        )
        result.fast_forward(
            float(candidate.scheduler["progress"]),
            total_optimizer_steps=total_optimizer_steps,
        )
        return result
    elif name == "inverse_sqrt":
        value = schedulers.LambdaLR(primary, lambda step: 1.0 / math.sqrt(max(1, step + 1)))
    elif name == "warmup_stable_decay":
        value = schedulers.LambdaLR(primary, lambda step: min(1.0, (step + 1) / 2.0) if step < 2 else max(0.1, 1.0 - (step - 2) / 8.0))
    else:
        raise LocalExecutionError(f"unsupported deployment scheduler {name!r}")
    result = _Scheduler(
        value,
        step_unit=step_unit,
        maximum_steps=maximum_steps,
        minimum_learning_rates=minimum_learning_rates,
    )
    result.fast_forward(
        float(candidate.scheduler["progress"]),
        total_optimizer_steps=total_optimizer_steps,
    )
    return result


def five_epoch_optimizer_steps(
    candidate: TargetCandidate,
    *,
    expected_train_examples: int,
) -> int:
    """Return the exact optimizer-step horizon used by the five-epoch run."""

    if type(expected_train_examples) is not int or expected_train_examples < 1:
        raise LocalExecutionError("expected training examples must be positive")
    expected_batches = math.ceil(expected_train_examples / candidate.microbatch_size)
    return 5 * math.ceil(
        expected_batches / candidate.gradient_accumulation_steps
    )


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    mover = getattr(value, "to", None)
    if callable(mover):
        return mover(device)
    return value


def _resize_sequence_fixture(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.ndim < 2 or length < 1:
        raise LocalExecutionError("sequence fixture resize requires rank >=2 and positive length")
    if value.shape[1] == length:
        return value
    indices = torch.arange(length, device=value.device).remainder(value.shape[1])
    return value.index_select(1, indices)


def bind_candidate_batch_shape(
    candidate: TargetCandidate,
    adapter: TaskAdapter,
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Apply the representative's exact declared tensor shape to a tiny fixture."""

    microbatch = candidate.microbatch_size
    if microbatch < 1:
        raise LocalExecutionError("candidate fixture batch must be positive")

    def batch_axis(tensor: torch.Tensor, count: int = microbatch) -> torch.Tensor:
        if tensor.ndim < 1 or tensor.shape[0] < 1:
            raise LocalExecutionError("fixture tensor has no reusable batch axis")
        indices = torch.arange(count, device=tensor.device).remainder(tensor.shape[0])
        return tensor.index_select(0, indices)

    def width_axis(tensor: torch.Tensor, width: int) -> torch.Tensor:
        if width < 1:
            raise LocalExecutionError("candidate fixture width must be positive")
        if tensor.shape[-1] >= width:
            return tensor[..., :width]
        repeats = math.ceil(width / tensor.shape[-1])
        return tensor.repeat(*([1] * (tensor.ndim - 1)), repeats)[..., :width]

    inputs = dict(adapter.build_model_inputs(batch))
    target = adapter.build_targets(batch)
    modality = adapter.entry.modality
    if modality == "vision":
        height = int(candidate.input_signature.get("height", 0))
        width = int(candidate.input_signature.get("width", 0))
        image = inputs.get("image")
        if min(height, width) < 1 or not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise LocalExecutionError("vision fixture cannot satisfy its declared shape")
        inputs["image"] = torch.nn.functional.interpolate(
            batch_axis(image), size=(height, width), mode="bilinear", align_corners=False
        )
        target = batch_axis(target)
        if adapter.task_kind == "image_restoration":
            target = torch.nn.functional.interpolate(
                target, size=(height, width), mode="bilinear", align_corners=False
            )
    elif modality == "nlp":
        length = int(candidate.input_signature.get("sequence_length", 0))
        decoder_length = int(candidate.input_signature.get("decoder_length", length))
        if length < 1:
            raise LocalExecutionError("NLP candidate has no declared sequence length")
        for name in ("token_ids", "attention_mask"):
            if isinstance(inputs.get(name), torch.Tensor):
                inputs[name] = batch_axis(_resize_sequence_fixture(inputs[name], length))
        if isinstance(inputs.get("decoder_ids"), torch.Tensor):
            inputs["decoder_ids"] = batch_axis(
                _resize_sequence_fixture(inputs["decoder_ids"], decoder_length)
            )
        target = batch_axis(target)
        if adapter.task_kind == "teacher_forced_seq2seq":
            target = _resize_sequence_fixture(target, decoder_length)
    elif modality == "audio":
        samples = int(candidate.input_signature.get("clip_samples", 0))
        waveform = inputs.get("waveform")
        if samples < 1 or not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise LocalExecutionError("audio fixture cannot satisfy its declared shape")
        inputs["waveform"] = torch.nn.functional.interpolate(
            batch_axis(waveform), size=samples, mode="linear", align_corners=False
        )
        inputs["sample_rate"] = int(candidate.input_signature["sample_rate"])
        target = batch_axis(target)
    elif modality == "tabular":
        feature_count = int(candidate.input_signature.get("feature_count", 0))
        categorical_fields = int(candidate.input_signature.get("categorical_fields", 0))
        dense = inputs.get("dense")
        categorical = inputs.get("categorical")
        if not isinstance(dense, torch.Tensor) or not isinstance(categorical, torch.Tensor):
            raise LocalExecutionError("tabular fixture tensors are missing")
        inputs["dense"] = width_axis(batch_axis(dense), feature_count)
        inputs["categorical"] = width_axis(
            batch_axis(categorical), categorical_fields
        )
        target = batch_axis(target)
    elif modality == "graph":
        node_count = int(candidate.input_signature.get("node_count", 0))
        edge_count = int(candidate.input_signature.get("edge_count", 0))
        node_width = int(candidate.input_signature.get("node_feature_width", 0))
        edge_width = int(candidate.input_signature.get("edge_feature_width", 0))
        nodes = inputs.get("node_features")
        edges = inputs.get("edge_features")
        if min(node_count, edge_count, node_width, edge_width) < 1 or not isinstance(
            nodes, torch.Tensor
        ) or not isinstance(edges, torch.Tensor):
            raise LocalExecutionError("graph fixture cannot satisfy its declared shape")
        base_nodes = width_axis(nodes, node_width)
        base_edges = width_axis(edges, edge_width)
        node_indices = torch.arange(microbatch * node_count, device=nodes.device)
        inputs["node_features"] = base_nodes.index_select(
            0, node_indices.remainder(base_nodes.shape[0])
        )
        edge_indices = torch.arange(microbatch * edge_count, device=nodes.device)
        local_edge = edge_indices.remainder(edge_count)
        graph = torch.div(edge_indices, edge_count, rounding_mode="floor")
        source = graph * node_count + local_edge.remainder(node_count)
        destination = graph * node_count + (local_edge + 1).remainder(node_count)
        inputs["edge_index"] = torch.stack((source, destination))
        inputs["edge_features"] = base_edges.index_select(
            0, edge_indices.remainder(base_edges.shape[0])
        )
        inputs["graph_index"] = torch.arange(microbatch, device=nodes.device).repeat_interleave(
            node_count
        )
        inputs["graph_count"] = microbatch
        try:
            from torch_geometric.data import Batch, Data

            graphs = []
            for graph_id in range(microbatch):
                node_start = graph_id * node_count
                edge_start = graph_id * edge_count
                graphs.append(
                    Data(
                        x=inputs["node_features"][
                            node_start : node_start + node_count
                        ],
                        edge_index=(
                            inputs["edge_index"][
                                :, edge_start : edge_start + edge_count
                            ]
                            - node_start
                        ),
                        edge_attr=inputs["edge_features"][
                            edge_start : edge_start + edge_count
                        ],
                    )
                )
            inputs["pyg_batch"] = Batch.from_data_list(graphs)
        except ImportError as error:
            if candidate.family_id == "cgcnn":
                raise LocalExecutionError(
                    "CGCNN requires the declared torch-geometric dependency"
                ) from error
            inputs["pyg_batch"] = None
        target = batch_axis(target)
    else:
        raise LocalExecutionError(f"unsupported fixture modality {modality!r}")
    return {**dict(batch), "inputs": inputs, "target": target}


def _all_finite(value: Any) -> bool:
    tensors = tensor_outputs(value)
    return bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _state_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        values = value.coalesce().values() if value.is_sparse else value
        return bool(torch.isfinite(values).all())
    if isinstance(value, Mapping):
        return all(_state_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_state_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _scheduler_state_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(
            (key == "mode_worse" and isinstance(item, float) and math.isinf(item))
            or _scheduler_state_finite(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return all(_scheduler_state_finite(item) for item in value)
    return _state_finite(value)


def _comparable_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() <= 16:
            return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "values": tensor.tolist()}
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sum": float(tensor.float().sum())}
    if isinstance(value, Mapping):
        return {str(key): _comparable_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_comparable_state(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity"
    return value


def _nested_close(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.allclose(left.float(), right.float(), rtol=rtol, atol=atol)
    if isinstance(left, Mapping) and isinstance(right, Mapping) and tuple(left) == tuple(right):
        return all(_nested_close(left[key], right[key], rtol=rtol, atol=atol) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(_nested_close(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right))
    return left == right


def _precision_context(candidate: TargetCandidate, device: torch.device) -> Any:
    policy = str(candidate.precision_policy["policy_id"])
    if device.type != "cuda" or policy in {"fp32_ieee", "fp32_tf32"}:
        return nullcontext()
    if PROFILE.is_a10:
        dtype = torch.float16 if policy == "fp16_grad_scaler" else torch.bfloat16
    else:
        # Volta has no BF16 or TF32 execution mode.
        dtype = torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def configure_precision_backends(candidate: TargetCandidate) -> Mapping[str, bool]:
    """Set process-global TF32 flags from one candidate, never from hardware defaults."""

    enabled = candidate.precision_policy["policy_id"] == "fp32_tf32"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    return {"matmul_allow_tf32": enabled, "cudnn_allow_tf32": enabled}


def _build_model(candidate: TargetCandidate, adapter: TaskAdapter, device: torch.device) -> FamilyModel:
    module = importlib.import_module(candidate.factory_id)
    model = module.build_model(
        output_width=adapter.target_width,
        task_kind=adapter.task_kind,
        seed=int(candidate.seed_policy["seed"]),
        architecture_parameters=candidate.architecture_parameters,
    )
    if not isinstance(model, FamilyModel) or model.family_id != candidate.family_id:
        raise LocalExecutionError("factory returned the wrong family type")
    return model.to(device).train()


def _forward_loss(
    model: FamilyModel,
    adapter: TaskAdapter,
    batch: Mapping[str, Any],
    checkpoint_enabled: bool,
) -> tuple[Any, torch.Tensor]:
    inputs = adapter.build_model_inputs(batch)
    if checkpoint_enabled:
        output = checkpoint(
            lambda: model(inputs),
            use_reentrant=False,
        )
    else:
        output = model(inputs)
    loss = model.compute_loss(output, batch, adapter)
    return output, loss


def _gradients_finite(model: FamilyModel) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    return bool(gradients) and all(_state_finite(gradient) for gradient in gradients)


def _gradient_failure_names(model: FamilyModel) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not _state_finite(parameter.grad)
    )


def _parameter_snapshot(model: FamilyModel) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in _trainable_parameters(model))


def _parameters_changed(before: tuple[torch.Tensor, ...], model: FamilyModel) -> bool:
    after = _trainable_parameters(model)
    return len(before) == len(after) and any(not torch.equal(left, right.detach()) for left, right in zip(before, after))


def _update(
    *,
    candidate: TargetCandidate,
    model: FamilyModel,
    batch: Mapping[str, Any],
    adapter: TaskAdapter,
    optimizer: _OptimizerBundle,
    scheduler: _Scheduler,
    forward_loss: Callable[[], tuple[Any, torch.Tensor]],
    scaler: torch.amp.GradScaler | None,
) -> tuple[Any, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    if candidate.optimizer["name"] == "lbfgs":
        holder: dict[str, Any] = {}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(candidate, device):
                output, loss = forward_loss()
            loss.backward()
            holder["output"] = output
            holder["loss"] = loss
            return loss

        optimizer.step(closure)
        output, loss = holder["output"], holder["loss"]
    else:
        with _precision_context(candidate, device):
            output, loss = forward_loss()
        if scaler is None:
            loss.backward()
            if not _gradients_finite(model):
                raise LocalExecutionError("training update produced missing/non-finite gradients")
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.optimizers[0])
            if not _gradients_finite(model):
                raise LocalExecutionError(
                    "scaled update produced missing/non-finite gradients: "
                    f"{_gradient_failure_names(model)}"
                )
            for item in optimizer.optimizers:
                scaler.step(item)
            scaler.update()
    scheduler.step(loss, unit="optimizer_step")
    if not _all_finite(output) or not bool(torch.isfinite(loss)):
        raise LocalExecutionError("training update produced non-finite output/loss")
    adapter.validate_output_shape(model.metric_output(output), adapter.build_targets(batch))
    return output, loss


@dataclass(frozen=True)
class LocalExecutionResult:
    version: str
    validation_signature_id: str
    representative_configuration_id: str
    family_id: str
    task_id: str
    requested_execution_mode: str
    compile_backend: str
    observed_hardware_id: str
    target_hardware_id: str
    accepted_v100_measurement: bool
    validation_harness_sha256: str
    validation_environment_sha256: str
    eager_update_passed: bool
    compile_passed: bool
    compiled_update_count: int
    outputs_finite: bool
    loss_finite: bool
    gradients_finite: bool
    parameter_change_verified: bool
    scheduler_state_verified: bool
    output_shape_verified: bool
    eager_compiled_equivalent: bool
    fixture_fingerprint: str

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def validate(self) -> None:
        if self.version != LOCAL_EXECUTION_RESULT_VERSION:
            raise LocalExecutionError("local execution result version mismatch")
        for value in (
            self.validation_signature_id,
            self.representative_configuration_id,
            self.fixture_fingerprint,
            self.validation_harness_sha256,
            self.validation_environment_sha256,
        ):
            if len(value) != 64:
                raise LocalExecutionError("local execution result has an invalid SHA-256 identity")
        if self.validation_harness_sha256 != validation_harness_sha256():
            raise LocalExecutionError("local execution result targets another validation harness")
        if self.target_hardware_id != "nvidia_tesla_v100_sxm2_32gb_nrp":
            raise LocalExecutionError("local execution target identity drifted")
        if self.accepted_v100_measurement is not False:
            raise LocalExecutionError("RTX verification cannot become V100 label evidence")
        if self.compiled_update_count != 2:
            raise LocalExecutionError("local execution must perform exactly two compiled updates")
        checks = (
            self.eager_update_passed,
            self.compile_passed,
            self.outputs_finite,
            self.loss_finite,
            self.gradients_finite,
            self.parameter_change_verified,
            self.scheduler_state_verified,
            self.output_shape_verified,
            self.eager_compiled_equivalent,
        )
        if any(value is not True for value in checks):
            raise LocalExecutionError("local execution result contains a failed trainability gate")


def validate_candidate_execution(
    candidate: TargetCandidate,
    signature: ValidationSignature,
    *,
    device: str | torch.device = "cuda",
    compile_backend: str = "inductor",
) -> LocalExecutionResult:
    """Run one eager update and two compiled updates on a tiny task fixture."""

    candidate.validate()
    signature.validate()
    if signature.representative_candidate_id != candidate.candidate_id:
        raise LocalExecutionError("signature representative differs from the candidate")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise LocalExecutionError("CUDA is unavailable for the RTX verification gate")
    if resolved_device.type == "cuda":
        observed_hardware = torch.cuda.get_device_name(resolved_device)
        configure_precision_backends(candidate)
    else:
        observed_hardware = f"local_{resolved_device.type}_test_fixture"
    adapter = adapter_for_task(candidate.task_id)
    fixture_fingerprint, real_batch = build_local_real_format_batch(adapter)
    batch = _move(real_batch, resolved_device)
    batch = bind_candidate_batch_shape(candidate, adapter, batch)
    total_optimizer_steps = five_epoch_optimizer_steps(
        candidate,
        expected_train_examples=4_096,
    )

    eager_model = _build_model(candidate, adapter, resolved_device)
    initial_state = {name: value.detach().clone() for name, value in eager_model.state_dict().items()}
    eager_optimizer = _build_optimizer(candidate, eager_model)
    eager_scheduler = _build_scheduler(
        candidate,
        eager_optimizer,
        total_optimizer_steps=total_optimizer_steps,
    )
    eager_before = _parameter_snapshot(eager_model)
    with _precision_context(candidate, resolved_device):
        eager_reference, eager_reference_loss = _forward_loss(
            eager_model,
            adapter,
            batch,
            bool(candidate.activation_checkpointing["enabled"]),
        )
    eager_scaler = (
        torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1_000)
        if resolved_device.type == "cuda" and candidate.precision_policy["gradient_scaler"]
        else None
    )
    _, eager_update_loss = _update(
        candidate=candidate,
        model=eager_model,
        batch=batch,
        adapter=adapter,
        optimizer=eager_optimizer,
        scheduler=eager_scheduler,
        forward_loss=lambda: _forward_loss(
            eager_model,
            adapter,
            batch,
            bool(candidate.activation_checkpointing["enabled"]),
        ),
        scaler=eager_scaler,
    )
    eager_scheduler.step(eager_update_loss, unit="epoch")
    if not _parameters_changed(eager_before, eager_model):
        raise LocalExecutionError("eager reference update changed no parameter")

    compiled_model = _build_model(candidate, adapter, resolved_device)
    compiled_model.load_state_dict(initial_state)
    compiled_optimizer = _build_optimizer(candidate, compiled_model)
    compiled_scheduler = _build_scheduler(
        candidate,
        compiled_optimizer,
        total_optimizer_steps=total_optimizer_steps,
    )
    compiled_before = _parameter_snapshot(compiled_model)
    scheduler_before = canonical_value(_comparable_state(compiled_scheduler.state_dict()))

    def compiled_forward_loss() -> tuple[Any, torch.Tensor]:
        return _forward_loss(
            compiled_model,
            adapter,
            batch,
            bool(candidate.activation_checkpointing["enabled"]),
        )

    compiled = torch.compile(
        compiled_forward_loss,
        backend=compile_backend,
        fullgraph=False,
        dynamic=False,
    )
    with _precision_context(candidate, resolved_device):
        compiled_reference, compiled_reference_loss = compiled()
    tolerance = (
        2e-2
        if candidate.precision_policy["autocast"]
        else 1e-3
        if candidate.precision_policy["policy_id"] in {"fp32_ieee", "fp32_tf32"}
        else 1e-5
    )
    equivalent = _nested_close(
        eager_reference,
        compiled_reference,
        rtol=tolerance,
        atol=tolerance,
    ) and torch.allclose(
        eager_reference_loss.float(),
        compiled_reference_loss.float(),
        rtol=tolerance,
        atol=tolerance,
    )
    if not equivalent:
        raise LocalExecutionError("eager and compiled reference results differ")
    compiled_scaler = (
        torch.amp.GradScaler("cuda", init_scale=1.0, growth_interval=1_000)
        if resolved_device.type == "cuda" and candidate.precision_policy["gradient_scaler"]
        else None
    )
    last_output: Any = compiled_reference
    last_loss = compiled_reference_loss
    for _ in range(2):
        last_output, last_loss = _update(
            candidate=candidate,
            model=compiled_model,
            batch=batch,
            adapter=adapter,
            optimizer=compiled_optimizer,
            scheduler=compiled_scheduler,
            forward_loss=compiled,
            scaler=compiled_scaler,
        )
        compiled_scheduler.step(last_loss, unit="epoch")
    scheduler_after_state = compiled_scheduler.state_dict()
    scheduler_after = canonical_value(_comparable_state(scheduler_after_state))
    scheduler_verified = (
        scheduler_before == scheduler_after
        if candidate.scheduler["name"] == "none"
        or float(candidate.scheduler["progress"]) == 1.0
        else scheduler_before != scheduler_after
    )
    result = LocalExecutionResult(
        version=LOCAL_EXECUTION_RESULT_VERSION,
        validation_signature_id=signature.signature_id,
        representative_configuration_id=candidate.candidate_id,
        family_id=candidate.family_id,
        task_id=candidate.task_id,
        requested_execution_mode=str(candidate.execution["mode"]),
        compile_backend=compile_backend,
        observed_hardware_id=observed_hardware,
        target_hardware_id=candidate.target_hardware_id,
        accepted_v100_measurement=False,
        validation_harness_sha256=validation_harness_sha256(),
        validation_environment_sha256=validation_environment_sha256(
            resolved_device, compile_backend
        ),
        eager_update_passed=True,
        compile_passed=True,
        compiled_update_count=2,
        outputs_finite=_all_finite(last_output),
        loss_finite=bool(torch.isfinite(last_loss)),
        gradients_finite=_gradients_finite(compiled_model),
        parameter_change_verified=_parameters_changed(compiled_before, compiled_model),
        scheduler_state_verified=scheduler_verified,
        output_shape_verified=True,
        eager_compiled_equivalent=equivalent,
        fixture_fingerprint=fixture_fingerprint,
    )
    if not _state_finite(compiled_optimizer.state_dict()) or not _scheduler_state_finite(scheduler_after_state):
        raise LocalExecutionError("optimizer/scheduler state contains a non-finite value")
    result.validate()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    return result


__all__ = [
    "LOCAL_EXECUTION_RESULT_VERSION",
    "LocalExecutionError",
    "LocalExecutionResult",
    "validate_candidate_execution",
    "bind_candidate_batch_shape",
]
