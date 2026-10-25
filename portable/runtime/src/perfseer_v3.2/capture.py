"""Paired logical execution graphs with state-preserving eager replay checks."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

from perfseer_v3.graph_ir_v3 import GraphIRV3
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v31.capture_export import CaptureOptions, exported_program_to_graph_ir
from perfseer_v31.capture_training import capture as capture_training, graph_from_design as training_graph
from .training import rng_state, restore_rng

from .version import HARDWARE_ID, INPUT_SCHEMA_VERSION, INFERENCE_SCHEMA_VERSION, MODES


def inference_settings(training):
    return {"precision": training["precision"], "microbatch_size": training["microbatch_size"],
            "backend": training.get("backend", "unknown"), "optimizer": {"name": "none"},
            "scheduler": {"name": "none"}, "gradient_accumulation_steps": 1}


@contextmanager
def preserve_execution(model):
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    rng = rng_state()
    try:
        yield
    finally:
        model.load_state_dict(state)
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)


class InferenceWrapper(nn.Module):
    def __init__(self, model, precision):
        super().__init__()
        self.model, self.precision = model, precision

    def forward(self, *inputs):
        policies = {"bf16": torch.bfloat16, "bf16_amp": torch.bfloat16,
                    "mixed_structured": torch.bfloat16, "fp16_grad_scaler": torch.float16,
                    "fp16_amp": torch.float16, "fp32": None, "fp32_ieee": None,
                    "fp32_tf32": None, "tf32": None}
        if self.precision not in policies:
            raise ValueError(f"unsupported workload precision: {self.precision}")
        dtype = policies[self.precision]
        device = next(value.device.type for value in inputs if isinstance(value, torch.Tensor))
        with torch.no_grad(), torch.autocast(device, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            outputs, _ = tree_flatten(self.model(*inputs))
            return tuple(value for value in outputs if isinstance(value, torch.Tensor))


def capture_inference(model, inputs, training, *, architecture_key):
    with preserve_execution(model), torch.backends.mkldnn.flags(enabled=False), torch.no_grad():
        model.eval()
        wrapper = InferenceWrapper(model, training["precision"])
        parameters, buffers = dict(wrapper.named_parameters()), dict(wrapper.named_buffers())
        names = [*parameters, *buffers]
        values = [*parameters.values(), *buffers.values(), *inputs]

        def inference(*args):
            return torch.func.functional_call(wrapper, dict(zip(names, args[:len(names)], strict=True)), args[len(names):])

        with preserve_execution(model):
            expected = tuple(value.detach().clone() for value in inference(*values))
        if not expected or not all(torch.isfinite(value).all() for value in expected):
            raise ValueError("nonfinite or empty inference outputs")
        with preserve_execution(model):
            traced = make_fx(inference, tracing_mode="real")(*values)
        with preserve_execution(model):
            actual = traced(*values)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-5)
        placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
        kinds = [(name, "PARAMETER") for name in parameters] + [(name, "BUFFER") for name in buffers] + [(None, "USER_INPUT") for _ in inputs]
        input_specs = [SimpleNamespace(arg=SimpleNamespace(name=node.name), kind=kind, target=name)
                       for node, (name, kind) in zip(placeholders, kinds, strict=True)]
        outputs = next(node for node in traced.graph.nodes if node.op == "output").args[0]
        output_specs = [SimpleNamespace(arg=SimpleNamespace(name=node.name), kind="USER_OUTPUT") for node in outputs]
        program = SimpleNamespace(graph_module=traced, graph_signature=SimpleNamespace(input_specs=input_specs, output_specs=output_specs), range_constraints={})
        settings = inference_settings(training)
        graph = exported_program_to_graph_ir(
            program, registry=OperationRegistry.load(), capture_mode="strict",
            source_fingerprint=architecture_key, model_fingerprint=architecture_key,
            args=tuple(inputs), kwargs={},
            options=CaptureOptions(training_mode=False, precision=training["precision"],
                                   optimizer_config=settings["optimizer"], training_config=settings,
                                   target_hardware_id=HARDWARE_ID),
            replay_validated=True, replay_samples=1,
            capture_metadata={"inference_capture": {"backend": "fx_functional_eval", "quality": "strict"}},
        )
        graph = replace(graph, global_features=replace(graph.global_features,
                        total_parameter_numel=sum(value.numel() for value in parameters.values()),
                        total_parameter_bytes=sum(value.numel() * value.element_size() for value in parameters.values()),
                        total_buffer_bytes=sum(value.numel() * value.element_size() for value in buffers.values())),
                        metadata={**graph.metadata, "batch_size": training["microbatch_size"]})
        graph.validate()
        return {"version": INFERENCE_SCHEMA_VERSION, "graph": graph.to_dict()}, {
            "inference_capture": "strict", "forward_outputs_checked": len(expected),
            "nodes": len(graph.nodes), "edges": len(graph.tensor_edges)}


def capture(model, inputs, batch, adapter, training, *, architecture_key):
    if adapter is not None:
        raise ValueError("v3.2 paired capture requires the native calibration model interface")
    with preserve_execution(model):
        inference, inference_audit = capture_inference(model, inputs, training, architecture_key=architecture_key)
        train, training_audit = capture_training(model, inputs, batch, adapter, training, architecture_key=architecture_key)
    design = {"version": INPUT_SCHEMA_VERSION, "training": train, "inference": inference}
    graphs_from_design(design)
    return design, {"training": training_audit, "inference": inference_audit}


def graphs_from_design(design):
    if design.get("version") != INPUT_SCHEMA_VERSION or any(mode not in design for mode in MODES):
        raise ValueError("expected a v3.2 paired-graph input")
    train = training_graph(design["training"])
    inference = design["inference"]
    if inference.get("version") != INFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported inference graph version")
    graph = GraphIRV3.from_dict(inference["graph"])
    if graph.training_mode or graph.coverage.capture_quality != "strict" or not graph.coverage.replay_validated:
        raise ValueError("inference requires a strict, replay-verified eval graph")
    if any(node.phase != "forward" for node in graph.nodes) or graph.optimizer_config != {"name": "none"}:
        raise ValueError("inference graph contains training operations or optimizer state")
    if graph.training_config != inference_settings(graph.training_config):
        raise ValueError("inference settings contain training-only fields")
    graph.validate()
    return {"training": train, "inference": graph}
