"""Capture the connected forward/loss/backward graph without analytical fallback."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten

from .capture_export import CaptureOptions, exported_program_to_graph_ir
from perfseer_v3.capture_training import _append_optimizer_summary
from .liveness import apply_liveness
from perfseer_v3.op_registry import OperationRegistry
from .training import restore_rng, rng_state

from .io import fingerprint
from .version import HARDWARE_ID, INPUT_SCHEMA_VERSION


class TrainingWrapper(nn.Module):
    def __init__(self, model, adapter, precision, input_spec, leaves):
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.precision = precision
        self.input_spec = input_spec
        self.leaves = leaves

    def forward(self, *flat_inputs):
        values = iter(flat_inputs)
        inputs, batch = tree_unflatten([next(values) if is_tensor else value for is_tensor, value in self.leaves], self.input_spec)
        tensors, _ = tree_flatten(inputs)
        device_type = next(value.device.type for value in tensors if isinstance(value, torch.Tensor))
        policies = {"bf16": torch.bfloat16, "bf16_amp": torch.bfloat16,
                    "mixed_structured": torch.bfloat16, "fp16_grad_scaler": torch.float16,
                    "fp16_amp": torch.float16, "fp32": None, "fp32_ieee": None,
                    "fp32_tf32": None, "tf32": None}
        if self.precision not in policies:
            raise ValueError(f"unsupported workload precision: {self.precision}")
        enabled = policies[self.precision] is not None
        dtype = policies[self.precision] or torch.bfloat16
        with torch.backends.mkldnn.flags(enabled=False), torch.autocast(device_type, dtype=dtype, enabled=enabled):
            if self.adapter is None:
                output = self.model(*inputs)
                loss = torch.nn.functional.mse_loss(output.float(), torch.zeros_like(output).float())
            else:
                output = self.model(inputs)
                loss = self.model.compute_loss(output, batch, self.adapter)
        outputs, _ = tree_flatten(output)
        return (loss, *(value for value in outputs if isinstance(value, torch.Tensor)))


def _gradient_tolerances(precision):
    if precision in ("bf16", "bf16_amp", "mixed_structured"):
        return 5e-2, 1e-2
    return 2e-2, 2e-3


def capture(model, inputs, batch, adapter, training, *, architecture_key):
    """Return canonical model-design input plus eager/joint gradient parity evidence."""
    with torch.backends.mkldnn.flags(enabled=False):
        return _capture(model, inputs, batch, adapter, training, architecture_key=architecture_key)


def _capture(model, inputs, batch, adapter, training, *, architecture_key):
    model.train()
    flat_inputs, input_spec = tree_flatten((inputs, batch))
    leaves = [(isinstance(value, torch.Tensor), None if isinstance(value, torch.Tensor) else value) for value in flat_inputs]
    flat_inputs = [value for value in flat_inputs if isinstance(value, torch.Tensor)]
    wrapper = TrainingWrapper(model, adapter, training["precision"], input_spec, leaves)
    state = {name: value.detach().clone() for name, value in wrapper.state_dict().items()}
    rng = rng_state()
    eager_loss, *eager_outputs = wrapper(*flat_inputs)
    if not torch.isfinite(eager_loss):
        raise ValueError("nonfinite profiling loss")
    parameters = dict(wrapper.named_parameters())
    active = [parameter for parameter in parameters.values() if parameter.requires_grad]
    gradients = torch.autograd.grad(eager_loss, active, allow_unused=True)
    eager_gradients = dict(zip((name for name, value in parameters.items() if value.requires_grad), gradients))
    wrapper.load_state_dict(state)
    restore_rng(rng)
    buffers = dict(wrapper.named_buffers())
    state_names = [*parameters, *buffers]
    gradient_names = [name for name, value in eager_gradients.items() if value is not None]
    active_indices = [index for index, name in enumerate(parameters) if parameters[name].requires_grad]

    def joint_function(*values):
        bound = dict(zip(state_names, values[:len(state_names)], strict=True))
        loss, *outputs = torch.func.functional_call(wrapper, bound, values[len(state_names):])
        grads = torch.autograd.grad(loss, [values[i] for i in active_indices], allow_unused=True)
        return (loss, *(value for value in grads if value is not None), *outputs)

    joint_inputs = [*parameters.values(), *buffers.values(), *flat_inputs]
    joint = make_fx(joint_function, tracing_mode="real")(*joint_inputs)
    wrapper.load_state_dict(state)
    restore_rng(rng)
    placeholders = [node for node in joint.graph.nodes if node.op == "placeholder"]
    kinds = [(name, "PARAMETER") for name in parameters] + [(name, "BUFFER") for name in buffers] + [(None, "USER_INPUT") for _ in flat_inputs]
    input_specs = [SimpleNamespace(arg=SimpleNamespace(name=node.name), kind=kind, target=name)
                   for node, (name, kind) in zip(placeholders, kinds, strict=True)]
    output_nodes = next(node for node in joint.graph.nodes if node.op == "output").args[0]
    loss_output = output_nodes[0].name
    with torch.backends.mkldnn.flags(enabled=False):
        result = joint(*joint_inputs)
    gradient_map = {node.name: name for node, name in zip(output_nodes[1:1 + len(gradient_names)], gradient_names, strict=True)}
    forward_nodes = set()

    def ancestors(node):
        if node in forward_nodes:
            return
        forward_nodes.add(node)
        for parent in node.all_input_nodes:
            ancestors(parent)

    for node in output_nodes[1 + len(gradient_names):]:
        ancestors(node)
    output_specs = []
    checked = 0
    gradient_rtol, gradient_atol = _gradient_tolerances(training["precision"])
    for node, value in zip(output_nodes, result, strict=True):
        name = node.name if isinstance(node, torch.fx.Node) else str(node)
        kind = "GRADIENT_TO_PARAMETER" if name in gradient_map else "USER_OUTPUT"
        output_specs.append(SimpleNamespace(arg=SimpleNamespace(name=name), kind=kind))
        if name == loss_output:
            torch.testing.assert_close(value, eager_loss, rtol=1e-3, atol=1e-5)
        if name in gradient_map:
            expected = eager_gradients[gradient_map[name]]
            if expected is None:
                raise ValueError(f"unexpected joint gradient: {name}")
            finite_value = value._values() if value.is_sparse else value
            finite_expected = expected._values() if expected.is_sparse else expected
            if not torch.isfinite(finite_value).all() or not torch.isfinite(finite_expected).all():
                raise ValueError(f"nonfinite parameter gradient: {name}")
            torch.testing.assert_close(value, expected, rtol=gradient_rtol, atol=gradient_atol)
            checked += 1
    for expected, actual in zip(eager_outputs, result[1 + len(gradient_names):], strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
    if checked != sum(value is not None for value in eager_gradients.values()):
        raise ValueError("joint capture omitted parameter gradients")
    registry = OperationRegistry.load()
    optimizer = dict(training["optimizer"])
    optimizer["gradient_accumulation_steps"] = training["gradient_accumulation_steps"]
    program = SimpleNamespace(
        graph_module=joint,
        graph_signature=SimpleNamespace(input_specs=input_specs, output_specs=output_specs),
        range_constraints={},
    )
    graph = exported_program_to_graph_ir(
        program, registry=registry, capture_mode="strict", source_fingerprint=architecture_key,
        model_fingerprint=architecture_key, args=(inputs,), kwargs={},
        options=CaptureOptions(
            training_mode=True, precision=training["precision"], optimizer_config=optimizer,
            training_config=training, target_hardware_id=HARDWARE_ID,
        ), replay_validated=True, replay_samples=1,
    )
    fx_nodes = [node for node in joint.graph.nodes if node.op == "call_function" and isinstance(node.meta.get("val"), (torch.Tensor, tuple, list))]
    loss_position = next(index for index, node in enumerate(fx_nodes) if node.name == loss_output)
    if len(fx_nodes) != len(graph.nodes):
        raise ValueError("joint tensor nodes and GraphIR nodes differ")
    graph = replace(
        graph,
        nodes=tuple(replace(node, phase="forward" if fx_nodes[index] in forward_nodes else "loss" if index <= loss_position else "backward") for index, node in enumerate(graph.nodes)),
        coverage=replace(graph.coverage, backward_capture_quality="strict"),
        global_features=replace(graph.global_features,
                                total_parameter_numel=sum(value.numel() for value in parameters.values()),
                                total_parameter_bytes=sum(value.numel() * value.element_size() for value in parameters.values()),
                                total_buffer_bytes=sum(value.numel() * value.element_size() for value in buffers.values())),
        metadata={**graph.metadata, "batch_size": training["microbatch_size"], "backward_capture": {"backend": "fx_autograd_joint", "quality": "strict"}},
    )
    # Autocast leaves the measured workloads' parameters and optimizer state in
    # FP32. The v3 summary otherwise incorrectly assumes BF16 parameter gradients.
    summarized = _append_optimizer_summary(replace(graph, precision="fp32"), registry, optimizer["name"], optimizer)
    graph = apply_liveness(replace(summarized, precision=graph.precision))
    graph.validate()
    payload = graph.to_dict()
    nodes = payload.pop("nodes")
    edges = payload.pop("tensor_edges")
    design = {
        "version": INPUT_SCHEMA_VERSION,
        "INPUT_SPECS": payload.pop("input_signature"),
        "NODE_SPECS": nodes,
        "EDGE_SPECS": edges,
        "TRAINING_CONFIG": payload.pop("training_config"),
        "graph": payload,
    }
    return design, {"backward_capture": "strict", "gradient_parameters_checked": checked,
                    "forward_outputs_checked": len(eager_outputs), "nodes": len(nodes), "edges": len(edges)}


def graph_from_design(design):
    from perfseer_v3.graph_ir_v3 import GraphIRV3

    if design["version"] != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported model-design version")
    graph = GraphIRV3.from_dict({
        **design["graph"], "input_signature": design["INPUT_SPECS"],
        "nodes": design["NODE_SPECS"], "tensor_edges": design["EDGE_SPECS"],
        "training_config": design["TRAINING_CONFIG"],
    })
    if graph.coverage.backward_capture_quality != "strict":
        raise ValueError("v3.1 requires strictly captured backward graphs")
    return graph
