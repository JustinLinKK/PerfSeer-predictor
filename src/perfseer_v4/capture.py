"""Capture only the strict, replay-verified training execution graph."""

from perfseer_v31.capture_training import capture as capture_training, graph_from_design as training_graph
from perfseer_v32.capture import preserve_execution

from .version import INPUT_SCHEMA_VERSION


def capture(model, inputs, batch, adapter, training, *, architecture_key):
    with preserve_execution(model):
        train, audit = capture_training(model, inputs, batch, adapter, training, architecture_key=architecture_key)
    design = {"version": INPUT_SCHEMA_VERSION, "training": train}
    graph_from_design(design)
    return design, {"training": audit}


def graph_from_design(design):
    if design.get("version") != INPUT_SCHEMA_VERSION or set(design) != {"version", "training"}:
        raise ValueError("expected a v4 training-only input")
    graph = training_graph(design["training"])
    if not graph.training_mode or graph.coverage.capture_quality != "strict" or not graph.coverage.replay_validated:
        raise ValueError("training requires a strict, replay-verified graph")
    graph.validate()
    return graph
