"""Deterministic workload context from supplied tensors, never sample identities."""
import numpy as np
import torch

VERSION = "perfseer_timing_context_v1"


def static_context(feature):
    """Retain graph-size/operation mix and normalized workload detail explicitly."""
    blocks = [feature.u_cont.reshape(-1).numpy(), feature.quality.reshape(-1).numpy()]
    for tensor in (feature.x_cont, feature.edge_cont):
        array = tensor.numpy().astype(np.float64)
        if len(array):
            blocks.extend((array.mean(0), array.std(0), array.min(0), array.max(0)))
        else:
            blocks.extend([np.zeros(tensor.shape[1])] * 4)
    for tensor, width in ((feature.op_family_id, 64), (feature.dtype_id, 16),
                          (feature.edge_role_id, 16)):
        values = tensor.numpy().reshape(-1)
        if len(values) and (values.min() < 0 or values.max() >= width):
            raise ValueError("categorical vocabulary exceeds fixed context schema")
        counts = np.bincount(values, minlength=width)
        blocks.append(np.log1p(counts))
    blocks.append(np.log1p([len(feature.x_cont), len(feature.edge_cont)]))
    for tensor, width in ((feature.precision_id, 16), (feature.optimizer_family_id, 9)):
        value = int(tensor.item())
        if not 0 <= value < width:
            raise ValueError("invalid workload category")
        blocks.append(np.eye(width)[value])
    value = np.concatenate(blocks).astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite workload context")
    return value


def context_matrix(features):
    cached = {}
    result = []
    for feature in features:
        key = id(feature)
        if key not in cached:
            cached[key] = static_context(feature)
        result.append(cached[key])
    return np.stack(result)
