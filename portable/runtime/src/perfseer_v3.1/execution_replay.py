"""Independent native/captured execution dataflow evidence for profiling inputs.

Profiling measures executed operations, including workloads whose low-precision
arithmetic overflows. Numerical equality alone is insufficient in that case.
This recorder hashes the actual ATen dispatch DAG in each execution, independently
of FX, without changing an operation or retaining its activation tensors.
"""

from collections import Counter
import hashlib
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from .io import fingerprint

VERSION = "perfseer_execution_dataflow_v1"


class ExecutionTrace(TorchDispatchMode):
    def __init__(self, bindings):
        super().__init__()
        self.tensors = {}
        self.operations = Counter()
        self.kinds = Counter()
        self.constants = Counter()
        for name, value in bindings.items():
            self.bind(value, ("input", name, self.metadata(value)))

    @staticmethod
    def metadata(value):
        if value.layout != torch.strided:
            raise ValueError("execution replay requires dense tensor operands")
        return (list(value.shape), list(value.stride()), str(value.dtype), str(value.device))

    def bind(self, value, provenance):
        key = fingerprint(provenance)
        self.tensors[id(value)] = (weakref.ref(value), key)
        return key

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            prior = self.tensors.get(id(value))
            if prior is not None and prior[0]() is value:
                return prior[1]
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            key = self.bind(value, ("constant", self.metadata(value), hashlib.sha256(raw).hexdigest()))
            self.constants[key] += 1
            return key
        if isinstance(value, (tuple, list)):
            return [self.encode(v) for v in value]
        if isinstance(value, dict):
            return {key: self.encode(v) for key, v in value.items()}
        if isinstance(value, float):
            return {"float": value.hex()}
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # These are identity views with no arithmetic, allocation or mutation.
        # Autograd inserts detach nodes which explicit backward replay omits.
        if str(func) in ("aten.detach.default", "aten.alias.default"):
            tag = self.encode(args[0])
            result = func(*args, **kwargs)
            self.tensors[id(result)] = (weakref.ref(result), tag)
            return result
        bound = {arg.name: args[index] if index < len(args) else kwargs.get(arg.name, arg.default_value)
                 for index, arg in enumerate(func._schema.arguments)}
        operands = self.encode(bound)
        result = func(*args, **kwargs)
        values, _ = tree_flatten(result)
        outputs = []
        for index, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                outputs.append(self.bind(value, ("operation", str(func), operands, index, self.metadata(value))))
            else:
                outputs.append(self.encode(value))
        self.operations[fingerprint((str(func), operands, outputs))] += 1
        self.kinds[str(func)] += 1
        return result

    def receipt(self, outputs):
        # Encode outside this mode: inspecting a constant must not add nodes.
        return {"version": VERSION, "outputs": self.encode(outputs),
                "operations": dict(sorted(self.operations.items())),
                "operator_kinds": dict(sorted(self.kinds.items())),
                "constants": dict(sorted(self.constants.items()))}


def verify_execution(eager, replay):
    if eager != replay:
        raise ValueError("native/captured execution dataflow differs")
    return {"version": VERSION, "sha256": fingerprint(eager),
            "operations_checked": sum(eager["operations"].values()),
            "outputs_checked": len(eager["outputs"]),
            "operator_kinds": eager["operator_kinds"],
            "constant_tensors_checked": sum(eager["constants"].values())}
