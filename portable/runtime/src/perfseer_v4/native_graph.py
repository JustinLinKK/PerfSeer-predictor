"""Bounded symbolic forward graphs from native source, without workload execution.

The reader interprets syntax over declaration and tensor-metadata records. It
never imports source modules, allocates workload tensors, or runs torch. Edges
are source-level producer/consumer relations, not an execution trace. Backward,
optimizer, layout, aliasing and kernel details are explicitly unavailable.
"""

import ast
from collections import Counter
from dataclasses import dataclass
from itertools import islice
import math

from .native_static import Declaration, Reader, Returned, Symbol, parameter_count


VERSION = "perfseer_v4_native_symbolic_graph_v1"


@dataclass(frozen=True)
class Tensor:
    node: int
    shape: tuple
    dtype: str = "float32"

    def __bool__(self):
        raise ValueError("data-dependent tensor control flow is unsupported")


def tensors(value):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def broadcast(left, right):
    result = []
    for i in range(max(len(left), len(right))):
        a = left[-i - 1] if i < len(left) else 1
        b = right[-i - 1] if i < len(right) else 1
        if a != b and a != 1 and b != 1:
            raise ValueError(f"incompatible symbolic broadcast: {left}, {right}")
        result.append(max(a, b))
    return tuple(reversed(result))


def parameter_shapes(module):
    name, args, options = module.name, module.args, module.kwargs
    def arg(index, key, default=None):
        return args[index] if len(args) > index else options.get(key, default)
    if name == "Parameter":
        return [list(args[0].args)]
    if name == "Linear":
        incoming, outgoing = arg(0, "in_features"), arg(1, "out_features")
        return [[outgoing, incoming]] + ([[outgoing]] if arg(2, "bias", True) else [])
    if name in {"Conv1d", "Conv2d", "ConvTranspose2d"}:
        incoming, outgoing = arg(0, "in_channels"), arg(1, "out_channels")
        kernel = arg(2, "kernel_size")
        kernel = list(kernel) if isinstance(kernel, (list, tuple)) else [kernel] * (1 if name == "Conv1d" else 2)
        channels = [incoming, outgoing // options.get("groups", 1)] if name == "ConvTranspose2d" else [outgoing, incoming // options.get("groups", 1)]
        return [[*channels, *kernel]] + ([[outgoing]] if options.get("bias", True) else [])
    if name in {"Embedding", "EmbeddingBag"}:
        return [[arg(0, "num_embeddings"), arg(1, "embedding_dim")]]
    if name in {"LayerNorm", "RMSNorm"}:
        shape = arg(0, "normalized_shape")
        shape = list(shape) if isinstance(shape, (list, tuple)) else [shape]
        return [shape] * ((2 if name == "LayerNorm" else 1) if options.get("elementwise_affine", True) else 0)
    if name in {"BatchNorm1d", "BatchNorm2d", "GroupNorm"}:
        return [[arg(1 if name == "GroupNorm" else 0, "num_channels" if name == "GroupNorm" else "num_features")]] * (2 if options.get("affine", True) else 0)
    if name in {"RNN", "GRU", "LSTM"}:
        hidden = arg(1, "hidden_size"); directions = 2 if options.get("bidirectional", False) else 1
        gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[name]; shapes = []
        for layer in range(arg(2, "num_layers", 1)):
            incoming = arg(0, "input_size") if layer == 0 else hidden * directions
            shapes.extend(([[gates * hidden, incoming], [gates * hidden, hidden]] + ([[gates * hidden]] * 2 if options.get("bias", True) else [])) * directions)
        return shapes
    if name == "PReLU":
        return [[arg(0, "num_parameters", 1)]]
    return []


class GraphReader(Reader):
    def __init__(self, text):
        super().__init__(text)
        self.nodes, self.edges, self.branches = [], [], []
        self.paths, self.parameter_nodes = {}, {}
        self.declarations = []
        self.line, self.repeat = 0, 1
        self.tracing = False

    def add(self, op, inputs=(), shape=(), dtype=None, **attributes):
        values = list(tensors(inputs))
        dtype = dtype or (values[0].dtype if values else "float32")
        index = len(self.nodes)
        if index >= 20000:
            raise ValueError("symbolic graph exceeds 20000 nodes")
        node = {"id": index, "op": op, "shape": list(shape), "dtype": dtype,
                "phase": "forward", "source_line": self.line, "repeat": self.repeat,
                "parameters": 0, "attributes": attributes}
        self.nodes.append(node)
        counts = Counter(value.node for value in values)
        for source, count in counts.items():
            parent = self.nodes[source]
            slots = [slot for slot, value in enumerate(values) if value.node == source]
            self.edges.append({"source": source, "target": index, "role": "parameter" if parent["op"] == "parameter" else "activation",
                               "source_slot": 0, "target_slot": slots[0], "target_slots": slots, "shape": parent["shape"],
                               "dtype": parent["dtype"], "multiplicity": count})
        return Tensor(index, tuple(shape), dtype)

    def register(self, value, path="model", seen=None):
        seen = set() if seen is None else seen
        if isinstance(value, Declaration):
            if id(value) in seen:
                return
            seen.add(id(value)); self.paths[id(value)] = path
            self.declarations.append(value)
            for i, child in enumerate(value.args):
                self.register(child, path + "." + str(i), seen)
            for name, child in value.attributes.items():
                self.register(child, path + "." + name, seen)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                self.register(child, path + "." + str(i), seen)

    def parameter(self, value):
        if id(value) not in self.parameter_nodes:
            count = parameter_count(value)
            shape = tuple(value.args[0].args) if value.name == "Parameter" else ()
            result = self.add("parameter", shape=shape, module=self.paths.get(id(value), value.name),
                              module_type=value.name, count=count, frozen=value.frozen,
                              shape_known=value.name == "Parameter", parameter_shapes=parameter_shapes(value))
            if sum(math.prod(item) for item in parameter_shapes(value)) != count:
                raise ValueError("symbolic parameter shapes differ from count")
            self.nodes[result.node]["parameters"] = count
            self.parameter_nodes[id(value)] = result
        return self.parameter_nodes[id(value)]

    @staticmethod
    def sequence(value):
        if isinstance(value, Declaration) and value.name in {"Sequential", "ModuleList"}:
            return value.args[0] if len(value.args) == 1 and isinstance(value.args[0], (list, tuple)) else value.args
        return value

    def bounded_sequence(self, value):
        result = list(islice(self.sequence(value), 4097))
        if len(result) > 4096:
            raise ValueError("symbolic loop exceeds 4096 iterations")
        return result

    def statements(self, nodes, env):
        if not self.tracing:
            return super().statements(nodes, env)
        for node in nodes:
            self.tick(); self.line = node.lineno
            if isinstance(node, ast.If):
                condition = self.expr(node.test, env)
                self.branches.append({"line": node.lineno, "condition": ast.unparse(node.test), "selected": bool(condition)})
                self.statements(node.body if condition else node.orelse, env)
            elif isinstance(node, ast.Assert):
                if not self.expr(node.test, env):
                    raise ValueError("symbolic forward assertion failed")
            elif isinstance(node, ast.For):
                values = self.bounded_sequence(self.expr(node.iter, env))
                # Long scalar-index loops are summarized once. Module loops are
                # expanded, preserving each distinct module and its parameters.
                compressed = len(values) > 16 and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"
                old_repeat = self.repeat
                if compressed:
                    before = env.copy(); lengths = {key: len(value) for key, value in env.items() if isinstance(value, list)}
                    start = len(self.nodes); self.repeat *= len(values)
                    assigned = {part.id for statement in node.body for part in ast.walk(statement) if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
                    entries = {}
                    for name in assigned & before.keys():
                        previous = before[name]
                        if isinstance(previous, Tensor):
                            entries[name] = self.add("loop_input", [previous], previous.shape, previous.dtype, variable=name, iterations=len(values))
                            env[name] = entries[name]
                for value in values[:1] if compressed else values:
                    self.assign(node.target, value, env)
                    self.statements(node.body, env)
                if compressed:
                    for name, previous in before.items():
                        current = env[name]
                        if name in entries and isinstance(current, Tensor):
                            self.edges.append({"source": current.node, "target": entries[name].node, "role": "loop_carried",
                                               "source_slot": 0, "target_slot": 0, "shape": list(current.shape), "dtype": current.dtype,
                                               "multiplicity": len(values) - 1})
                        if name in lengths and len(current) > lengths[name]:
                            tail = current[lengths[name]:]
                            current.extend(tail * (len(values) - 1))
                    self.branches.append({"line": node.lineno, "loop_iterations": len(values), "summarized": True,
                                          "body_node_start": start, "body_node_stop": len(self.nodes)})
                    self.repeat = old_repeat
                self.statements(node.orelse, env)
            elif isinstance(node, ast.Raise):
                raise ValueError("selected symbolic source raises an exception")
            else:
                super().statements([node], env)

    def expr(self, node, env):
        if not self.tracing:
            return super().expr(node, env)
        self.tick()
        if isinstance(node, ast.Name) and node.id in {"F", "math", "enumerate", "zip", "tuple"}:
            return Symbol(node.id)
        if isinstance(node, ast.Slice):
            return slice(*(self.expr(value, env) if value is not None else None for value in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value, env)
            if isinstance(value, Declaration):
                if node.attr in value.attributes:
                    result = value.attributes[node.attr]
                    return self.parameter(result) if isinstance(result, Declaration) and result.name == "Parameter" else result
                definition = self.definitions.get(value.name)
                if isinstance(definition, ast.ClassDef):
                    method = next((n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == node.attr), None)
                    if method is not None:
                        return ("bound", value, method)
            if isinstance(value, Tensor):
                if node.attr == "shape":
                    return value.shape
                if node.attr == "ndim":
                    return len(value.shape)
                if node.attr == "dtype":
                    return value.dtype
                if node.attr == "device":
                    return "symbolic_device"
                return ("tensor_method", value, node.attr)
            if isinstance(value, dict) and node.attr in value:
                return value[node.attr]
        if isinstance(node, ast.Subscript):
            value, index = self.expr(node.value, env), self.expr(node.slice, env)
            if isinstance(value, Tensor):
                parts = index if isinstance(index, tuple) else (index,)
                shape, dimension = [], 0
                for part in parts:
                    if part is None:
                        shape.append(1); continue
                    size = value.shape[dimension]; dimension += 1
                    if isinstance(part, slice):
                        shape.append(len(range(*part.indices(size))))
                    elif not isinstance(part, int):
                        raise ValueError("unsupported symbolic tensor index")
                return self.add("getitem", [value], tuple(shape) + value.shape[dimension:], index=repr(index))
            return self.sequence(value)[index]
        if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return super().expr(node, env)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.UnaryOp)):
            if isinstance(node, ast.BinOp):
                left, right = self.expr(node.left, env), self.expr(node.right, env)
                operands = [left, right]; operation = type(node.op).__name__
            elif isinstance(node, ast.UnaryOp):
                operands = [self.expr(node.operand, env)]; operation = type(node.op).__name__
            else:
                operands = [self.expr(node.left, env), *(self.expr(x, env) for x in node.comparators)]
                operation = "compare_" + "_".join(type(op).__name__ for op in node.ops)
            values = list(tensors(operands))
            if values:
                shape = values[0].shape
                for value in values[1:]:
                    shape = broadcast(shape, value.shape)
                scalars = [x for x in operands if isinstance(x, (str, int, float, bool))]
                return self.add(operation, values, shape, "bool" if isinstance(node, ast.Compare) else None, scalars=scalars)
        return super().expr(node, env)

    def call(self, function, args, kwargs, env):
        if not self.tracing:
            return super().call(function, args, kwargs, env)
        if isinstance(function, tuple) and function[0] == "bound":
            return self.function(function[2], [function[1], *args], kwargs, env)
        if isinstance(function, Declaration):
            if function.name in self.definitions:
                definition = self.definitions[function.name]
                forward = next((n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
                if forward is None:
                    raise ValueError("module has no symbolic forward")
                return self.function(forward, [function, *args], kwargs, env)
            if function.name == "Sequential":
                result = args[0]
                for child in self.sequence(function):
                    result = self.call(child, [result], {}, env)
                return result
            return self.module(function, args, kwargs)
        if isinstance(function, tuple) and function[0] == "tensor_method":
            return self.operation(function[2], [function[1], *args], kwargs)
        if isinstance(function, Symbol):
            name = function.name
            if name in {"enumerate", "zip", "tuple"}:
                values = [self.sequence(value) for value in args]
                if name == "tuple":
                    return tuple(self.bounded_sequence(values[0]))
                return {"enumerate": enumerate, "zip": zip, "tuple": tuple}[name](*values, **kwargs)
            if name == "math.sqrt":
                return math.sqrt(*args)
            if name == "torch.finfo":
                return {"min": -3.4028234663852886e38}
            if name.startswith(("torch.", "F.")):
                return self.operation(name.split(".")[-1], args, kwargs)
        return super().call(function, args, kwargs, env)

    def module(self, module, args, kwargs):
        name, pargs, options = module.name, module.args, module.kwargs
        value = args[0]; shape = list(value.shape)
        def arg(index, key, default=None):
            return pargs[index] if len(pargs) > index else options.get(key, default)
        if name == "Linear":
            shape[-1] = arg(1, "out_features")
        elif name in {"Conv1d", "Conv2d", "ConvTranspose2d", "AvgPool2d"}:
            rank = 1 if name == "Conv1d" else 2
            def dims(item):
                return list(item) if isinstance(item, (list, tuple)) else [item] * rank
            pool = name == "AvgPool2d"
            kernel = dims(arg(0 if pool else 2, "kernel_size"))
            stride = dims(arg(1 if pool else 3, "stride", kernel if pool else 1))
            padding = dims(arg(2 if pool else 4, "padding", 0))
            dilation = dims(options.get("dilation", 1))
            for i in range(rank):
                old = shape[-rank + i]
                shape[-rank + i] = ((old - 1) * stride[i] - 2 * padding[i] + dilation[i] * (kernel[i] - 1) + options.get("output_padding", 0) + 1
                                    if name == "ConvTranspose2d" else (old + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1)
            if not pool:
                shape[1] = arg(1, "out_channels")
        elif name == "AdaptiveAvgPool2d":
            size = arg(0, "output_size"); shape[-2:] = size if isinstance(size, (list, tuple)) else [size, size]
        elif name == "PixelShuffle":
            factor = arg(0, "upscale_factor"); shape[1] //= factor ** 2; shape[-2] *= factor; shape[-1] *= factor
        elif name == "Upsample":
            if options.get("size") is not None:
                shape[2:] = options["size"]
            else:
                scale = options["scale_factor"]; shape[2:] = [int(x * scale) for x in shape[2:]]
        elif name == "Embedding":
            shape.append(arg(1, "embedding_dim"))
        elif name == "EmbeddingBag":
            shape = [args[1].shape[0], arg(1, "embedding_dim")]
        elif name in {"LSTM", "GRU", "RNN"}:
            shape[-1] = arg(1, "hidden_size") * (2 if options.get("bidirectional", False) else 1)
        count = parameter_count(module)
        inputs = [*args, kwargs]
        if count:
            inputs.append(self.parameter(module))
        result = self.add("nn." + name, inputs, shape, "float32" if name in {"Embedding", "EmbeddingBag"} else value.dtype,
                          module=self.paths.get(id(module), name), args=pargs, kwargs=options)
        if name in {"LSTM", "GRU", "RNN"}:
            state_shape = (arg(2, "num_layers", 1) * (2 if options.get("bidirectional", False) else 1), value.shape[0 if options.get("batch_first", False) else 1], arg(1, "hidden_size"))
            hidden = self.add("recurrent_hidden", [result], state_shape)
            state = (hidden, self.add("recurrent_cell", [result], state_shape)) if name == "LSTM" else hidden
            return result, state
        return result

    def operation(self, name, args, kwargs):
        values = list(tensors(args)); value = values[0] if values else None
        shape = list(value.shape) if value is not None else []
        dtype = value.dtype if value is not None else "float32"
        def arg(index, key, default=None):
            return args[index] if len(args) > index else kwargs.get(key, default)
        if name in {"view", "reshape"}:
            desired = list(args[1]) if len(args) == 2 and isinstance(args[1], (list, tuple)) else list(args[1:])
            if desired.count(-1) > 1:
                raise ValueError("ambiguous symbolic reshape")
            if -1 in desired:
                other = math.prod(x for x in desired if x != -1)
                if not other or math.prod(shape) % other:
                    raise ValueError("incompatible symbolic reshape")
                desired[desired.index(-1)] = math.prod(shape) // other
            if math.prod(desired) != math.prod(shape):
                raise ValueError("symbolic reshape changes element count")
            shape = desired
        elif name == "flatten":
            start, end = arg(1, "start_dim", 0), arg(2, "end_dim", -1)
            end %= len(shape); shape = shape[:start] + [math.prod(shape[start:end + 1])] + shape[end + 1:]
        elif name == "permute":
            order = args[1] if len(args) == 2 and isinstance(args[1], (list, tuple)) else args[1:]
            shape = [shape[i] for i in order]
        elif name == "transpose":
            a, b = args[1:]; shape[a], shape[b] = shape[b], shape[a]
        elif name in {"mean", "sum", "vector_norm", "logsumexp"}:
            dim = arg(1, "dim", None)
            dims = range(len(shape)) if dim is None else dim if isinstance(dim, (list, tuple)) else (dim,)
            dims = {d % len(shape) for d in dims}
            shape = [1 if i in dims else size for i, size in enumerate(shape)] if kwargs.get("keepdim", False) else [size for i, size in enumerate(shape) if i not in dims]
        elif name == "unsqueeze":
            dimension = args[1] % (len(shape) + 1); shape.insert(dimension, 1)
        elif name == "repeat_interleave":
            shape[arg(2, "dim")] *= args[1]
        elif name in {"chunk", "topk"}:
            dimension = arg(2, "dim", 0 if name == "chunk" else -1)
            count = args[1]
            if name == "chunk":
                step = math.ceil(shape[dimension] / count); pieces = []
                for start in range(0, shape[dimension], step):
                    part = shape.copy(); part[dimension] = min(step, shape[dimension] - start)
                    pieces.append(self.add(name, args, part, dtype, dimension=dimension, start=start))
                return tuple(pieces)
            shape[dimension] = count
            return self.add("topk_values", args, shape, dtype), self.add("topk_indices", args, shape, "int64")
        elif name in {"cat", "stack"}:
            dimension = arg(1, "dim", 0)
            if name == "cat":
                shape[dimension] = sum(v.shape[dimension] for v in args[0])
            else:
                shape.insert(dimension % (len(shape) + 1), len(args[0]))
        elif name == "matmul":
            left, right = args[:2]
            if len(left.shape) < 2 or len(right.shape) < 2 or left.shape[-1] != right.shape[-2]:
                raise ValueError("unsupported symbolic matmul shapes")
            shape = list(broadcast(left.shape[:-2], right.shape[:-2])) + [left.shape[-2], right.shape[-1]]
        elif name == "einsum":
            equation = args[0]; incoming, outgoing = equation.split("->"); dimensions = {}
            for symbols, tensor in zip(incoming.split(","), args[1:], strict=True):
                if len(symbols) != len(tensor.shape):
                    raise ValueError("unsupported symbolic einsum")
                for symbol, size in zip(symbols, tensor.shape):
                    if symbol in dimensions and dimensions[symbol] != size:
                        raise ValueError("incompatible symbolic einsum")
                    dimensions[symbol] = size
            shape = [dimensions[symbol] for symbol in outgoing]
        elif name == "scaled_dot_product_attention":
            shape = list(args[0].shape[:-1]) + [args[2].shape[-1]]
        elif name == "pad":
            for i, amount in enumerate(args[1]):
                shape[-(i // 2) - 1] += amount
        elif name in {"interpolate", "adaptive_avg_pool2d"}:
            size = arg(1, "size" if name == "interpolate" else "output_size")
            shape[2:] = size if isinstance(size, (list, tuple)) else [size] * (len(shape) - 2)
        elif name == "affine_grid":
            size = args[1]; shape = [size[0], *size[2:], len(size) - 2]
        elif name in {"zeros", "ones", "empty"}:
            shape = list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else list(args)
            dtype = kwargs.get("dtype", "float32")
        elif name == "eye":
            shape = [args[0], args[1] if len(args) > 1 else args[0]]; dtype = kwargs.get("dtype", "float32")
        elif name == "arange":
            shape = [len(range(*args))]; dtype = "int64"
        elif name in {"float", "long", "to"}:
            dtype = "float32" if name == "float" else "int64" if name == "long" else args[1]
        elif name not in {"relu", "gelu", "selu", "silu", "elu", "softplus", "sigmoid", "tanh", "softmax", "log_softmax", "normalize", "exp", "log", "abs", "remainder", "clamp_min", "detach", "masked_fill", "roll", "grid_sample", "zeros_like"}:
            raise ValueError("unsupported symbolic tensor operation: " + name)
        scalars = [x for x in args if isinstance(x, (str, int, float, bool))]
        options = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, tuple))}
        return self.add(name, [args, kwargs], shape, dtype, scalars=scalars, options=options)

    def graph(self, config, inputs):
        self.remaining = 1000000
        root = self.function(self.definitions["make_model"], [], {}, {"MODEL_CONFIG": config})
        self.register(root)
        allocated = [(parameter_count(module), module.frozen) for module in self.declarations
                     if module.name not in self.definitions and module.name not in {"Sequential", "ModuleList", "tensor_shape"}]
        self.tracing = True
        arguments = [self.add("input", shape=item["shape"], dtype=item["dtype"], name=item["name"]) for item in inputs]
        result = self.call(root, arguments, {}, {})
        outputs = list(tensors(result))
        if not outputs:
            raise ValueError("symbolic graph has no tensor outputs")
        for output in outputs:
            self.add("output", [output], output.shape, output.dtype)
        return {"version": VERSION, "nodes": self.nodes, "edges": self.edges,
                "metadata": {"graph_kind": "static_forward_dataflow", "branches": self.branches,
                             "backward_available": False, "optimizer_graph_available": False,
                             "kernel_capture_available": False, "layout_available": False,
                             "shape_source": "literal_inputs_and_symbolic_formulas",
                             "parameter_count": sum(n["parameters"] for n in self.nodes),
                             "parameter_count_scope": "parameters_referenced_by_selected_forward",
                             "allocated_parameter_count": sum(count for count, _ in allocated),
                             "allocated_trainable_parameter_count": sum(count for count, frozen in allocated if not frozen),
                             "unknown_fields": ["backward", "optimizer_graph", "aliasing", "strides", "kernel_backend", "runtime_layout", "activation_lifetime"]}}


def build_graph(text, model_config, input_specs):
    """Return JSON-compatible nodes/edges from source literals; labels are unused."""
    return GraphReader(text).graph(model_config, input_specs)
