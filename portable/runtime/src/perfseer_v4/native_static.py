"""Bounded AST interpretation of module declarations, without importing workloads.

Only literal arithmetic/control flow and symbolic module constructors are
supported. Module declarations are plain Python records: no torch import,
tensor allocation, source eval/exec, forward pass, or GPU measurement occurs.
"""

import ast
from collections import Counter
from dataclasses import dataclass, field
import math
import operator


VERSION = "perfseer_v4_native_static_modules_v2"


@dataclass
class Symbol:
    name: str


@dataclass
class Declaration:
    name: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    frozen: bool = False


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class Reader:
    def __init__(self, text):
        tree = ast.parse(text)
        self.definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.remaining = 0

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("static module reader exceeded its operation budget")

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value):
                raise ValueError("static unpacking differs")
            for item, v in zip(target.elts, value):
                self.assign(item, v, env)
        elif isinstance(target, ast.Attribute) and isinstance(self.expr(target.value, env), Declaration):
            self.expr(target.value, env).attributes[target.attr] = value
        else:
            raise ValueError("unsupported static assignment")

    def statements(self, nodes, env):
        for node in nodes:
            self.tick()
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value is not None:
                    value = self.expr(node.value, env)
                    for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        self.assign(target, value, env)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For):
                values = self.expr(node.iter, env)
                if len(values) > 4096:
                    raise ValueError("static loop exceeds 4096 iterations")
                for value in values:
                    self.assign(node.target, value, env)
                    self.statements(node.body, env)
                self.statements(node.orelse, env)
            elif isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, env) if node.value else None)
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.FunctionDef):
                env[node.name] = (node, env.copy())
            elif isinstance(node, ast.Pass):
                pass
            else:
                raise ValueError(f"unsupported static statement: {type(node).__name__}")

    def function(self, node, args, kwargs, outer):
        env = outer.copy()
        names = [a.arg for a in node.args.args]
        if len(args) > len(names) or set(kwargs) - set(names):
            raise ValueError("static constructor arguments differ")
        for name, value in zip(names[-len(node.args.defaults):], node.args.defaults):
            env[name] = self.expr(value, outer)
        env.update(zip(names, args)); env.update(kwargs)
        if any(name not in env for name in names):
            raise ValueError("missing static constructor argument")
        try:
            self.statements(node.body, env)
        except Returned as result:
            return result.value
        return None

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.definitions:
                return self.definitions[node.id]
            if node.id in {"nn", "torch", "int", "float", "str", "bool", "min", "max", "range", "len", "super", "isinstance", "Mapping", "getattr"}:
                return Symbol(node.id)
            raise ValueError(f"unknown static name: {node.id}")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = []
            for n in node.elts:
                values.extend(self.expr(n.value, env)) if isinstance(n, ast.Starred) else values.append(self.expr(n, env))
            return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
        if isinstance(node, ast.Dict):
            return {self.expr(k, env): self.expr(v, env) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            return self.expr(node.value, env)[self.expr(node.slice, env)]
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value, env)
            if isinstance(value, Symbol):
                return Symbol(value.name + "." + node.attr)
            if isinstance(value, Declaration) and node.attr in value.attributes:
                return value.attributes[node.attr]
            # Only allowlisted methods are interpreted by call(); never getattr source objects.
            return ("method", value, node.attr)
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.BoolOp):
            value = self.expr(node.values[0], env)
            for part in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
                value = self.expr(part, env)
            return value
        if isinstance(node, ast.UnaryOp):
            operations = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
            if type(node.op) in operations:
                return operations[type(node.op)](self.expr(node.operand, env))
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if isinstance(node.op, ast.Mult):
                for sequence, multiplier in ((left, right), (right, left)):
                    if isinstance(sequence, (str, tuple, list)) and isinstance(multiplier, int) and len(sequence) * max(multiplier, 0) > 4096:
                        raise ValueError("static sequence multiplication exceeds bounds")
            if isinstance(node.op, ast.Pow) and (abs(right) > 32 or abs(left) > 1000000):
                raise ValueError("static exponent exceeds bounds")
            if type(node.op) in operations:
                return operations[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                          ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
                          ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            left = self.expr(node.left, env)
            for op, n in zip(node.ops, node.comparators):
                right = self.expr(n, env)
                if not operations[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            if len(node.generators) != 1:
                raise ValueError("nested static comprehension unsupported")
            gen = node.generators[0]; values = self.expr(gen.iter, env)
            if len(values) > 4096:
                raise ValueError("static comprehension exceeds bounds")
            result = []
            for value in values:
                local = env.copy(); self.assign(gen.target, value, local)
                if all(self.expr(condition, local) for condition in gen.ifs):
                    result.append(self.expr(node.elt, local))
            return result
        if isinstance(node, ast.Call):
            # Parameter-value initialization does not affect module dimensions.
            name = ast.unparse(node.func)
            if name.startswith("nn.init.") or name.endswith(".data.copy_"):
                return None
            args = []
            for n in node.args:
                args.extend(self.expr(n.value, env)) if isinstance(n, ast.Starred) else args.append(self.expr(n, env))
            kwargs = {k.arg: self.expr(k.value, env) for k in node.keywords}
            return self.call(self.expr(node.func, env), args, kwargs, env)
        raise ValueError(f"unsupported static expression: {ast.dump(node)[:180]}")

    def call(self, function, args, kwargs, env):
        if isinstance(function, ast.ClassDef):
            result = Declaration(function.name)
            init = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            self.function(init, [result, *args], kwargs, env)
            return result
        if isinstance(function, ast.FunctionDef):
            return self.function(function, args, kwargs, env)
        if isinstance(function, tuple) and len(function) == 2 and isinstance(function[0], ast.FunctionDef):
            return self.function(function[0], args, kwargs, function[1])
        if isinstance(function, tuple) and function[0] == "method":
            _, value, name = function
            if isinstance(value, dict) and name == "get":
                return value.get(*args, **kwargs)
            if isinstance(value, list) and name in {"append", "extend"}:
                return value.append(*args) if name == "append" else value.extend(*args)
            if isinstance(value, str) and name in {"lower", "startswith", "endswith"}:
                return getattr(value, name)(*args, **kwargs)
            if isinstance(value, Declaration) and name == "parameters":
                return [value]
            if isinstance(value, Declaration) and name == "requires_grad_":
                value.frozen = not args[0]
                return value
            raise ValueError(f"unsupported static method: {name}")
        if isinstance(function, Symbol):
            name = function.name
            builtins = {"int": int, "float": float, "str": str, "bool": bool, "min": min, "max": max, "range": range, "len": len}
            if name in builtins:
                return builtins[name](*args, **kwargs)
            if name == "super":
                return Symbol("super")
            if name == "super.__init__":
                return None
            if name == "isinstance" and isinstance(args[1], Symbol) and args[1].name == "Mapping":
                return isinstance(args[0], dict)
            if name.startswith("nn.") and name.count(".") == 1:
                return Declaration(name[3:], args, kwargs)
            if name in {"torch.zeros", "torch.ones", "torch.empty"}:
                shape = list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
                return Declaration("tensor_shape", shape)
        raise ValueError(f"unsupported static call: {function}")

    def describe(self, config, inputs):
        self.remaining = 200000
        root = self.function(self.definitions["make_model"], [], {}, {"MODEL_CONFIG": config})
        counts, parameters, dimensions = Counter(), Counter(), Counter()
        visited = set()
        def visit(value):
            if isinstance(value, Declaration):
                if id(value) in visited:
                    return
                visited.add(id(value))
                if value.name in self.definitions or value.name in {"Sequential", "ModuleList"}:
                    for child in [*value.args, *value.attributes.values()]:
                        visit(child)
                else:
                    count = parameter_count(value)
                    counts[value.name] += 1
                    parameters[value.name] += count
                    parameters["total"] += count
                    parameters["trainable"] += 0 if value.frozen else count
                    for i, argument in enumerate(value.args[:3]):
                        if isinstance(argument, (int, float)) and not isinstance(argument, bool):
                            dimensions[f"{value.name}.arg{i}.sum"] += argument
                            dimensions[f"{value.name}.arg{i}.max"] = max(dimensions[f"{value.name}.arg{i}.max"], argument)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
        visit(root)
        if not counts:
            raise ValueError("no static module declarations found")
        volume = sum(math.prod(item["shape"]) for item in inputs)
        return {"counts": dict(counts), "parameters": dict(parameters), "dimensions": dict(dimensions),
                "input_elements": volume}


def parameter_count(module):
    name, args, kwargs = module.name, module.args, module.kwargs
    def arg(index, key, default=None):
        return args[index] if len(args) > index else kwargs.get(key, default)
    def product(value, rank=1):
        return math.prod(value) if isinstance(value, (list, tuple)) else value ** rank
    if name == "Parameter":
        if len(args) != 1 or not isinstance(args[0], Declaration) or args[0].name != "tensor_shape":
            raise ValueError("unsupported static parameter shape")
        return math.prod(args[0].args)
    if name == "Linear":
        return arg(0, "in_features") * arg(1, "out_features") + int(arg(2, "bias", True)) * arg(1, "out_features")
    if name in {"Conv1d", "Conv2d", "ConvTranspose2d"}:
        rank = 1 if name == "Conv1d" else 2
        incoming, outgoing = arg(0, "in_channels"), arg(1, "out_channels")
        return outgoing * (incoming // kwargs.get("groups", 1)) * product(arg(2, "kernel_size"), rank) + int(kwargs.get("bias", True)) * outgoing
    if name in {"Embedding", "EmbeddingBag"}:
        return arg(0, "num_embeddings") * arg(1, "embedding_dim")
    if name in {"LayerNorm", "RMSNorm"}:
        return product(arg(0, "normalized_shape")) * (2 if name == "LayerNorm" else 1) * int(kwargs.get("elementwise_affine", True))
    if name in {"BatchNorm1d", "BatchNorm2d", "GroupNorm"}:
        return arg(1 if name == "GroupNorm" else 0, "num_channels" if name == "GroupNorm" else "num_features") * 2 * int(kwargs.get("affine", True))
    if name in {"RNN", "GRU", "LSTM"}:
        incoming, hidden = arg(0, "input_size"), arg(1, "hidden_size")
        directions = 2 if kwargs.get("bidirectional", False) else 1
        gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[name]
        return sum(directions * gates * hidden * ((incoming if layer == 0 else hidden * directions) + hidden + 2 * int(kwargs.get("bias", True)))
                   for layer in range(arg(2, "num_layers", 1)))
    if name == "PReLU":
        return arg(0, "num_parameters", 1)
    if name in {"ReLU", "ELU", "GELU", "Hardswish", "Mish", "SELU", "SiLU", "Identity", "AvgPool2d",
                "AdaptiveAvgPool2d", "PixelShuffle", "Upsample", "AlphaDropout", "Dropout"}:
        return 0
    raise ValueError(f"unsupported static module: {name}")
