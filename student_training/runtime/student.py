"""TorchScript-deployable CPU student for the node-164 teacher.

The student keeps the teacher's output semantics (six columns: wall/GPU/epoch ms, NVML busy %,
avg/peak VRAM MiB) and its useful inductive biases at ~1/400 of the parameters:
  * additive, positive per-operation timing cost + overhead (like AdditiveTimingBranch),
  * soft hurdle for NVML busy (sigmoid(logit) * positive magnitude, like the soft busy export),
  * log-space VRAM point estimate plus a conservative upper head for 100 MB capacity decisions,
  * Horus workload summaries computed in-graph from the same fixed registry tables,
  * an identity-initialised hardware FiLM adapter so the student can itself be transferred.
forward() takes plain tensors only (see pack_batch), so torch.jit.script works and the scripted
file runs without the perfseer packages. Every graph is predicted independently of its batch.
"""
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


class MessageLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.message = mlp(3 * width, width, width)
        self.update = mlp(3 * width, width, width)
        self.edge_norm = nn.LayerNorm(width)
        self.node_norm = nn.LayerNorm(width)

    def forward(self, nodes: Tensor, edges: Tensor, context: Tensor, edge_index: Tensor, node_graph: Tensor):
        source = edge_index[0]
        destination = edge_index[1]
        message = self.message(torch.cat((nodes[source], nodes[destination], edges), -1))
        edges = self.edge_norm(edges + message)
        incoming = torch.zeros_like(nodes).index_add_(0, destination, edges)
        nodes = self.node_norm(nodes + self.update(torch.cat((nodes, incoming, context[node_graph]), -1)))
        return nodes, edges


class PerfSeerStudent(nn.Module):
    def __init__(self, dims: Dict[str, int], horus_indices: List[int], horus_op_groups: List[int],
                 target_medians: List[float], mean_node_count: float, sm_positive_fraction: float,
                 sm_positive_median: float, width: int = 128, layers: int = 3):
        super().__init__()
        self.width = width
        self.op = nn.Embedding(dims["num_exact_ops"], 32)
        self.phase = nn.Embedding(dims["num_phases"], 8)
        self.dtype = nn.Embedding(dims["num_dtypes"], 8)
        self.role = nn.Embedding(dims["num_edge_roles"], 8)
        self.precision = nn.Embedding(dims["num_dtypes"], 8)
        self.optimizer = nn.Embedding(dims["num_optimizers"], 8)
        self.register_buffer("horus_fields", torch.tensor(horus_indices, dtype=torch.long))
        self.register_buffer("horus_op_groups", torch.tensor(horus_op_groups, dtype=torch.long))
        self.horus_groups = int(max(horus_op_groups)) + 1
        horus_width = len(horus_indices) + 2 * self.horus_groups + 1
        self.node_encoder = mlp(dims["node_dim"] + dims["node_flag_dim"] + 48, width, width)
        self.edge_encoder = mlp(dims["edge_dim"] + dims["edge_flag_dim"] + 8, width, width)
        self.global_encoder = mlp(dims["global_dim"] + dims["quality_dim"] + 16 + 2 + horus_width, width, width)
        self.layers = nn.ModuleList([MessageLayer(width) for _ in range(layers)])
        self.readout = mlp(4 * width, width, width)
        self.readout_norm = nn.LayerNorm(width)
        self.hardware = nn.Sequential(nn.Linear(2 * dims["hardware_dim"], 32), nn.SiLU(), nn.Linear(32, 2 * width))
        self.node_cost = mlp(2 * width, width // 2, 2)
        self.overhead = nn.Linear(width, 2)
        self.correction = nn.Linear(width, 2)
        self.busy = mlp(width, width // 2, 2)
        self.memory = mlp(width, width // 2, 2)
        self.memory_upper = nn.Linear(width, 2)
        self.log_calibration = nn.Parameter(torch.zeros(2))
        medians = torch.tensor(target_medians, dtype=torch.float32)
        self.register_buffer("per_node_scale", .8 * medians[:2] / float(mean_node_count))
        self.register_buffer("overhead_scale", .2 * medians[:2])
        self.register_buffer("log_memory_median", medians[4:6].log())
        for layer in (self.hardware[-1], self.node_cost[-1], self.overhead, self.correction, self.memory[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.busy[-1].weight)
        nn.init.zeros_(self.memory_upper.weight)
        nn.init.constant_(self.memory_upper.bias, -2.)
        with torch.no_grad():
            probability = torch.tensor(sm_positive_fraction).clamp(.01, .99)
            magnitude = (torch.tensor(sm_positive_median) / 100).clamp(.01, .99)
            self.busy[-1].bias.copy_(torch.stack((torch.logit(probability), torch.logit(magnitude))))

    def horus(self, u_cont: Tensor, op_exact_id: Tensor, node_graph: Tensor) -> Tensor:
        graphs = u_cont.size(0)
        counts = torch.zeros(graphs * self.horus_groups, dtype=torch.float32, device=u_cont.device)
        flat = node_graph * self.horus_groups + self.horus_op_groups[op_exact_id]
        counts = counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32)).view(graphs, self.horus_groups)
        total = counts.sum(-1, keepdim=True)
        return torch.cat((u_cont.float()[:, self.horus_fields], torch.log1p(counts),
                          counts / total.clamp_min(1.), torch.log1p(total)), -1)

    def forward(self, x_cont: Tensor, op_exact_id: Tensor, phase_id: Tensor, dtype_id: Tensor, node_flags: Tensor,
                edge_index: Tensor, edge_cont: Tensor, edge_role_id: Tensor, edge_flags: Tensor, node_graph: Tensor,
                u_cont: Tensor, quality: Tensor, precision_id: Tensor, optimizer_id: Tensor,
                hardware_cont: Tensor, hardware_mask: Tensor, raw_steps: Tensor) -> Dict[str, Tensor]:
        graphs = u_cont.size(0)
        width = self.width
        nodes = self.node_encoder(torch.cat((x_cont.float(), node_flags.float(), self.op(op_exact_id),
                                             self.phase(phase_id), self.dtype(dtype_id)), -1))
        edges = self.edge_encoder(torch.cat((edge_cont.float(), edge_flags.float(), self.role(edge_role_id)), -1))
        ones = torch.ones((nodes.size(0), 1), dtype=torch.float32, device=nodes.device)
        node_count = torch.zeros((graphs, 1), dtype=torch.float32, device=nodes.device).index_add_(0, node_graph, ones)
        edge_graph = node_graph[edge_index[0]]
        edge_ones = torch.ones((edges.size(0), 1), dtype=torch.float32, device=nodes.device)
        edge_count = torch.zeros((graphs, 1), dtype=torch.float32, device=nodes.device).index_add_(0, edge_graph, edge_ones)
        context = self.global_encoder(torch.cat((u_cont.float(), quality.float(), self.precision(precision_id),
                                                 self.optimizer(optimizer_id), torch.log1p(node_count),
                                                 torch.log1p(edge_count), self.horus(u_cont, op_exact_id, node_graph)), -1))
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, context, edge_index, node_graph)
        pooled_sum = torch.zeros((graphs, width), dtype=nodes.dtype, device=nodes.device).index_add_(0, node_graph, nodes)
        pooled_max = torch.zeros((graphs, width), dtype=nodes.dtype, device=nodes.device).scatter_reduce(
            0, node_graph.unsqueeze(1).expand(-1, width), nodes, reduce="amax", include_self=False)
        embedding = self.readout_norm(self.readout(torch.cat(
            (context, torch.log1p(pooled_sum.abs()) * pooled_sum.sign(), pooled_sum / node_count.clamp_min(1.), pooled_max), -1)))
        mask = hardware_mask.float()
        film = self.hardware(torch.cat((hardware_cont.float() * (1 - mask), mask), -1))
        embedding = embedding * (1 + film[:, :width]) + film[:, width:]
        normalizer = F.softplus(torch.zeros(1, device=nodes.device))
        node_cost = (F.softplus(self.node_cost(torch.cat((nodes, embedding[node_graph]), -1))) + 1e-6) / normalizer
        base = torch.zeros((graphs, 2), dtype=nodes.dtype, device=nodes.device).index_add_(0, node_graph, node_cost * self.per_node_scale)
        overhead = (F.softplus(self.overhead(embedding)) + 1e-6) / normalizer * self.overhead_scale
        log_time = torch.log(base + overhead) + .5 * torch.tanh(self.correction(embedding)) + self.log_calibration
        time = torch.exp(log_time)
        busy_logits = self.busy(embedding)
        busy_positive = 100 * torch.sigmoid(busy_logits[:, 1])
        busy = torch.sigmoid(busy_logits[:, 0]) * busy_positive
        log_memory = self.log_memory_median + self.memory(embedding)
        memory = torch.exp(log_memory)
        memory_upper = memory * torch.exp(F.softplus(self.memory_upper(embedding)))
        epoch = time[:, 0] * raw_steps.to(time.dtype)
        prediction = torch.stack((time[:, 0], time[:, 1], epoch, busy, memory[:, 0], memory[:, 1]), -1)
        return {"prediction": prediction, "log_time": log_time, "busy_logit": busy_logits[:, 0],
                "busy_positive": busy_positive, "log_memory": log_memory, "memory_upper": memory_upper,
                "embedding": embedding}


INPUT_NAMES = ("x_cont", "op_exact_id", "phase_id", "dtype_id", "node_flags", "edge_index", "edge_cont",
               "edge_role_id", "edge_flags", "node_graph", "u_cont", "quality", "precision_id", "optimizer_id",
               "hardware_cont", "hardware_mask", "raw_steps")


def pack_batch(batch, raw_steps):
    """Convert a normalised perfseer GraphFeaturesV3 batch into the student's tensor tuple."""
    return (batch.x_cont, batch.op_exact_id, batch.phase_id, batch.dtype_id, batch.node_flags, batch.edge_index,
            batch.edge_cont, batch.edge_role_id, batch.edge_flags, batch.batch, batch.u_cont, batch.quality,
            batch.precision_id, batch.optimizer_id, batch.hardware_cont, batch.hardware_missing_mask,
            torch.as_tensor(raw_steps))


def student_from_teacher(payload, width=128, layers=3):
    """Size the student from the released checkpoint's own configuration (no data access)."""
    config, init, physics = payload["vram_config"], payload["initialization"], payload["physics_config"]
    dims = {"num_exact_ops": config["num_exact_ops"], "num_phases": config["num_phases"],
            "num_dtypes": config["num_dtypes"], "num_edge_roles": config["num_edge_roles"],
            "num_optimizers": config["num_optimizers"], "node_dim": config["node_continuous_dim"],
            "edge_dim": config["edge_continuous_dim"], "global_dim": config["global_continuous_dim"],
            "quality_dim": config["quality_dim"], "node_flag_dim": config["node_flag_dim"],
            "edge_flag_dim": config["edge_flag_dim"], "hardware_dim": config["hardware_continuous_dim"]}
    return PerfSeerStudent(dims, physics["horus_global_indices"], physics["horus_op_groups"],
                           init["target_medians"], init["mean_node_count"], init["sm_positive_fraction"],
                           init["sm_positive_median"], width=width, layers=layers)
