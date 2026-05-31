"""Configurable optimized SeerNet variants."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax


@dataclass
class SeerNetConfig:
    node_dim: int
    edge_dim: int
    global_dim: int
    hidden: int = 256
    num_blocks: int = 1
    num_outputs: int = 1
    head_hidden: int | None = None
    activation: str = "relu"
    dropout: float = 0.0
    encoder_norm: str = "none"
    block_norm: str = "none"
    residual: str = "direct"
    residual_gate_init: float = 0.1
    residual_gate_mode: str = "scalar_per_stream"
    use_synmm: bool = True
    global_agg: str = "synmm"
    attention_pool: bool = False
    use_gnpb: bool = True
    include_u_in_edge_update: bool = True
    mlp_z_num_linear_layers: int = 3
    softmax_agg_mode: str = "learned_score"
    metric_heads: str = "separate"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeerNetConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _activation(name: str) -> nn.Module:
    key = (name or "relu").lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}")


def _norm(name: str, dim: int) -> nn.Module:
    return nn.LayerNorm(dim) if (name or "none").lower() == "layernorm" else nn.Identity()


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden: int,
    num_linear_layers: int = 2,
    activation: str = "relu",
    dropout: float = 0.0,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build an MLP where ``num_linear_layers`` includes the output layer."""

    if num_linear_layers < 1:
        raise ValueError("num_linear_layers must be >= 1")
    layers: list[nn.Module] = []
    cur = in_dim
    for _ in range(num_linear_layers - 1):
        layers.append(nn.Linear(cur, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        cur = hidden
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class SynMM(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.lin = nn.Linear(2 * dim, out_dim)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        v_max = scatter(v, batch, dim=0, dim_size=size, reduce="max")
        v_mean = scatter(v, batch, dim=0, dim_size=size, reduce="mean")
        return self.lin(torch.cat([v_max, v_mean], dim=-1))


class SynMMPlus(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.lin = nn.Linear(4 * dim, out_dim)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        v_max = scatter(v, batch, dim=0, dim_size=size, reduce="max")
        v_mean = scatter(v, batch, dim=0, dim_size=size, reduce="mean")
        v_sum = scatter(v, batch, dim=0, dim_size=size, reduce="sum")
        sq_mean = scatter(v * v, batch, dim=0, dim_size=size, reduce="mean")
        v_std = torch.sqrt(torch.clamp(sq_mean - v_mean * v_mean, min=0.0) + 1e-12)
        return self.lin(torch.cat([v_max, v_mean, v_sum, v_std], dim=-1))


class AttentionGraphPool(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.pre = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)
        self.out = nn.Linear(3 * dim, out_dim)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        logits = self.score(torch.tanh(self.pre(v)))
        alpha = softmax(logits, batch, num_nodes=size)
        weighted = scatter(v * alpha, batch, dim=0, dim_size=size, reduce="sum")
        v_mean = scatter(v, batch, dim=0, dim_size=size, reduce="mean")
        v_max = scatter(v, batch, dim=0, dim_size=size, reduce="max")
        return self.out(torch.cat([weighted, v_mean, v_max], dim=-1))


class SoftmaxNodeAgg(nn.Module):
    def __init__(self, dim: int, mode: str = "learned_score") -> None:
        super().__init__()
        self.mode = mode
        self.score = nn.Linear(dim, 1)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        if self.mode == "feature_softmax":
            logits = v.mean(dim=-1, keepdim=True)
        else:
            logits = self.score(v)
        alpha = softmax(logits, batch, num_nodes=size)
        return scatter(v * alpha, batch, dim=0, dim_size=size, reduce="sum")


class ResidualGate(nn.Module):
    def __init__(self, hidden: int, init_value: float = 0.1, mode: str = "scalar") -> None:
        super().__init__()
        init_value = min(max(init_value, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_value / (1.0 - init_value))
        shape = (hidden,) if mode == "vector" else (1,)
        self.logit = nn.Parameter(torch.full(shape, init_logit))

    def forward(self, old: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return old + torch.sigmoid(self.logit).to(update.dtype) * update


class SeerBlock(nn.Module):
    def __init__(self, cfg: SeerNetConfig) -> None:
        super().__init__()
        h = cfg.hidden
        self.cfg = cfg
        self.use_gnpb = cfg.use_gnpb
        self.include_u_in_edge_update = cfg.include_u_in_edge_update
        self.block_norm = cfg.block_norm.lower()

        e_in_dim = 3 * h + (h if cfg.include_u_in_edge_update else 0)
        self.mlp_e = make_mlp(e_in_dim, h, h, 2, cfg.activation, cfg.dropout)
        self.mlp_v = make_mlp(3 * h, h, h, 2, cfg.activation, cfg.dropout)
        self.agg_z = SoftmaxNodeAgg(h, cfg.softmax_agg_mode)
        z_layers = max(1, int(cfg.mlp_z_num_linear_layers))
        self.mlp_z = make_mlp(h, h, h, z_layers, cfg.activation, cfg.dropout)
        if cfg.attention_pool or cfg.global_agg == "attention":
            self.agg_u = AttentionGraphPool(h, h)
        elif cfg.global_agg == "synmm_plus":
            self.agg_u = SynMMPlus(h, h)
        elif cfg.use_synmm:
            self.agg_u = SynMM(h, h)
        else:
            self.agg_u = nn.Linear(h, h)
        self.mlp_u = make_mlp(3 * h, h, h, 2, cfg.activation, cfg.dropout)

        self.norm_v = nn.LayerNorm(h) if self.block_norm == "prenorm" else nn.Identity()
        self.norm_e = nn.LayerNorm(h) if self.block_norm == "prenorm" else nn.Identity()
        self.norm_u = nn.LayerNorm(h) if self.block_norm == "prenorm" else nn.Identity()
        self.norm_z = nn.LayerNorm(h) if self.block_norm == "prenorm" else nn.Identity()
        self.post_v = nn.LayerNorm(h) if self.block_norm == "postnorm" else nn.Identity()
        self.post_e = nn.LayerNorm(h) if self.block_norm == "postnorm" else nn.Identity()
        self.post_u = nn.LayerNorm(h) if self.block_norm == "postnorm" else nn.Identity()
        self.post_z = nn.LayerNorm(h) if self.block_norm == "postnorm" else nn.Identity()

    def _maybe_prenorm(
        self, v: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.block_norm == "prenorm":
            return self.norm_v(v), self.norm_e(e), self.norm_u(u), self.norm_z(z)
        return v, e, u, z

    def _maybe_postnorm(
        self, v: torch.Tensor, e: torch.Tensor, u: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.block_norm == "postnorm":
            return self.post_v(v), self.post_e(e), self.post_u(u), self.post_z(z)
        return v, e, u, z

    def forward(
        self,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
        batch: torch.Tensor,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        v_in_stream, e_in_stream, u_in_stream, z_in_stream = self._maybe_prenorm(v, e, u, z)

        if e_in_stream.size(0) == 0:
            e_new = e_in_stream
            ebar = v_in_stream.new_zeros(v_in_stream.shape)
        else:
            parts = [e_in_stream, v_in_stream[src], v_in_stream[dst]]
            if self.include_u_in_edge_update:
                parts.append(u_in_stream[batch[src]])
            e_new = self.mlp_e(torch.cat(parts, dim=-1))
            ebar = scatter(e_new, dst, dim=0, dim_size=v.size(0), reduce="mean")

        u_node = u_in_stream[batch]
        v_new = self.mlp_v(torch.cat([ebar, v_in_stream + z_in_stream, u_node], dim=-1))

        if self.use_gnpb:
            zbar = self.agg_z(v_new, batch, size)
            z_graph = scatter(z_in_stream, batch, dim=0, dim_size=size, reduce="mean")
            z_new_graph = self.mlp_z(zbar + z_graph)
            z_new = z_new_graph[batch]
        else:
            z_new_graph = u_in_stream.new_zeros(u_in_stream.shape)
            z_new = z_in_stream.new_zeros(z_in_stream.shape)

        if isinstance(self.agg_u, nn.Linear):
            vbar_u = self.agg_u(scatter(v_new, batch, dim=0, dim_size=size, reduce="mean"))
        else:
            vbar_u = self.agg_u(v_new, batch, size)
        u_new = self.mlp_u(torch.cat([vbar_u, z_new_graph, u_in_stream], dim=-1))
        return self._maybe_postnorm(v_new, e_new, u_new, z_new)


class SeerTrunk(nn.Module):
    def __init__(self, cfg: SeerNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.node_enc = nn.Sequential(nn.Linear(cfg.node_dim, h), _norm(cfg.encoder_norm, h))
        self.edge_enc = nn.Sequential(nn.Linear(cfg.edge_dim, h), _norm(cfg.encoder_norm, h))
        self.global_enc = nn.Sequential(nn.Linear(cfg.global_dim, h), _norm(cfg.encoder_norm, h))
        self.z_init = SoftmaxNodeAgg(h, cfg.softmax_agg_mode)
        self.blocks = nn.ModuleList(SeerBlock(cfg) for _ in range(cfg.num_blocks))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        gate_mode = "vector" if cfg.residual_gate_mode == "vector_per_stream" else "scalar"
        self.gates = nn.ModuleList()
        if cfg.residual.lower() == "gated":
            for _ in range(cfg.num_blocks):
                self.gates.append(
                    nn.ModuleDict(
                        {
                            "v": ResidualGate(h, cfg.residual_gate_init, gate_mode),
                            "e": ResidualGate(h, cfg.residual_gate_init, gate_mode),
                            "u": ResidualGate(h, cfg.residual_gate_init, gate_mode),
                            "z": ResidualGate(h, cfg.residual_gate_init, gate_mode),
                        }
                    )
                )

    def _apply_residual(
        self,
        idx: int,
        old: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        update: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self.cfg.residual.lower()
        if mode == "none":
            return update
        if mode == "gated":
            gates = self.gates[idx]
            return (
                gates["v"](old[0], self.dropout(update[0])),
                gates["e"](old[1], self.dropout(update[1])),
                gates["u"](old[2], self.dropout(update[2])),
                gates["z"](old[3], self.dropout(update[3])),
            )
        return tuple(old[i] + self.dropout(update[i]) for i in range(4))  # type: ignore[return-value]

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        size = int(getattr(data, "num_graphs", int(batch.max().item()) + 1 if batch.numel() else 1))

        v = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        u = self.global_enc(u)
        if self.cfg.use_gnpb:
            z_graph = self.z_init(v, batch, size)
            z = z_graph[batch]
        else:
            z = torch.zeros_like(v)

        for idx, block in enumerate(self.blocks):
            old = (v, e, u, z)
            update = block(v, edge_index, e, u, z, batch, size)
            v, e, u, z = self._apply_residual(idx, old, update)
        return u


class SeerNet(nn.Module):
    def __init__(self, cfg: SeerNetConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = SeerNetConfig.from_dict(kwargs)
        elif kwargs:
            merged = cfg.to_dict()
            merged.update(kwargs)
            cfg = SeerNetConfig.from_dict(merged)
        self.cfg = cfg
        self.trunk = SeerTrunk(cfg)
        head_hidden = cfg.head_hidden or cfg.hidden
        self.head = make_mlp(cfg.hidden, cfg.num_outputs, head_hidden, 2, cfg.activation, cfg.dropout)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.trunk(data))


class SeerNetMulti(nn.Module):
    def __init__(self, cfg: SeerNetConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = SeerNetConfig.from_dict({**kwargs, "num_outputs": kwargs.get("num_outputs", 6)})
        elif kwargs:
            merged = cfg.to_dict()
            merged.update(kwargs)
            cfg = SeerNetConfig.from_dict(merged)
        cfg.num_outputs = int(cfg.num_outputs or 6)
        self.cfg = cfg
        self.trunk = SeerTrunk(cfg)
        head_hidden = cfg.head_hidden or cfg.hidden
        if cfg.metric_heads == "shared":
            self.head = make_mlp(cfg.hidden, cfg.num_outputs, head_hidden, 2, cfg.activation, cfg.dropout)
            self.heads = None
        else:
            self.head = None
            self.heads = nn.ModuleList(
                make_mlp(cfg.hidden, 1, head_hidden, 2, cfg.activation, cfg.dropout)
                for _ in range(cfg.num_outputs)
            )

    def forward(self, data) -> torch.Tensor:
        u = self.trunk(data)
        if self.head is not None:
            return self.head(u)
        assert self.heads is not None
        return torch.cat([head(u) for head in self.heads], dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
