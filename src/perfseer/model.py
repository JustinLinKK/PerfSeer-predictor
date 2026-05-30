"""PerfSeer SeerNet model (single-metric performance predictor).

This module implements the graph-network building blocks described in the
PerfSeer paper (arXiv:2502.01206), section 3.2 and Figures 2-3:

  * ``SynMM``   - Synergistic Max-Mean node->global aggregation (Fig. 3).
  * ``SeerBlock`` - one graph-network block faithful to eqs. (1)-(8), including
                  the Global-Node Perspective Boost (GNPB) per-node global node
                  ``z`` (softmax initialisation + eqs. (5)-(6) update).
  * ``SeerNet``  - a stack of ``SeerBlock`` followed by a 2-layer MLP head, with
                  a configurable number of outputs (default 1 = single metric).

Tensor / batching conventions (PyTorch Geometric ``Data`` / ``Batch``):
  data.x          : [N_total, node_dim]    node features (concatenated over batch)
  data.edge_index : [2, E_total]           COO edges, row 0 = source, row 1 = target
  data.edge_attr  : [E_total, edge_dim]    edge features
  data.u          : [B, global_dim]        per-graph global features (B = num graphs)
  data.batch      : [N_total]              maps each node to its graph id in [0, B)
  data.num_graphs : int B

``forward(data)`` returns a tensor of shape ``[B, num_outputs]`` in the
standardized-log target space (the inverse transform to the original metric
space is handled outside the model, in metrics/eval code).

All comments are in English per project convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ``scatter`` and ``softmax`` are the canonical PyG aggregation helpers. They
# operate on the flat node/edge tensors using an index vector that maps each
# row to a destination group (graph id or target-node id).
from torch_geometric.utils import scatter, softmax


def _mlp(in_dim: int, out_dim: int, hidden: int, num_layers: int = 1) -> nn.Sequential:
    """Build an MLP update function ``phi`` used throughout the SeerBlock.

    ``num_layers`` counts the number of *hidden* layers:
      * ``num_layers == 1`` -> Linear(in,h) -> ReLU -> Linear(h,out)   (1 hidden layer)
      * ``num_layers == 2`` -> Linear(in,h) -> ReLU -> Linear(h,h) -> ReLU -> Linear(h,out)

    This matches the paper: phi^e/phi^v/phi^u use 1 hidden layer at 256 channels,
    while phi^z (MLP_z) uses 2 layers.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class SynMM(nn.Module):
    """Synergistic Max-Mean aggregation (paper Fig. 3, eq. 7).

    Given per-node features ``V'`` and a ``batch`` index, compute, per graph:
        vbar1 = max  over nodes
        vbar2 = mean over nodes
        vbar  = Linear( concat(vbar1, vbar2) )
    i.e. a learned linear combination of the max- and mean-aggregated node
    features. Output dimensionality equals ``out_dim`` (defaults to ``dim``).
    """

    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        # Linear maps the concatenated [max || mean] (2*dim) -> out_dim.
        self.lin = nn.Linear(2 * dim, out_dim)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        # v: [N, dim], batch: [N], size = number of graphs B.
        v_max = scatter(v, batch, dim=0, dim_size=size, reduce="max")   # [B, dim]
        v_mean = scatter(v, batch, dim=0, dim_size=size, reduce="mean")  # [B, dim]
        return self.lin(torch.cat([v_max, v_mean], dim=-1))             # [B, out_dim]


class SoftmaxNodeAgg(nn.Module):
    """Attention-style softmax aggregation of node features into a single
    per-graph vector (paper eq. 5: rho^{v->z}, and the GNPB initialisation).

    A linear layer produces a scalar logit per node; logits are normalised with
    a per-graph softmax (``torch_geometric.utils.softmax``); the node features
    are then summed weighted by those attention coefficients. This yields one
    vector per graph, which GNPB broadcasts back to every node as the global
    node ``z``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, v: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
        # v: [N, dim] -> per-graph aggregated [B, dim].
        logits = self.score(v)                       # [N, 1]
        alpha = softmax(logits, batch, num_nodes=v.size(0))  # [N, 1], per-graph normalised
        weighted = v * alpha                         # [N, dim]
        return scatter(weighted, batch, dim=0, dim_size=size, reduce="sum")  # [B, dim]


class SeerBlock(nn.Module):
    """One SeerNet graph-network block implementing eqs. (1)-(8) with GNPB.

    Update order per forward pass (s_j = source node of edge j, t_j = target):
      (2) e'_j  = MLP_e( [ e_j , v_{s_j} , v_{t_j} , u_broadcast ] )
      (3) ebar_i = mean over incoming edges of e'_j   (aggregated at target node)
      (4) v'_i  = MLP_v( [ ebar_i , (v_i + z_i) , u_broadcast ] )
      (5) zbar' = softmax-aggregation of V' (per graph)
      (6) z'    = MLP_z( zbar' + z_graph ) , then broadcast to nodes
      (7) vbar'_u = SynMM(V')                          (per graph)
      (8) u'    = MLP_u( [ vbar'_u , z' , u ] )

    All feature dimensions are kept equal to ``hidden`` across the block so that
    blocks can be stacked. ``z`` is stored per *node* (GNPB), but every node in a
    graph shares the same value (broadcast from the per-graph z).
    """

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        h = hidden

        # (2) edge update phi^e: input = [e, v_src, v_dst, u] -> 4*h ; output h.
        self.mlp_e = _mlp(4 * h, h, h, num_layers=1)
        # (4) node update phi^v: input = [ebar, (v+z), u] -> 3*h ; output h.
        self.mlp_v = _mlp(3 * h, h, h, num_layers=1)
        # (5) rho^{v->z}: softmax node aggregation producing zbar' of dim h.
        self.agg_z = SoftmaxNodeAgg(h)
        # (6) global-node update phi^z (MLP_z, 2 layers): input zbar'+z (dim h) -> h.
        self.mlp_z = _mlp(h, h, h, num_layers=2)
        # (7) rho^{v->u}: SynMM node->global aggregation producing vbar'_u of dim h.
        self.agg_u = SynMM(h, out_dim=h)
        # (8) global update phi^u: input = [vbar'_u, z', u] -> 3*h ; output h.
        self.mlp_u = _mlp(3 * h, h, h, num_layers=1)

    def forward(
        self,
        v: torch.Tensor,          # [N, h]   node features
        edge_index: torch.Tensor,  # [2, E]   row0 = source, row1 = target
        e: torch.Tensor,          # [E, h]   edge features
        u: torch.Tensor,          # [B, h]   global features
        z: torch.Tensor,          # [N, h]   per-node global node (GNPB)
        batch: torch.Tensor,      # [N]      node -> graph id
        size: int,                # B        number of graphs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # --- (2) edge update --------------------------------------------------
        # Broadcast the per-graph global u to each edge using the edge's graph id
        # (the source node's graph id; src and dst share the same graph).
        u_edge = u[batch[src]]                              # [E, h]
        e_in = torch.cat([e, v[src], v[dst], u_edge], dim=-1)  # [E, 4h]
        e_new = self.mlp_e(e_in)                            # [E, h]

        # --- (3) edge -> node aggregation (mean of incoming edges) ------------
        # Each edge contributes to its target node t_j = dst.
        ebar = scatter(e_new, dst, dim=0, dim_size=v.size(0), reduce="mean")  # [N, h]

        # --- (4) node update --------------------------------------------------
        u_node = u[batch]                                   # [N, h]
        v_in = torch.cat([ebar, v + z, u_node], dim=-1)     # [N, 3h]  (GNPB: v_i + z_i)
        v_new = self.mlp_v(v_in)                            # [N, h]

        # --- (5)+(6) global-node (z) update -----------------------------------
        zbar = self.agg_z(v_new, batch, size)               # [B, h]  per-graph zbar'
        z_graph = scatter(z, batch, dim=0, dim_size=size, reduce="mean")  # [B, h] recover per-graph z
        z_new_graph = self.mlp_z(zbar + z_graph)            # [B, h]  eq. (6): MLP_z(zbar' + z)
        z_new = z_new_graph[batch]                          # [N, h]  broadcast back to nodes

        # --- (7) node -> global aggregation (SynMM) ---------------------------
        vbar_u = self.agg_u(v_new, batch, size)             # [B, h]

        # --- (8) global update ------------------------------------------------
        u_in = torch.cat([vbar_u, z_new_graph, u], dim=-1)  # [B, 3h]
        u_new = self.mlp_u(u_in)                            # [B, h]

        return v_new, e_new, u_new, z_new


class SeerNet(nn.Module):
    """Full single-metric SeerNet: input encoders -> stacked SeerBlocks -> head.

    Pipeline:
      1. Linear encoders lift raw node/edge/global features to ``hidden`` dim.
      2. GNPB initialisation: the per-node global node ``z`` is initialised by a
         softmax-aggregation of the *encoded* initial node features (one vector
         per graph, broadcast to all its nodes).
      3. ``num_blocks`` SeerBlocks update (v, e, u, z) in sequence; residual
         connections keep features stable when more than one block is stacked.
      4. The final per-graph global embedding ``u'`` is passed through a 2-layer
         MLP head (256 hidden) producing ``num_outputs`` values (default 1).

    A single block + head is sized to roughly ~1.02M parameters at hidden=256.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden: int = 256,
        num_blocks: int = 1,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.num_outputs = num_outputs

        # (1) input encoders: lift each feature stream to the common hidden dim.
        self.node_enc = nn.Linear(node_dim, hidden)
        self.edge_enc = nn.Linear(edge_dim, hidden)
        self.global_enc = nn.Linear(global_dim, hidden)

        # GNPB initialisation: softmax aggregation of encoded node features.
        self.z_init = SoftmaxNodeAgg(hidden)

        # Stacked graph-network blocks (each operates fully in hidden dim).
        self.blocks = nn.ModuleList(SeerBlock(hidden) for _ in range(num_blocks))

        # 2-layer MLP head on the final global embedding u' -> num_outputs.
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, data) -> torch.Tensor:
        # Read PyG batched fields. ``batch`` may be absent for a single graph.
        x, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        # Number of graphs in the batch.
        size = int(getattr(data, "num_graphs", int(batch.max().item()) + 1))

        # (1) encode raw features to hidden dim.
        v = self.node_enc(x)            # [N, h]
        e = self.edge_enc(edge_attr)    # [E, h]
        u = self.global_enc(u)          # [B, h]

        # (2) GNPB init: per-graph z from softmax-aggregated initial node feats,
        # then broadcast so every node of a graph starts with the same z.
        z_graph = self.z_init(v, batch, size)   # [B, h]
        z = z_graph[batch]                       # [N, h]

        # (3) run stacked blocks with residual connections.
        for block in self.blocks:
            v_new, e_new, u_new, z_new = block(v, edge_index, e, u, z, batch, size)
            v = v + v_new
            e = e + e_new
            u = u + u_new
            z = z + z_new

        # (4) head on the final global embedding -> [B, num_outputs].
        return self.head(u)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters (handy for the ~1.02M check)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
