import torch
from torch_geometric.data import Data

from perfseer_optimized.model import SeerNetConfig
from perfseer_optimized.train_v12 import (
    METRIC_HEAD_WIDTHS,
    SixMetricTwelveLabelSeerNet,
    TARGET_NAMES,
)


def test_six_metric_heads_emit_twelve_labels():
    model = SixMetricTwelveLabelSeerNet(
        SeerNetConfig(
            node_dim=40,
            edge_dim=14,
            global_dim=110,
            hidden=32,
            num_blocks=1,
            head_hidden=32,
            num_outputs=6,
        )
    )
    data = Data(
        x=torch.zeros(2, 40),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.zeros(1, 14),
        u=torch.zeros(1, 110),
        batch=torch.zeros(2, dtype=torch.long),
        num_graphs=1,
    )

    assert len(model.heads) == 6
    assert [head[-1].out_features for head in model.heads] == list(METRIC_HEAD_WIDTHS)
    assert model(data).shape == (1, len(TARGET_NAMES))
