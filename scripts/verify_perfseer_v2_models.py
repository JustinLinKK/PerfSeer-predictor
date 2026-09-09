#!/usr/bin/env python3
"""Verify the active PerfSeer v2 student and teacher architecture contracts."""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src/perfseer-optimized/model.py"
CONFIGS = [
    {
        "name": "student",
        "path": ROOT / "src/perfseer-optimized/configs/train_deploy_model/v2_student.yaml",
        "blocks": 2,
        "hidden": 192,
        "pool": "SynMM",
        "gate_shape": (1,),
    },
    {
        "name": "teacher",
        "path": ROOT / "src/perfseer-optimized/configs/train_hardware_teacher/v2_teacher.yaml",
        "blocks": 8,
        "hidden": 1024,
        "pool": "AttentionGraphPool",
        "gate_shape": (1024,),
    },
]


def load_model_module():
    spec = importlib.util.spec_from_file_location("perfseer_model_verify", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA verifier requires an allocated NVIDIA GPU.")
    module = load_model_module()
    device = torch.device("cuda")
    results = []
    for expected in CONFIGS:
        model_values = yaml.safe_load(expected["path"].read_text(encoding="utf8"))["model"].copy()
        model_values.pop("name")
        model_values.update(node_dim=78, edge_dim=16, global_dim=164, num_outputs=6)
        config = module.SeerNetConfig.from_dict(model_values)
        model = module.SeerNetMulti(config).to(device).eval()

        assert len(model.trunk.blocks) == expected["blocks"]
        assert len(model.heads) == 6
        assert all(type(block.agg_u).__name__ == expected["pool"] for block in model.trunk.blocks)
        assert all(tuple(gates["v"].logit.shape) == expected["gate_shape"] for gates in model.trunk.gates)

        data = SimpleNamespace(
            x=torch.randn(4, 78, device=device),
            edge_index=torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long, device=device),
            edge_attr=torch.randn(6, 16, device=device),
            u=torch.randn(1, 164, device=device),
            batch=torch.zeros(4, dtype=torch.long, device=device),
            num_graphs=1,
        )
        with torch.no_grad():
            output = model(data)
        assert tuple(output.shape) == (1, 6)
        results.append(
            {
                "model": expected["name"],
                "blocks": expected["blocks"],
                "hidden": expected["hidden"],
                "pool": expected["pool"],
                "gate_shape": list(expected["gate_shape"]),
                "output_shape": list(output.shape),
                "parameters": module.count_parameters(model),
            }
        )
        del model, data, output
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps({"cuda_device": torch.cuda.get_device_name(0), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
