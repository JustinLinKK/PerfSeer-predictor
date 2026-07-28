"""Validate and re-hash a deterministic PerfSeer GraphIRV3 JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.graph_ir_v3 import GraphIRV3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    args = parser.parse_args(argv)
    graph = GraphIRV3.load(args.graph)
    print(
        f"valid graph_sha256={graph.graph_sha256} "
        f"nodes={len(graph.nodes)} tensor_edges={len(graph.tensor_edges)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

