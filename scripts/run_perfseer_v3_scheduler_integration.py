#!/usr/bin/env python3
"""Exercise the versioned scheduler wrapper and print its structured result."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.graph_ir_v3 import GraphIRV3
from perfseer_v3.runtime import PerfSeerV3Runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args(argv)
    result = PerfSeerV3Runtime(args.artifact).predict_graph(GraphIRV3.load(args.graph))
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
