#!/usr/bin/env python3
"""Full-data P1 target continuation from an explicit P_A100 warm start."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import main
if __name__ == "__main__":
    main("full")
