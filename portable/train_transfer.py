#!/usr/bin/env python3
"""Selected T7-only A100-to-A10 transfer from an explicit P_A100 warm start."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import main
if __name__ == "__main__":
    main("T7")
