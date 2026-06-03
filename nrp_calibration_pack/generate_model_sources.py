"""CLI entrypoint for generating the NRP calibration source-model pack."""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nrp_calibration_pack.build_pack import main  # noqa: E402


if __name__ == "__main__":
    main()
